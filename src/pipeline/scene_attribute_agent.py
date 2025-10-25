"""
Scene Attribute Agent for PROVE pipeline.
Uses LLM agent to orchestrate iterative information gathering from Qwen VL for ENTIRE SCENES,
then generates binary verification questions for probability extraction.

Architecture (mirroring AttributeAgent but for scenes, not objects):
1. Agent analyzes scene attribute subquestion and referenced images
2. Agent decides: Need more visual information about a scene? → Ask Qwen VL
3. Agent accumulates knowledge through Q&A with Qwen about full images
4. When ready: Agent generates binary questions for verification
5. Binary questions → Qwen with FULL IMAGES → Logit probability extraction

KEY DIFFERENCE from AttributeAgent:
- Works with FULL IMAGES (no bounding boxes, no specific objects)
- Asks about SCENE-LEVEL attributes (environment, lighting, weather, vegetation, etc.)
- image_id instead of object_id throughout
"""

import os
from typing import List, Dict, Optional
from PIL import Image
from dataclasses import dataclass, field
from collections import defaultdict

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ImageData
from src.core.probability import get_verifier_probability
from src.language.output_models import (
    SceneAgentDecision,
    QwenSceneInformationRequest,
    BinarySceneAttributeQuestion
)


@dataclass
class SceneAttributeResult:
    """Intermediate result for a single scene attribute value extraction."""
    image_id: str  # e.g., "image_a"
    attribute_class: str  # e.g., "environment_type", "lighting"
    attribute_value: str  # e.g., "outdoor", "bright"
    confidence: float  # probability from verification


class SceneAttributeAgentError(RuntimeError):
    """Custom exception for scene attribute agent failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


@dataclass
class SceneAgentState:
    """
    Tracks agent's reasoning and information gathering process for SCENES.
    Maintains conversation memory across agent loop iterations.
    """
    original_question: str
    referenced_images: List[str]  # ["image_a", "image_b"]

    # Information gathered from Qwen VL about entire scenes
    qwen_qa_history: List[Dict[str, str]] = field(default_factory=list)
    information_gathered: Dict[str, str] = field(default_factory=dict)  # {image_id: description}

    # Final binary questions for verification
    binary_questions: List[BinarySceneAttributeQuestion] = field(default_factory=list)

    # Agent's reasoning trace (for debugging/explainability)
    reasoning_trace: List[str] = field(default_factory=list)

    def add_qwen_interaction(self, request: QwenSceneInformationRequest, answer: str):
        """Record Q&A interaction with Qwen VL about a scene."""
        self.qwen_qa_history.append({
            "image_id": request.image_id,
            "question": request.question,
            "answer": answer,
            "reasoning": request.reasoning
        })

        # Update information gathered for this image
        if request.image_id in self.information_gathered:
            self.information_gathered[request.image_id] += f" | {answer}"
        else:
            self.information_gathered[request.image_id] = answer

    def add_reasoning(self, reasoning: str):
        """Add agent reasoning step to trace."""
        self.reasoning_trace.append(reasoning)


class SceneAttributeAgent:
    """
    Scene attribute extraction agent using LLM orchestration.

    The agent follows an iterative loop for ENTIRE SCENES:
    1. Analyze current knowledge state about scenes
    2. Decide: Need more info about a scene? → Ask Qwen VL (with full image)
    3. Have enough info? → Generate binary questions
    4. Verify binary questions using full images → Extract probabilities
    """

    def __init__(self, max_qwen_calls: int = 15, debug: bool = False):
        """
        Initialize scene attribute agent.

        Args:
            max_qwen_calls: Maximum Qwen VL calls per subquestion (prevents infinite loops)
            debug: If True, saves full images and prints detailed verification info
        """
        self.model_manager = ModelManager()
        self.max_qwen_calls = max_qwen_calls
        self.debug = debug

        # Create debug directory if needed
        if self.debug:
            os.makedirs("debug_scene_attributes", exist_ok=True)

    def process_scene_attribute_subquestions(
        self,
        scene_attribute_subquestions: List[BinarySubquestion],
        image_paths: Dict[str, str],
        images: Dict[str, ImageData],
        image_contexts: Dict[str, str] = None  # Optional for compatibility
    ) -> Dict[str, int]:
        """
        Process scene attribute subquestions using agentic approach.

        Args:
            scene_attribute_subquestions: List of scene_attribute binary subquestions
            image_paths: Dict mapping image_id to file path
            images: ImageData structure containing objects per image
            image_contexts: Optional (not used by agent, kept for API compatibility)

        Returns:
            Dict[str, int]: Summary of scene attributes extracted per image

        Raises:
            SceneAttributeAgentError: If processing fails
        """
        try:
            total_scene_attributes_extracted = 0
            scene_attributes_per_image = {}

            # Collect all scene attribute results first
            all_results = []

            # Process each scene attribute subquestion with agent
            for i, subquestion in enumerate(scene_attribute_subquestions, 1):
                if subquestion.subquestion_type != "scene_attribute":
                    continue

                print(f"\n  Processing subquestion {i}/{len(scene_attribute_subquestions)}: {subquestion.question}")

                # Run agentic extraction for this subquestion
                results = self.process_single_subquestion(subquestion, image_paths, images)
                all_results.extend(results)

                print(f"    ✓ Extracted {len(results)} scene attribute values")

            # Group results by (image_id, attribute_class)
            grouped = defaultdict(list)

            for scene_result in all_results:
                # Group by (image_id, attribute_class)
                key = (scene_result.image_id, scene_result.attribute_class)
                grouped[key].append({
                    "value": scene_result.attribute_value,
                    "confidence": scene_result.confidence
                })

            # Store in knowledge base using proper API
            for (image_id, attribute_class), attribute_values in grouped.items():
                # Initialize scene_attributes if needed
                if not hasattr(images[image_id], 'scene_attributes') or images[image_id].scene_attributes is None:
                    images[image_id].scene_attributes = {}

                # Store scene attributes
                if attribute_class not in images[image_id].scene_attributes:
                    images[image_id].scene_attributes[attribute_class] = []

                images[image_id].scene_attributes[attribute_class].extend(attribute_values)

                scene_attributes_per_image[image_id] = scene_attributes_per_image.get(image_id, 0) + 1
                total_scene_attributes_extracted += 1

            return scene_attributes_per_image

        except Exception as e:
            raise SceneAttributeAgentError(f"Failed to process scene attribute subquestions: {str(e)}")

    def process_single_subquestion(
        self,
        subquestion: BinarySubquestion,
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> List[SceneAttributeResult]:
        """
        Process single scene attribute subquestion using agentic loop.
        Discovers relevant IMAGES (not objects) from natural language question.

        Args:
            subquestion: Scene attribute subquestion to process (no image IDs in question)
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            List[SceneAttributeResult]: Extracted scene attribute results with image IDs
        """
        # 1. Discover which images are relevant to this question
        print(f"    Discovering relevant images for: {subquestion.question}")
        relevant_images = self._discover_relevant_images(subquestion.question, images)
        print(f"    → Discovered {len(relevant_images)} relevant images: {relevant_images}")

        # 2. Initialize agent state with discovered images
        state = SceneAgentState(
            original_question=subquestion.question,
            referenced_images=relevant_images
        )

        print(f"    Starting agentic loop (max {self.max_qwen_calls} Qwen calls)...")

        # 3. Agentic planning loop
        for iteration in range(self.max_qwen_calls):
            print(f"      Iteration {iteration + 1}: Agent deciding next action...")

            # Agent decides: ask Qwen for info OR generate binary questions
            decision = self._agent_decide_next_action(state)
            state.add_reasoning(decision.reasoning)

            if decision.action == "ask_qwen":
                # Execute Qwen VL query on FULL IMAGE
                print(f"        → Asking Qwen: {decision.qwen_request.question}")
                print(f"          Image: {decision.qwen_request.image_id}")

                answer = self._ask_qwen_vl(decision.qwen_request, image_paths)
                state.add_qwen_interaction(decision.qwen_request, answer)

                print(f"          Answer: {answer[:100]}...")

            elif decision.action == "generate_binary_questions":
                # Agent has enough info, generate final verification questions
                print(f"        → Agent ready! Generating {len(decision.binary_questions)} binary questions")
                state.binary_questions = decision.binary_questions

                for bq in state.binary_questions:
                    print(f"          • {bq.binary_question}")

                break

        if not state.binary_questions:
            print(f"    Warning: Agent did not generate binary questions after {self.max_qwen_calls} iterations")
            return []

        # 4. Verify binary questions and extract probabilities
        print(f"    Verifying {len(state.binary_questions)} binary questions...")
        results = []

        for bq in state.binary_questions:
            probability = self._verify_binary_question(bq, image_paths)

            results.append(SceneAttributeResult(
                image_id=bq.image_id,
                attribute_class=bq.attribute_class,
                attribute_value=bq.attribute_value,
                confidence=probability
            ))

            print(f"      {bq.image_id}.{bq.attribute_class} = {bq.attribute_value} (p={probability:.3f})")

        return results

    def _discover_relevant_images(
        self,
        question: str,
        images: Dict[str, ImageData]
    ) -> List[str]:
        """
        LLM analyzes natural language question to discover which IMAGE IDs are relevant.

        Args:
            question: Natural language scene attribute question (e.g., "Are both images outdoor?")
            images: All available images with IDs

        Returns:
            List of relevant image IDs (e.g., ["image_a", "image_b"])
        """
        llm_client = self.model_manager.get_llm_client()

        # Build list of available images
        image_list = "\n".join([f"  - {img_id}" for img_id in sorted(images.keys())])

        messages = [{
            "role": "system",
            "content": "You analyze questions to identify which images are relevant."
        }, {
            "role": "user",
            "content": f"""QUESTION: {question}

AVAILABLE IMAGES:
{image_list}

TASK: Identify which image IDs are relevant to answer this question.

Rules:
1. Parse the question to understand what images are mentioned
2. If question mentions "both images", "all images", or is comparative, include ALL images
3. If question mentions specific image (like "IMAGE_A", "the left image"), identify that image
4. Return ONLY the image IDs, no explanations

Examples:
- "Are both images outdoor?" → ["image_a", "image_b"]
- "Is IMAGE_A taken during daytime?" → ["image_a"]
- "Do all images show grass?" → ["image_a", "image_b"]

Output JSON:
{{
  "image_ids": ["image_a", "image_b", ...]
}}"""
        }]

        # Use image discovery method for images (not object discovery!)
        response = llm_client.discover_images(messages)
        return response.image_ids  # Returns list of image IDs

    def _agent_decide_next_action(self, state: SceneAgentState) -> SceneAgentDecision:
        """
        LLM agent decides: ask Qwen for more scene info OR generate binary questions.

        Args:
            state: Current agent state with conversation history

        Returns:
            SceneAgentDecision: Agent's decision with action and rationale
        """
        llm_client = self.model_manager.get_llm_client()

        # Format gathered information for prompt
        info_summary = "\n".join([
            f"  - {img_id}: {info}"
            for img_id, info in state.information_gathered.items()
        ]) if state.information_gathered else "  (No information gathered yet)"

        # Format Q&A history
        qa_history = "\n".join([
            f"  Q: {qa['question']} (Image: {qa['image_id']})\n  A: {qa['answer']}"
            for qa in state.qwen_qa_history
        ]) if state.qwen_qa_history else "  (No questions asked yet)"

        prompt = f"""You are an intelligent agent extracting visual scene attributes from images to answer complex questions.

**Original Question**: {state.original_question}

**Referenced Images**: {', '.join(state.referenced_images)}

**Information Gathered So Far**:
{info_summary}

**Previous Q&A with Vision Model (Qwen VL)**:
{qa_history}

---

## Your Task

Decide if you need MORE information from the vision model about a scene, OR if you have ENOUGH information to generate final binary verification questions.

## Decision Rules

### ❓ **Choose "ask_qwen"** if you need to:
- Determine scene-level attributes: environment type (indoor/outdoor), lighting (bright/dim), weather (sunny/rainy), vegetation (grass/trees), time of day
- Get visual details about the overall scene composition
- Learn about setting, atmosphere, or context
- Understand spatial arrangement or scene layout

### ✅ **Choose "generate_binary_questions"** when you:
- Know all relevant scene attribute values for all referenced images
- Can formulate specific Yes/No questions like "Is this an outdoor environment?"
- Have gathered enough information to answer the original question
- Can verify scene attributes with binary questions

## Important Guidelines

1. **Binary questions must be SPECIFIC and use NATURAL LANGUAGE**:
   - Good: "Is this an outdoor environment?"
   - Bad: "Is image_a outdoor?" (don't use image IDs in question text)
   - Bad: "What type of environment is this?" (not a binary question)

2. **Binary questions reference ONE image and ONE scene attribute value**
   - The image_id field identifies which image (e.g., "image_a")
   - The binary_question uses natural language (e.g., "Is this an outdoor environment?")

3. **Generate questions that collectively answer the original question**

4. **For comparison questions** (e.g., "same environment"), generate:
   - Questions for each image's actual attribute ("Is this outdoor?")
   - Questions for comparison candidates as needed

5. **Work with scene-level attributes**: environment_type, lighting, weather, vegetation, time_of_day, sky_color, setting, etc.

---

## Output Format

**If you need more information:**
```json
{{
  "action": "ask_qwen",
  "reasoning": "I need to know the environment type of image_a to compare it with image_b",
  "qwen_request": {{
    "image_id": "image_a",
    "question": "What type of environment is shown in this image?",
    "reasoning": "Need environment type information for comparison"
  }}
}}
```

**If ready to generate binary questions:**
```json
{{
  "action": "generate_binary_questions",
  "reasoning": "I now know: image_a is outdoor with grass, image_b is outdoor with grass. I can formulate binary questions to verify these scene attributes and determine if both images have the same environment.",
  "binary_questions": [
    {{
      "image_id": "image_a",
      "attribute_class": "environment_type",
      "attribute_value": "outdoor",
      "binary_question": "Is this an outdoor environment?"
    }},
    {{
      "image_id": "image_a",
      "attribute_class": "vegetation",
      "attribute_value": "grass",
      "binary_question": "Does this scene contain grass?"
    }}
  ]
}}
```

Respond in strict JSON format only."""

        response = llm_client.chat_with_validation(
            messages=[{"role": "user", "content": prompt}],
            output_model=SceneAgentDecision,
            temperature=0.3
        )

        return response

    def _ask_qwen_vl(
        self,
        request: QwenSceneInformationRequest,
        image_paths: Dict[str, str]
    ) -> str:
        """
        Ask Qwen VL an open-ended question about a FULL SCENE (entire image).

        Args:
            request: Information request with image_id and question
            image_paths: Image file paths

        Returns:
            str: Qwen's answer to the question
        """
        qwen_client = self.model_manager.get_qwen_vl()

        # Load FULL image (no cropping!)
        image = Image.open(image_paths[request.image_id])

        # Direct question about the scene
        prompt = request.question

        # Run open-ended inference (no logits needed for information gathering)
        response, _ = qwen_client.run_inference_with_logits(image, prompt)

        return response.strip()

    def _verify_binary_question(
        self,
        bq: BinarySceneAttributeQuestion,
        image_paths: Dict[str, str]
    ) -> float:
        """
        Verify binary scene question using Qwen VL with FULL IMAGE and logit probability extraction.

        Args:
            bq: Binary scene attribute question
            image_paths: Image file paths

        Returns:
            float: Probability that the statement is true (0.0 to 1.0)
        """
        qwen_client = self.model_manager.get_qwen_vl()

        # Load FULL image (no cropping!)
        image = Image.open(image_paths[bq.image_id])

        # Simple, direct prompt for scene verification
        prompt = f"{bq.binary_question} Answer Yes or No.\n\nAnswer:"

        # Run verification with logits
        response, logits = qwen_client.run_inference_with_logits(image, prompt)

        # Extract probability using verbalizer summing (Yes/No logits)
        probability = get_verifier_probability(
            logits,
            response,
            qwen_client.processor.tokenizer,
            debug=self.debug  # Pass debug flag for detailed logit breakdown
        )

        # DEBUG: Save full images and print full verification details
        if self.debug:
            # Save the full image
            debug_filename = f"debug_scene_attributes/{bq.image_id}__{bq.attribute_class}__{bq.attribute_value}.png"
            image.save(debug_filename)

            # Print full debugging info
            print("\n" + "=" * 80)
            print(f"🔍 SCENE ATTRIBUTE VERIFICATION DEBUG")
            print("=" * 80)
            print(f"Image: {bq.image_id}")
            print(f"Scene Attribute: {bq.attribute_class} = {bq.attribute_value}")
            print(f"Binary Question: {bq.binary_question}")
            print(f"\nFull image size: {image.size}")
            print(f"Full image saved: {debug_filename}")
            print(f"\n--- FULL PROMPT TO VLM ---")
            print(prompt)
            print(f"--- END PROMPT ---")
            print(f"\n--- VLM RESPONSE ---")
            print(f'"{response}"')
            print(f"--- END RESPONSE ---")
            print(f"\nExtracted Probability: {probability:.4f}")
            print("=" * 80 + "\n")

        return probability
