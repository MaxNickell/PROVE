"""
Attribute Agent for PROVE pipeline.
Uses LLM agent to orchestrate iterative information gathering from Qwen VL,
then generates binary verification questions for probability extraction.

Architecture:
1. Agent analyzes attribute subquery and referenced objects
2. Agent decides: Need more visual information? → Ask Qwen VL
3. Agent accumulates knowledge through Q&A with Qwen
4. When ready: Agent generates binary questions for verification
5. Binary questions → Qwen with logits → Probability extraction
"""

import os
from typing import List, Dict, Tuple, Optional
from PIL import Image
from dataclasses import dataclass, field
from collections import defaultdict

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ObjectDetection, AttributeData, AttributeValue, ImageData
from src.core.probability import get_verifier_probability
from src.language.output_models import AgentDecision, QwenInformationRequest, BinaryAttributeQuestion
from src.pipeline.base_agent import BaseVerificationAgent


@dataclass
class AttributeResult:
    """Intermediate result for a single attribute value extraction."""
    object_id: str  # Full object ID (e.g., "field_a_0")
    attribute_class: str  # e.g., "color", "size"
    attribute_value: str  # e.g., "brown", "large"
    confidence: float  # probability from verification


class AttributeAgentError(RuntimeError):
    """Custom exception for attribute agent failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


@dataclass
class AgentState:
    """
    Tracks agent's reasoning and information gathering process.
    Maintains conversation memory across agent loop iterations.
    """
    original_question: str
    referenced_objects: List[str]
    target_claims: Dict[str, List[str]] = field(default_factory=dict)  # NEW: What must be verified from subquestion

    # Information gathered from Qwen VL
    qwen_qa_history: List[Dict[str, str]] = field(default_factory=list)
    information_gathered: Dict[str, str] = field(default_factory=dict)  # {object_id: description}

    # Final binary questions for verification
    binary_questions: List[BinaryAttributeQuestion] = field(default_factory=list)

    # Agent's reasoning trace (for debugging/explainability)
    reasoning_trace: List[str] = field(default_factory=list)

    def add_qwen_interaction(self, request: QwenInformationRequest, answer: str):
        """Record Q&A interaction with Qwen VL."""
        self.qwen_qa_history.append({
            "object_id": request.object_id,
            "question": request.question,
            "answer": answer,
            "reasoning": request.reasoning
        })

        # Update information gathered for this object
        if request.object_id in self.information_gathered:
            self.information_gathered[request.object_id] += f" | {answer}"
        else:
            self.information_gathered[request.object_id] = answer

    def add_reasoning(self, reasoning: str):
        """Add agent reasoning step to trace."""
        self.reasoning_trace.append(reasoning)


class AttributeAgent(BaseVerificationAgent[BinaryAttributeQuestion]):
    """
    Attribute extraction agent using LLM orchestration.

    The agent follows an iterative loop:
    1. Analyze current knowledge state
    2. Decide: Need more info? → Ask Qwen VL
    3. Have enough info? → Generate binary questions
    4. Verify binary questions → Extract probabilities
    """

    def __init__(self, max_qwen_calls: int = 15, debug: bool = False):
        """
        Initialize attribute agent.

        Args:
            max_qwen_calls: Maximum Qwen VL calls per subquery (prevents infinite loops)
            debug: If True, saves cropped images and prints detailed verification info
        """
        super().__init__()  # Initialize BaseVerificationAgent
        self.max_qwen_calls = max_qwen_calls
        self.debug = debug

        # Create debug directory if needed
        if self.debug:
            os.makedirs("debug_attributes", exist_ok=True)

    def _extract_target_claims_from_subquestion(self, subquestion: str) -> Dict[str, List[str]]:
        """
        Extract the specific attribute values that need to be verified from subquestion.

        Examples:
        - "Is the shirt black?" → {"attribute_values": ["black"]}
        - "Is the dog brown or tan?" → {"attribute_values": ["brown", "tan"]}
        - "Does the car have a metallic finish?" → {"attribute_values": ["metallic"]}

        Args:
            subquestion: Natural language subquestion

        Returns:
            Dict with "attribute_values" key containing list of values to verify
        """
        return self._extract_target_claims_generic(
            subquestion=subquestion,
            claim_type="attribute values",
            claim_key="attribute_values",
            examples=[
                '"Is the shirt black?" → {"attribute_values": ["black"]}',
                '"Is the dog brown or tan?" → {"attribute_values": ["brown", "tan"]}',
                '"Are the walls white?" → {"attribute_values": ["white"]}',
                '"Is the car metallic red?" → {"attribute_values": ["metallic red"]}'
            ]
        )

    def _validate_binary_questions(
        self,
        binary_questions: List[BinaryAttributeQuestion],
        target_claims: Dict[str, List[str]],
        subquestion: str
    ) -> bool:
        """
        Validate that binary questions include DIRECT verification of target claims.

        Args:
            binary_questions: Generated binary questions
            target_claims: Target claims extracted from subquestion
            subquestion: Original subquestion

        Returns:
            bool: True if all target claims are covered, False otherwise
        """
        return self._validate_binary_questions_generic(
            binary_questions=binary_questions,
            target_claims=target_claims,
            claim_key="attribute_values",
            value_extractor=lambda q: q.attribute_value,
            subquestion=subquestion
        )

    def _generate_fallback_questions(
        self,
        target_claims: Dict[str, List[str]],
        objects: List[str],
        existing_questions: List[BinaryAttributeQuestion]
    ) -> List[BinaryAttributeQuestion]:
        """
        Generate minimal binary questions to directly verify missing target claims.

        Args:
            target_claims: Target claims that need verification
            objects: List of relevant object IDs
            existing_questions: Already generated questions

        Returns:
            List of fallback binary questions for missing targets
        """
        if not objects:
            return []

        # Use base class method to get missing claim values
        missing_values = self._get_missing_claims(
            target_claims=target_claims,
            claim_key="attribute_values",
            existing_questions=existing_questions,
            value_extractor=lambda q: q.attribute_value
        )

        if not missing_values:
            return []

        print(f"  → Generating fallback questions for missing values: {missing_values}")

        fallback_questions = []
        for obj_id in objects:
            for value in missing_values:
                # Extract object type from ID (e.g., "shirt_a_1" → "shirt")
                obj_type = obj_id.split('_')[0]

                fallback_questions.append(BinaryAttributeQuestion(
                    object_id=obj_id,
                    attribute_class="color",  # Default to color, could be more sophisticated
                    attribute_value=value,
                    binary_question=f"Is the {obj_type} {value}?"
                ))

        return fallback_questions

    def process_attribute_subquestions(
        self,
        attribute_subquestions: List[BinarySubquestion],
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> Dict[str, int]:
        """
        Process attribute subquestions using agentic approach.

        Args:
            attribute_subquestions: List of attribute binary subquestions
            image_paths: Dict mapping image_id to file path
            images: ImageData structure containing objects per image

        Returns:
            Dict[str, int]: Summary of attributes extracted per image

        Raises:
            AttributeAgentError: If processing fails
        """
        try:
            total_attributes_extracted = 0
            attributes_per_image = {}

            # Collect all attribute results first
            all_results = []

            # Process each attribute subquestion with agent
            for i, subquestion in enumerate(attribute_subquestions, 1):
                if subquestion.subquestion_type != "attribute":
                    continue

                print(f"\n  Processing subquestion {i}/{len(attribute_subquestions)}: {subquestion.question}")

                # Run agentic extraction for this subquestion
                results = self.process_single_subquestion(subquestion, image_paths, images)
                all_results.extend(results)

                print(f"    ✓ Extracted {len(results)} attribute values")

            # Group results by (image_id, object_index) and construct AttributeData
            grouped = defaultdict(lambda: defaultdict(list))

            for attr_result in all_results:
                # Parse object_id to get image_id and object_index
                parts = attr_result.object_id.split('_')
                if len(parts) < 3:
                    continue

                image_letter = parts[-2]
                object_index = int(parts[-1])
                image_id = f"image_{image_letter}"

                # Group by (image_id, object_index, attribute_class)
                grouped[(image_id, object_index)][attr_result.attribute_class].append(
                    AttributeValue(value=attr_result.attribute_value, confidence=attr_result.confidence)
                )

            # Store in knowledge base using proper API
            for (image_id, object_index), attributes_dict in grouped.items():
                attr_data = AttributeData(attributes=attributes_dict)
                images[image_id].attributes[object_index] = attr_data
                attributes_per_image[image_id] = attributes_per_image.get(image_id, 0) + len(attributes_dict)
                total_attributes_extracted += len(attributes_dict)

            return attributes_per_image

        except Exception as e:
            raise AttributeAgentError(f"Failed to process attribute subquestions: {str(e)}")

    def process_single_subquestion(
        self,
        subquestion: BinarySubquestion,
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> List[AttributeResult]:
        """
        Process single attribute subquestion using agentic loop.
        NOW: Discovers relevant objects from natural language question.

        Args:
            subquestion: Attribute subquestion to process (no object IDs)
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            List[AttributeResult]: Extracted attribute results with object IDs
        """
        # 1. Extract target claims from subquestion (what must be verified)
        target_claims = self._extract_target_claims_from_subquestion(subquestion.question)
        if target_claims.get("attribute_values"):
            print(f"    Target claims to verify: {target_claims['attribute_values']}")

        # 2. Discover which objects are relevant to this question
        print(f"    Discovering relevant objects for: {subquestion.question}")
        relevant_objects = self._discover_relevant_objects(subquestion.question, images)
        print(f"    → Discovered {len(relevant_objects)} relevant objects: {relevant_objects}")

        # 3. Initialize agent state with discovered objects AND target claims
        state = AgentState(
            original_question=subquestion.question,
            referenced_objects=relevant_objects,
            target_claims=target_claims
        )

        print(f"    Starting agentic loop (max {self.max_qwen_calls} Qwen calls)...")

        # 2. Agentic planning loop
        for iteration in range(self.max_qwen_calls):
            print(f"      Iteration {iteration + 1}: Agent deciding next action...")

            # Agent decides: ask Qwen for info OR generate binary questions
            decision = self._agent_decide_next_action(state)
            state.add_reasoning(decision.reasoning)

            if decision.action == "ask_qwen":
                # Execute Qwen VL query
                print(f"        → Asking Qwen: {decision.qwen_request.question}")
                print(f"          Object: {decision.qwen_request.object_id}")

                answer = self._ask_qwen_vl(decision.qwen_request, image_paths, images)
                state.add_qwen_interaction(decision.qwen_request, answer)

                print(f"          Answer: {answer[:100]}...")

            elif decision.action == "generate_binary_questions":
                # Agent has enough info, generate final verification questions
                print(f"        → Agent ready! Generating {len(decision.binary_questions)} binary questions")
                state.binary_questions = decision.binary_questions

                # Validate that target claims are covered
                if not self._validate_binary_questions(
                    state.binary_questions,
                    target_claims,
                    subquestion.question
                ):
                    # Generate fallback questions for missing target claims
                    fallback_questions = self._generate_fallback_questions(
                        target_claims,
                        relevant_objects,
                        state.binary_questions
                    )
                    state.binary_questions.extend(fallback_questions)
                    print(f"        → Added {len(fallback_questions)} fallback questions for missing target claims")

                for bq in state.binary_questions:
                    print(f"          • {bq.binary_question}")

                break

        if not state.binary_questions:
            print(f"    Warning: Agent did not generate binary questions after {self.max_qwen_calls} iterations")
            # Generate fallback questions as last resort
            if target_claims.get("attribute_values") and relevant_objects:
                print(f"    Generating fallback questions for target claims...")
                state.binary_questions = self._generate_fallback_questions(
                    target_claims,
                    relevant_objects,
                    []
                )
            if not state.binary_questions:
                return []

        # 3. Verify binary questions and extract probabilities
        print(f"    Verifying {len(state.binary_questions)} binary questions...")
        results = []

        for bq in state.binary_questions:
            probability = self._verify_binary_question(bq, image_paths, images)

            results.append(AttributeResult(
                object_id=bq.object_id,
                attribute_class=bq.attribute_class,
                attribute_value=bq.attribute_value,
                confidence=probability
            ))

            print(f"      {bq.object_id}.{bq.attribute_class} = {bq.attribute_value} (p={probability:.3f})")

        return results

    def _discover_relevant_objects(
        self,
        question: str,
        images: Dict[str, ImageData]
    ) -> List[str]:
        """
        LLM analyzes natural language question to discover which object IDs are relevant.

        Args:
            question: Natural language attribute question (e.g., "Are the buffalos brown?")
            images: All detected objects with IDs

        Returns:
            List of relevant object IDs (e.g., ["buffalo_a_3", "buffalo_b_4"])
        """
        llm_client = self.model_manager.get_llm_client()

        # Build complete object list with IDs
        object_lines = []
        for image_id, image_data in sorted(images.items()):
            object_lines.append(f"\n{image_id.upper()}:")
            for obj in image_data.objects:
                image_letter = image_id.replace("image_", "")
                obj_id = f"{obj.label}_{image_letter}_{obj.object_id}"
                object_lines.append(f"  - {obj_id} ({obj.label})")

        object_context = "\n".join(object_lines)

        messages = [{
            "role": "system",
            "content": "You analyze questions to identify which detected objects are relevant."
        }, {
            "role": "user",
            "content": f"""QUESTION: {question}

DETECTED OBJECTS:
{object_context}

TASK: Identify which object IDs are relevant to answer this question.

Rules:
1. Parse the question to understand what object classes are mentioned
2. Include ALL instances of relevant classes (e.g., if question asks about "buffalos", include all buffalo IDs)
3. For comparative questions, include objects from multiple images
4. Return ONLY the object IDs, no explanations

Output JSON:
{{
  "object_ids": ["buffalo_a_3", "buffalo_b_4", ...]
}}"""
        }]

        response = llm_client.discover_objects(messages)
        return response.object_ids

    def _agent_decide_next_action(self, state: AgentState) -> AgentDecision:
        """
        LLM agent decides: ask Qwen for more info OR generate binary questions.

        Args:
            state: Current agent state with conversation history

        Returns:
            AgentDecision: Agent's decision with action and rationale
        """
        llm_client = self.model_manager.get_llm_client()

        # Format gathered information for prompt
        info_summary = "\n".join([
            f"  - {obj_id}: {info}"
            for obj_id, info in state.information_gathered.items()
        ]) if state.information_gathered else "  (No information gathered yet)"

        # Format Q&A history
        qa_history = "\n".join([
            f"  Q: {qa['question']} (Object: {qa['object_id']})\n  A: {qa['answer']}"
            for qa in state.qwen_qa_history
        ]) if state.qwen_qa_history else "  (No questions asked yet)"

        # Format target claims
        target_claims_text = ""
        if state.target_claims.get("attribute_values"):
            values_str = ', '.join(state.target_claims["attribute_values"])
            target_claims_text = f"\n**Target Claims to Verify**: {values_str}\n(These specific values MUST be directly verified in your binary questions)"

        prompt = f"""You are an intelligent agent extracting visual attributes from images to answer complex questions.

**Original Question**: {state.original_question}

**Referenced Objects**: {', '.join(state.referenced_objects)}
{target_claims_text}

**Information Gathered So Far**:
{info_summary}

**Previous Q&A with Vision Model (Qwen VL)**:
{qa_history}

---

## Your Task

Decide if you need MORE information from the vision model, OR if you have ENOUGH information to generate final binary verification questions.

## Decision Rules

### ❓ **Choose "ask_qwen"** if you need to:
- Determine what attribute value an object has (color, size, texture, shape, orientation, etc.)
- Get visual details you cannot infer from context
- Learn specific properties about object appearance
- Understand visual attributes for comparison

### ✅ **Choose "generate_binary_questions"** when you:
- Have gathered enough visual information about the objects
- **CRITICAL: Can directly verify the target claims (if specified)**
- Can formulate specific Yes/No questions
- Even if visual evidence suggests different values, you can still generate verification questions

## CRITICAL RULE: Subquestion-Aligned Verification

Your binary questions MUST include direct verification of target claims, even if:
- Visual evidence suggests a different value
- The claim appears unlikely based on what you observed
- You think the answer will be "No"

**Why:** The ProbLog reasoner needs the probability that the claim is TRUE, not just probabilities for what you observed.

**Example:**
  Original Question: "Is the shirt black?"
  Target Claim: "black"
  Visual Evidence: "The shirt appears to be green"

  CORRECT binary questions:
    ✓ "Is the shirt black?" (directly verifies the target claim - REQUIRED!)
    ✓ "Is the shirt green?" (confirms alternative, helps reasoning - OPTIONAL)

  INCORRECT binary questions:
    ✗ Only "Is the shirt green?" (doesn't verify the claim!)

## Important Guidelines

1. **Binary questions must be SPECIFIC and use NATURAL LANGUAGE**:
   - Good: "Is the dog brown?"
   - Bad: "Is dog_a_1 brown?" (don't use object IDs in question text)
   - Bad: "What color is the dog?" (not a binary question)

2. **Binary questions reference ONE object and ONE attribute value**
   - The object_id field identifies which object (e.g., "dog_a_1")
   - The binary_question uses natural language (e.g., "Is the dog brown?")

3. **ALWAYS include questions for target claims (if specified)**
   - Even if visual evidence suggests otherwise
   - This ensures we can compute probabilities for the actual question being asked

4. **For comparison questions** (e.g., "same color"), generate:
   - Questions for each object's actual attribute ("Is the dog brown?")
   - Questions for comparison candidates ("Is the cat brown?", "Is the cat tan?")

5. **Work with ANY attribute category**: color, size, texture, shape, orientation, position, etc.

---

## Output Format

**If you need more information:**
```json
{{
  "action": "ask_qwen",
  "reasoning": "I need to know the color of dog_a_1 to compare it with other dogs",
  "qwen_request": {{
    "object_id": "dog_a_1",
    "question": "What color is this dog?",
    "reasoning": "Need color information for comparison"
  }}
}}
```

**If ready to generate binary questions:**
```json
{{
  "action": "generate_binary_questions",
  "reasoning": "I now know: dog_a_1 is brown, dog_a_2 is tan, dog_b_1 is brown, dog_b_2 is white. I can formulate binary questions to verify these attributes and determine if dogs in image_1 have same colors as dogs in image_2.",
  "binary_questions": [
    {{
      "object_id": "dog_a_1",
      "attribute_class": "color",
      "attribute_value": "brown",
      "binary_question": "Is the dog brown?"
    }},
    {{
      "object_id": "dog_a_2",
      "attribute_class": "color",
      "attribute_value": "tan",
      "binary_question": "Is the dog tan?"
    }}
  ]
}}
```

Respond in strict JSON format only."""

        response = llm_client.chat_with_validation(
            messages=[{"role": "user", "content": prompt}],
            output_model=AgentDecision,
            temperature=0.3
        )

        return response

    def _ask_qwen_vl(
        self,
        request: QwenInformationRequest,
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> str:
        """
        Ask Qwen VL an open-ended question about a specific object.

        Args:
            request: Information request with object_id and question
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            str: Qwen's answer to the question
        """
        qwen_client = self.model_manager.get_qwen_vl()

        # Find object and its image
        obj, image_id = self._find_object_by_id(request.object_id, images)
        if not obj or not image_id:
            return f"Error: Object {request.object_id} not found"

        # Load image
        image = Image.open(image_paths[image_id])

        # Crop to object with margin - focuses VLM attention (consistent with verification)
        cropped_image = self._crop_with_margin(image, obj.bbox, margin=0.15)

        # Direct question (no bbox coordinates needed!)
        prompt = request.question

        # Run open-ended inference (no logits needed for information gathering)
        response, _ = qwen_client.run_inference_with_logits(cropped_image, prompt)

        return response.strip()

    def _crop_with_margin(
        self,
        image: Image.Image,
        bbox: List[float],
        margin: float = 0.15,
        min_size: int = 32
    ) -> Image.Image:
        """
        Crop image to bounding box with percentage margin on all sides.
        Ensures minimum dimensions for VLM processing.

        Args:
            image: PIL Image to crop
            bbox: Bounding box [x1, y1, x2, y2]
            margin: Percentage margin to add (0.15 = 15% of bbox size)
            min_size: Minimum dimension in pixels (default 32, Qwen needs 28)

        Returns:
            Cropped PIL Image with margin (guaranteed >= min_size x min_size)
        """
        x1, y1, x2, y2 = bbox
        width, height = image.size

        # Calculate margin in pixels
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        margin_x = bbox_width * margin
        margin_y = bbox_height * margin

        # Apply margin with bounds checking
        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(width, x2 + margin_x)
        crop_y2 = min(height, y2 + margin_y)

        # Check if crop meets minimum size requirements
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        # Expand if too small (maintain centering)
        if crop_width < min_size:
            deficit = min_size - crop_width
            expand_left = deficit / 2
            expand_right = deficit / 2

            crop_x1 = max(0, crop_x1 - expand_left)
            crop_x2 = min(width, crop_x2 + expand_right)

            # If hit boundary, expand other side more
            if crop_x1 == 0:
                crop_x2 = min(width, crop_x2 + expand_left)
            if crop_x2 == width:
                crop_x1 = max(0, crop_x1 - expand_right)

        if crop_height < min_size:
            deficit = min_size - crop_height
            expand_top = deficit / 2
            expand_bottom = deficit / 2

            crop_y1 = max(0, crop_y1 - expand_top)
            crop_y2 = min(height, crop_y2 + expand_bottom)

            # If hit boundary, expand other side more
            if crop_y1 == 0:
                crop_y2 = min(height, crop_y2 + expand_top)
            if crop_y2 == height:
                crop_y1 = max(0, crop_y1 - expand_bottom)

        return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def _verify_binary_question(
        self,
        bq: BinaryAttributeQuestion,
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> float:
        """
        Verify binary question using Qwen VL with logit probability extraction.

        Args:
            bq: Binary attribute question
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            float: Probability that the statement is true (0.0 to 1.0)
        """
        qwen_client = self.model_manager.get_qwen_vl()

        # Find object
        obj, image_id = self._find_object_by_id(bq.object_id, images)
        if not obj or not image_id:
            print(f"    Warning: Object {bq.object_id} not found for verification")
            return 0.5  # Default probability

        # Load image
        image = Image.open(image_paths[image_id])

        # Crop to object with margin - removes distracting context
        cropped_image = self._crop_with_margin(image, obj.bbox, margin=0.15)

        # Simple, direct prompt (NO bbox coordinates in text!)
        prompt = f"{bq.binary_question} Answer Yes or No.\n\nAnswer:"

        # Run verification with logits
        response, logits = qwen_client.run_inference_with_logits(cropped_image, prompt)

        # Extract probability using verbalizer summing (Yes/No logits)
        probability = get_verifier_probability(
            logits,
            response,
            qwen_client.processor.tokenizer,
            debug=self.debug  # Pass debug flag for detailed logit breakdown
        )

        # DEBUG: Save crops and print full verification details
        if self.debug:
            # Save the cropped image
            debug_filename = f"debug_attributes/{bq.object_id}__{bq.attribute_class}__{bq.attribute_value}.png"
            cropped_image.save(debug_filename)

            # Print full debugging info
            print("\n" + "=" * 80)
            print(f"🔍 ATTRIBUTE VERIFICATION DEBUG")
            print("=" * 80)
            print(f"Object: {bq.object_id} ({obj.label})")
            print(f"Attribute: {bq.attribute_class} = {bq.attribute_value}")
            print(f"Binary Question: {bq.binary_question}")
            print(f"\nOriginal bbox: {obj.bbox}")
            print(f"Cropped image size: {cropped_image.size}")
            print(f"Cropped image saved: {debug_filename}")
            print(f"\n--- FULL PROMPT TO VLM ---")
            print(prompt)
            print(f"--- END PROMPT ---")
            print(f"\n--- VLM RESPONSE ---")
            print(f'"{response}"')
            print(f"--- END RESPONSE ---")
            print(f"\nExtracted Probability: {probability:.4f}")
            print("=" * 80 + "\n")

        return probability

    def _find_object_by_id(
        self,
        object_id: str,
        images: Dict[str, ImageData]
    ) -> Tuple[Optional[ObjectDetection], Optional[str]]:
        """
        Find object by ID across all images.

        Args:
            object_id: Object ID to find (e.g., "dog_a_1")
            images: ImageData structure

        Returns:
            Tuple[ObjectDetection, image_id] or (None, None) if not found
        """
        try:
            # Parse object ID format: label_imageid_objectid (e.g., "sky_a_3")
            parts = object_id.split('_')
            if len(parts) < 3:
                return None, None

            image_letter = parts[-2]  # Second to last part (e.g., "a")
            object_index = int(parts[-1])

            # Convert image letter back to full key (e.g., "a" -> "image_a")
            image_id = f"image_{image_letter}"

            if image_id in images:
                objects = images[image_id].objects
                if object_index < len(objects):
                    obj = objects[object_index]
                    # Verify the object_id field matches
                    if obj.object_id == object_index:
                        return obj, image_id

            return None, None
        except (ValueError, IndexError):
            return None, None
