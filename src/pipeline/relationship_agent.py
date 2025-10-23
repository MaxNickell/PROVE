"""
Relationship Agent for PROVE pipeline.
Uses LLM agent to orchestrate iterative information gathering from Qwen VL about spatial
and interaction relationships, then generates binary verification questions.

Architecture (Mirrors attribute_agent.py):
1. LLM Reasoner: Analyzes relationship subquery and referenced objects
2. LLM Planner: Decides what relationship information to gather from VLM
3. VLM Perceiver: Describes spatial/interaction relationships with colored boxes
4. VLM Verifier: Verifies binary relationship questions with probabilities

Key Feature: Colored bounding boxes (RED for subject, BLUE for object) in both phases
"""

from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw
from dataclasses import dataclass, field
from collections import defaultdict

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ObjectDetection, IntraRelation, ImageData
from src.core.probability import get_verifier_probability
from src.language.output_models import RelationshipAgentDecision, QwenRelationshipRequest, BinaryRelationshipQuestion


@dataclass
class RelationshipResult:
    """Intermediate result for a single relationship extraction."""
    subject_id: str  # Full object ID (e.g., "bird_a_0")
    object_id: str   # Full object ID (e.g., "buffalo_a_1")
    relation: str    # e.g., "perched_on", "near", "touching"
    confidence: float  # probability from verification


class RelationshipAgentError(RuntimeError):
    """Custom exception for relationship agent failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


@dataclass
class RelationshipAgentState:
    """
    Tracks agent's reasoning and information gathering for relationships.
    Maintains conversation memory across agent loop iterations.
    """
    original_question: str
    referenced_objects: List[str]

    # Object pairs to investigate for relationships
    object_pairs: List[Tuple[str, str]] = field(default_factory=list)  # [(subj_id, obj_id), ...]

    # Information gathered from Qwen VL about relationships
    relationship_descriptions: Dict[Tuple[str, str], str] = field(default_factory=dict)
    qwen_qa_history: List[Dict[str, str]] = field(default_factory=list)

    # Final binary questions for verification
    binary_questions: List[BinaryRelationshipQuestion] = field(default_factory=list)

    # Agent's reasoning trace (for debugging/explainability)
    reasoning_trace: List[str] = field(default_factory=list)

    def add_relationship_info(
        self,
        object_pair: Tuple[str, str],
        description: str,
        request: QwenRelationshipRequest
    ):
        """Record relationship description from Qwen VL."""
        self.qwen_qa_history.append({
            "subject_id": object_pair[0],
            "object_id": object_pair[1],
            "question": request.question,
            "answer": description,
            "reasoning": request.reasoning
        })

        # Store description for this object pair
        if object_pair in self.relationship_descriptions:
            self.relationship_descriptions[object_pair] += f" | {description}"
        else:
            self.relationship_descriptions[object_pair] = description

    def add_reasoning(self, reasoning: str):
        """Add agent reasoning step to trace."""
        self.reasoning_trace.append(reasoning)


class RelationshipAgent:
    """
    Relationship extraction agent using LLM orchestration.

    The agent follows an iterative loop:
    1. Analyze current knowledge state about spatial/interaction relationships
    2. Decide: Need more info? → Ask Qwen VL to describe relationship
    3. Have enough info? → Generate binary verification questions
    4. Verify binary questions → Extract probabilities with colored boxes
    """

    def __init__(self, max_qwen_calls: int = 15):
        """
        Initialize relationship agent.

        Args:
            max_qwen_calls: Maximum Qwen VL calls per subquery (prevents infinite loops)
        """
        self.model_manager = ModelManager()
        self.max_qwen_calls = max_qwen_calls

    def process_relationship_subquestions(
        self,
        relationship_subquestions: List[BinarySubquestion],
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> List[IntraRelation]:
        """
        Process relationship subquestions using agentic approach.

        Args:
            relationship_subquestions: List of relationship binary subquestions
            image_paths: Dict mapping image_id to file path
            images: ImageData structure containing objects per image

        Returns:
            List[IntraRelation]: Extracted relationships with probabilities

        Raises:
            RelationshipAgentError: If processing fails
        """
        try:
            # Validate input
            if not relationship_subquestions:
                return []

            non_relationship_subquestions = [sq for sq in relationship_subquestions if sq.subquery_type != "relationship"]
            if non_relationship_subquestions:
                invalid_types = [sq.subquery_type for sq in non_relationship_subquestions]
                raise RelationshipAgentError(
                    f"RelationshipAgent only accepts relationship subquestions. "
                    f"Received {len(non_relationship_subquestions)} non-relationship subquestions: {set(invalid_types)}"
                )

            # Collect all relationship results
            all_results = []

            # Process each relationship subquestion with agent
            for i, subquestion in enumerate(relationship_subquestions, 1):
                if subquestion.subquery_type != "relationship":
                    continue

                print(f"\n  Processing subquestion {i}/{len(relationship_subquestions)}: {subquestion.question}")
                print(f"  Referenced objects: {subquestion.referenced_objects}")

                # Run agentic extraction for this subquestion
                results = self.process_single_subquestion(subquestion, image_paths, images)
                all_results.extend(results)

                print(f"    ✓ Extracted {len(results)} relationships")

            # Convert RelationshipResult to IntraRelation
            intra_relations = []
            for result in all_results:
                intra_relations.append(IntraRelation(
                    subject_id=result.subject_id,
                    object_id=result.object_id,
                    relation=result.relation,
                    probability=result.confidence
                ))

            return intra_relations

        except Exception as e:
            raise RelationshipAgentError(f"Failed to process relationship subquestions: {str(e)}")

    def process_single_subquestion(
        self,
        subquestion: BinarySubquestion,
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> List[RelationshipResult]:
        """
        Process single relationship subquestion using agentic loop.

        Args:
            subquestion: Relationship subquestion to process
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            List[RelationshipResult]: Extracted relationship results
        """
        # 1. Initialize agent state with object pairs from referenced objects
        object_pairs = self._extract_object_pairs_from_references(subquestion.referenced_objects)

        state = RelationshipAgentState(
            original_question=subquestion.question,
            referenced_objects=subquestion.referenced_objects,
            object_pairs=object_pairs
        )

        print(f"    Starting agentic loop (max {self.max_qwen_calls} Qwen calls)...")
        print(f"    Object pairs to investigate: {len(object_pairs)}")

        # 2. Agentic planning loop
        for iteration in range(self.max_qwen_calls):
            print(f"      Iteration {iteration + 1}: Agent deciding next action...")

            # Agent decides: ask Qwen for relationship info OR generate binary questions
            decision = self._agent_decide_next_action(state)
            state.add_reasoning(decision.reasoning)

            if decision.action == "ask_qwen":
                # Execute Qwen VL query to describe relationship
                print(f"        → Asking Qwen about relationship between:")
                print(f"          Subject: {decision.qwen_request.subject_id} (RED)")
                print(f"          Object: {decision.qwen_request.object_id} (BLUE)")

                answer = self._ask_qwen_about_relationship(
                    decision.qwen_request, image_paths, images
                )

                object_pair = (decision.qwen_request.subject_id, decision.qwen_request.object_id)
                state.add_relationship_info(object_pair, answer, decision.qwen_request)

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

        # 3. Verify binary questions and extract probabilities
        print(f"    Verifying {len(state.binary_questions)} binary questions...")
        results = []

        for bq in state.binary_questions:
            probability = self._verify_binary_relationship(bq, image_paths, images)

            results.append(RelationshipResult(
                subject_id=bq.subject_id,
                object_id=bq.object_id,
                relation=bq.relation,
                confidence=probability
            ))

            print(f"      {bq.subject_id} {bq.relation} {bq.object_id} (p={probability:.3f})")

        return results

    def _extract_object_pairs_from_references(
        self,
        referenced_objects: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Extract potential object pairs to investigate from referenced objects.
        For relationships, we typically need to examine pairs of objects.

        Args:
            referenced_objects: List of object IDs from subquestion

        Returns:
            List of (subject_id, object_id) tuples to investigate
        """
        pairs = []

        # Generate all possible pairs from referenced objects
        for i in range(len(referenced_objects)):
            for j in range(i + 1, len(referenced_objects)):
                # Add both directions since relationships can be directional
                pairs.append((referenced_objects[i], referenced_objects[j]))
                pairs.append((referenced_objects[j], referenced_objects[i]))

        return pairs

    def _agent_decide_next_action(self, state: RelationshipAgentState) -> RelationshipAgentDecision:
        """
        LLM agent decides: ask Qwen about relationship OR generate binary questions.

        Args:
            state: Current agent state with conversation history

        Returns:
            RelationshipAgentDecision: Agent's decision with action and rationale
        """
        llm_client = self.model_manager.get_llm_client()

        # Format gathered information for prompt
        relationships_summary = "\n".join([
            f"  - {pair[0]} & {pair[1]}: {desc}"
            for pair, desc in state.relationship_descriptions.items()
        ]) if state.relationship_descriptions else "  (No relationship information gathered yet)"

        # Format Q&A history
        qa_history = "\n".join([
            f"  Q: {qa['question']}\n  Subject: {qa['subject_id']} (red) | Object: {qa['object_id']} (blue)\n  A: {qa['answer']}"
            for qa in state.qwen_qa_history
        ]) if state.qwen_qa_history else "  (No questions asked yet)"

        prompt = f"""You are an intelligent agent extracting spatial and interaction relationships between objects to answer complex questions.

**Original Question**: {state.original_question}

**Referenced Objects**: {', '.join(state.referenced_objects)}

**Object Pairs to Investigate**: {[f"{p[0]} & {p[1]}" for p in state.object_pairs]}

**Relationship Information Gathered So Far**:
{relationships_summary}

**Previous Q&A with Vision Model (Qwen VL)**:
{qa_history}

---

## Your Task

Decide if you need MORE visual information about spatial/interaction relationships, OR if you have ENOUGH information to generate final binary verification questions.

## Decision Rules

### ❓ **Choose "ask_qwen"** if you need to:
- Understand the spatial relationship between two objects (above, below, near, far, touching, inside, etc.)
- Determine interaction relationships (perched_on, carrying, holding, lifting, supporting, etc.)
- Clarify ambiguous positioning or contact between objects
- Get visual details about how objects relate to each other

### ✅ **Choose "generate_binary_questions"** when you:
- Know which spatial/interaction relationships likely exist between object pairs
- Can formulate specific Yes/No questions like "Is the bird perched on the buffalo?"
- Have gathered enough visual information to answer the original question
- Can verify relationships with binary questions

## Important: Colored Bounding Boxes

Remember that in all visual queries:
- **Subject object** is marked with a **RED** bounding box
- **Object** is marked with a **BLUE** bounding box
- Reference colors in your questions: "the bird (red)" and "the buffalo (blue)"

## Output Format

**If you need more information:**
```json
{{
  "action": "ask_qwen",
  "reasoning": "I need to understand the spatial relationship between the bird and buffalo to determine if they are touching or just nearby",
  "qwen_request": {{
    "subject_id": "bird_a_0",
    "object_id": "buffalo_a_1",
    "question": "Describe the spatial relationship between the bird (red) and the buffalo (blue). Is the bird touching, near, or on top of the buffalo?",
    "reasoning": "Need to determine if bird is perched on buffalo or just nearby for answering the similarity question"
  }}
}}
```

**If ready to generate binary questions:**
```json
{{
  "action": "generate_binary_questions",
  "reasoning": "Based on visual descriptions, I know the bird in image_a is perched on the buffalo, and the bird in image_b is also perched on the cow. I can now verify these specific relationships.",
  "binary_questions": [
    {{
      "subject_id": "bird_a_0",
      "object_id": "buffalo_a_1",
      "relation": "perched_on",
      "binary_question": "Is the bird perched on the buffalo?"
    }},
    {{
      "subject_id": "bird_b_0",
      "object_id": "cow_b_1",
      "relation": "perched_on",
      "binary_question": "Is the bird perched on the cow?"
    }}
  ]
}}
```

Respond in strict JSON format only."""

        response = llm_client.chat_with_validation(
            messages=[{"role": "user", "content": prompt}],
            output_model=RelationshipAgentDecision,
            temperature=0.3
        )

        return response

    def _ask_qwen_about_relationship(
        self,
        request: QwenRelationshipRequest,
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> str:
        """
        Ask Qwen VL to describe spatial/interaction relationship with colored boxes.

        Args:
            request: Relationship information request with subject and object IDs
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            str: Qwen's description of the relationship
        """
        qwen_client = self.model_manager.get_qwen_vl()

        # Find both objects
        subject_obj, subject_image_id = self._find_object_by_id(request.subject_id, images)
        object_obj, object_image_id = self._find_object_by_id(request.object_id, images)

        if not subject_obj or not object_obj:
            return f"Error: Could not find objects {request.subject_id} or {request.object_id}"

        if subject_image_id != object_image_id:
            return f"Error: Objects are in different images ({subject_image_id} vs {object_image_id})"

        # Load image
        image = Image.open(image_paths[subject_image_id])

        # Draw colored boxes (RED for subject, BLUE for object)
        annotated_image = self._draw_colored_boxes(
            image, subject_obj.bbox, object_obj.bbox
        )

        # Format open-ended question with color references
        prompt = f"""The {subject_obj.label} is marked in red and the {object_obj.label} is marked in blue.

{request.question}"""

        # Run open-ended inference (no logits needed for perception)
        response, _ = qwen_client.run_inference_with_logits(annotated_image, prompt)

        return response.strip()

    def _verify_binary_relationship(
        self,
        bq: BinaryRelationshipQuestion,
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> float:
        """
        Verify binary relationship question using Qwen VL with logit probability extraction.

        Args:
            bq: Binary relationship question
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            float: Probability that the relationship exists (0.0 to 1.0)
        """
        qwen_client = self.model_manager.get_qwen_vl()

        # Find both objects
        subject_obj, subject_image_id = self._find_object_by_id(bq.subject_id, images)
        object_obj, object_image_id = self._find_object_by_id(bq.object_id, images)

        if not subject_obj or not object_obj:
            print(f"    Warning: Objects not found for verification")
            return 0.5  # Default probability

        if subject_image_id != object_image_id:
            print(f"    Warning: Objects in different images")
            return 0.5

        # Load image
        image = Image.open(image_paths[subject_image_id])

        # Draw colored boxes (RED for subject, BLUE for object)
        annotated_image = self._draw_colored_boxes(
            image, subject_obj.bbox, object_obj.bbox
        )

        # Format binary question with color references
        prompt = f"""The {subject_obj.label} is marked in red and the {object_obj.label} is marked in blue.

{bq.binary_question} Answer Yes or No.

Answer:"""

        # Run verification with logits for probability extraction
        response, logits = qwen_client.run_inference_with_logits(annotated_image, prompt)

        # Extract probability using verbalizer summing (Yes/No logits)
        probability = get_verifier_probability(
            logits,
            response,
            qwen_client.processor.tokenizer
        )

        return probability

    def _draw_colored_boxes(
        self,
        image: Image.Image,
        subject_bbox: List[float],
        object_bbox: List[float]
    ) -> Image.Image:
        """
        Draw colored bounding boxes on image.
        RED for subject, BLUE for object.

        Args:
            image: PIL Image
            subject_bbox: [x1, y1, x2, y2] for subject (will be red)
            object_bbox: [x1, y1, x2, y2] for object (will be blue)

        Returns:
            PIL Image with colored boxes drawn
        """
        # Create a copy to avoid modifying original
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Draw RED box for subject (thick lines)
        draw.rectangle(subject_bbox, outline="red", width=4)

        # Draw BLUE box for object (thick lines)
        draw.rectangle(object_bbox, outline="blue", width=4)

        return annotated_image

    def _find_object_by_id(
        self,
        object_id: str,
        images: Dict[str, ImageData]
    ) -> Tuple[Optional[ObjectDetection], Optional[str]]:
        """
        Find object by ID across all images.

        Args:
            object_id: Object ID to find (e.g., "bird_a_0")
            images: ImageData structure

        Returns:
            Tuple[ObjectDetection, image_id] or (None, None) if not found
        """
        try:
            # Parse object ID format: label_imageid_objectid (e.g., "bird_a_0")
            parts = object_id.split('_')
            if len(parts) < 3:
                return None, None

            simple_image_id = parts[-2]  # Second to last part (e.g., "a")
            object_index = int(parts[-1])

            # Convert simple image ID back to full key (e.g., "a" -> "image_a")
            image_id = f"image_{simple_image_id}"

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
