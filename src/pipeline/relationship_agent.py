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

import os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw
from dataclasses import dataclass, field
from collections import defaultdict

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ObjectDetection, IntraRelation, ImageData
from src.core.probability import get_verifier_probability
from src.language.output_models import RelationshipAgentDecision, QwenRelationshipRequest, BinaryRelationshipQuestion
from src.pipeline.base_agent import BaseVerificationAgent


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
    target_claims: Dict[str, List[str]] = field(default_factory=dict)  # NEW: What relationship types must be verified

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


class RelationshipAgent(BaseVerificationAgent[BinaryRelationshipQuestion]):
    """
    Relationship extraction agent using LLM orchestration.

    The agent follows an iterative loop:
    1. Analyze current knowledge state about spatial/interaction relationships
    2. Decide: Need more info? → Ask Qwen VL to describe relationship
    3. Have enough info? → Generate binary verification questions
    4. Verify binary questions → Extract probabilities with colored boxes
    """

    def __init__(self, max_qwen_calls: int = 15, debug: bool = False):
        """
        Initialize relationship agent.

        Args:
            max_qwen_calls: Maximum Qwen VL calls per subquery (prevents infinite loops)
            debug: If True, saves cropped images and prints detailed verification info
        """
        super().__init__()  # Initialize BaseVerificationAgent
        self.max_qwen_calls = max_qwen_calls
        self.debug = debug

        # Create debug directory if needed
        if self.debug:
            os.makedirs("debug_relationships", exist_ok=True)

    def _extract_target_claims_from_subquestion(self, subquestion: str) -> Dict[str, List[str]]:
        """
        Extract the specific relationship types that need to be verified from subquestion.

        Examples:
        - "Is the candy on top of the table?" → {"relations": ["on_top_of"]}
        - "Is the bird perched on the buffalo?" → {"relations": ["perched_on"]}
        - "Is the book near or touching the lamp?" → {"relations": ["near", "touching"]}

        Args:
            subquestion: Natural language subquestion

        Returns:
            Dict with "relations" key containing list of relation types to verify
        """
        return self._extract_target_claims_generic(
            subquestion=subquestion,
            claim_type="relationship types",
            claim_key="relations",
            examples=[
                '"Is the candy on top of the table?" → {"relations": ["on_top_of"]}',
                '"Is the bird perched on the buffalo?" → {"relations": ["perched_on"]}',
                '"Is the book near or touching the lamp?" → {"relations": ["near", "touching"]}',
                '"Is the cup below the shelf?" → {"relations": ["below"]}'
            ]
        )

    def _validate_binary_questions(
        self,
        binary_questions: List[BinaryRelationshipQuestion],
        target_claims: Dict[str, List[str]],
        subquestion: str
    ) -> bool:
        """
        Validate that binary questions include DIRECT verification of target relationship claims.

        Args:
            binary_questions: Generated binary relationship questions
            target_claims: Target claims extracted from subquestion
            subquestion: Original subquestion

        Returns:
            bool: True if all target claims are covered, False otherwise
        """
        return self._validate_binary_questions_generic(
            binary_questions=binary_questions,
            target_claims=target_claims,
            claim_key="relations",
            value_extractor=lambda q: q.relation,
            subquestion=subquestion
        )

    def _generate_fallback_questions(
        self,
        target_claims: Dict[str, List[str]],
        object_pairs: List[Tuple[str, str]],
        existing_questions: List[BinaryRelationshipQuestion]
    ) -> List[BinaryRelationshipQuestion]:
        """
        Generate minimal binary questions to directly verify missing target relationship claims.

        Args:
            target_claims: Target claims that need verification
            object_pairs: List of relevant object pairs
            existing_questions: Already generated questions

        Returns:
            List of fallback binary questions for missing targets
        """
        if not object_pairs:
            return []

        # Use base class method to get missing claim values
        missing_relations = self._get_missing_claims(
            target_claims=target_claims,
            claim_key="relations",
            existing_questions=existing_questions,
            value_extractor=lambda q: q.relation
        )

        if not missing_relations:
            return []

        print(f"  → Generating fallback questions for missing relations: {missing_relations}")

        fallback_questions = []
        for subj_id, obj_id in object_pairs:
            for relation in missing_relations:
                # Extract object types from IDs
                subj_type = subj_id.split('_')[0]
                obj_type = obj_id.split('_')[0]

                # Format relation for natural language
                relation_text = relation.replace("_", " ")

                fallback_questions.append(BinaryRelationshipQuestion(
                    subject_id=subj_id,
                    object_id=obj_id,
                    relation=relation,
                    binary_question=f"Is the {subj_type} {relation_text} the {obj_type}?"
                ))

        return fallback_questions

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

            non_relationship_subquestions = [sq for sq in relationship_subquestions if sq.subquestion_type != "relationship"]
            if non_relationship_subquestions:
                invalid_types = [sq.subquestion_type for sq in non_relationship_subquestions]
                raise RelationshipAgentError(
                    f"RelationshipAgent only accepts relationship subquestions. "
                    f"Received {len(non_relationship_subquestions)} non-relationship subquestions: {set(invalid_types)}"
                )

            # Collect all relationship results
            all_results = []

            # Process each relationship subquestion with agent
            for i, subquestion in enumerate(relationship_subquestions, 1):
                if subquestion.subquestion_type != "relationship":
                    continue

                print(f"\n  Processing subquestion {i}/{len(relationship_subquestions)}: {subquestion.question}")

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
        NOW: Discovers relevant object pairs from natural language question.

        Args:
            subquestion: Relationship subquestion to process (no object IDs)
            image_paths: Image file paths
            images: ImageData structure

        Returns:
            List[RelationshipResult]: Extracted relationship results
        """
        # 1. Extract target claims from subquestion (what relationships must be verified)
        target_claims = self._extract_target_claims_from_subquestion(subquestion.question)
        if target_claims.get("relations"):
            print(f"    Target relationship claims to verify: {target_claims['relations']}")

        # 2. Discover which object pairs are relevant to this question
        print(f"    Discovering relevant object pairs for: {subquestion.question}")
        object_pairs = self._discover_relevant_object_pairs(subquestion.question, images)
        print(f"    → Discovered {len(object_pairs)} relevant pairs")

        # 3. Initialize agent state with discovered pairs AND target claims
        state = RelationshipAgentState(
            original_question=subquestion.question,
            referenced_objects=[],  # No longer used
            object_pairs=object_pairs,
            target_claims=target_claims
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

                # Validate that target relationship claims are covered
                if not self._validate_binary_questions(
                    state.binary_questions,
                    target_claims,
                    subquestion.question
                ):
                    # Generate fallback questions for missing target claims
                    fallback_questions = self._generate_fallback_questions(
                        target_claims,
                        object_pairs,
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
            if target_claims.get("relations") and object_pairs:
                print(f"    Generating fallback questions for target relationship claims...")
                state.binary_questions = self._generate_fallback_questions(
                    target_claims,
                    object_pairs,
                    []
                )
            if not state.binary_questions:
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

    def _discover_relevant_object_pairs(
        self,
        question: str,
        images: Dict[str, ImageData]
    ) -> List[Tuple[str, str]]:
        """
        LLM analyzes natural language question to discover relevant object pairs.

        Args:
            question: Natural language relationship question
                     (e.g., "Is a bird perched on the buffalo?")
            images: All detected objects

        Returns:
            List of (subject_id, object_id) tuples
        """
        llm_client = self.model_manager.get_llm_client()

        # Build object list
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
            "content": "You analyze relationship questions to identify which object pairs are relevant."
        }, {
            "role": "user",
            "content": f"""QUESTION: {question}

DETECTED OBJECTS:
{object_context}

TASK: Identify which object PAIRS are relevant for this relationship question.

Rules:
1. Parse the question to understand what relationship is being asked (e.g., "perched on", "near", "holding")
2. Identify subject and object classes (e.g., "Is X on Y?" → subject=X, object=Y)
3. Include ALL plausible pairs if multiple instances exist
4. For "in both images" questions, generate pairs from both images
5. Subject is the active entity, object is the passive entity

Output JSON:
{{
  "object_pairs": [
    {{"subject_id": "bird_a_0", "object_id": "buffalo_a_3"}},
    {{"subject_id": "bird_b_2", "object_id": "cow_b_4"}}
  ]
}}"""
        }]

        response = llm_client.discover_object_pairs(messages)
        return [(pair.subject_id, pair.object_id) for pair in response.object_pairs]

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

        # Format target claims
        target_claims_text = ""
        if state.target_claims.get("relations"):
            relations_str = ', '.join(state.target_claims["relations"])
            target_claims_text = f"\n**Target Relationship Claims to Verify**: {relations_str}\n(These specific relationships MUST be directly verified in your binary questions)"

        prompt = f"""You are an intelligent agent extracting spatial and interaction relationships between objects to answer complex questions.

**Original Question**: {state.original_question}

**Referenced Objects**: {', '.join(state.referenced_objects)}

**Object Pairs to Investigate**: {[f"{p[0]} & {p[1]}" for p in state.object_pairs]}
{target_claims_text}

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
- Have gathered enough visual information about spatial/interaction relationships
- **CRITICAL: Can directly verify the target relationship claims (if specified)**
- Can formulate specific Yes/No questions
- Even if visual evidence suggests different relationships, you can still generate verification questions

## CRITICAL RULE: Subquestion-Aligned Verification

Your binary questions MUST include direct verification of target relationship claims, even if:
- Visual evidence suggests a different spatial relationship
- The claim appears unlikely based on what you observed
- You think the answer will be "No"

**Why:** The ProbLog reasoner needs the probability that the claimed relationship is TRUE, not just probabilities for what you observed.

**Example:**
  Original Question: "Is the candy on top of the table?"
  Target Claim: "on_top_of"
  Visual Evidence: "The candy is below the table"

  CORRECT binary questions:
    ✓ "Is the candy on top of the table?" (directly verifies target claim - REQUIRED!)
    ✓ "Is the candy below the table?" (confirms alternative - OPTIONAL)

  INCORRECT binary questions:
    ✗ Only "Is the candy below the table?" (doesn't verify the claim!)

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

        # Crop to union of both objects with margin - focuses VLM attention (consistent with verification)
        cropped_image, adj_subj_bbox, adj_obj_bbox = self._crop_to_union_bbox(
            image, subject_obj.bbox, object_obj.bbox, margin=0.15
        )

        # Draw colored boxes on CROPPED image (RED for subject, BLUE for object)
        annotated_image = self._draw_colored_boxes(
            cropped_image, adj_subj_bbox, adj_obj_bbox
        )

        # Format open-ended question with color references
        prompt = f"""The {subject_obj.label} is marked in RED and the {object_obj.label} is marked in BLUE.

{request.question}"""

        # Run open-ended inference (no logits needed for perception)
        response, _ = qwen_client.run_inference_with_logits(annotated_image, prompt)

        return response.strip()

    def _crop_to_union_bbox(
        self,
        image: Image.Image,
        bbox1: List[float],
        bbox2: List[float],
        margin: float = 0.15
    ) -> Tuple[Image.Image, List[float], List[float]]:
        """
        Crop to union of two bounding boxes with margin.
        Returns cropped image and adjusted bbox coordinates relative to crop.

        Args:
            image: PIL Image to crop
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            margin: Percentage margin to add (0.15 = 15% of union bbox size)

        Returns:
            Tuple of (cropped_image, adjusted_bbox1, adjusted_bbox2)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate union bounding box
        union_x1 = min(x1_1, x1_2)
        union_y1 = min(y1_1, y1_2)
        union_x2 = max(x2_1, x2_2)
        union_y2 = max(y2_1, y2_2)

        # Apply margin
        width, height = image.size
        union_width = union_x2 - union_x1
        union_height = union_y2 - union_y1
        margin_x = union_width * margin
        margin_y = union_height * margin

        crop_x1 = max(0, union_x1 - margin_x)
        crop_y1 = max(0, union_y1 - margin_y)
        crop_x2 = min(width, union_x2 + margin_x)
        crop_y2 = min(height, union_y2 + margin_y)

        # Crop image
        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Adjust bbox coordinates relative to crop
        adjusted_bbox1 = [
            x1_1 - crop_x1,
            y1_1 - crop_y1,
            x2_1 - crop_x1,
            y2_1 - crop_y1
        ]
        adjusted_bbox2 = [
            x1_2 - crop_x1,
            y1_2 - crop_y1,
            x2_2 - crop_x1,
            y2_2 - crop_y1
        ]

        return cropped, adjusted_bbox1, adjusted_bbox2

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

        # Crop to union of both objects with margin - removes distracting context
        cropped_image, adj_subj_bbox, adj_obj_bbox = self._crop_to_union_bbox(
            image, subject_obj.bbox, object_obj.bbox, margin=0.15
        )

        # Draw colored boxes on CROPPED image (easier to see, no distractors)
        annotated_image = self._draw_colored_boxes(
            cropped_image, adj_subj_bbox, adj_obj_bbox
        )

        # Format binary question with color references (NO bbox coordinates in text!)
        prompt = f"""The {subject_obj.label} is marked in RED and the {object_obj.label} is marked in BLUE.

{bq.binary_question}

Answer Yes or No.\n\nAnswer:"""

        # Run verification with logits for probability extraction
        response, logits = qwen_client.run_inference_with_logits(annotated_image, prompt)

        # DEBUG: Save crops and print full verification details (BEFORE verbalizer to group output)
        if self.debug:
            # Save the annotated crop
            debug_filename = f"debug_relationships/{bq.subject_id}__{bq.object_id}__{bq.relation}.png"
            annotated_image.save(debug_filename)

            # Print full debugging info FIRST (before verbalizer breakdown)
            print("\n" + "=" * 80)
            print(f"🔍 RELATIONSHIP VERIFICATION DEBUG")
            print("=" * 80)
            print(f"Subject: {bq.subject_id} ({subject_obj.label})")
            print(f"Object: {bq.object_id} ({object_obj.label})")
            print(f"Relation: {bq.relation}")
            print(f"Binary Question: {bq.binary_question}")
            print(f"\nOriginal bboxes:")
            print(f"  Subject: {subject_obj.bbox}")
            print(f"  Object: {object_obj.bbox}")
            print(f"\nAdjusted bboxes (on cropped image):")
            print(f"  Subject (RED): {adj_subj_bbox}")
            print(f"  Object (BLUE): {adj_obj_bbox}")
            print(f"\nCropped image size: {cropped_image.size}")
            print(f"Annotated crop saved: {debug_filename}")
            print(f"\n--- FULL PROMPT TO VLM ---")
            print(prompt)
            print(f"--- END PROMPT ---")
            print(f"\n--- VLM RESPONSE ---")
            print(f'"{response}"')
            print(f"--- END RESPONSE ---")

        # Extract probability using verbalizer summing (Yes/No logits)
        # This will print verbalizer breakdown if debug=True
        probability = get_verifier_probability(
            logits,
            response,
            qwen_client.processor.tokenizer,
            debug=self.debug  # Pass debug flag for detailed logit breakdown
        )

        if self.debug:
            print(f"\n✓ Final Result: {bq.subject_id} {bq.relation} {bq.object_id} (probability: {probability:.4f})")
            print("=" * 80 + "\n")

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

        # Draw RED box for subject (3px for visibility without overwhelming)
        draw.rectangle(subject_bbox, outline="red", width=3)

        # Draw BLUE box for object (3px for visibility without overwhelming)
        draw.rectangle(object_bbox, outline="blue", width=3)

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
