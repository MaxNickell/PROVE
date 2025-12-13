"""
Unified Evidence Collection Agent for PROVE pipeline.

Single intelligent agent that handles ANY subquestion by autonomously collecting
all needed evidence (attributes, relationships, scene data, counts).

Key Design Principles:
1. Agent starts with detected objects (no "what exists?" questions)
2. NEVER filters candidates early - collects probabilities for ALL
3. Agent collects evidence, ProbLog does inference/composition
4. Stores all facts (even low probability) for complete knowledge
"""

from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ImageData, ObjectDetection
from src.core.probability import get_verifier_probability
from src.language.output_models import PerceiveDecision, VerifyDecision, DoneDecision


@dataclass
class EvidenceCollection:
    """
    All evidence collected for one subquestion.

    This is what the agent returns - raw probabilistic facts.
    ProbLog will later compose these to answer the subquestion.
    """
    subquestion: str

    # Attribute evidence: (entity_id, attribute_class, value, probability)
    # Example: ("dog_a_1", "color", "orange", 0.87)
    # Stores ALL checks, even low probability ones
    attributes: List[Tuple[str, str, str, float]] = field(default_factory=list)

    # Relationship evidence: (subject_id, object_id, relation, probability)
    # Example: ("dog_a_1", "table_a_5", "on_top_of", 0.92)
    relationships: List[Tuple[str, str, str, float]] = field(default_factory=list)

    # Count distributions (reuse existing structure)
    # Key: object class, Value: distribution data
    counts: Dict[str, Any] = field(default_factory=dict)


    # Agent's reasoning process (for debugging/transparency)
    reasoning_trace: List[str] = field(default_factory=list)

    # Property-granular verification tracking to prevent duplicates
    # Key: tuple representing what was verified
    # Examples:
    #   ("bird_a_2", "color"): attribute verification
    #   ("dog_a_1", "table_a_3", "on_top_of"): relationship verification
    #   ("image_a", "bird", "count"): count verification
    verifications_completed: Dict[Tuple, bool] = field(default_factory=dict)

    # Perceive tracking to prevent duplicate questions
    # Key: tuple (entity_id, question) representing what was asked
    # Examples: ("dog_a_1", "What color is this dog?")
    perceive_completed: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def add_attribute(self, entity_id: str, attribute_class: str, value: str, probability: float):
        """Add attribute evidence and track verification."""
        self.attributes.append((entity_id, attribute_class, value, probability))
        # Track that we verified this (entity, property)
        self.verifications_completed[(entity_id, attribute_class)] = True

    def add_relationship(self, subject_id: str, object_id: str, relation: str, probability: float):
        """Add relationship evidence and track verification."""
        self.relationships.append((subject_id, object_id, relation, probability))
        # Track that we verified this (subject, object, relation)
        self.verifications_completed[(subject_id, object_id, relation)] = True

    def add_count_distribution(self, image_id: str, object_class: str, distribution: Dict[int, float]):
        """Store count distribution and track verification."""
        key = f"{image_id}_{object_class}"
        self.counts[key] = distribution
        # Track that we verified this (image, class, count)
        self.verifications_completed[(image_id, object_class, "count")] = True

    def add_reasoning(self, step: str):
        """Add reasoning step to trace."""
        self.reasoning_trace.append(step)

    def add_perceive_result(self, entity_id: str, question: str, answer: str):
        """Track that we asked a perceive question and store the answer."""
        self.perceive_completed[(entity_id, question)] = answer

    def has_been_perceived(self, entity_id: str, question: str) -> bool:
        """Check if this exact perceive question was already asked."""
        return (entity_id, question) in self.perceive_completed

    def get_perceive_result(self, entity_id: str, question: str) -> str:
        """Get previous perceive result if available."""
        return self.perceive_completed.get((entity_id, question), "")


@dataclass
class UnifiedAgentState:
    """
    Tracks agent's state during evidence collection for one subquestion.

    This maintains the agent's memory across iterations of the agentic loop.
    """

    # The question being answered
    original_question: str

    # Candidate entities from object detection
    # Key: entity type (e.g., "dog", "table", "cat")
    # Value: list of full entity IDs (e.g., ["dog_a_0", "dog_a_1"])
    # These come from detected objects - agent doesn't ask "what exists?"
    candidate_entities: Dict[str, List[str]] = field(default_factory=dict)

    # Evidence accumulated so far
    evidence: 'EvidenceCollection' = field(default_factory=lambda: EvidenceCollection(subquestion=""))

    # Conversation history with Qwen VL (perceive actions)
    # List of {"entity": entity_id, "question": str, "answer": str}
    qwen_qa_history: List[Dict[str, str]] = field(default_factory=list)

    # Agent's chain of thought reasoning
    reasoning_trace: List[str] = field(default_factory=list)

    # Current iteration count (for max iteration limit)
    iteration: int = 0

    def add_reasoning(self, step: str):
        """Add reasoning step to both state and evidence trace."""
        self.reasoning_trace.append(step)
        self.evidence.add_reasoning(step)




class UnifiedAgent:
    """
    Unified evidence collection agent.

    Handles ANY subquestion by autonomously collecting all needed evidence:
    - Attributes (object properties)
    - Relationships (spatial/interaction)
    - Scene attributes (environmental)
    - Counts (quantity distributions)

    Agent follows natural dependencies (e.g., "find orange dogs, then check if on table")
    but doesn't filter candidates. Instead, it collects probabilistic evidence at each
    step, and ProbLog naturally filters via probability composition.

    Architecture: Reasoner → Planner → Perceiver → Verifier (iterative loop)
    """

    def __init__(self, max_iterations: int = 20, mode: str = "probabilistic"):
        """Initialize unified agent.

        Args:
            max_iterations: Maximum iterations for evidence collection
            mode: Execution mode - "probabilistic" or "deterministic"
        """
        self.model_manager = ModelManager()
        self.max_iterations = max_iterations
        self.mode = mode

    def collect_evidence(
        self,
        subquestion: BinarySubquestion,
        images: Dict[str, ImageData],
        image_paths: Dict[str, str]
    ) -> EvidenceCollection:
        """
        Main entry point: collect all evidence needed for subquestion.

        This is the public API - given a subquestion, returns all probabilistic
        facts collected to answer it.

        Args:
            subquestion: The question to collect evidence for
            images: ImageData with detected objects (from object detection step)
            image_paths: Paths to image files for VLM access

        Returns:
            EvidenceCollection with all probabilistic facts

        Raises:
            RuntimeError: If evidence collection fails critically
        """
        # Initialize agent state with candidates from detected objects
        state = self._initialize_state(subquestion, images)

        # Main agent loop - agent decides when evidence collection is complete
        done = False
        while state.iteration < self.max_iterations and not done:
            state.iteration += 1

            # Agent reasoner & planner: decide next action
            decision = self._agent_reasoner_and_planner(state, images)

            # Execute action based on decision
            if decision.action == "perceive":
                # Ask VLM open-ended questions to gather information
                self._execute_perceive_action(decision, state, images, image_paths)
            elif decision.action == "verify":
                # Verify binary questions and collect probabilities
                self._execute_verify_action(decision, state, images, image_paths)
            elif decision.action == "done":
                # Evidence collection complete
                done = True
                state.add_reasoning("Evidence collection complete")

        return state.evidence

    def _apply_mode_mapping(self, probability: float) -> float:
        """Apply mode-specific probability mapping.

        Args:
            probability: Raw probability from VLM (0.0 to 1.0)

        Returns:
            float: Mapped probability based on execution mode
        """
        if self.mode == "deterministic":
            # Map to binary: <50% → 0%, ≥50% → 100%
            return 1.0 if probability >= 0.5 else 0.0
        else:
            # Probabilistic mode: return as-is
            return probability

    def _initialize_state(
        self,
        subquestion: BinarySubquestion,
        images: Dict[str, ImageData]
    ) -> UnifiedAgentState:
        """
        Initialize agent state with candidate entities from detected objects.

        Key principle: Agent starts with detected objects, doesn't ask VLM "what exists?"

        Args:
            subquestion: The question to answer
            images: ImageData with detected objects

        Returns:
            UnifiedAgentState with candidates populated
        """
        state = UnifiedAgentState(original_question=subquestion.question)
        state.evidence.subquestion = subquestion.question

        # TODO: Use LLM to identify which entity types are relevant from question
        # For now, use all entity types from detected objects
        entity_types_found = set()
        for image_id, image_data in images.items():
            for obj in image_data.objects:
                entity_types_found.add(obj.label.lower())

        # Populate candidates from detected objects
        for entity_type in entity_types_found:
            candidates = []
            for image_id, image_data in images.items():
                image_letter = image_id.replace("image_", "")
                for obj in image_data.objects:
                    if obj.label.lower() == entity_type:
                        # Create full entity ID: label_imageletter_objectid
                        entity_id = f"{obj.label}_{image_letter}_{obj.object_id}"
                        candidates.append(entity_id)

            if candidates:
                state.candidate_entities[entity_type] = candidates
                state.add_reasoning(f"Identified {len(candidates)} candidate {entity_type}(s)")

        return state

    def _agent_reasoner_and_planner(
        self,
        state: UnifiedAgentState,
        images: Dict[str, ImageData]
    ):
        """
        Agent analyzes current state and decides next action.

        This is the "brain" of the agent. It examines:
        - Original question
        - Candidates identified
        - Evidence collected so far
        - What's still missing

        Then decides: perceive (gather info), verify (check facts), or done.

        Args:
            state: Current agent state
            images: ImageData for context

        Returns:
            Decision with next action
        """
        from src.language.output_models import UnifiedAgentDecision

        llm_client = self.model_manager.get_llm_client()

        # Build context for LLM decision
        candidates_text = self._format_candidates(state.candidate_entities)

        system_prompt = """You collect minimal evidence to answer questions probabilistically.

ACTIONS (all require "action" and "reasoning" fields):
- perceive: Ask VLM open-ended questions about entities
- verify: Get binary evidence (attribute/relationship/count)
- done: When evidence is sufficient

COMPLETE JSON EXAMPLES:

PERCEIVE:
{"action": "perceive", "reasoning": "Need to understand what material this furniture is made of", "target": "chair_a_2", "question": "What material is this chair made of?"}

VERIFY ATTRIBUTE:
{"action": "verify", "reasoning": "Need to check if this laptop is silver colored", "verify_type": "attribute", "targets": ["laptop_b_1"], "property": "color", "value": "silver", "verification_question": "Is this laptop silver?"}

VERIFY RELATIONSHIP:
{"action": "verify", "reasoning": "Need to check if the phone is positioned on the desk", "verify_type": "relationship", "targets": ["phone_a_3", "desk_a_1"], "property": "on_top_of", "verification_question": "Is the phone on top of the desk?"}

VERIFY COUNT:
{"action": "verify", "reasoning": "Need to count cars in this parking scene", "verify_type": "count", "targets": ["image_b"], "property": "car", "verification_question": null}

DONE:
{"action": "done", "reasoning": "All necessary evidence has been collected to answer the question"}

EDGE CASE EXAMPLES:

MULTIPLE ATTRIBUTES:
{"action": "verify", "reasoning": "Need to verify both material and color properties", "verify_type": "attribute", "targets": ["pot_a_1"], "property": "material", "value": "stainless_steel", "verification_question": "Is this pot made of stainless steel?"}

COMPLEX SPATIAL:
{"action": "verify", "reasoning": "Need to check containment relationship", "verify_type": "relationship", "targets": ["plant_b_2", "vase_b_4"], "property": "inside", "verification_question": "Is the plant inside the vase?"}

CONDITIONAL VERIFICATION:
{"action": "perceive", "reasoning": "Need to determine screen condition before checking if phone is functional", "target": "phone_a_1", "question": "What is the condition of this phone screen?"}

STATE VERIFICATION:
{"action": "verify", "reasoning": "Need to check if appliance is currently active", "verify_type": "attribute", "targets": ["oven_a_3"], "property": "state", "value": "on", "verification_question": "Is this oven turned on?"}

COMPARATIVE SIZE:
{"action": "verify", "reasoning": "Need to compare relative sizes of furniture pieces", "verify_type": "relationship", "targets": ["chair_b_1", "sofa_b_3"], "property": "smaller_than", "verification_question": "Is the chair smaller than the sofa?"}

TARGET STRUCTURE EXAMPLES:

✓ CORRECT ATTRIBUTE (1 target, 1 property, 1 value):
{"verify_type": "attribute", "targets": ["chair_a_2"], "property": "material", "value": "wood", "verification_question": "Is this chair made of wood?"}

✗ WRONG ATTRIBUTE (multiple targets):
{"verify_type": "attribute", "targets": ["chair_a_2", "chair_a_3"], "property": "material", "value": "wood"}
// INSTEAD: Create 2 separate verify actions

✓ CORRECT RELATIONSHIP (exactly 2 targets):
{"verify_type": "relationship", "targets": ["laptop_b_1", "desk_b_2"], "property": "on_top_of", "verification_question": "Is the laptop on top of the desk?"}

✗ WRONG RELATIONSHIP (1 or 3+ targets):
{"verify_type": "relationship", "targets": ["laptop_b_1"], "property": "on_top_of"}
{"verify_type": "relationship", "targets": ["laptop_b_1", "desk_b_2", "mouse_b_3"], "property": "near"}
// INSTEAD: Exactly 2 targets for each relationship

✓ CORRECT COUNT (1 image target):
{"verify_type": "count", "targets": ["image_a"], "property": "car"}

MULTI-STEP VERIFICATION APPROACH:

For complex questions requiring multiple checks, break into separate actions:

EXAMPLE: "Are both chairs red and made of wood?"
// WRONG - trying to check multiple properties at once
{"verify_type": "attribute", "targets": ["chair_a_1"], "property": "color_and_material", "value": "red_wood"}

// CORRECT - separate verification for each property and each chair
Action 1: {"verify_type": "attribute", "targets": ["chair_a_1"], "property": "color", "value": "red"}
Action 2: {"verify_type": "attribute", "targets": ["chair_a_1"], "property": "material", "value": "wood"}
Action 3: {"verify_type": "attribute", "targets": ["chair_a_2"], "property": "color", "value": "red"}
Action 4: {"verify_type": "attribute", "targets": ["chair_a_2"], "property": "material", "value": "wood"}

VERIFICATION QUESTION EXAMPLES:
✓ CORRECT (Binary Yes/No):
- "Is this laptop silver?"
- "Are the chair and desk made of the same material?"
- "Is the plant inside the pot?"
- "Does this car have four doors?"
- "Is this screen cracked?"

✗ WRONG (Open-ended):
- "What material is this chair made of?" (use "Is this chair made of wood?" instead)
- "How many cars are in the parking lot?" (count verification handles this automatically)
- "Which appliance is larger?" (break into "Is the refrigerator larger than the stove?")
- "What condition is this phone in?" (use "Is this phone damaged?" instead)

PERCEIVE QUESTION EXAMPLES:
✓ CORRECT (Generic):
- "What material is this furniture made of?"
- "What is the condition of this screen?"
- "How are these kitchen appliances arranged?"
- "What type of plant is this?"

✗ WRONG (Image references):
- "What material is the chair in image A made of?" (VLM doesn't know what "image A" means)
- "How are the cars arranged in image B?" (use "How are these cars arranged?")
- "What color is the laptop in the right image?" (use "What color is this laptop?")

RULES:
- Count targets: Use "image_a"/"image_b" (pure image IDs)
- Other targets: Use entity IDs like "chair_a_2", "laptop_b_1", "car_a_0"
- CRITICAL: verification_question MUST be answerable with YES or NO only
- NEVER use open-ended questions like "What material is...", "How many...", "Which..."
- Verification questions MUST be binary: "Is this metal?", "Are they matching?", "Is X inside Y?"
- PERCEIVE questions must be generic - do NOT reference "image A" or "image B" in question text
- VLM only sees cropped regions, not full labeled images

TARGET CONSTRAINTS (CRITICAL):
- ATTRIBUTE verification: EXACTLY 1 target, 1 property, 1 value
- RELATIONSHIP verification: EXACTLY 2 targets, 1 relationship property
- COUNT verification: EXACTLY 1 image target, 1 object class
- For multiple checks: Create separate verify actions, do NOT combine targets

- Stop when answer is clear (don't over-collect)
- Check completed verifications to avoid duplicates
- ALL actions MUST include "action" and "reasoning" fields

Output JSON only."""

        # Build context with available object classes
        available_classes = self._format_available_object_classes(state.candidate_entities)

        user_prompt = f"""Question: "{state.original_question}"

CANDIDATE ENTITIES
------------------
{candidates_text}

AVAILABLE OBJECT CLASSES
------------------------
{available_classes}

VERIFICATIONS COMPLETED
-----------------------
{self._format_verifications_completed(state.evidence.verifications_completed, state.evidence)}

PERCEIVE QUESTIONS ASKED
------------------------
{self._format_perceive_completed(state.evidence.perceive_completed)}

EVIDENCE SUMMARY
----------------
- Attributes: {len(state.evidence.attributes)} facts
- Relationships: {len(state.evidence.relationships)} facts
- Counts: {len(state.evidence.counts)} distributions

RECENT ACTIONS
--------------
{self._format_recent_actions(state)}

Iteration: {state.iteration}/{self.max_iterations}

YOUR JOB
--------
1. Check "Verifications Completed" - what's already done?
2. Determine what evidence is still missing
3. Choose the single most useful next action
4. Output ONE JSON action object

Stop only when ALL required evidence is collected → action = "done"

RESPOND WITH JSON ONLY:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Get response from LLM
            response = llm_client.chat(messages, temperature=0.3)

            # Extract and parse JSON
            json_str = llm_client._extract_json(response)
            import json
            parsed_json = json.loads(json_str)

            # Discriminate based on action field
            action = parsed_json.get("action")

            if action == "perceive":
                from src.language.output_models import PerceiveDecision
                decision = PerceiveDecision(**parsed_json)
            elif action == "verify":
                from src.language.output_models import VerifyDecision
                decision = VerifyDecision(**parsed_json)
            elif action == "done":
                from src.language.output_models import DoneDecision
                decision = DoneDecision(**parsed_json)
            else:
                raise ValueError(f"Unknown action: {action}")

            state.add_reasoning(f"Decision: {decision.action} - {decision.reasoning}")
            return decision

        except Exception as e:
            print(f"    ⚠ Warning: Agent decision failed: {e}")
            # Fallback: mark done if we've tried enough iterations
            if state.iteration >= self.max_iterations // 2:
                from src.language.output_models import DoneDecision
                return DoneDecision(
                    action="done",
                    reasoning="Fallback: max iterations approaching"
                )
            else:
                raise RuntimeError(f"Agent reasoning failed: {e}")

    def _format_candidates(self, candidate_entities: Dict[str, List[str]]) -> str:
        """Format candidate entities for LLM prompt, grouped by image for clarity."""
        if not candidate_entities:
            return "None identified yet"

        # Group entities by image
        by_image: Dict[str, List[Tuple[str, str]]] = {}  # image_id -> [(entity_type, entity_id), ...]

        for entity_type, entity_ids in candidate_entities.items():
            for entity_id in entity_ids:
                # Extract image letter from entity_id (format: label_imageletter_objectid)
                parts = entity_id.split('_')
                if len(parts) >= 3:
                    image_letter = parts[-2]
                    image_key = f"Image {image_letter.upper()}"

                    if image_key not in by_image:
                        by_image[image_key] = []
                    by_image[image_key].append((entity_type, entity_id))

        # Format grouped by image
        lines = []
        for image_key in sorted(by_image.keys()):
            lines.append(f"{image_key}:")

            # Group by entity type within this image
            entities_in_image = by_image[image_key]
            by_type: Dict[str, List[str]] = {}
            for entity_type, entity_id in entities_in_image:
                if entity_type not in by_type:
                    by_type[entity_type] = []
                by_type[entity_type].append(entity_id)

            # Format each entity type
            for entity_type in sorted(by_type.keys()):
                entity_ids = by_type[entity_type]
                count = len(entity_ids)
                entity_word = "entity" if count == 1 else "entities"
                id_list = ', '.join(entity_ids[:5]) + ('...' if len(entity_ids) > 5 else '')
                lines.append(f"  - {entity_type}: {count} {entity_word} ({id_list})")

        return "\n".join(lines)

    def _format_verifications_completed(self, verifications_completed: Dict[Tuple, bool], evidence: 'EvidenceCollection') -> str:
        """
        Format completed verifications showing actual results, not just completion.
        Shows what evidence was collected to help agent avoid repetition.
        """
        if not verifications_completed:
            return "None yet"

        # Group by type with results
        attributes = []
        relationships = []
        counts = []

        for key in verifications_completed.keys():
            if len(key) == 2:
                # Attribute: (entity_id, property)
                entity_id, property = key
                # Find the actual attribute result
                result_text = self._get_attribute_result_summary(entity_id, property, evidence)
                attributes.append(f"    - ({entity_id}, {property}): {result_text}")
            elif len(key) == 3:
                if key[2] == "count":
                    # Count: (image_id, class, "count")
                    image_id, object_class = key[0], key[1]
                    # Find the actual count result
                    result_text = self._get_count_result_summary(image_id, object_class, evidence)
                    counts.append(f"    - ({image_id}, {object_class}): {result_text}")
                else:
                    # Relationship: (subject, object, relation)
                    subject_id, object_id, relation = key
                    # Find the actual relationship result
                    result_text = self._get_relationship_result_summary(subject_id, object_id, relation, evidence)
                    relationships.append(f"    - ({subject_id}, {object_id}, {relation}): {result_text}")

        lines = []
        if attributes:
            lines.append("  Attributes:")
            lines.extend(attributes[:10])  # Show max 10
            if len(attributes) > 10:
                lines.append(f"    ... and {len(attributes) - 10} more")

        if relationships:
            lines.append("  Relationships:")
            lines.extend(relationships[:10])
            if len(relationships) > 10:
                lines.append(f"    ... and {len(relationships) - 10} more")

        if counts:
            lines.append("  Counts:")
            lines.extend(counts)

        return "\n".join(lines) if lines else "None yet"

    def _format_perceive_completed(self, perceive_completed: Dict[Tuple[str, str], str]) -> str:
        """
        Format completed perceive questions with their answers.
        Shows what information was already gathered via VLM.
        """
        if not perceive_completed:
            return "None yet"

        lines = []
        for i, ((entity_id, question), answer) in enumerate(perceive_completed.items(), 1):
            # Truncate long answers for readability
            short_answer = answer[:100] + "..." if len(answer) > 100 else answer
            lines.append(f"  {i}. {entity_id}: \"{question}\" → \"{short_answer}\"")

            # Limit to 10 most recent
            if i >= 10:
                remaining = len(perceive_completed) - 10
                if remaining > 0:
                    lines.append(f"  ... and {remaining} more")
                break

        return "\n".join(lines)

    def _get_attribute_result_summary(self, entity_id: str, property: str, evidence: 'EvidenceCollection') -> str:
        """Get summary of attribute verification result."""
        for attr_entity_id, attr_property, value, probability in evidence.attributes:
            if attr_entity_id == entity_id and attr_property == property:
                return f"{value} (p={probability:.3f})"
        return "verified"

    def _get_relationship_result_summary(self, subject_id: str, object_id: str, relation: str, evidence: 'EvidenceCollection') -> str:
        """Get summary of relationship verification result."""
        for rel_subject_id, rel_object_id, rel_relation, probability in evidence.relationships:
            if rel_subject_id == subject_id and rel_object_id == object_id and rel_relation == relation:
                return f"p={probability:.3f}"
        return "verified"

    def _get_count_result_summary(self, image_id: str, object_class: str, evidence: 'EvidenceCollection') -> str:
        """Get summary of count verification result."""
        key = f"{image_id}_{object_class}"
        if key in evidence.counts:
            distribution = evidence.counts[key]
            # Find the most likely count value
            max_prob = 0.0
            most_likely_count = 0
            for count_val, prob in distribution.items():
                if prob > max_prob:
                    max_prob = prob
                    most_likely_count = count_val
            return f"most likely {most_likely_count} (p={max_prob:.3f})"
        return "verified"

    def _format_available_object_classes(self, candidate_entities: Dict[str, List[str]]) -> str:
        """Format available object classes that agent can choose from for count verification."""
        if not candidate_entities:
            return "None available"

        classes = list(candidate_entities.keys())
        return ", ".join(sorted(classes))

    def _format_recent_actions(self, state: UnifiedAgentState, max_actions: int = 3) -> str:
        """
        Format last N actions with their results for LLM context.
        Helps LLM see what was just done to avoid immediate repetition.
        """
        if not state.reasoning_trace:
            return "None yet"

        # Get last N reasoning steps
        recent = state.reasoning_trace[-max_actions:]

        lines = []
        start_iter = max(1, state.iteration - len(recent) + 1)
        for i, step in enumerate(recent):
            iter_num = start_iter + i
            lines.append(f"Iteration {iter_num}: {step}")

        return "\n".join(lines)

    def _normalize_count_target(self, target: str) -> Optional[str]:
        """
        Normalize malformed count targets to proper image IDs.

        Examples:
            "image_a" → "image_a" (already correct)
            "image_b_3" → "image_b" (remove extra suffix)
            "bottle_a_2" → "image_a" (extract image from entity ID)
            "a" → "image_a" (add image_ prefix)

        Args:
            target: Raw target string from LLM

        Returns:
            Normalized image ID (e.g., "image_a") or None if unparseable
        """
        target = target.strip()

        # Already correct format
        if target.startswith("image_") and len(target.split("_")) == 2:
            return target

        # Handle malformed image IDs: "image_b_3" → "image_b"
        if target.startswith("image_"):
            parts = target.split("_")
            if len(parts) >= 2:
                return f"image_{parts[1]}"

        # Handle entity IDs: "bottle_a_2" → "image_a"
        if "_" in target:
            parts = target.split("_")
            if len(parts) >= 2:
                # Look for image letter (usually second-to-last part)
                for i in range(1, len(parts)):
                    letter = parts[i]
                    if len(letter) == 1 and letter.isalpha():
                        return f"image_{letter}"

        # Handle bare letters: "a" → "image_a", "b" → "image_b"
        if len(target) == 1 and target.isalpha():
            return f"image_{target}"

        return None

    def _resolve_entity(
        self,
        entity_id: str,
        images: Dict[str, ImageData]
    ) -> Tuple[Optional[str], Optional[ObjectDetection]]:
        """
        Resolve entity_id to (image_id, ObjectDetection).

        Args:
            entity_id: Entity ID in format "label_imageletter_objectid" (e.g., "dog_a_1")
            images: ImageData dictionary

        Returns:
            Tuple[Optional[str], Optional[ObjectDetection]]: (image_id, object) or (None, None)
        """
        try:
            parts = entity_id.split('_')
            if len(parts) < 3:
                return None, None

            image_letter = parts[-2]
            object_index = int(parts[-1])
            image_id = f"image_{image_letter}"

            if image_id not in images:
                return None, None

            # Find object by index
            for obj in images[image_id].objects:
                if obj.object_id == object_index:
                    return image_id, obj

            return None, None

        except (ValueError, IndexError):
            return None, None

    def _crop_to_entity(
        self,
        image: Image.Image,
        bbox: List[float],
        margin: float = 0.15,
        min_size: int = 32
    ) -> Image.Image:
        """
        Crop image to bounding box with percentage margin.

        Args:
            image: PIL Image to crop
            bbox: Bounding box [x1, y1, x2, y2]
            margin: Percentage margin (0.15 = 15%)
            min_size: Minimum dimension in pixels

        Returns:
            Cropped PIL Image with margin
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

        # Ensure minimum size
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        if crop_width < min_size:
            deficit = min_size - crop_width
            crop_x1 = max(0, crop_x1 - deficit / 2)
            crop_x2 = min(width, crop_x2 + deficit / 2)

        if crop_height < min_size:
            deficit = min_size - crop_height
            crop_y1 = max(0, crop_y1 - deficit / 2)
            crop_y2 = min(height, crop_y2 + deficit / 2)

        return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def _crop_to_union(
        self,
        image: Image.Image,
        bbox1: List[float],
        bbox2: List[float],
        margin: float = 0.15
    ) -> Tuple[Image.Image, List[float], List[float]]:
        """
        Crop to union of two bounding boxes with margin.

        Args:
            image: PIL Image to crop
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]
            margin: Percentage margin

        Returns:
            Tuple[Image, adjusted_bbox1, adjusted_bbox2]: Cropped image and adjusted bboxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        width, height = image.size

        # Union bbox
        union_x1 = min(x1_1, x1_2)
        union_y1 = min(y1_1, y1_2)
        union_x2 = max(x2_1, x2_2)
        union_y2 = max(y2_1, y2_2)

        # Add margin
        union_width = union_x2 - union_x1
        union_height = union_y2 - union_y1
        margin_x = union_width * margin
        margin_y = union_height * margin

        crop_x1 = max(0, union_x1 - margin_x)
        crop_y1 = max(0, union_y1 - margin_y)
        crop_x2 = min(width, union_x2 + margin_x)
        crop_y2 = min(height, union_y2 + margin_y)

        # Crop
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

    def _draw_colored_boxes(
        self,
        image: Image.Image,
        bbox1: List[float],
        bbox2: List[float]
    ) -> Image.Image:
        """
        Draw colored bounding boxes (RED for subject, BLUE for object).

        Args:
            image: PIL Image
            bbox1: Subject bbox [x1, y1, x2, y2] - RED
            bbox2: Object bbox [x1, y1, x2, y2] - BLUE

        Returns:
            Image with colored boxes drawn
        """
        from PIL import ImageDraw

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # RED for subject
        draw.rectangle(bbox1, outline="red", width=4)

        # BLUE for object
        draw.rectangle(bbox2, outline="blue", width=4)

        return annotated

    def _check_if_already_perceived(
        self,
        decision: 'PerceiveDecision',
        state: UnifiedAgentState
    ) -> str:
        """
        Check if this perceive question was already asked.
        Returns previous answer if found, empty string if not asked yet.
        """
        return state.evidence.get_perceive_result(decision.target, decision.question)

    def _execute_perceive_action(
        self,
        decision: 'PerceiveDecision',
        state: UnifiedAgentState,
        images: Dict[str, ImageData],
        image_paths: Dict[str, str]
    ):
        """
        Execute perceive action: Ask VLM open-ended question.

        This is for information gathering - "What color is this dog?"
        Results stored in QA history for agent's future decisions.

        Args:
            decision: PerceiveDecision with target and question
            state: Current agent state
            images: ImageData for entity lookup
            image_paths: Paths to image files
        """
        # Check if we already asked this exact question
        previous_answer = self._check_if_already_perceived(decision, state)
        if previous_answer:
            print(f"  [Perceive] {decision.question} (already asked)")
            print(f"    → {previous_answer} (cached)")
            state.add_reasoning(f"Skipped duplicate perceive: already asked '{decision.question}' about {decision.target}")
            return

        # Resolve entity
        image_id, entity = self._resolve_entity(decision.target, images)

        if not entity or not image_id:
            state.add_reasoning(f"⚠ Could not resolve entity {decision.target}")
            return

        # Load image and crop to entity
        image = Image.open(image_paths[image_id])
        cropped = self._crop_to_entity(image, entity.bbox)

        # Ask VLM
        qwen_client = self.model_manager.get_qwen_vl()
        response = qwen_client.run_inference(cropped, decision.question)

        # Store in history and track as completed
        qa_entry = {
            "entity": decision.target,
            "question": decision.question,
            "answer": response
        }
        state.qwen_qa_history.append(qa_entry)
        state.evidence.add_perceive_result(decision.target, decision.question, response)
        state.add_reasoning(f"VLM: {response[:80]}...")

        # Always print VLM Q&A
        print(f"  [Perceive] {decision.question}")
        print(f"    → {response}")

    def _check_if_already_verified(
        self,
        decision: 'VerifyDecision',
        state: UnifiedAgentState
    ) -> List[Tuple]:
        """
        Check if any targets in decision are already verified.
        Returns list of already-verified keys for logging.
        """
        already_done = []

        if decision.verify_type == "attribute":
            for entity_id in decision.targets:
                key = (entity_id, decision.property)
                if key in state.evidence.verifications_completed:
                    already_done.append(key)

        elif decision.verify_type == "relationship":
            if len(decision.targets) >= 2:
                key = (decision.targets[0], decision.targets[1], decision.property)
                if key in state.evidence.verifications_completed:
                    already_done.append(key)

        elif decision.verify_type == "count":
            for target in decision.targets:
                # Parse target to get image_id
                image_id = target if target.startswith("image_") else f"image_{target.split('_')[-2]}"
                key = (image_id, decision.property, "count")
                if key in state.evidence.verifications_completed:
                    already_done.append(key)

        return already_done

    def _execute_verify_action(
        self,
        decision: 'VerifyDecision',
        state: UnifiedAgentState,
        images: Dict[str, ImageData],
        image_paths: Dict[str, str]
    ):
        """
        Execute verify action: Collect probabilities via binary questions.

        This is where we get probabilistic evidence. Key principle:
        Check ALL candidates - don't filter early.

        Args:
            decision: VerifyDecision with verify details
            state: Current agent state
            images: ImageData for entity lookup
            image_paths: Paths to image files
        """
        # Check for duplicates
        already_done = self._check_if_already_verified(decision, state)
        if already_done:
            state.add_reasoning(f"Skipped duplicate verification: {already_done}")
            return

        # Validate count verification uses available object classes
        if decision.verify_type == "count":
            # Validate object class exists
            if decision.property not in state.candidate_entities:
                print(f"  ⚠ Warning: Invalid object class '{decision.property}' for count verification")
                print(f"    Available classes: {list(state.candidate_entities.keys())}")
                state.add_reasoning(f"Failed: Invalid object class '{decision.property}' not in detected objects")
                return

            # Auto-correct malformed count targets
            corrected_targets = []
            for target in decision.targets:
                corrected_target = self._normalize_count_target(target)
                if corrected_target:
                    if corrected_target != target:
                        print(f"  → Auto-corrected count target: '{target}' → '{corrected_target}'")
                    corrected_targets.append(corrected_target)
                else:
                    print(f"  ⚠ Warning: Cannot parse count target '{target}' - skipping")

            if not corrected_targets:
                state.add_reasoning("Failed: No valid count targets after correction")
                return

            # Update decision with corrected targets
            decision.targets = corrected_targets

        if decision.verify_type == "attribute":
            self._verify_attributes(decision, state, images, image_paths)
        elif decision.verify_type == "relationship":
            self._verify_relationships(decision, state, images, image_paths)
        elif decision.verify_type == "count":
            self._verify_counts(decision, state, images)

    def _verify_attributes(
        self,
        decision: 'VerifyDecision',
        state: UnifiedAgentState,
        images: Dict[str, ImageData],
        image_paths: Dict[str, str]
    ):
        """
        Verify attributes using LLM-generated binary questions.

        Key: Check every entity, store all probabilities (even low ones).

        Args:
            decision: VerifyDecision with attribute details
            state: Current agent state
            images: ImageData for entity lookup
            image_paths: Paths to image files
        """
        qwen_client = self.model_manager.get_qwen_vl()

        for entity_id in decision.targets:
            # Resolve entity
            image_id, entity = self._resolve_entity(entity_id, images)

            if not entity or not image_id:
                print(f"      ⚠ Could not resolve entity: {entity_id}")
                continue

            # Load image and crop
            image = Image.open(image_paths[image_id])
            cropped = self._crop_to_entity(image, entity.bbox)

            # Get verification question
            if decision.verification_question:
                # Use LLM-generated question (preferred)
                question = decision.verification_question
            else:
                # Fallback: construct simple binary question
                if decision.value:
                    question = f"Is this {entity.label} {decision.value}?"
                else:
                    # Default to existence check
                    property_phrase = decision.property.replace('_', ' ')
                    question = f"Does this {entity.label} have {property_phrase}?"

            # Always add binary instruction with strict formatting
            prompt = f"""{question}

Respond with ONLY "Yes" or "No". Do not add punctuation or explanation.

Answer:"""

            # Run verification with logits for probability extraction
            response, logits = qwen_client.run_inference_with_logits(cropped, prompt)

            # Extract probability from Yes/No tokens
            raw_probability = get_verifier_probability(logits, response, qwen_client.processor.tokenizer)

            # Apply mode-specific mapping
            probability = self._apply_mode_mapping(raw_probability)

            # Store in evidence
            state.evidence.add_attribute(
                entity_id,
                decision.property,
                decision.value or response.strip(),
                probability
            )

            # Always print verification Q&A
            print(f"  [Verify] {question}")
            print(f"    → {response.strip()} (p={probability:.3f})")

        state.add_reasoning(f"Verified {len(decision.targets)} attributes")

    def _verify_relationships(
        self,
        decision: 'VerifyDecision',
        state: UnifiedAgentState,
        images: Dict[str, ImageData],
        image_paths: Dict[str, str]
    ):
        """
        Verify relationships for entity pairs.

        Key: Check all relevant pairs, store all probabilities.

        Args:
            decision: VerifyDecision with relationship details
            state: Current agent state
            images: ImageData for entity lookup
            image_paths: Paths to image files
        """
        qwen_client = self.model_manager.get_qwen_vl()

        # Generate all pairs from targets
        # Assume targets are provided as pairs, or generate all combinations
        if len(decision.targets) < 2:
            print(f"      ⚠ Need at least 2 entities for relationship verification")
            state.add_reasoning("Not enough entities for relationship verification")
            return

        # For now, check all pairs in targets
        # TODO: Agent should be smarter about which pairs to check
        for i in range(len(decision.targets)):
            for j in range(i + 1, len(decision.targets)):
                subject_id = decision.targets[i]
                object_id = decision.targets[j]

                # Resolve both entities
                subject_image_id, subject_entity = self._resolve_entity(subject_id, images)
                object_image_id, object_entity = self._resolve_entity(object_id, images)

                if not subject_entity or not object_entity:
                    print(f"      ⚠ Could not resolve entities: {subject_id}, {object_id}")
                    continue

                if subject_image_id != object_image_id:
                    print(f"      ⚠ Entities in different images: {subject_id}, {object_id}")
                    continue

                # Load image
                image = Image.open(image_paths[subject_image_id])

                # Crop to union with colored boxes
                cropped, adj_subj_bbox, adj_obj_bbox = self._crop_to_union(
                    image, subject_entity.bbox, object_entity.bbox, margin=0.15
                )

                # Draw colored boxes
                annotated = self._draw_colored_boxes(cropped, adj_subj_bbox, adj_obj_bbox)

                # Get verification question
                if decision.verification_question:
                    # Use LLM-generated question
                    question = decision.verification_question
                else:
                    # Fallback: construct from property
                    if decision.property:
                        relation_phrase = decision.property.replace('_', ' ')
                        question = f"Is the {subject_entity.label} {relation_phrase} the {object_entity.label}?"
                    else:
                        question = f"Describe the spatial relationship between the {subject_entity.label} and the {object_entity.label}."

                prompt = f"""The {subject_entity.label} is marked in RED and the {object_entity.label} is marked in BLUE.

{question}

Respond with ONLY "Yes" or "No". Do not add punctuation or explanation.

Answer:"""

                # Run verification with logits
                response, logits = qwen_client.run_inference_with_logits(annotated, prompt)

                # Extract probability
                raw_probability = get_verifier_probability(logits, response, qwen_client.processor.tokenizer)

                # Apply mode-specific mapping
                probability = self._apply_mode_mapping(raw_probability)

                # Store in evidence
                state.evidence.add_relationship(
                    subject_id,
                    object_id,
                    decision.property or "spatial_relation",
                    probability
                )

                # Always print verification Q&A
                print(f"  [Verify] {question}")
                print(f"    → {response.strip()} (p={probability:.3f})")

        state.add_reasoning(f"Verified relationships for {len(decision.targets)} entities")

    def _verify_counts(
        self,
        decision: 'VerifyDecision',
        state: UnifiedAgentState,
        images: Dict[str, ImageData]
    ):
        """
        Call count processor to get probabilistic count distributions.

        This is like a "tool call" - agent passes image_id and object_class,
        count processor returns full probability distribution.

        Args:
            decision: VerifyDecision with count details
                - targets: List of image_ids (e.g., ["image_a", "image_b"])
                - property: Object class to count (e.g., "bird", "dog")
            state: Current agent state
            images: ImageData for count computation
        """
        from src.pipeline.count_processor import CountProcessor, CountRequirement

        count_processor = CountProcessor()
        object_class = decision.property

        for target in decision.targets:
            # Parse target - could be image_id or entity_id
            if target.startswith("image_"):
                image_id = target
            else:
                # Extract image from entity_id (e.g., "bird_a_2" -> "image_a")
                parts = target.split('_')
                if len(parts) >= 2:
                    image_letter = parts[-2]
                    image_id = f"image_{image_letter}"
                else:
                    continue

            if image_id not in images:
                continue

            # Create count requirement
            requirement = CountRequirement(
                image_id=image_id,
                object_class=object_class,
                required_for_subquestions=[state.original_question]
            )

            # Compute count distribution using Poisson-Binomial
            count_result = count_processor._compute_poisson_binomial_count(
                requirement, images
            )

            if count_result:
                # Store in evidence
                state.evidence.add_count_distribution(
                    count_result.image_id,
                    count_result.object_class,
                    count_result.distribution
                )

                # Store in ImageData for ProbLog
                if image_id in images:
                    if not images[image_id].counts:
                        images[image_id].counts = {}
                    images[image_id].counts[object_class] = {
                        "distribution": count_result.distribution
                    }

                # Always print count verification with most likely count
                most_likely_count = max(count_result.distribution.items(), key=lambda x: x[1])
                print(f"  [Verify] Count of {object_class} in {image_id}")
                print(f"    → {most_likely_count[0]} (p={most_likely_count[1]:.3f})")

        state.add_reasoning(f"Computed count distributions for {object_class}")
