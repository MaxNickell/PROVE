"""
Simplified ReAct Evidence Agent for PROVE pipeline.

A ReAct-style agent that collects evidence to answer questions.
Uses Think → Act → Observe loop until sufficient evidence is gathered.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from PIL import Image

import os

from src.core.model_manager import ModelManager
from src.core.types import ImageData
from src.language.output_models import (
    AgentAction,
    PerceiveAction,
    VerifyAttributeAction,
    VerifyRelationshipAction,
    VerifyCountAction,
    DoneAction
)


@dataclass
class EntityCandidate:
    """A detected entity available for evidence collection."""
    entity_id: str      # e.g., "dog_a_0"
    image_id: str       # e.g., "image_a"
    object_class: str   # e.g., "dog"
    bbox: List[float]   # [x1, y1, x2, y2]
    confidence: float   # detection confidence


@dataclass
class CountEvidence:
    """A single count query result."""
    query_type: str  # "at_least", "at_most", "exactly", "more", "fewer", "equal", "total_*"
    object_class: str
    probability: float
    image_id: str = None  # For single-image queries
    image_id_a: str = None  # For comparison/total queries
    image_id_b: str = None  # For comparison/total queries
    value: int = None  # For queries with N


@dataclass
class EvidenceCollection:
    """All evidence collected for a question."""
    question: str

    # Attribute evidence: (entity_id, attribute, probability)
    attributes: List[Tuple[str, str, float]] = field(default_factory=list)

    # Relationship evidence: (subject_id, object_id, relation, probability)
    relationships: List[Tuple[str, str, str, float]] = field(default_factory=list)

    # Count evidence: list of CountEvidence objects
    counts: List[CountEvidence] = field(default_factory=list)

    # Turn-by-turn action history: [{"thought": "...", "action": "...", "result": "..."}]
    action_history: List[Dict[str, str]] = field(default_factory=list)

    def add_attribute(self, entity_id: str, attribute: str, probability: float):
        self.attributes.append((entity_id, attribute, probability))

    def add_relationship(self, subject_id: str, object_id: str, relation: str, probability: float):
        self.relationships.append((subject_id, object_id, relation, probability))

    def add_count(self, count_evidence: CountEvidence):
        self.counts.append(count_evidence)

    def add_action(self, thought: str, action: str, result: str):
        self.action_history.append({
            "thought": thought,
            "action": action,
            "result": result
        })


class UnifiedAgent:
    """
    ReAct-style evidence collection agent.

    Given a question and detected entities, collects minimal evidence
    needed to answer the question through an iterative Think → Act → Observe loop.
    """

    # Minimum verify actions before "done" is accepted
    MIN_VERIFY_ACTIONS = 1

    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations
        self.model_manager = ModelManager()
        # Detect Nova models for prompt adjustments
        model_id = os.environ.get("LLAMA33_MODEL_ID", "")
        self.is_nova = "nova" in model_id.lower()

    def collect_evidence(
        self,
        question: str,
        images: Dict[str, ImageData],
        image_paths: Dict[str, str]
    ) -> EvidenceCollection:
        """
        Main entry point: collect evidence needed for question.

        Args:
            question: The question to collect evidence for
            images: ImageData with detected objects
            image_paths: Paths to image files

        Returns:
            EvidenceCollection with all gathered evidence
        """
        # Build list of entity candidates from detected objects
        candidates = self._build_candidates(images)

        # Initialize evidence collection
        evidence = EvidenceCollection(question=question)

        # Determine image IDs for prompt generation
        image_ids = sorted(image_paths.keys())

        # ReAct loop
        for iteration in range(self.max_iterations):
            # Think: Get LLM decision on next action
            action = self._get_llm_decision(question, candidates, evidence, iteration, image_ids)

            if action is None:
                break

            # Check for done
            if isinstance(action, DoneAction):
                n_verify = len(evidence.attributes) + len(evidence.relationships) + len(evidence.counts)
                if n_verify < self.MIN_VERIFY_ACTIONS and iteration < self.max_iterations - 1:
                    # Reject premature done — no probabilistic evidence collected
                    evidence.add_action(
                        action.thought, "done (REJECTED)",
                        "REJECTED: You have NOT collected any probabilistic evidence yet. "
                        "You MUST use verify_attribute, verify_relationship, or verify_count "
                        "to collect evidence before stopping. Look at the question again and "
                        "verify each claim. Do NOT say done until you have verified something."
                    )
                    print(f"  [Done REJECTED] iteration={iteration}, verify_actions={n_verify}")
                    continue
                evidence.add_action(action.thought, "done", "Agent stopped")
                break

            # Act & Observe (execute_action records the turn)
            self._execute_action(action, candidates, images, image_paths, evidence)

        return evidence

    def _build_candidates(self, images: Dict[str, ImageData]) -> List[EntityCandidate]:
        """Extract entity candidates from detected objects."""
        candidates = []

        for image_id, image_data in images.items():
            image_letter = image_id.replace("image_", "")

            for obj in image_data.objects:
                entity_id = f"{obj.label.replace(' ', '_')}_{image_letter}_{obj.object_id}"
                candidates.append(EntityCandidate(
                    entity_id=entity_id,
                    image_id=image_id,
                    object_class=obj.label,
                    bbox=obj.bbox,
                    confidence=obj.confidence
                ))

        return candidates

    def _get_llm_decision(
        self,
        question: str,
        candidates: List[EntityCandidate],
        evidence: EvidenceCollection,
        iteration: int,
        image_ids: List[str] = None
    ) -> Optional[AgentAction]:
        """Get next action from LLM using ReAct prompting with Pydantic validation."""

        llm_client = self.model_manager.get_llm_client()

        # Default to two-image setup for backward compatibility
        if image_ids is None:
            image_ids = ["image_a", "image_b"]

        system_prompt = self._build_system_prompt(image_ids)

        # Format candidates grouped by image
        candidates_text = self._format_candidates(candidates)

        # Format action history
        action_history_text = self._format_action_history(evidence)

        # Get unique object classes for count verification
        object_classes = sorted(set(c.object_class for c in candidates))
        object_classes_text = ", ".join(object_classes) if object_classes else "None"

        user_prompt = f"""QUESTION: "{question}"

DETECTED OBJECTS:
{candidates_text}

OBJECT CLASSES FOR COUNTING:
{object_classes_text}

ACTION HISTORY:
{action_history_text}

Iteration: {iteration + 1}/{self.max_iterations}

What is your next action? Output JSON only:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # B fix: retry up to 3 times on LLM failure instead of giving up immediately
        max_retries = 3
        for attempt in range(max_retries):
            try:
                action = llm_client.parse_agent_action(messages, temperature=0)
                return action

            except Exception as e:
                print(f"  Warning: LLM decision failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                # C1 fix: only fallback to done if we have actual verify evidence,
                # not just perceive actions
                n_verify = len(evidence.attributes) + len(evidence.relationships) + len(evidence.counts)
                if n_verify > 0:
                    return DoneAction(thought="Fallback: stopping due to error", action="done")
                return None

    def _execute_action(
        self,
        action: AgentAction,
        candidates: List[EntityCandidate],
        images: Dict[str, ImageData],
        image_paths: Dict[str, str],
        evidence: EvidenceCollection
    ):
        """Execute the decided action and update evidence."""

        if isinstance(action, PerceiveAction):
            self._execute_perceive(action, candidates, image_paths, evidence)

        elif isinstance(action, VerifyAttributeAction):
            self._execute_verify_attribute(action, candidates, image_paths, evidence)

        elif isinstance(action, VerifyRelationshipAction):
            self._execute_verify_relationship(action, candidates, image_paths, evidence)

        elif isinstance(action, VerifyCountAction):
            self._execute_verify_count(action, candidates, images, evidence)

    def _execute_perceive(
        self,
        action: PerceiveAction,
        candidates: List[EntityCandidate],
        image_paths: Dict[str, str],
        evidence: EvidenceCollection
    ):
        """Ask VLM an open-ended question about an entity or the whole image."""

        # Build action string for logging
        if action.entity_id:
            action_str = f'perceive(image_id={action.image_id}, entity_id={action.entity_id}, question="{action.question}")'
        else:
            action_str = f'perceive(image_id={action.image_id}, question="{action.question}")'

        # Get image path
        image_path = image_paths.get(action.image_id)
        if not image_path:
            evidence.add_action(action.thought, action_str, "Failed: image path not found")
            return

        try:
            image = Image.open(image_path)

            if action.entity_id:
                # Entity-level perceive: crop to entity
                entity = self._find_entity(action.entity_id, candidates)
                if not entity:
                    hint = self._available_entities_hint(action.image_id, candidates)
                    evidence.add_action(action.thought, action_str, f"Failed: entity {action.entity_id} not found. {hint}")
                    return

                if entity.image_id != action.image_id:
                    evidence.add_action(action.thought, action_str, f"Failed: entity is in {entity.image_id}, not {action.image_id}")
                    return

                x1, y1, x2, y2 = [int(c) for c in entity.bbox]
                image = image.crop((x1, y1, x2, y2))
                log_target = action.entity_id
            else:
                # Image-level perceive: use whole image
                log_target = action.image_id

            # Ask VLM
            qwen_vl = self.model_manager.get_qwen_vl()
            answer = qwen_vl.run_inference(image, action.question).strip()

            # Record action with result
            evidence.add_action(action.thought, action_str, f'"{answer}"')
            print(f"  [Perceive] {log_target}: {action.question}")
            print(f"    → {answer}")

        except Exception as e:
            evidence.add_action(action.thought, action_str, f"Failed: {e}")
            print(f"  [Perceive] {log_target} → Failed: {e}")

    def _execute_verify_attribute(
        self,
        action: VerifyAttributeAction,
        candidates: List[EntityCandidate],
        image_paths: Dict[str, str],
        evidence: EvidenceCollection
    ):
        """Verify if an entity has a specific attribute using BLIP-ITM."""
        action_str = f'verify_attribute(image_id={action.image_id}, entity_id={action.entity_id}, attribute={action.attribute}, verification="{action.verification}")'

        # Find the entity
        entity = self._find_entity(action.entity_id, candidates)
        if not entity:
            hint = self._available_entities_hint(action.image_id, candidates)
            evidence.add_action(action.thought, action_str, f"Failed: entity {action.entity_id} not found. {hint}")
            return

        # Validate image_id matches entity
        if entity.image_id != action.image_id:
            evidence.add_action(action.thought, action_str, f"Failed: entity is in {entity.image_id}, not {action.image_id}")
            return

        # Get image path
        image_path = image_paths.get(action.image_id)
        if not image_path:
            evidence.add_action(action.thought, action_str, f"Failed: image path not found")
            return

        try:
            # Use BLIP-ITM verifier for attribute verification
            blip_verifier = self.model_manager.get_blip_verifier()
            probability = blip_verifier.verify_attribute(
                image=image_path,
                bbox=entity.bbox,
                verification=action.verification
            )

            # Store evidence for probability computation
            evidence.add_attribute(action.entity_id, action.attribute, probability)

            # Record action with result
            evidence.add_action(action.thought, action_str, f"p={probability:.3f}")
            print(f"  [Verify Attribute] {action.entity_id}.{action.attribute}")
            print(f"    verification: \"{action.verification}\"")
            print(f"    → p={probability:.3f}")

        except Exception as e:
            evidence.add_action(action.thought, action_str, f"Failed: {e}")
            print(f"  [Verify Attribute] {action.entity_id} → Failed: {e}")

    def _execute_verify_relationship(
        self,
        action: VerifyRelationshipAction,
        candidates: List[EntityCandidate],
        image_paths: Dict[str, str],
        evidence: EvidenceCollection
    ):
        """Verify if two entities have a relationship using BLIP-ITM."""
        action_str = f'verify_relationship(image_id={action.image_id}, subject_id={action.subject_id}, object_id={action.object_id}, relation={action.relation}, verification="{action.verification}")'

        # Find both entities
        subject = self._find_entity(action.subject_id, candidates)
        obj = self._find_entity(action.object_id, candidates)

        if not subject or not obj:
            missing = []
            if not subject:
                missing.append(action.subject_id)
            if not obj:
                missing.append(action.object_id)
            hint = self._available_entities_hint(action.image_id, candidates)
            evidence.add_action(action.thought, action_str, f"Failed: entity {', '.join(missing)} not found. {hint}")
            return

        # Validate both entities are in the specified image
        if subject.image_id != action.image_id:
            evidence.add_action(action.thought, action_str, f"Failed: subject is in {subject.image_id}, not {action.image_id}")
            return

        if obj.image_id != action.image_id:
            evidence.add_action(action.thought, action_str, f"Failed: object is in {obj.image_id}, not {action.image_id}")
            return

        image_path = image_paths.get(action.image_id)
        if not image_path:
            evidence.add_action(action.thought, action_str, "Failed: image path not found")
            return

        try:
            # Use BLIP-ITM verifier for relationship verification
            blip_verifier = self.model_manager.get_blip_verifier()
            probability = blip_verifier.verify_relationship(
                image=image_path,
                bbox1=subject.bbox,
                bbox2=obj.bbox,
                verification=action.verification
            )

            # Store evidence for probability computation
            evidence.add_relationship(action.subject_id, action.object_id, action.relation, probability)

            # Record action with result
            evidence.add_action(action.thought, action_str, f"p={probability:.3f}")
            print(f"  [Verify Relationship] {action.subject_id} {action.relation} {action.object_id}")
            print(f"    verification: \"{action.verification}\"")
            print(f"    → p={probability:.3f}")

        except Exception as e:
            evidence.add_action(action.thought, action_str, f"Failed: {e}")
            print(f"  [Verify Relationship] {action.subject_id} {action.relation} {action.object_id} → Failed: {e}")

    def _execute_verify_count(
        self,
        action: VerifyCountAction,
        candidates: List[EntityCandidate],
        images: Dict[str, ImageData],
        evidence: EvidenceCollection
    ):
        """Execute count verification queries using Poisson-Binomial distributions."""
        query_type = action.query_type
        object_class = action.object_class

        # Build action string for logging
        if query_type in ["at_least", "at_most", "exactly"]:
            action_str = f'verify_count(query_type={query_type}, object_class={object_class}, image_id={action.image_id}, value={action.value})'
        elif query_type in ["more", "fewer", "equal"]:
            action_str = f'verify_count(query_type={query_type}, object_class={object_class}, image_id_a={action.image_id_a}, image_id_b={action.image_id_b})'
        else:  # total_*
            action_str = f'verify_count(query_type={query_type}, object_class={object_class}, image_id_a={action.image_id_a}, image_id_b={action.image_id_b}, value={action.value})'

        try:
            # Validate required fields before computing
            if query_type in ["at_least", "at_most", "exactly"]:
                if not action.image_id:
                    evidence.add_action(action.thought, action_str, f"Failed: missing image_id for {query_type} query")
                    print(f"  [Verify Count] {query_type}({object_class}) → Failed: missing image_id")
                    return
                if action.value is None:
                    evidence.add_action(action.thought, action_str, f"Failed: missing value for {query_type} query")
                    print(f"  [Verify Count] {query_type}({object_class}) → Failed: missing value")
                    return
            elif query_type in ["more", "fewer", "equal"]:
                if not action.image_id_a or not action.image_id_b:
                    evidence.add_action(action.thought, action_str, f"Failed: missing image_id_a or image_id_b for {query_type} query")
                    print(f"  [Verify Count] {query_type}({object_class}) → Failed: missing image_id_a/b")
                    return
            else:  # total_*
                if not action.image_id_a or not action.image_id_b:
                    evidence.add_action(action.thought, action_str, f"Failed: missing image_id_a or image_id_b for {query_type} query")
                    print(f"  [Verify Count] {query_type}({object_class}) → Failed: missing image_id_a/b")
                    return
                if action.value is None:
                    evidence.add_action(action.thought, action_str, f"Failed: missing value for {query_type} query")
                    print(f"  [Verify Count] {query_type}({object_class}) → Failed: missing value")
                    return

            # Compute the probability based on query type
            if query_type in ["at_least", "at_most", "exactly"]:
                probability = self._compute_single_image_count(
                    query_type, object_class, action.image_id, action.value, candidates
                )
                count_ev = CountEvidence(
                    query_type=query_type,
                    object_class=object_class,
                    probability=probability,
                    image_id=action.image_id,
                    value=action.value
                )
            elif query_type in ["more", "fewer", "equal"]:
                probability = self._compute_comparison_count(
                    query_type, object_class, action.image_id_a, action.image_id_b, candidates
                )
                count_ev = CountEvidence(
                    query_type=query_type,
                    object_class=object_class,
                    probability=probability,
                    image_id_a=action.image_id_a,
                    image_id_b=action.image_id_b
                )
            else:  # total_*
                probability = self._compute_total_count(
                    query_type, object_class, action.image_id_a, action.image_id_b, action.value, candidates
                )
                count_ev = CountEvidence(
                    query_type=query_type,
                    object_class=object_class,
                    probability=probability,
                    image_id_a=action.image_id_a,
                    image_id_b=action.image_id_b,
                    value=action.value
                )

            # Store evidence
            evidence.add_count(count_ev)

            # Record action with result
            evidence.add_action(action.thought, action_str, f"p={probability:.3f}")
            print(f"  [Verify Count] {query_type}({object_class}) → p={probability:.3f}")

        except Exception as e:
            evidence.add_action(action.thought, action_str, f"Failed: {e}")
            print(f"  [Verify Count] {query_type}({object_class}) → Failed: {e}")

    def _compute_single_image_count(
        self,
        query_type: str,
        object_class: str,
        image_id: str,
        value: int,
        candidates: List[EntityCandidate]
    ) -> float:
        """Compute P(count >= N), P(count <= N), or P(count == N) for single image."""
        # Get distribution for this image
        dist = self._get_count_distribution(object_class, image_id, candidates)

        if query_type == "at_least":
            # P(count >= N) = sum of P(k) for k >= N
            return sum(p for k, p in dist.items() if k >= value)
        elif query_type == "at_most":
            # P(count <= N) = sum of P(k) for k <= N
            return sum(p for k, p in dist.items() if k <= value)
        elif query_type == "exactly":
            # P(count == N)
            return dist.get(value, 0.0)
        else:
            raise ValueError(f"Unknown single-image query type: {query_type}")

    def _compute_comparison_count(
        self,
        query_type: str,
        object_class: str,
        image_id_a: str,
        image_id_b: str,
        candidates: List[EntityCandidate]
    ) -> float:
        """Compute P(count_a > count_b), P(count_a < count_b), or P(count_a == count_b)."""
        dist_a = self._get_count_distribution(object_class, image_id_a, candidates)
        dist_b = self._get_count_distribution(object_class, image_id_b, candidates)

        probability = 0.0
        for count_a, prob_a in dist_a.items():
            for count_b, prob_b in dist_b.items():
                if query_type == "more" and count_a > count_b:
                    probability += prob_a * prob_b
                elif query_type == "fewer" and count_a < count_b:
                    probability += prob_a * prob_b
                elif query_type == "equal" and count_a == count_b:
                    probability += prob_a * prob_b

        return probability

    def _compute_total_count(
        self,
        query_type: str,
        object_class: str,
        image_id_a: str,
        image_id_b: str,
        value: int,
        candidates: List[EntityCandidate]
    ) -> float:
        """Compute P(total >= N), P(total <= N), or P(total == N) across both images."""
        dist_a = self._get_count_distribution(object_class, image_id_a, candidates)
        dist_b = self._get_count_distribution(object_class, image_id_b, candidates)

        # Convolve distributions to get total distribution
        total_dist = {}
        for count_a, prob_a in dist_a.items():
            for count_b, prob_b in dist_b.items():
                total = count_a + count_b
                total_dist[total] = total_dist.get(total, 0.0) + prob_a * prob_b

        if query_type == "total_at_least":
            return sum(p for k, p in total_dist.items() if k >= value)
        elif query_type == "total_at_most":
            return sum(p for k, p in total_dist.items() if k <= value)
        elif query_type == "total_exactly":
            return total_dist.get(value, 0.0)
        else:
            raise ValueError(f"Unknown total query type: {query_type}")

    def _get_count_distribution(
        self,
        object_class: str,
        image_id: str,
        candidates: List[EntityCandidate]
    ) -> Dict[int, float]:
        """Get Poisson-Binomial distribution for object class in image."""
        matching = [c for c in candidates
                   if c.image_id == image_id and c.object_class == object_class]

        if not matching:
            return {0: 1.0}  # No detections = count is 0 with certainty

        probabilities = [c.confidence for c in matching]
        return self._compute_poisson_binomial(probabilities)

    def _compute_poisson_binomial(self, probabilities: List[float]) -> Dict[int, float]:
        """Compute Poisson-Binomial distribution using DP."""
        P = [1.0]

        for p in probabilities:
            new_P = [0.0] * (len(P) + 1)
            for k in range(len(P)):
                new_P[k] += P[k] * (1 - p)
                new_P[k + 1] += P[k] * p
            P = new_P

        return {k: P[k] for k in range(len(P))}

    def _find_entity(self, entity_id: str, candidates: List[EntityCandidate]) -> Optional[EntityCandidate]:
        """Find entity by ID."""
        for c in candidates:
            if c.entity_id == entity_id:
                return c
        return None

    def _available_entities_hint(self, image_id: str, candidates: List[EntityCandidate]) -> str:
        """Format available entity IDs for a given image for error messages."""
        matching = [c.entity_id for c in candidates if c.image_id == image_id]
        if not matching:
            return f"No entities detected in {image_id}."
        return f"Available entities in {image_id}: {', '.join(matching)}"

    def _format_candidates(self, candidates: List[EntityCandidate]) -> str:
        """Format candidates for LLM prompt with explicit labeling."""
        if not candidates:
            return "None"

        # Group by image
        by_image: Dict[str, List[EntityCandidate]] = {}
        for c in candidates:
            if c.image_id not in by_image:
                by_image[c.image_id] = []
            by_image[c.image_id].append(c)

        lines = []
        # Format with explicit labels: Image A, image_id: image_a
        for image_id in sorted(by_image.keys()):
            # Convert image_id to display name (image_a -> Image A)
            image_letter = image_id.replace("image_", "").upper()
            lines.append(f"Image {image_letter}, image_id: {image_id}")
            for c in by_image[image_id]:
                # Explicit format: object_id: <id>, object_class: <class>
                lines.append(f"  - object_id: {c.entity_id}, object_class: {c.object_class}")
            lines.append("")  # Empty line between images

        return "\n".join(lines).rstrip()

    def _format_action_history(self, evidence: EvidenceCollection) -> str:
        """Format turn-by-turn action history for LLM prompt."""
        if not evidence.action_history:
            return "None yet"

        lines = []
        for i, turn in enumerate(evidence.action_history, 1):
            lines.append(f"[Turn {i}]")
            lines.append(f"Thought: {turn['thought']}")
            lines.append(f"Action: {turn['action']}")
            lines.append(f"Result: {turn['result']}")
            lines.append("")  # Empty line between turns

        return "\n".join(lines).rstrip()

    def _build_system_prompt(self, image_ids: List[str]) -> str:
        """Build system prompt dynamically based on the number of images."""
        n_images = len(image_ids)
        multi_image = n_images > 1

        # Image header
        if n_images == 1:
            image_word = "ONE image"
            images_list = f"- Image A, image_id: {image_ids[0]}"
        else:
            image_word = f"{n_images} images"
            image_labels = []
            for img_id in image_ids:
                letter = img_id.replace("image_", "").upper()
                image_labels.append(f"- Image {letter}, image_id: {img_id}")
            if n_images == 2:
                image_labels[0] += " (the question may refer to Image A as the left or the first image)"
                image_labels[1] += " (the question may refer to Image B as the right or the second image)"
            images_list = "\n".join(image_labels)

        # Valid image_id values for the rules section
        valid_ids = ", ".join(f'"{i}"' for i in image_ids)

        # Evidence chain examples — adapt based on single vs multi
        chain_examples = """
1. Count + Relationship: "At least one dog is sitting on the couch"
   → Verify the count of dogs AND verify the "sitting on" relationship between dog and couch

2. Count + Attribute: "There are two red cars"
   → Verify the count of cars AND verify the "red" attribute for each car

3. Attribute + Relationship: "The large cat is next to the bowl"
   → Verify the "large" attribute for the cat AND verify the "next to" relationship between cat and bowl

4. Multiple Attributes: "The car is red and shiny"
   → Verify the "red" attribute AND verify the "shiny" attribute

5. Relationship Chain: "The bird is on the branch which is above the water"
   → Verify the "on" relationship between bird and branch AND verify the "above" relationship between branch and water"""

        if multi_image:
            chain_examples += """

6. Count Comparison + Attribute: "There are more striped shirts in image A than B"
   → Verify the count comparison between images AND verify the "striped" attribute for each of the shirts"""

        # Example image_id for action examples
        ex_img = image_ids[0]
        ex_letter = ex_img.replace("image_", "")

        # Count action section
        count_section = f"""4. verify_count - Verify count-related queries. Returns probability (0.0-1.0).
   Required fields: thought, action, query_type, object_class

   Single-image queries (also requires: image_id, value):
   - "at_least": P(count >= N) - {{"thought": "...", "action": "verify_count", "query_type": "at_least", "object_class": "dog", "image_id": "{ex_img}", "value": 2}}
   - "at_most": P(count <= N) - {{"thought": "...", "action": "verify_count", "query_type": "at_most", "object_class": "dog", "image_id": "{ex_img}", "value": 2}}
   - "exactly": P(count == N) - {{"thought": "...", "action": "verify_count", "query_type": "exactly", "object_class": "dog", "image_id": "{ex_img}", "value": 2}}"""

        if multi_image:
            img_a, img_b = image_ids[0], image_ids[1]
            count_section += f"""

   Cross-image comparison (also requires: image_id_a, image_id_b):
   - "more": P(count_a > count_b) - {{"thought": "...", "action": "verify_count", "query_type": "more", "object_class": "dog", "image_id_a": "{img_a}", "image_id_b": "{img_b}"}}
   - "fewer": P(count_a < count_b) - {{"thought": "...", "action": "verify_count", "query_type": "fewer", "object_class": "dog", "image_id_a": "{img_a}", "image_id_b": "{img_b}"}}
   - "equal": P(count_a == count_b) - {{"thought": "...", "action": "verify_count", "query_type": "equal", "object_class": "dog", "image_id_a": "{img_a}", "image_id_b": "{img_b}"}}

   Total across images (also requires: image_id_a, image_id_b, value):
   - "total_exactly": P(total == N) - {{"thought": "...", "action": "verify_count", "query_type": "total_exactly", "object_class": "dog", "image_id_a": "{img_a}", "image_id_b": "{img_b}", "value": 5}}
   - "total_at_least": P(total >= N) - {{"thought": "...", "action": "verify_count", "query_type": "total_at_least", "object_class": "dog", "image_id_a": "{img_a}", "image_id_b": "{img_b}", "value": 5}}
   - "total_at_most": P(total <= N) - {{"thought": "...", "action": "verify_count", "query_type": "total_at_most", "object_class": "dog", "image_id_a": "{img_a}", "image_id_b": "{img_b}", "value": 5}}"""

        system_prompt = f"""You are a ReAct evidence agent collecting evidence to answer a visual question about {image_word}.

IMAGES:
{images_list}

GOAL:
Collect all evidence required to answer the question. When done, the collected evidence alone should be sufficient to determine if the statement is true or false.

EVIDENCE CHAINS:
Questions often contain multiple connected claims. You must verify ALL parts of the chain.

Common patterns:
{chain_examples}

Verifying only PART of a chain is INCOMPLETE - you must verify ALL parts.

ACTIONS (output ONE as JSON):

1. perceive - Ask open-ended question to gather information
   Required fields: thought, action, image_id, question

   Entity-level (also requires: entity_id):
   - {{"thought": "I need to know the dog's color", "action": "perceive", "image_id": "{ex_img}", "entity_id": "dog_{ex_letter}_0", "question": "What color is this dog?"}}

   Image-level (no entity_id needed):
   - {{"thought": "I need to understand the scene", "action": "perceive", "image_id": "{ex_img}", "question": "What is happening in this image?"}}

2. verify_attribute - Check if entity has specific attribute. Returns probability (0.0-1.0).
   Required fields: thought, action, image_id, entity_id, attribute, verification
   - attribute: the attribute being verified (e.g., "orange", "wooden", "showing teeth")
   - verification: natural language describing the attribute (e.g., "an orange dog", "a dog showing its teeth")
   Examples:
   - {{"thought": "Verifying the dog is orange", "action": "verify_attribute", "image_id": "{ex_img}", "entity_id": "dog_{ex_letter}_0", "attribute": "orange", "verification": "an orange dog"}}

3. verify_relationship - Check relationship between two entities in SAME image. Returns probability (0.0-1.0).
   Supports both spatial relations (e.g., on, next to, left of, behind, above, etc.) and interactions (e.g., wearing, holding, eating, looking at, sitting on, etc.).
   Required fields: thought, action, image_id, subject_id, object_id, relation, verification
   - relation: the relationship being verified (e.g., "on top of", "wearing")
   - verification: natural language describing the relationship (e.g., "a bird on top of a buffalo", "a man wearing a coat")
   Examples:
   - Spatial: {{"thought": "Checking if bird is on buffalo", "action": "verify_relationship", "image_id": "{ex_img}", "subject_id": "bird_{ex_letter}_0", "object_id": "buffalo_{ex_letter}_1", "relation": "on top of", "verification": "a bird on top of a buffalo"}}

{count_section}

5. done - Stop ONLY when ALL evidence has been collected
   Required fields: thought, action
   Example: {{"thought": "I have verified ALL attributes, relationships, and counts mentioned in the question", "action": "done"}}

PERCEPTION SEMANTICS:
- Perceive gathers contextual information to help you decide what to verify
- Perceive does NOT collect probabilistic evidence - it only provides textual context
- Image-level perceive: Use to understand the overall scene, context, or relationships in the whole image
- Entity-level perceive: Use to gather specific information about a particular object
- Use perceive to investigate, then verify to collect probabilistic evidence

VERIFICATION SEMANTICS:
- Verification returns a probability score (0.0-1.0) representing the model's confidence
- This score is DETERMINISTIC - verifying the same thing again will return the same result
- Low probability is valid evidence that something is likely FALSE
- High probability is valid evidence that something is likely TRUE
- Do NOT re-verify the same attribute or relationship - once you have the probability, that IS the evidence

RULES:
- COMPLETENESS IS REQUIRED: Verify ALL attributes, relationships, and counts in the question
- Do NOT stop early - even if partial evidence seems sufficient to answer, you must verify everything
- If the question mentions multiple entities/attributes, verify EACH ONE
- Perceive alone is NOT sufficient - you must verify to collect evidence
- Stop (done) ONLY after verifying every claim in the question
- Do NOT repeat actions - check ACTION HISTORY before each action
- Output valid JSON only
- entity_id must match exactly from the DETECTED OBJECTS list
- image_id must be one of: {valid_ids}
- For verify_count, object_class MUST be one of the exact names from OBJECT CLASSES FOR COUNTING"""

        # Nova-specific prompt additions — Nova models tend to stop too early
        if self.is_nova:
            system_prompt += """

CRITICAL ADDITIONAL RULES:
- You MUST perform at least one verify_attribute, verify_relationship, or verify_count action BEFORE saying done
- Do NOT say done after only perceive actions — perceive alone does NOT collect probabilistic evidence
- If you are unsure what to verify, re-read the question carefully and identify ALL claims that need verification
- If you tried to verify something and got a low probability, that IS valid evidence — do NOT stop because of low scores
- You should typically perform 5-15 actions per question. If you have done fewer than 5, you are likely missing evidence
- After each verify action, ask yourself: "Have I verified EVERY claim in the question?" If not, continue"""

        return system_prompt
