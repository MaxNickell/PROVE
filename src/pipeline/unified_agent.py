"""
Simplified ReAct Evidence Agent for PROVE pipeline.

A ReAct-style agent that collects evidence to answer subquestions.
Uses Think → Act → Observe loop until sufficient evidence is gathered.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ImageData
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
class EvidenceCollection:
    """All evidence collected for one subquestion."""
    subquestion: str

    # Attribute evidence: (entity_id, attribute_class, value, probability)
    attributes: List[Tuple[str, str, str, float]] = field(default_factory=list)

    # Relationship evidence: (subject_id, object_id, relation, probability)
    relationships: List[Tuple[str, str, str, float]] = field(default_factory=list)

    # Count distributions: {"image_a_dog": {0: 0.1, 1: 0.3, 2: 0.6}}
    counts: Dict[str, Dict[int, float]] = field(default_factory=dict)

    # Agent's reasoning trace
    reasoning_trace: List[str] = field(default_factory=list)

    # Perceive Q&A history: [{"entity_id": "...", "question": "...", "answer": "..."}]
    perceive_history: List[Dict[str, str]] = field(default_factory=list)

    def add_attribute(self, entity_id: str, attribute: str, value: str, probability: float):
        self.attributes.append((entity_id, attribute, value, probability))

    def add_relationship(self, subject_id: str, object_id: str, relation: str, probability: float):
        self.relationships.append((subject_id, object_id, relation, probability))

    def add_count(self, image_id: str, object_class: str, distribution: Dict[int, float]):
        key = f"{image_id}_{object_class}"
        self.counts[key] = distribution

    def add_perceive(self, entity_id: str, question: str, answer: str):
        self.perceive_history.append({
            "entity_id": entity_id,
            "question": question,
            "answer": answer
        })


class UnifiedAgent:
    """
    ReAct-style evidence collection agent.

    Given a subquestion and detected entities, collects minimal evidence
    needed to answer the question through an iterative Think → Act → Observe loop.
    """

    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations
        self.model_manager = ModelManager()

    def collect_evidence(
        self,
        subquestion: BinarySubquestion,
        images: Dict[str, ImageData],
        image_paths: Dict[str, str]
    ) -> EvidenceCollection:
        """
        Main entry point: collect evidence needed for subquestion.

        Args:
            subquestion: The question to collect evidence for
            images: ImageData with detected objects
            image_paths: Paths to image files

        Returns:
            EvidenceCollection with all gathered evidence
        """
        # Build list of entity candidates from detected objects
        candidates = self._build_candidates(images)

        # Initialize evidence collection
        evidence = EvidenceCollection(subquestion=subquestion.question)
        evidence.reasoning_trace.append(f"Starting evidence collection for: {subquestion.question}")
        evidence.reasoning_trace.append(f"Available candidates: {len(candidates)} entities")

        # ReAct loop
        for iteration in range(self.max_iterations):
            # Think: Get LLM decision on next action
            action = self._get_llm_decision(subquestion.question, candidates, evidence, iteration)

            if action is None:
                evidence.reasoning_trace.append("Decision failed, stopping")
                break

            # Log the thought
            evidence.reasoning_trace.append(f"[{iteration+1}] {action.thought}")

            # Check for done
            if isinstance(action, DoneAction):
                evidence.reasoning_trace.append("Agent decided evidence is sufficient")
                break

            # Act & Observe
            self._execute_action(action, candidates, images, image_paths, evidence)

        return evidence

    def _build_candidates(self, images: Dict[str, ImageData]) -> List[EntityCandidate]:
        """Extract entity candidates from detected objects."""
        candidates = []

        for image_id, image_data in images.items():
            image_letter = image_id.replace("image_", "")

            for obj in image_data.objects:
                entity_id = f"{obj.label}_{image_letter}_{obj.object_id}"
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
        subquestion: str,
        candidates: List[EntityCandidate],
        evidence: EvidenceCollection,
        iteration: int
    ) -> Optional[AgentAction]:
        """Get next action from LLM using ReAct prompting with Pydantic validation."""

        llm_client = self.model_manager.get_llm_client()

        system_prompt = """You are a ReAct evidence agent collecting evidence to answer a visual question about TWO images.

IMAGES:
- Image A, image_id: image_a
- Image B, image_id: image_b

ACTIONS (output ONE as JSON):

1. perceive - Ask open-ended question about an entity to gather information
   Required fields: thought, action, image_id, entity_id, question
   Example: {"thought": "I need to know the dog's color", "action": "perceive", "image_id": "image_a", "entity_id": "dog_a_0", "question": "What color is this dog?"}

2. verify_attribute - Check if entity has specific attribute (returns Yes/No probability)
   Required fields: thought, action, image_id, entity_id, attribute, value
   Example: {"thought": "Verifying the dog is orange", "action": "verify_attribute", "image_id": "image_a", "entity_id": "dog_a_0", "attribute": "color", "value": "orange"}

3. verify_relationship - Check spatial relationship between two entities in SAME image
   Required fields: thought, action, image_id, subject_id, object_id, relation
   Example: {"thought": "Checking if bird is on buffalo", "action": "verify_relationship", "image_id": "image_a", "subject_id": "bird_a_0", "object_id": "buffalo_a_1", "relation": "on_top_of"}

4. verify_count - Count objects of a class in an image
   Required fields: thought, action, image_id, object_class
   Example: {"thought": "Need to count dogs in image A", "action": "verify_count", "image_id": "image_a", "object_class": "dog"}

5. done - Stop when sufficient evidence collected
   Required fields: thought, action
   Example: {"thought": "I have verified the dog is orange with high probability", "action": "done"}

RULES:
- Use perceive to investigate BEFORE knowing what to verify
- Use verify actions to collect evidence
- Stop (done) as soon as you have sufficient evidence to confidently answer the question
- Output valid JSON only
- entity_id must match exactly from the DETECTED OBJECTS list
- image_id must be "image_a" or "image_b" """

        # Format candidates grouped by image
        candidates_text = self._format_candidates(candidates)

        # Format evidence collected so far
        evidence_text = self._format_evidence(evidence)

        # Format perceive history
        perceive_text = self._format_perceive_history(evidence.perceive_history)

        user_prompt = f"""QUESTION: "{subquestion}"

DETECTED OBJECTS:
{candidates_text}

EVIDENCE COLLECTED SO FAR:
{evidence_text}

PERCEIVE HISTORY:
{perceive_text}

Iteration: {iteration + 1}/{self.max_iterations}

What is your next action? Output JSON only:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            action = llm_client.parse_agent_action(messages, temperature=0.2)
            return action

        except Exception as e:
            print(f"  Warning: LLM decision failed: {e}")
            # Fallback to done if we've collected some evidence
            if evidence.attributes or evidence.relationships or evidence.counts:
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
        """Ask VLM an open-ended question about an entity."""

        # Find the entity
        entity = self._find_entity(action.entity_id, candidates)
        if not entity:
            evidence.reasoning_trace.append(f"Perceive failed: entity {action.entity_id} not found")
            return

        # Validate image_id matches entity
        if entity.image_id != action.image_id:
            evidence.reasoning_trace.append(f"Perceive failed: entity {action.entity_id} is in {entity.image_id}, not {action.image_id}")
            return

        # Get cropped image
        image_path = image_paths.get(action.image_id)
        if not image_path:
            evidence.reasoning_trace.append(f"Perceive failed: image path not found for {action.image_id}")
            return

        try:
            image = Image.open(image_path)
            x1, y1, x2, y2 = [int(c) for c in entity.bbox]
            cropped = image.crop((x1, y1, x2, y2))

            # Ask VLM
            qwen = self.model_manager.get_qwen_vl()
            answer = qwen.run_inference(cropped, action.question)

            # Store result
            evidence.add_perceive(action.entity_id, action.question, answer.strip())
            print(f"  [Perceive] {action.entity_id}: {action.question}")
            print(f"    → {answer.strip()}")

        except Exception as e:
            evidence.reasoning_trace.append(f"Perceive failed: {e}")

    def _execute_verify_attribute(
        self,
        action: VerifyAttributeAction,
        candidates: List[EntityCandidate],
        image_paths: Dict[str, str],
        evidence: EvidenceCollection
    ):
        """Verify if an entity has a specific attribute using BLIP-ITM."""

        # Find the entity
        entity = self._find_entity(action.entity_id, candidates)
        if not entity:
            evidence.reasoning_trace.append(f"Verify attribute failed: entity {action.entity_id} not found")
            return

        # Validate image_id matches entity
        if entity.image_id != action.image_id:
            evidence.reasoning_trace.append(f"Verify attribute failed: entity {action.entity_id} is in {entity.image_id}, not {action.image_id}")
            return

        # Get image path
        image_path = image_paths.get(action.image_id)
        if not image_path:
            evidence.reasoning_trace.append(f"Verify attribute failed: image path not found for {action.image_id}")
            return

        try:
            # Use BLIP-ITM verifier for attribute verification
            blip_verifier = self.model_manager.get_blip_verifier()
            probability = blip_verifier.verify_attribute(
                image=image_path,
                bbox=entity.bbox,
                object_class=entity.object_class,
                attr_value=action.value
            )

            # Store evidence
            evidence.add_attribute(action.entity_id, action.attribute, action.value, probability)
            print(f"  [Verify Attribute] {action.entity_id}.{action.attribute}={action.value}")
            print(f"    → p={probability:.3f}")

        except Exception as e:
            evidence.reasoning_trace.append(f"Verify attribute failed: {e}")

    def _execute_verify_relationship(
        self,
        action: VerifyRelationshipAction,
        candidates: List[EntityCandidate],
        image_paths: Dict[str, str],
        evidence: EvidenceCollection
    ):
        """Verify if two entities have a relationship using BLIP-ITM."""

        # Find both entities
        subject = self._find_entity(action.subject_id, candidates)
        obj = self._find_entity(action.object_id, candidates)

        if not subject or not obj:
            evidence.reasoning_trace.append(f"Verify relationship failed: entity not found")
            return

        # Validate both entities are in the specified image
        if subject.image_id != action.image_id:
            evidence.reasoning_trace.append(f"Verify relationship failed: subject {action.subject_id} is in {subject.image_id}, not {action.image_id}")
            return

        if obj.image_id != action.image_id:
            evidence.reasoning_trace.append(f"Verify relationship failed: object {action.object_id} is in {obj.image_id}, not {action.image_id}")
            return

        image_path = image_paths.get(action.image_id)
        if not image_path:
            evidence.reasoning_trace.append(f"Verify relationship failed: image path not found for {action.image_id}")
            return

        try:
            # Use BLIP-ITM verifier for relationship verification
            blip_verifier = self.model_manager.get_blip_verifier()
            probability = blip_verifier.verify_relationship(
                image=image_path,
                bbox1=subject.bbox,
                bbox2=obj.bbox,
                obj1_class=subject.object_class,
                obj2_class=obj.object_class,
                relation=action.relation
            )

            # Store evidence
            evidence.add_relationship(action.subject_id, action.object_id, action.relation, probability)
            print(f"  [Verify Relationship] {action.subject_id} {action.relation} {action.object_id}")
            print(f"    → p={probability:.3f}")

        except Exception as e:
            evidence.reasoning_trace.append(f"Verify relationship failed: {e}")

    def _execute_verify_count(
        self,
        action: VerifyCountAction,
        candidates: List[EntityCandidate],
        images: Dict[str, ImageData],
        evidence: EvidenceCollection
    ):
        """Compute count distribution for an object class using Poisson-Binomial."""

        # Find all candidates of this class in the specified image
        matching = [c for c in candidates
                   if c.image_id == action.image_id and c.object_class == action.object_class]

        if not matching:
            # No detections = count is 0 with certainty
            evidence.add_count(action.image_id, action.object_class, {0: 1.0})
            print(f"  [Verify Count] {action.object_class} in {action.image_id}: 0 (no detections)")
            return

        # Get detection confidences
        probabilities = [c.confidence for c in matching]

        # Compute Poisson-Binomial distribution
        distribution = self._compute_poisson_binomial(probabilities)

        # Store evidence
        evidence.add_count(action.image_id, action.object_class, distribution)

        # Find most likely count
        most_likely = max(distribution, key=distribution.get)
        print(f"  [Verify Count] {action.object_class} in {action.image_id}")
        print(f"    → Most likely: {most_likely} (p={distribution[most_likely]:.3f})")

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

    def _format_evidence(self, evidence: EvidenceCollection) -> str:
        """Format collected evidence for LLM prompt."""
        parts = []

        if evidence.attributes:
            parts.append("Attributes:")
            for entity_id, attr, value, prob in evidence.attributes:
                parts.append(f"  - {entity_id}.{attr}={value} (p={prob:.2f})")

        if evidence.relationships:
            parts.append("Relationships:")
            for subj, obj, rel, prob in evidence.relationships:
                parts.append(f"  - {subj} {rel} {obj} (p={prob:.2f})")

        if evidence.counts:
            parts.append("Counts:")
            for key, dist in evidence.counts.items():
                most_likely = max(dist, key=dist.get)
                parts.append(f"  - {key}: most likely {most_likely} (p={dist[most_likely]:.2f})")

        if not parts:
            return "None collected yet"

        return "\n".join(parts)

    def _format_perceive_history(self, history: List[Dict[str, str]]) -> str:
        """Format perceive Q&A history for LLM prompt."""
        if not history:
            return "None"

        lines = []
        for item in history:
            lines.append(f"Q ({item['entity_id']}): {item['question']}")
            lines.append(f"A: {item['answer']}")

        return "\n".join(lines)
