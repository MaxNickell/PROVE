"""
ProbLog fact builder for PROVE pipeline.
Converts evidence collections into ProbLog facts.
"""

from typing import List, Dict, Tuple, Set
from src.core.types import ImageData, ProbLogFact


class ProbLogFactBuilder:
    """
    Build ProbLog facts from evidence collections.

    Predicates:
    - entity(image_id, entity_id, category)
    - attribute(image_id, entity_id, value)
    - relation(image_id, subject_id, object_id, relation_type)
    - count(image_id, category, count_value)
    """

    SUGAR_RULES = """% Sugar rules for cleaner queries
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C).
has_relationship(I,A,B,R) :- relation(I,A,B,R)."""

    def build_facts(
        self,
        evidence: 'EvidenceCollection',
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build ProbLog facts from evidence collection.

        Args:
            evidence: Evidence collected for a subquestion
            images: ImageData for entity metadata

        Returns:
            List of ProbLogFact objects
        """
        facts = []

        # Get all entity IDs referenced in evidence
        entity_ids = self._get_referenced_entities(evidence, images)

        # Build entity facts
        facts.extend(self._build_entity_facts(entity_ids, images))

        # Build attribute facts
        facts.extend(self._build_attribute_facts(evidence.attributes))

        # Build relation facts
        facts.extend(self._build_relation_facts(evidence.relationships))

        # Build count facts
        facts.extend(self._build_count_facts(evidence.counts))

        return facts

    @staticmethod
    def threshold_facts(facts: List[ProbLogFact], threshold: float = 0.5) -> List[ProbLogFact]:
        """
        Convert probabilistic facts to deterministic by thresholding.

        Args:
            facts: Probabilistic facts
            threshold: Threshold value (p >= threshold → 1.0, else → 0.0)

        Returns:
            Deterministic facts
        """
        return [
            ProbLogFact(
                probability=1.0 if f.probability >= threshold else 0.0,
                predicate=f.predicate,
                arguments=f.arguments.copy()
            )
            for f in facts
        ]

    @staticmethod
    def facts_to_string(facts: List[ProbLogFact]) -> str:
        """Convert facts to ProbLog program string."""
        return "\n".join(f.to_prolog_string() for f in facts)

    def _get_referenced_entities(
        self,
        evidence: 'EvidenceCollection',
        images: Dict[str, ImageData]
    ) -> Set[str]:
        """Extract all entity IDs referenced in evidence."""
        entities = set()

        # From attributes: (entity_id, attr_class, value, prob)
        for entity_id, _, _, _ in evidence.attributes:
            entities.add(entity_id)

        # From relationships: (subj_id, obj_id, relation, prob)
        for subj_id, obj_id, _, _ in evidence.relationships:
            entities.add(subj_id)
            entities.add(obj_id)

        # From counts: find entities of that category in that image
        for count_key in evidence.counts.keys():
            # count_key format: "image_a_dog"
            parts = count_key.rsplit('_', 1)
            if len(parts) == 2:
                image_id, category = parts
                if image_id in images:
                    for obj in images[image_id].objects:
                        if obj.label == category:
                            letter = image_id.replace("image_", "")
                            entities.add(f"{obj.label}_{letter}_{obj.object_id}")

        return entities

    def _build_entity_facts(
        self,
        entity_ids: Set[str],
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """Build entity facts for referenced entities only."""
        facts = []

        for image_id, image_data in images.items():
            letter = image_id.replace("image_", "")
            for obj in image_data.objects:
                entity_id = f"{obj.label}_{letter}_{obj.object_id}"
                if entity_id in entity_ids:
                    facts.append(ProbLogFact(
                        probability=obj.confidence,
                        predicate="entity",
                        arguments=[image_id, entity_id, obj.label]
                    ))

        return facts

    def _build_attribute_facts(
        self,
        attributes: List[Tuple[str, str, str, float]]
    ) -> List[ProbLogFact]:
        """Build attribute facts from evidence."""
        facts = []

        for entity_id, _, value, prob in attributes:
            # Extract image_id from entity_id (e.g., "bird_a_3" -> "image_a")
            parts = entity_id.split('_')
            if len(parts) >= 2:
                image_id = f"image_{parts[-2]}"
                facts.append(ProbLogFact(
                    probability=prob,
                    predicate="attribute",
                    arguments=[image_id, entity_id, value]
                ))

        return facts

    def _build_relation_facts(
        self,
        relationships: List[Tuple[str, str, str, float]]
    ) -> List[ProbLogFact]:
        """Build relation facts from evidence."""
        facts = []

        for subj_id, obj_id, relation, prob in relationships:
            # Extract image_id from subject_id
            parts = subj_id.split('_')
            if len(parts) >= 2:
                image_id = f"image_{parts[-2]}"
                facts.append(ProbLogFact(
                    probability=prob,
                    predicate="relation",
                    arguments=[image_id, subj_id, obj_id, relation]
                ))

        return facts

    def _build_count_facts(
        self,
        counts: Dict[str, Dict[int, float]]
    ) -> List[ProbLogFact]:
        """Build count facts from evidence (full distribution)."""
        facts = []

        for count_key, distribution in counts.items():
            # count_key format: "image_a_dog"
            parts = count_key.rsplit('_', 1)
            if len(parts) == 2:
                image_id, category = parts
                for count_value, probability in distribution.items():
                    facts.append(ProbLogFact(
                        probability=probability,
                        predicate="count",
                        arguments=[image_id, category, str(count_value)]
                    ))

        return facts
