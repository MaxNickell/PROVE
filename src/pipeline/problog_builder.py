"""
ProbLog fact builder for PROVE pipeline.
Converts evidence collections into ProbLog facts.
"""

from typing import List, Dict, Tuple, Set, TYPE_CHECKING
from src.core.types import ImageData, ProbLogFact

if TYPE_CHECKING:
    from src.pipeline.unified_agent import CountEvidence


class ProbLogFactBuilder:
    """
    Build ProbLog facts from evidence collections.

    Predicates:
    - entity(image_id, entity_id, category)
    - attribute(image_id, entity_id, value)
    - relation(image_id, subject_id, object_id, relation_type)

    Count predicates (query-driven):
    - count_at_least(image_id, class, N)
    - count_at_most(image_id, class, N)
    - count_exactly(image_id, class, N)
    - count_more(image_id_a, image_id_b, class)
    - count_fewer(image_id_a, image_id_b, class)
    - count_equal(image_id_a, image_id_b, class)
    - count_total_exactly(image_id_a, image_id_b, class, N)
    - count_total_at_least(image_id_a, image_id_b, class, N)
    - count_total_at_most(image_id_a, image_id_b, class, N)
    """

    def build_facts(
        self,
        evidence: 'EvidenceCollection',
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build ProbLog facts from evidence collection.

        Args:
            evidence: Evidence collected for the question
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

        # From attributes: (entity_id, attribute, prob)
        for entity_id, _, _ in evidence.attributes:
            entities.add(entity_id)

        # From relationships: (subj_id, obj_id, relation, prob)
        for subj_id, obj_id, _, _ in evidence.relationships:
            entities.add(subj_id)
            entities.add(obj_id)

        # Count evidence no longer adds entities - counts are standalone predicates

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
                entity_id = f"{obj.label.replace(' ', '_')}_{letter}_{obj.object_id}"
                if entity_id in entity_ids:
                    facts.append(ProbLogFact(
                        probability=obj.confidence,
                        predicate="entity",
                        arguments=[image_id, entity_id, obj.label]
                    ))

        return facts

    def _build_attribute_facts(
        self,
        attributes: List[Tuple[str, str, float]]
    ) -> List[ProbLogFact]:
        """Build attribute facts from evidence."""
        facts = []

        for entity_id, attribute, prob in attributes:
            # Extract image_id from entity_id (e.g., "bird_a_3" -> "image_a")
            parts = entity_id.split('_')
            if len(parts) >= 2:
                image_id = f"image_{parts[-2]}"
                facts.append(ProbLogFact(
                    probability=prob,
                    predicate="attribute",
                    arguments=[image_id, entity_id, attribute]
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
        counts: List['CountEvidence']
    ) -> List[ProbLogFact]:
        """Build count facts from evidence (query-driven format)."""
        facts = []

        for count_ev in counts:
            query_type = count_ev.query_type
            object_class = count_ev.object_class
            probability = count_ev.probability

            if query_type in ["at_least", "at_most", "exactly"]:
                # Single-image: count_<query_type>(image_id, class, N)
                predicate = f"count_{query_type}"
                facts.append(ProbLogFact(
                    probability=probability,
                    predicate=predicate,
                    arguments=[count_ev.image_id, object_class, str(count_ev.value)]
                ))

            elif query_type in ["more", "fewer", "equal"]:
                # Cross-image comparison: count_<query_type>(image_id_a, image_id_b, class)
                predicate = f"count_{query_type}"
                facts.append(ProbLogFact(
                    probability=probability,
                    predicate=predicate,
                    arguments=[count_ev.image_id_a, count_ev.image_id_b, object_class]
                ))

            elif query_type.startswith("total_"):
                # Total across both: count_<query_type>(image_id_a, image_id_b, class, N)
                predicate = f"count_{query_type}"
                facts.append(ProbLogFact(
                    probability=probability,
                    predicate=predicate,
                    arguments=[count_ev.image_id_a, count_ev.image_id_b, object_class, str(count_ev.value)]
                ))

        return facts
