"""
ProbLog knowledge base builder for PROVE pipeline.
Converts extracted evidence into probabilistic logical facts using exact specification format.
"""

from typing import List, Dict, Any
from src.core.types import ImageData, ProbLogFact


class ProbLogBuilderError(RuntimeError):
    """Custom exception for ProbLog building failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class ProbLogBuilder:
    """
    Build ProbLog knowledge base from ImageData using exact specification format:
    - entity(image_id, entity_id, category, x1, y1, x2, y2)
    - relation(image_id, entity_a, entity_b, relation_type)
    - attribute(image_id, entity_id, attr_value)
    - scene_attr(image_id, attr_value)
    - count(image_id, category, value)
    """

    def __init__(self):
        """Initialize ProbLog builder."""
        pass

    def build_knowledge_base(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build complete ProbLog knowledge base from ImageData structure.

        Args:
            images: ImageData structure containing all evidence per image

        Returns:
            List[ProbLogFact]: Complete knowledge base as probabilistic facts

        Raises:
            ProbLogBuilderError: If building fails
        """
        try:
            facts = []

            # Build all fact types following exact specification
            facts.extend(self._build_entity_facts(images))
            facts.extend(self._build_attribute_facts(images))
            facts.extend(self._build_relation_facts(images))
            facts.extend(self._build_scene_attr_facts(images))
            facts.extend(self._build_count_facts(images))

            # Validate facts
            validated_facts = self._validate_facts(facts)

            return validated_facts

        except Exception as err:
            raise ProbLogBuilderError(f"ProbLog knowledge base building failed: {err}")

    def _build_entity_facts(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build entity facts: entity(image_id, entity_id, category, x1, y1, x2, y2).

        Args:
            images: ImageData structure containing objects per image

        Returns:
            List[ProbLogFact]: Entity facts with bbox coordinates
        """
        facts = []

        for image_id, image_data in images.items():
            for obj in image_data.objects:
                # Create entity_id following format: category_image_objectid
                simple_image_id = image_id.replace("image_", "")
                entity_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"

                # Extract bbox coordinates as integers
                x1, y1, x2, y2 = [int(coord) for coord in obj.bbox]

                # Create entity fact
                fact = ProbLogFact(
                    probability=obj.confidence,
                    predicate="entity",
                    arguments=[image_id, entity_id, obj.label, str(x1), str(y1), str(x2), str(y2)]
                )
                facts.append(fact)

        return facts

    def _build_attribute_facts(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build attribute facts: attribute(image_id, entity_id, attr_value).

        Args:
            images: ImageData structure containing attributes per image

        Returns:
            List[ProbLogFact]: Attribute facts with attr_value format
        """
        facts = []

        for image_id, image_data in images.items():
            for object_id, attr_data in image_data.attributes.items():
                # Find the target object to get its label
                target_obj = None
                for obj in image_data.objects:
                    if str(obj.object_id) == str(object_id):
                        target_obj = obj
                        break

                if not target_obj:
                    print(f"Warning: Could not find object {object_id} in image {image_id}")
                    continue

                # Create entity_id following format: category_image_objectid
                simple_image_id = image_id.replace("image_", "")
                entity_id = f"{target_obj.label}_{simple_image_id}_{object_id}"

                # Convert each attribute class and its values
                for attribute_class, values in attr_data.attributes.items():
                    for value in values:
                        # Use just the attribute value (no class prefix)
                        attr_value = value.value

                        fact = ProbLogFact(
                            probability=value.confidence,
                            predicate="attribute",
                            arguments=[image_id, entity_id, attr_value]
                        )
                        facts.append(fact)

        return facts

    def _build_relation_facts(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build relation facts: relation(image_id, entity_a, entity_b, relation_type).

        Args:
            images: ImageData structure containing relationships per image

        Returns:
            List[ProbLogFact]: Relationship facts
        """
        facts = []

        for image_id, image_data in images.items():
            for relation in image_data.relationships:
                # Use the string IDs directly (already in correct format like "cattle_a_1")
                entity_a = relation.subject_id
                entity_b = relation.object_id
                relation_type = relation.relation

                # Create relationship fact
                fact = ProbLogFact(
                    probability=relation.probability,
                    predicate="relation",
                    arguments=[image_id, entity_a, entity_b, relation_type]
                )
                facts.append(fact)

        return facts

    def _build_scene_attr_facts(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build scene attribute facts: scene_attr(image_id, attr_value).

        Args:
            images: ImageData structure containing scene attributes per image

        Returns:
            List[ProbLogFact]: Scene attribute facts
        """
        facts = []

        for image_id, image_data in images.items():
            for attr_class, attr_values in image_data.scene_attributes.items():
                for attr_value in attr_values:
                    # Use just the attribute value (no class prefix)
                    scene_attr_value = attr_value['value']

                    fact = ProbLogFact(
                        probability=attr_value['confidence'],
                        predicate="scene_attr",
                        arguments=[image_id, scene_attr_value]
                    )
                    facts.append(fact)

        return facts

    def _build_count_facts(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build count facts: count(image_id, category, value).

        Encodes COMPLETE Poisson-Binomial distributions by creating separate facts
        for each possible count value with its probability.

        Args:
            images: ImageData structure containing count distributions per image

        Returns:
            List[ProbLogFact]: Count distribution facts
        """
        facts = []

        for image_id, image_data in images.items():
            for category, count_data in image_data.counts.items():
                distribution = count_data['distribution']

                # Create separate fact for each count value in the distribution
                for count_str, probability in distribution.items():
                    count_value = int(count_str)

                    fact = ProbLogFact(
                        probability=probability,
                        predicate="count",
                        arguments=[image_id, category, str(count_value)]
                    )
                    facts.append(fact)

        return facts

    def _validate_facts(
        self,
        facts: List[ProbLogFact]
    ) -> List[ProbLogFact]:
        """
        Validate and clean ProbLog facts.

        Args:
            facts: Raw facts to validate

        Returns:
            List[ProbLogFact]: Validated facts
        """
        validated_facts = []

        for fact in facts:
            try:
                # Validate probability range
                if not (0.0 <= fact.probability <= 1.0):
                    print(f"Warning: Invalid probability {fact.probability} for fact {fact.predicate}")
                    continue

                # Validate predicate
                if not fact.predicate or not fact.predicate.strip():
                    print(f"Warning: Empty predicate for fact")
                    continue

                # Validate arguments
                if not fact.arguments:
                    print(f"Warning: Empty arguments for fact {fact.predicate}")
                    continue

                # Clean arguments (remove empty strings, convert to string)
                clean_args = []
                for arg in fact.arguments:
                    if arg is not None:
                        clean_arg = str(arg).strip()
                        if clean_arg:
                            clean_args.append(clean_arg)

                if not clean_args:
                    print(f"Warning: No valid arguments for fact {fact.predicate}")
                    continue

                # Create cleaned fact
                validated_fact = ProbLogFact(
                    probability=fact.probability,
                    predicate=fact.predicate.strip(),
                    arguments=clean_args
                )

                validated_facts.append(validated_fact)

            except Exception as e:
                print(f"Warning: Failed to validate fact {fact.predicate}: {e}")
                continue

        return validated_facts

    def facts_to_prolog_string(
        self,
        facts: List[ProbLogFact]
    ) -> str:
        """
        Convert facts to complete ProbLog program string.

        Args:
            facts: List of ProbLog facts

        Returns:
            str: Complete ProbLog program
        """
        fact_strings = []

        for fact in facts:
            fact_string = fact.to_prolog_string()
            fact_strings.append(fact_string)

        # Add specification header
        header = """% PROVE Pipeline Knowledge Base - Specification Format
% Generated from visual evidence extraction

% entity(image_id: str, entity_id: str, category: str, x1: int, y1: int, x2: int, y2: int).
% relation(image_id: str, entity_a: str, entity_b: str, relation_type: str).
% attribute(image_id: str, entity_id: str, attr_value: str).
% scene_attr(image_id: str, attr_value: str).
% count(image_id: str, category: str, value: int).

"""

        # Group facts by predicate for better organization
        grouped_facts = {}
        for fact_string in fact_strings:
            predicate = fact_string.split('::')[1].split('(')[0]
            if predicate not in grouped_facts:
                grouped_facts[predicate] = []
            grouped_facts[predicate].append(fact_string)

        # Build organized program with specification order
        program_parts = [header]
        predicate_order = ["entity", "relation", "attribute", "scene_attr", "count"]

        for predicate in predicate_order:
            if predicate in grouped_facts:
                program_parts.append(f"% {predicate} facts")
                program_parts.extend(grouped_facts[predicate])
                program_parts.append("")  # Empty line between sections

        return "\n".join(program_parts)

    def get_building_summary(
        self,
        facts: List[ProbLogFact]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for knowledge base building.

        Args:
            facts: List of ProbLog facts

        Returns:
            Dict with summary information
        """
        if not facts:
            return {
                "total_facts": 0,
                "predicates": {},
                "avg_confidence": 0.0
            }

        total_facts = len(facts)
        predicate_counts = {}
        all_probabilities = []

        for fact in facts:
            predicate_counts[fact.predicate] = predicate_counts.get(fact.predicate, 0) + 1
            all_probabilities.append(fact.probability)

        avg_confidence = sum(all_probabilities) / len(all_probabilities) if all_probabilities else 0.0

        return {
            "total_facts": total_facts,
            "predicates": predicate_counts,
            "avg_confidence": avg_confidence,
            "specification_compliance": "EXACT" if set(predicate_counts.keys()).issubset({"entity", "relation", "attribute", "scene_attr", "count"}) else "PARTIAL"
        }


# Example usage and testing
if __name__ == "__main__":
    # Test ProbLog builder with specification format
    builder = ProbLogBuilder()

    print("✓ ProbLog builder ready with specification format!")
    print("✓ Supports: entity, relation, attribute, scene_attr, count predicates")
    print("✓ Encodes complete count distributions as separate facts")