"""
ProbLog knowledge base builder for PROVE pipeline.
Converts extracted evidence into probabilistic logical facts deterministically.
"""

from typing import List, Dict, Any
from src.core.types import (
    ObjectDetection, AttributeData, IntraRelation,
    ProbLogFact, AttributeValue, ImageData
)


class ProbLogBuilderError(RuntimeError):
    """Custom exception for ProbLog building failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class ProbLogBuilder:
    """
    Build ProbLog knowledge base from extracted evidence.
    Converts Python objects into structured probabilistic logical facts.
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
            images: Clean ImageData structure containing all evidence per image

        Returns:
            List[ProbLogFact]: Complete knowledge base as probabilistic facts

        Raises:
            ProbLogBuilderError: If building fails
        """
        try:
            facts = []

            # Convert object facts from ImageData
            object_facts = self._build_object_facts_from_images(images)
            facts.extend(object_facts)

            # Convert attribute facts from ImageData
            attribute_facts = self._build_attribute_facts_from_images(images)
            facts.extend(attribute_facts)

            # Convert relationship facts from ImageData
            relation_facts = self._build_relation_facts_from_images(images)
            facts.extend(relation_facts)

            # Validate facts
            validated_facts = self._validate_facts(facts)

            return validated_facts

        except Exception as err:
            raise ProbLogBuilderError(f"ProbLog knowledge base building failed: {err}")

    def _build_object_facts_from_images(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build object existence facts from ImageData structure.

        Args:
            images: ImageData structure containing objects per image

        Returns:
            List[ProbLogFact]: Object existence facts
        """
        facts = []

        for image_id, image_data in images.items():
            for obj in image_data.objects:
                # Create unique object identifier
                obj_id = f"{obj.label}_{image_id}_{obj.object_id}"

                # Object existence fact
                fact = ProbLogFact(
                    probability=obj.confidence,
                    predicate="object",
                    arguments=[obj_id, obj.label, image_id]
                )
                facts.append(fact)

                # Object location fact (optional)
                bbox_str = f"bbox_{int(obj.bbox[0])}_{int(obj.bbox[1])}_{int(obj.bbox[2])}_{int(obj.bbox[3])}"
                location_fact = ProbLogFact(
                    probability=obj.confidence,
                    predicate="location",
                    arguments=[obj_id, bbox_str]
                )
                facts.append(location_fact)

        return facts

    def _build_attribute_facts_from_images(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build attribute facts from ImageData structure.

        Args:
            images: ImageData structure containing attributes per image

        Returns:
            List[ProbLogFact]: Attribute facts
        """
        facts = []

        for image_id, image_data in images.items():
            for object_id, attr_data in image_data.attributes.items():
                # Find the target object to get its label
                target_obj = None
                for obj in image_data.objects:
                    if obj.object_id == object_id:
                        target_obj = obj
                        break

                if not target_obj:
                    print(f"Warning: Could not find object {object_id} in image {image_id}")
                    continue

                # Create full object ID following label_image_index format
                full_obj_id = f"{target_obj.label}_{image_id}_{object_id}"

                # Convert each attribute class and its values
                for attribute_class, values in attr_data.attributes.items():
                    for value in values:
                        fact = ProbLogFact(
                            probability=value.confidence,
                            predicate="attribute",
                            arguments=[full_obj_id, attribute_class, value.value]
                        )
                        facts.append(fact)

        return facts

    def _build_relation_facts_from_images(
        self,
        images: Dict[str, ImageData]
    ) -> List[ProbLogFact]:
        """
        Build relationship facts from ImageData structure.

        Args:
            images: ImageData structure containing relationships per image

        Returns:
            List[ProbLogFact]: Relationship facts
        """
        facts = []

        for image_id, image_data in images.items():
            for relation in image_data.relationships:
                # Use the string IDs directly since they are already in the correct format
                # relation.subject_id and relation.object_id are now strings like "bird_a_0"
                subject_full_id = relation.subject_id
                object_full_id = relation.object_id

                # Create relationship fact
                fact = ProbLogFact(
                    probability=relation.probability,
                    predicate="relation",
                    arguments=[subject_full_id, object_full_id, relation.relation]
                )
                facts.append(fact)

        return facts

    def _build_object_facts(
        self,
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[ProbLogFact]:
        """
        Build object existence facts from detected objects.
        
        Args:
            all_objects: Detected objects per image
            
        Returns:
            List[ProbLogFact]: Object existence facts
        """
        facts = []
        
        for image_id, objects in all_objects.items():
            for obj in objects:
                # Create unique object identifier
                obj_id = f"{obj.label}_{image_id}_{obj.object_id}"
                
                # Object existence fact
                fact = ProbLogFact(
                    probability=obj.confidence,
                    predicate="object",
                    arguments=[obj_id, obj.label, image_id]
                )
                facts.append(fact)
                
                # Object location fact (optional)
                bbox_str = f"bbox_{int(obj.bbox[0])}_{int(obj.bbox[1])}_{int(obj.bbox[2])}_{int(obj.bbox[3])}"
                location_fact = ProbLogFact(
                    probability=obj.confidence,
                    predicate="location",
                    arguments=[obj_id, bbox_str]
                )
                facts.append(location_fact)
        
        return facts
    
    def _build_attribute_facts_nested(
        self,
        attributes_nested: Dict[str, Dict[int, AttributeData]],
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[ProbLogFact]:
        """
        Build attribute facts from nested attribute data structure.

        Args:
            attributes_nested: Nested structure {image_id: {object_id: AttributeData}}
            all_objects: Original objects for ID mapping

        Returns:
            List[ProbLogFact]: Attribute facts
        """
        facts = []

        # Iterate through nested structure
        for image_id, objects_attributes in attributes_nested.items():
            if image_id not in all_objects:
                print(f"Warning: Could not find image {image_id} in all_objects")
                continue

            for object_id, attr_data in objects_attributes.items():
                # Find the target object
                target_obj = None
                for obj in all_objects[image_id]:
                    if obj.object_id == object_id:
                        target_obj = obj
                        break

                if not target_obj:
                    print(f"Warning: Could not find object {object_id} in image {image_id}")
                    continue

                # Create full object ID following label_image_index format
                full_obj_id = f"{target_obj.label}_{image_id}_{object_id}"

                # Convert each attribute class and its values
                for attribute_class, values in attr_data.attributes.items():
                    for value in values:
                        fact = ProbLogFact(
                            probability=value.confidence,
                            predicate="attribute",
                            arguments=[full_obj_id, attribute_class, value.value]
                        )
                        facts.append(fact)

        return facts
    
    def _build_relation_facts(
        self,
        relations: List[IntraRelation],
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[ProbLogFact]:
        """
        Build relationship facts from extracted relations.
        
        Args:
            relations: Extracted relationships
            all_objects: Original objects for ID mapping
            
        Returns:
            List[ProbLogFact]: Relationship facts
        """
        facts = []
        
        # Create object ID mapping
        object_id_map = self._create_object_id_map(all_objects)
        
        for relation in relations:
            # For legacy relations or when image_id not specified, try to find objects
            # This assumes intra-image relationships
            found_image_id = None
            subject_obj = None
            target_obj = None

            # Try to find the objects in any image
            for img_id, objects in all_objects.items():
                if not subject_obj:
                    for obj in objects:
                        if obj.object_id == relation.subject_id:
                            subject_obj = obj
                            found_image_id = img_id
                            break

                if subject_obj and found_image_id == img_id:
                    for obj in objects:
                        if obj.object_id == relation.object_id:
                            target_obj = obj
                            break

                if subject_obj and target_obj:
                    break

            if not subject_obj:
                print(f"Warning: Could not find subject object {relation.subject_id}")
                continue

            if not target_obj:
                print(f"Warning: Could not find target object {relation.object_id}")
                continue

            # Create full object IDs
            subject_full_id = f"{subject_obj.label}_{found_image_id}_{relation.subject_id}"
            object_full_id = f"{target_obj.label}_{found_image_id}_{relation.object_id}"
            
            # Create relationship fact
            fact = ProbLogFact(
                probability=relation.probability,
                predicate="relation",
                arguments=[subject_full_id, object_full_id, relation.relation]
            )
            facts.append(fact)
        
        return facts

    def _build_relation_facts_nested(
        self,
        relationships_nested: Dict[str, List[IntraRelation]],
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[ProbLogFact]:
        """
        Build relationship facts from nested relationship data structure.

        Args:
            relationships_nested: Nested structure {image_id: [IntraRelation]}
            all_objects: Original objects for ID mapping

        Returns:
            List[ProbLogFact]: Relationship facts
        """
        facts = []

        # Iterate through nested structure
        for image_id, relations in relationships_nested.items():
            if image_id not in all_objects:
                print(f"Warning: Could not find image {image_id} in all_objects")
                continue

            for relation in relations:
                # Find subject object using simple subject_id
                subject_obj = None
                for obj in all_objects[image_id]:
                    if obj.object_id == relation.subject_id:
                        subject_obj = obj
                        break

                if not subject_obj:
                    print(f"Warning: Could not find subject object {relation.subject_id} in image {image_id}")
                    continue

                # Find target object using simple object_id
                target_obj = None
                for obj in all_objects[image_id]:
                    if obj.object_id == relation.object_id:
                        target_obj = obj
                        break

                if not target_obj:
                    print(f"Warning: Could not find target object {relation.object_id} in image {image_id}")
                    continue

                # Create full object IDs following label_image_index format
                subject_full_id = f"{subject_obj.label}_{image_id}_{relation.subject_id}"
                object_full_id = f"{target_obj.label}_{image_id}_{relation.object_id}"

                # Create relationship fact
                fact = ProbLogFact(
                    probability=relation.probability,
                    predicate="relation",
                    arguments=[subject_full_id, object_full_id, relation.relation]
                )
                facts.append(fact)

        return facts

    def _create_object_id_map(
        self,
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create mapping from full object IDs to object data.
        
        Args:
            all_objects: All detected objects
            
        Returns:
            Dict mapping full_object_id to object data
        """
        id_map = {}
        
        for image_id, objects in all_objects.items():
            for obj in objects:
                full_id = f"{obj.label}_{image_id}_{obj.object_id}"
                id_map[full_id] = {
                    'object_id': obj.object_id,
                    'label': obj.label,
                    'image_id': image_id,
                    'confidence': obj.confidence,
                    'bbox': obj.bbox
                }
        
        return id_map
    
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
                    predicate=fact.predicate.strip().lower(),
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
        
        # Add header comment
        header = "% PROVE Pipeline Knowledge Base\n% Generated from visual evidence extraction\n\n"
        
        # Group facts by predicate for better organization
        grouped_facts = {}
        for fact_string in fact_strings:
            predicate = fact_string.split('::')[1].split('(')[0]
            if predicate not in grouped_facts:
                grouped_facts[predicate] = []
            grouped_facts[predicate].append(fact_string)
        
        # Build organized program
        program_parts = [header]
        
        for predicate in sorted(grouped_facts.keys()):
            program_parts.append(f"% {predicate.capitalize()} facts")
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
            "confidence_distribution": {
                "high (>0.8)": len([p for p in all_probabilities if p > 0.8]),
                "medium (0.5-0.8)": len([p for p in all_probabilities if 0.5 <= p <= 0.8]),
                "low (<0.5)": len([p for p in all_probabilities if p < 0.5])
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test ProbLog builder
    builder = ProbLogBuilder()
    
    # Sample data
    from src.core.types import ObjectDetection, AttributeData, AttributeValue, IntraRelation
    
    all_objects = {
        "image_a": [
            ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95),
            ObjectDetection(1, "weight", [150.0, 50.0, 300.0, 250.0], 0.88)
        ]
    }
    
    # Test with nested attribute structure matching new KnowledgeBase format
    attributes_nested = {
        "image_a": {
            0: AttributeData(
                attributes={
                    "muscle_mass": [AttributeValue("high", 0.89)],
                    "body_size": [AttributeValue("large", 0.82)]
                }
            )
        }
    }
    
    # Test with nested relationships structure matching new KnowledgeBase format
    relationships_nested = {
        "image_a": [
            IntraRelation(
                subject_id=0,
                object_id=1,
                relation="lifting",
                probability=0.91
            )
        ]
    }
    
    try:
        # Build knowledge base
        facts = builder.build_knowledge_base(all_objects, attributes_nested, relationships_nested)
        
        # Get summary
        summary = builder.get_building_summary(facts)
        
        # Generate ProbLog program
        prolog_program = builder.facts_to_prolog_string(facts)
        
        print(f"✓ Built knowledge base with {len(facts)} facts")
        print(f"✓ Summary: {summary}")
        print(f"✓ ProbLog program generated ({len(prolog_program)} characters)")
        print("✓ ProbLog builder ready!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")