"""
Contextual subquery generator for PROVE pipeline.
Generates binary subquestions using visual context and detected objects to resolve ultimate question ambiguity.
"""

from typing import List, Dict, Any

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, BinarySubquery, ImageData


class SubqueryGeneratorError(RuntimeError):
    """Custom exception for subquery generation failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class SubqueryGenerator:
    """
    Generate contextual binary subquestions that resolve ultimate question ambiguity.
    Uses detailed image captions + detected objects + ultimate question to create specific reasoning questions.
    """
    
    def __init__(self):
        """Initialize generator with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def generate_binary_subqueries(
        self,
        ultimate_question: str,
        images: Dict[str, ImageData]  # Clean ImageData structure
    ) -> List[BinarySubquery]:
        """
        Generate object-aware binary subquestions that collectively answer the ultimate question.

        Args:
            ultimate_question: Main comparative question to answer
            images: ImageData structure containing objects, captions, and context per image

        Returns:
            List[BinarySubquery]: Binary questions with object references and types

        Raises:
            SubqueryGeneratorError: If generation fails
        """
        try:
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()
            
            # Create structured context for LLM from ImageData
            context = self._build_structured_context_from_images(images)
            
            # Generate binary subqueries using LLM
            prompt = self._create_subquery_prompt(ultimate_question, context)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at breaking down ambiguous comparative questions into specific binary subquestions using visual context. Generate binary questions that reference specific detected objects and can be answered Yes/No. Return strict JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Use Pydantic validation for robust JSON parsing
            response = llm_client.generate_subqueries(
                messages,
                temperature=0.3
            )
            
            # Extract objects from ImageData for conversion
            all_objects = {image_id: image_data.objects for image_id, image_data in images.items()}

            # Convert Pydantic response to BinarySubquery objects
            subqueries = self._convert_to_binary_subqueries(response.subqueries, all_objects)
            
            return subqueries
            
        except Exception as err:
            raise SubqueryGeneratorError(f"Binary subquery generation failed: {err}")
    
    def _build_structured_context_from_images(
        self,
        images: Dict[str, ImageData]
    ) -> str:
        """
        Build structured context string from ImageData structure.

        Args:
            images: ImageData structure containing objects, captions, and context per image

        Returns:
            str: Formatted context for LLM
        """
        context_parts = []

        for image_id in sorted(images.keys()):
            image_data = images[image_id]
            caption = image_data.scene_context.get("caption", "No caption available")
            objects = image_data.objects

            # Format objects with IDs
            object_list = []
            for obj in objects:
                # Create object ID in format: label_imageid_index (using simple image key)
                # Convert "image_a" to "a", "image_b" to "b" for simpler parsing
                simple_image_id = image_id.replace("image_", "")
                obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                object_list.append(f"{obj_id} ({obj.label}, conf={obj.confidence:.2f})")

            objects_str = ", ".join(object_list)

            context_parts.append(f"""
Image {image_id.upper()}:
Context: {caption}
Objects: {objects_str}""")

        return "\n".join(context_parts)
    
    def _create_subquery_prompt(self, ultimate_question: str, context: str) -> str:
        """
        Create simplified prompt for binary subquery generation.

        Args:
            ultimate_question: Main comparative question
            context: Structured visual context

        Returns:
            str: Formatted prompt for LLM
        """
        prompt = f"""Your task is to break down the ultimate question into binary (Yes/No) subquestions.

Ultimate Question: "{ultimate_question}"

Visual Context:
{context}

PROCESS:
1. First, understand what the ultimate question is asking
2. Consider what each scene depicts based on the image captions provided
3. Then break the ultimate question into binary subquestions that will collectively answer it

SUBQUESTION CATEGORIES:
- **attribute**: Characteristics of specific detected objects
- **relationship**: Spatial or interaction relations between detected objects
- **scene_attribute**: Scene-level characteristics that apply to the entire image, not individual objects
- **count**: Quantities of object classes in an image

RULES:
- Each subquestion must be answerable with Yes/No
- Only reference objects from the Objects list using their exact IDs
- List all relevant object IDs in "referenced_objects"
- Generate subquestions across all 4 categories
"""

        return prompt
    
    def _convert_to_binary_subqueries(
        self,
        subqueries: List,
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[BinarySubquery]:
        """
        Convert Pydantic-validated subqueries to BinarySubquery objects.
        Trust LLM-provided object references and types, only validate object ID existence.

        Args:
            subqueries: List of SubqueryItem objects from Pydantic validation
            all_objects: Original objects for validation

        Returns:
            List[BinarySubquery]: BinarySubquery instances
        """
        binary_subqueries = []

        # Get valid object IDs for validation (use same format as context building)
        valid_object_ids = set()
        for image_id, objects in all_objects.items():
            for obj in objects:
                # Use same format as _build_structured_context: strip "image_" prefix
                simple_image_id = image_id.replace("image_", "")
                obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                valid_object_ids.add(obj_id)

        for subquery_item in subqueries:
            try:
                # Extract data from SubqueryItem (Pydantic already validated types)
                question = subquery_item.question.strip()
                subquery_type = subquery_item.subquery_type.strip()

                # Trust LLM-provided referenced_objects, but validate they exist
                referenced_objects = getattr(subquery_item, 'referenced_objects', [])

                # Validate that all referenced objects exist in our valid set
                invalid_objects = [obj_id for obj_id in referenced_objects if obj_id not in valid_object_ids]
                if invalid_objects:
                    print(f"Warning: Skipping subquery with invalid object IDs: {invalid_objects}")
                    continue

                # Create BinarySubquery instance with validated data
                binary_subquery = BinarySubquery(
                    question=question,
                    referenced_objects=referenced_objects,
                    subquery_type=subquery_type
                )

                binary_subqueries.append(binary_subquery)

            except Exception as e:
                print(f"Warning: Failed to parse subquery: {e}")
                continue

        return binary_subqueries
    
    def validate_subqueries(self, subqueries: List[BinarySubquery]) -> bool:
        """
        Validate that generated subqueries have basic required structure.
        Pydantic handles type validation, we just check basic content.

        Args:
            subqueries: List of BinarySubquery instances

        Returns:
            bool: True if all subqueries are valid
        """
        try:
            for subquery in subqueries:
                # Check required attributes exist
                assert hasattr(subquery, 'question')
                assert hasattr(subquery, 'referenced_objects')
                assert hasattr(subquery, 'subquery_type')

                # Validate basic content (non-empty)
                assert subquery.question.strip()
                assert subquery.subquery_type.strip()
                assert isinstance(subquery.referenced_objects, list)

            return True

        except (AssertionError, AttributeError):
            return False
    
    def get_subqueries_summary(
        self, 
        subqueries: List[BinarySubquery]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for generated subqueries.
        
        Args:
            subqueries: List of BinarySubquery instances
            
        Returns:
            Dict[str, Any]: Summary information
        """
        if not subqueries:
            return {"count": 0, "types": {}, "avg_question_length": 0}
        
        # Count by type
        type_counts = {}
        question_lengths = []
        unique_objects = set()
        
        for subquery in subqueries:
            # Count types
            subquery_type = subquery.subquery_type
            type_counts[subquery_type] = type_counts.get(subquery_type, 0) + 1
            
            # Track question lengths
            question_lengths.append(len(subquery.question.split()))
            
            # Track unique objects
            unique_objects.update(subquery.referenced_objects)
        
        return {
            "count": len(subqueries),
            "types": type_counts,
            "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            "unique_objects_referenced": len(unique_objects),
            "sample_questions": [sq.question for sq in subqueries[:3]]
        }
    
    


# Example usage and testing
if __name__ == "__main__":
    # Test subquery generator
    generator = SubqueryGenerator()
    
    # Sample data
    ultimate_question = "Who is more powerful between these two people?"
    
    # Create test ImageData structure
    from src.core.types import ObjectDetection, ImageData
    images = {
        "image_a": ImageData(
            objects=[
                ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95),
                ObjectDetection(1, "weight", [150.0, 50.0, 300.0, 250.0], 0.88)
            ],
            attributes={},
            relationships=[],
            scene_context={"caption": "Muscular man lifting heavy weights in gym setting"}
        ),
        "image_b": ImageData(
            objects=[
                ObjectDetection(0, "person", [20.0, 30.0, 110.0, 210.0], 0.92),
                ObjectDetection(1, "equipment", [160.0, 60.0, 250.0, 180.0], 0.85)
            ],
            attributes={},
            relationships=[],
            scene_context={"caption": "Athletic woman doing pull-ups with defined muscle tone"}
        )
    }

    # Test generation
    try:
        subqueries = generator.generate_binary_subqueries(
            ultimate_question, images
        )

        # Validate subqueries
        is_valid = generator.validate_subqueries(subqueries)
        summary = generator.get_subqueries_summary(subqueries)
        
        print(f"✓ Generated {len(subqueries)} binary subqueries")
        print(f"✓ Validation: {is_valid}")
        print(f"✓ Summary: {summary}")
        
        for i, subquery in enumerate(subqueries):
            print(f"  {i+1}. {subquery.question}")
            
        print("✓ Subquery generator ready!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")