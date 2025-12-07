"""
Contextual subquestion generator for PROVE pipeline.
Generates binary subquestions using visual context and detected objects to resolve ultimate question ambiguity.
"""

from typing import List, Dict, Any

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, BinarySubquestion, ImageData




class SubquestionGenerator:
    """
    Generate contextual binary subquestions that resolve ultimate question ambiguity.
    Uses detailed image captions + detected objects + ultimate question to create specific reasoning questions.
    """

    def __init__(self):
        """Initialize generator with ModelManager singleton."""
        self.model_manager = ModelManager()

    def generate_binary_subquestions(
        self,
        ultimate_question: str,
        images: Dict[str, ImageData]  # Clean ImageData structure
    ) -> List[BinarySubquestion]:
        """
        Generate object-aware binary subquestions that collectively answer the ultimate question.

        Args:
            ultimate_question: Main comparative question to answer
            images: ImageData structure containing objects, captions, and context per image

        Returns:
            List[BinarySubquestion]: Binary questions with object references and types

        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()

            # Generate binary subquestions using LLM
            prompt = self._create_subquestion_prompt(ultimate_question, images)

            messages = [
                {
                    "role": "system",
                    "content": "Break down comparative questions into minimal binary subquestions. Output JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Use Pydantic validation for robust JSON parsing
            response = llm_client.generate_subquestions(
                messages,
                temperature=0.3
            )

            # Extract objects from ImageData for conversion
            all_objects = {image_id: image_data.objects for image_id, image_data in images.items()}

            # Convert Pydantic response to BinarySubquestion objects
            subquestions = self._convert_to_binary_subquestions(response.subquestions, all_objects)

            return subquestions

        except Exception as err:
            raise RuntimeError(f"Binary subquestion generation failed: {err}")
    
    
    def _create_subquestion_prompt(self, ultimate_question: str, images: Dict[str, ImageData]) -> str:
        """Create simplified prompt for binary subquestion generation."""
        image_ids = sorted(images.keys())
        if len(image_ids) < 2:
            raise ValueError("Expected at least 2 images for subquestion generation")

        caption_a = images[image_ids[0]].scene_context.get("caption", "No caption available")
        caption_b = images[image_ids[1]].scene_context.get("caption", "No caption available")

        def format_object_list(image_data: ImageData) -> str:
            class_counts = {}
            for obj in image_data.objects:
                class_counts[obj.label] = class_counts.get(obj.label, 0) + 1
            return ", ".join([
                f"{label} ({count})" if count > 1 else label
                for label, count in sorted(class_counts.items())
            ])

        objects_a = format_object_list(images[image_ids[0]])
        objects_b = format_object_list(images[image_ids[1]])

        prompt = f"""Break down the ultimate question into binary (Yes/No) subquestions using the visual context.

RULES:
- Each subquestion must be answerable with Yes or No
- Use object classes from the object lists
- Generate minimal subquestions needed to answer ultimate question
- Output strict JSON format

EXAMPLE:
Ultimate Question: Are there more people wearing blue shirts in image A than image B?
Output: {{"subquestions": ["In image A, how many people are wearing blue shirts?", "In image B, how many people are wearing blue shirts?"]}}

IMAGE A
Caption: {caption_a}
Objects: {objects_a}

IMAGE B
Caption: {caption_b}
Objects: {objects_b}

Ultimate Question: {ultimate_question}

Generate subquestions:"""

        return prompt
    
    def _convert_to_binary_subquestions(
        self,
        subquestions: List[str],
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[BinarySubquestion]:
        """
        Convert list of question strings to BinarySubquestion objects.

        Args:
            subquestions: List of question strings from Pydantic validation
            all_objects: Original objects (unused now, but kept for compatibility)

        Returns:
            List[BinarySubquestion]: BinarySubquestion instances
        """
        binary_subquestions = []

        for question in subquestions:
            try:
                if not isinstance(question, str):
                    continue

                # Create BinarySubquestion instance (no type field anymore)
                binary_subquestion = BinarySubquestion(question=question.strip())
                binary_subquestions.append(binary_subquestion)

            except Exception as e:
                print(f"Warning: Failed to parse subquestion: {e}")
                continue

        return binary_subquestions
    
    def validate_subquestions(self, subquestions: List[BinarySubquestion]) -> bool:
        """
        Validate that generated subquestions have basic required structure.
        Pydantic handles type validation, we just check basic content.

        Args:
            subquestions: List of BinarySubquestion instances

        Returns:
            bool: True if all subquestions are valid
        """
        try:
            for subquestion in subquestions:
                # Check required attributes exist
                assert hasattr(subquestion, 'question')

                # Validate basic content (non-empty)
                assert subquestion.question.strip()

            return True

        except (AssertionError, AttributeError):
            return False
    
    def get_subquestions_summary(
        self,
        subquestions: List[BinarySubquestion]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for generated subquestions.

        Args:
            subquestions: List of BinarySubquestion instances

        Returns:
            Dict[str, Any]: Summary information
        """
        if not subquestions:
            return {"count": 0, "types": {}, "avg_question_length": 0}

        # Count by type
        type_counts = {}
        question_lengths = []
        unique_objects = set()

        for subquestion in subquestions:
            # Count types
            subquestion_type = subquestion.subquestion_type
            type_counts[subquestion_type] = type_counts.get(subquestion_type, 0) + 1

            # Track question lengths
            question_lengths.append(len(subquestion.question.split()))

        return {
            "count": len(subquestions),
            "types": type_counts,
            "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            "sample_questions": [sq.question for sq in subquestions[:3]]
        }
    
    


if __name__ == "__main__":
    print("✓ Subquestion generator ready!")