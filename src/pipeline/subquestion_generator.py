"""
Contextual subquestion generator for PROVE pipeline.
Generates binary subquestions using visual context and detected objects to resolve ultimate question ambiguity.
"""

from typing import List, Dict

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ImageData




class SubquestionGenerator:
    """
    Generate contextual binary subquestions that resolve ultimate question ambiguity.
    Uses image captions + detected objects + ultimate question to create specific reasoning questions.
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
            List[BinarySubquestion]: Binary questions with object references

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

            # Convert Pydantic response to BinarySubquestion objects
            subquestions = self._convert_to_binary_subquestions(response.subquestions)

            return subquestions

        except Exception as err:
            raise RuntimeError(f"Binary subquestion generation failed: {err}")
    
    
    def _create_subquestion_prompt(self, ultimate_question: str, images: Dict[str, ImageData]) -> str:
        """Create prompt for binary subquestion generation using captions and objects."""
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

        prompt = f"""TASK
Decompose the ultimate question into simpler binary (Yes/No) subquestions
about attributes, relationships, or counts of the detected objects.

RULES
1. Every subquestion must be answerable with Yes or No
2. Always specify the image: "in image A", "in image B", or "in both images"
3. Only reference objects from the DETECTED OBJECTS lists
4. Generate the minimal set needed to answer the ultimate question

EXAMPLES

Ultimate: "Are there more birds in one image than the other?"
Output: {{"subquestions": ["Are there more birds in image A than in image B?"]}}

Ultimate: "Is there a white bird sitting on a buffalo?"
Output: {{"subquestions": ["Is there a white bird sitting on a buffalo in image A?", "Is there a white bird sitting on a buffalo in image B?"]}}

Ultimate: "Do both images show a person wearing a hat?"
Output: {{"subquestions": ["Is there a person wearing a hat in image A?", "Is there a person wearing a hat in image B?"]}}

Ultimate: "Are all the cats sleeping?"
Output: {{"subquestions": ["Is every cat in image A sleeping?", "Is every cat in image B sleeping?"]}}

Ultimate: "Are the cars the same color?"
Output: {{"subquestions": ["Are the cars in image A the same color as the cars in image B?"]}}

Ultimate: "Do both images have the same number of dogs?"
Output: {{"subquestions": ["Are there the same number of dogs in image A as in image B?"]}}

Ultimate: "Is every child holding a balloon?"
Output: {{"subquestions": ["Is every child in image A holding a balloon?", "Is every child in image B holding a balloon?"]}}

DETECTED OBJECTS
Image A: {objects_a}
Image B: {objects_b}

IMAGE CAPTIONS
Image A: {caption_a}
Image B: {caption_b}

ULTIMATE QUESTION: {ultimate_question}

Output JSON:"""

        return prompt
    
    def _convert_to_binary_subquestions(
        self,
        subquestions: List[str]
    ) -> List[BinarySubquestion]:
        """
        Convert list of question strings to BinarySubquestion objects.

        Args:
            subquestions: List of question strings from Pydantic validation

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


if __name__ == "__main__":
    print("✓ Subquestion generator ready!")