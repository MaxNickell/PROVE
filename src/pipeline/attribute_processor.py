"""
Attribute processor for PROVE pipeline.
Processes attribute subqueries individually: planning + extraction per subquery.
Follows per-subquery architecture - no cross-subquery consolidation.
"""

from typing import List, Dict, Any, Tuple
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquery, ObjectDetection, AttributeValue, AttributeData, ImageData
from src.core.probability import get_verifier_probability
from src.vision.florence2 import Florence2
from src.vision.qwen_vl import QwenVL


class AttributeProcessorError(RuntimeError):
    """Custom exception for attribute processing failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class AttributeProcessor:
    """
    Process attribute subqueries individually using per-subquery architecture.
    Each subquery: planning → region description → question generation → binary verification → immediate storage
    """

    def __init__(self):
        """Initialize processor with ModelManager singleton."""
        self.model_manager = ModelManager()

    def process_attribute_subqueries(
        self,
        attribute_subqueries: List[BinarySubquery],
        image_paths: Dict[str, str],
        images: Dict[str, ImageData]
    ) -> Dict[str, int]:
        """
        Process attribute subqueries individually - no cross-subquery consolidation.

        Args:
            attribute_subqueries: List of attribute binary subqueries
            image_paths: Dict mapping image_id to file path (e.g., {'image_a': './test_images/img0.png'})
            images: ImageData structure containing objects per image

        Returns:
            Dict[str, int]: Summary of attributes extracted per image

        Raises:
            AttributeProcessorError: If processing fails
        """
        try:
            # Load models
            llm_client = self.model_manager.get_llm_client()
            florence2 = self.model_manager.get_florence2()
            qwen_client = self.model_manager.get_qwen_vl()

            total_attributes_extracted = 0
            attributes_per_image = {}

            # Process each attribute subquery independently
            for i, subquery in enumerate(attribute_subqueries, 1):
                if subquery.subquery_type != "attribute":
                    continue

                print(f"  Processing subquery {i}/{len(attribute_subqueries)}: {subquery.question}")

                # Step 1: Determine attribute classes needed for referenced objects in this subquery
                attribute_requirements = self._determine_attribute_classes_for_subquery(
                    llm_client, subquery, images
                )

                if not attribute_requirements:
                    print(f"    No attribute requirements determined for this subquery")
                    continue

                print(f"    Determined {len(attribute_requirements)} attribute requirements")

                # Step 2: Extract attributes for each requirement immediately
                for req in attribute_requirements:
                    attributes_extracted = self._extract_attributes_for_requirement(
                        llm_client, florence2, qwen_client, req, image_paths, images, subquery.question
                    )

                    if attributes_extracted > 0:
                        total_attributes_extracted += attributes_extracted
                        attributes_per_image[req["image_id"]] = attributes_per_image.get(req["image_id"], 0) + attributes_extracted

            return attributes_per_image

        except Exception as e:
            raise AttributeProcessorError(f"Failed to process attribute subqueries: {str(e)}")

    def _determine_attribute_classes_for_subquery(
        self,
        llm_client,
        subquery: BinarySubquery,
        images: Dict[str, ImageData]
    ) -> List[Dict[str, Any]]:
        """
        Determine which attribute classes are needed for referenced_objects in this specific subquery.

        Args:
            llm_client: LLM client for analysis
            subquery: Single attribute subquery to analyze
            images: ImageData structure

        Returns:
            List[Dict]: List of {image_id, object_id, attribute_classes} for this subquery only
        """
        try:
            # Build context for referenced objects only (not all objects)
            referenced_context = []
            for obj_id in subquery.referenced_objects:
                # Parse object ID to find image and object info
                obj_info = self._find_object_by_id(obj_id, images)
                if obj_info:
                    referenced_context.append(f"{obj_id} ({obj_info['label']})")
                else:
                    referenced_context.append(f"{obj_id} (unknown)")

            if not referenced_context:
                return []

            objects_str = ", ".join(referenced_context)

            prompt = f"""Analyze this question to determine what attribute classes are needed for the referenced objects.

Question: "{subquery.question}"
Referenced Objects: {objects_str}

Task: Determine which visual attribute classes need to be extracted from which referenced objects to answer this specific question.

Rules:
- Only include objects explicitly referenced in the subquery
- Only include attribute classes directly needed to answer this specific question
- Use specific attribute class names (not generic descriptions)
- If no attributes needed, return empty dict: {{}}
Answer:"""

            messages = [{"role": "user", "content": prompt}]

            # Use Pydantic validation for guaranteed structure
            response = llm_client.plan_attributes(messages, temperature=0.2)

            # Convert Pydantic response to requirements list
            requirements = []
            for obj_id, attr_classes in response.attribute_requirements.items():
                obj_info = self._find_object_by_id(obj_id, images)
                if obj_info and attr_classes:
                    requirements.append({
                        "image_id": obj_info["image_id"],
                        "object_id": obj_info["object_index"],
                        "attribute_classes": attr_classes,
                        "full_object_id": obj_id
                    })

            return requirements

        except Exception as e:
            print(f"Warning: Failed to determine attribute classes for subquery '{subquery.question}': {e}")
            return []

    def _extract_attributes_for_requirement(
        self,
        llm_client,
        florence2: Florence2,
        qwen_client: QwenVL,
        requirement: Dict[str, Any],
        image_paths: Dict[str, str],
        images: Dict[str, ImageData],
        subquery_question: str
    ) -> int:
        """
        Extract attributes for a single requirement immediately.

        Args:
            llm_client: LLM client
            florence2: Florence-2 client
            qwen_client: Qwen VLM client
            requirement: Single attribute requirement
            image_paths: Dict mapping image_id to file path
            images: ImageData structure
            subquery_question: Original subquery question

        Returns:
            int: Number of attributes extracted
        """
        try:
            image_id = requirement["image_id"]
            object_index = requirement["object_id"]
            attribute_classes = requirement["attribute_classes"]

            # Find the object
            if image_id not in images or object_index >= len(images[image_id].objects):
                print(f"    Warning: Object not found: {image_id}[{object_index}]")
                return 0

            obj = images[image_id].objects[object_index]

            # Get image path directly from mapping
            if image_id not in image_paths:
                print(f"    Warning: Image path not found for {image_id}")
                return 0

            image_path = image_paths[image_id]

            # Load image
            image = Image.open(image_path)

            total_extracted = 0

            # Process each attribute class
            for attribute_class in attribute_classes:
                print(f"      Extracting {attribute_class} for {requirement['full_object_id']}")

                # Step 1: Generate region description
                region_description = self._get_region_description(florence2, image, obj)

                # Step 2: Generate attribute candidates
                candidates = self._generate_attribute_candidates(
                    llm_client, attribute_class, obj.label, region_description, subquery_question
                )

                if not candidates:
                    print(f"        No candidates generated for {attribute_class}")
                    continue

                # Step 3: Verify each candidate with Qwen
                verified_values = []
                for candidate in candidates:
                    confidence = self._verify_attribute_with_qwen(
                        qwen_client, image, obj, attribute_class, candidate
                    )
                    verified_values.append(AttributeValue(value=candidate, confidence=confidence))

                # Step 4: Store results immediately
                if verified_values:
                    self._store_attribute_results(images, image_id, object_index, attribute_class, verified_values)
                    total_extracted += len(verified_values)
                    print(f"        Stored {len(verified_values)} values for {attribute_class}")

            return total_extracted

        except Exception as e:
            print(f"    Warning: Failed to extract attributes for requirement: {e}")
            return 0

    def _get_region_description(self, florence2: Florence2, image: Image.Image, obj: ObjectDetection) -> str:
        """Generate dense caption for object region."""
        try:
            # Crop to object region
            x1, y1, x2, y2 = obj.bbox
            cropped_region = image.crop((x1, y1, x2, y2))

            # Generate dense caption
            caption = florence2.describe_region(cropped_region)
            return caption
        except Exception as e:
            print(f"        Warning: Failed to generate region description: {e}")
            return f"A {obj.label} object"

    def _generate_attribute_candidates(
        self,
        llm_client,
        attribute_class: str,
        object_label: str,
        region_description: str,
        subquery_question: str
    ) -> List[str]:
        """Generate candidate values for an attribute class."""
        try:
            prompt = f"""Generate candidate values for a specific attribute class based on the region description and question context.

Question: "{subquery_question}"
Object Label: {object_label}
Attribute Class: {attribute_class}
Region Description: "{region_description}"

Task: Generate most likely candidate values for the '{attribute_class}' attribute of this {object_label}.

Consider the question context - the candidates should help answer the specific question being asked.

Respond in this exact JSON format:
{{
  "candidates": ["value1", "value2", ..."]
}}

Answer:"""

            messages = [{"role": "user", "content": prompt}]

            # Use Pydantic validation for guaranteed structure
            response = llm_client.generate_candidates(messages, temperature=0.3)

            # Return validated candidates directly
            return response.candidates

        except Exception as e:
            print(f"        Warning: Failed to generate candidates: {e}")
            return []

    def _verify_attribute_with_qwen(
        self,
        qwen_client: QwenVL,
        image: Image.Image,
        obj: ObjectDetection,
        attribute_class: str,
        candidate_value: str
    ) -> float:
        """Verify attribute value with Qwen binary verification."""
        try:
            # Crop image to object region for focused evaluation
            x1, y1, x2, y2 = obj.bbox
            cropped_image = image.crop((x1, y1, x2, y2))

            # Create binary question with cropped image
            question = f"""Look at this image showing a {obj.label}.

Question: Does this {obj.label} have {attribute_class} "{candidate_value}"? Answer Yes or No.

Answer:"""

            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(cropped_image, question)

            # Use unified verifier probability extraction
            prob_statement_true = get_verifier_probability(
                logits,
                response,
                qwen_client.processor.tokenizer
            )

            return prob_statement_true

        except Exception as e:
            print(f"        Warning: Failed to verify attribute with Qwen: {e}")
            return 0.5

    def _store_attribute_results(
        self,
        images: Dict[str, ImageData],
        image_id: str,
        object_index: int,
        attribute_class: str,
        values: List[AttributeValue]
    ) -> None:
        """Store attribute results immediately in ImageData structure."""
        try:
            # Ensure attribute structure exists
            if object_index not in images[image_id].attributes:
                images[image_id].attributes[object_index] = AttributeData(attributes={})

            # Store the attribute values
            images[image_id].attributes[object_index].attributes[attribute_class] = values

        except Exception as e:
            print(f"        Warning: Failed to store attribute results: {e}")

    def _find_object_by_id(self, object_id: str, images: Dict[str, ImageData]) -> Dict[str, Any]:
        """Find object information by full object ID."""
        try:
            # Parse object_id format: "label_image_index" (e.g., "bird_a_0")
            parts = object_id.split('_')
            if len(parts) < 3:
                return None

            label = parts[0]
            image_suffix = parts[1]  # "a" or "b"
            object_index = int(parts[2])

            image_id = f"image_{image_suffix}"

            # Check if object exists
            if image_id in images and object_index < len(images[image_id].objects):
                obj = images[image_id].objects[object_index]
                if obj.label == label:  # Verify label matches
                    return {
                        "image_id": image_id,
                        "object_index": object_index,
                        "label": obj.label,
                        "confidence": obj.confidence
                    }

            return None

        except Exception as e:
            print(f"Warning: Failed to parse object ID '{object_id}': {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Test attribute processor
    processor = AttributeProcessor()

    # Sample data would go here for testing
    print("✓ Attribute processor ready for per-subquery processing!")