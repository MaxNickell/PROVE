"""
Scene attribute processor for PROVE pipeline.
Handles scene_attribute subqueries using LLM decomposition + Qwen binary verification.
Follows the established subquery analysis pattern used by attribute and relationship extractors.
"""

from typing import List, Dict, Any, Tuple
import json
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, BinarySubquery, ImageData
from src.vision.qwen_vl import QwenVL


class ContextProcessorError(RuntimeError):
    """Custom exception for scene attribute processing failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class SceneAttributeCandidate:
    """Scene attribute candidate for binary verification."""
    def __init__(self, image_id: str, attribute_class: str, candidate_value: str, binary_question: str):
        self.image_id = image_id
        self.attribute_class = attribute_class
        self.candidate_value = candidate_value
        self.binary_question = binary_question
        self.required_for_subqueries = []

    def to_dict(self):
        return {
            "image_id": self.image_id,
            "attribute_class": self.attribute_class,
            "candidate_value": self.candidate_value,
            "binary_question": self.binary_question,
            "required_for_subqueries": self.required_for_subqueries
        }


class SceneAttributeResult:
    """Scene attribute result with confidence score."""
    def __init__(self, image_id: str, attribute_class: str, value: str, confidence: float, binary_question: str, subquery: str):
        self.image_id = image_id
        self.attribute_class = attribute_class
        self.value = value
        self.confidence = confidence
        self.binary_question = binary_question
        self.subquery = subquery

    def to_dict(self):
        return {
            "attribute_class": self.attribute_class,
            "value": self.value,
            "confidence": self.confidence,
            "binary_question": self.binary_question,
            "subquery": self.subquery
        }


class ContextProcessor:
    """
    Process scene_attribute subqueries using LLM decomposition + Qwen binary verification.
    Follows the established pattern: Subquery → LLM Analysis → Atomic Binary Questions → Qwen Verification → Results
    """

    def __init__(self):
        """Initialize processor with ModelManager singleton."""
        self.model_manager = ModelManager()

    def process_scene_attribute_subqueries(
        self,
        scene_subqueries: List[BinarySubquery],
        image_paths: List[str],
        images: Dict[str, ImageData]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process scene_attribute subqueries using proper subquery decomposition.

        Args:
            scene_subqueries: List of scene_attribute binary subqueries
            image_paths: List of image file paths
            images: ImageData structure containing objects and captions per image

        Returns:
            Dict[str, Dict[str, Any]]: Scene context per image with attribute facts
            Format: {"image_a": {"caption": str, "scene_attributes": [SceneAttributeResult.to_dict()]}}

        Raises:
            ContextProcessorError: If processing fails
        """
        try:
            # Initialize scene context
            scene_context = {}
            for image_id, image_data in images.items():
                scene_context[image_id] = {
                    "caption": image_data.scene_context.get("caption", ""),
                    "scene_attributes": []
                }

            if not scene_subqueries:
                return scene_context

            # Load models
            llm_client = self.model_manager.get_llm_client()
            qwen_client = self.model_manager.get_qwen_vl()

            # Step 1: Analyze all scene subqueries to determine scene attribute candidates
            all_candidates = self._analyze_scene_subqueries_for_candidates(
                llm_client, scene_subqueries, images
            )

            print(f"Generated {len(all_candidates)} scene attribute candidates from {len(scene_subqueries)} subqueries")

            # Step 2: Verify each candidate with Qwen binary verification
            verified_results = []
            for candidate in all_candidates:
                result = self._verify_scene_attribute_candidate(
                    qwen_client, candidate, image_paths
                )
                if result:
                    verified_results.append(result)

            print(f"Verified {len(verified_results)} scene attributes")

            # Step 3: Group results by image and store
            for result in verified_results:
                if result.image_id in scene_context:
                    scene_context[result.image_id]["scene_attributes"].append(result.to_dict())

            return scene_context

        except Exception as e:
            raise ContextProcessorError(f"Failed to process scene_attribute subqueries: {str(e)}")

    def _analyze_scene_subqueries_for_candidates(
        self,
        llm_client,
        scene_subqueries: List[BinarySubquery],
        images: Dict[str, ImageData]
    ) -> List[SceneAttributeCandidate]:
        """
        Analyze all scene subqueries to determine what scene attributes need verification.
        Decomposes compound subqueries into atomic binary questions.

        Args:
            llm_client: LLM client for analysis
            scene_subqueries: List of scene_attribute subqueries
            images: ImageData structure with captions

        Returns:
            List[SceneAttributeCandidate]: All scene attribute candidates that need verification
        """
        all_candidates = []

        for subquery in scene_subqueries:
            if subquery.subquery_type != "scene_attribute":
                continue

            # Analyze this subquery to determine required scene attributes
            candidates = self._analyze_single_subquery_for_scene_attributes(
                llm_client, subquery, images
            )

            # Add subquery reference to candidates
            for candidate in candidates:
                candidate.required_for_subqueries = [subquery.question]

            all_candidates.extend(candidates)

        return all_candidates

    def _analyze_single_subquery_for_scene_attributes(
        self,
        llm_client,
        subquery: BinarySubquery,
        images: Dict[str, ImageData]
    ) -> List[SceneAttributeCandidate]:
        """
        Analyze a single scene subquery to determine required scene attribute verifications.

        Args:
            llm_client: LLM client
            subquery: Scene attribute subquery to analyze
            images: ImageData structure

        Returns:
            List[SceneAttributeCandidate]: Required scene attribute verifications for this subquery
        """
        try:
            # Build context about available images and their captions
            image_context = {}
            for image_id, image_data in images.items():
                image_context[image_id] = image_data.scene_context.get("caption", "")

            prompt = f"""Analyze this scene attribute subquery to determine what atomic scene attributes need verification.

Subquery: "{subquery.question}"

Available Images and Descriptions:
{json.dumps(image_context, indent=2)}

Task: Break this subquery into atomic scene attribute verifications that can be answered with binary Yes/No questions.

For each atomic verification needed, provide:
1. image_id: Which image to verify
2. attribute_class: Scene attribute category (environment_type, lighting, weather, vegetation, time_of_day, etc.)
3. candidate_value: The specific value to verify
4. binary_question: A clear Yes/No question for VLM verification

Respond in this exact JSON format:
{{
  "scene_attribute_candidates": [
    {{
      "image_id": "image_a",
      "attribute_class": "environment_type",
      "candidate_value": "outdoor",
      "binary_question": "Is this an outdoor environment?"
    }}
  ]
}}

Examples:
- "Do both images show outdoor settings?" → Need environment_type=outdoor for both images
- "Is IMAGE_A taken during daytime with blue sky?" → Need time_of_day=daytime AND sky_color=blue for IMAGE_A
- "Does IMAGE_B have grass?" → Need vegetation=grass for IMAGE_B

Answer:"""

            response = llm_client.generate_response(prompt, temperature=0.2)

            # Parse JSON response
            try:
                result = json.loads(response)
                candidates_data = result.get("scene_attribute_candidates", [])

                candidates = []
                for data in candidates_data:
                    candidate = SceneAttributeCandidate(
                        image_id=data.get("image_id", ""),
                        attribute_class=data.get("attribute_class", ""),
                        candidate_value=data.get("candidate_value", ""),
                        binary_question=data.get("binary_question", "")
                    )
                    candidates.append(candidate)

                return candidates

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse LLM response for scene analysis: {e}")
                print(f"Response was: {response}")
                return []

        except Exception as e:
            print(f"Warning: Failed to analyze scene subquery '{subquery.question}': {e}")
            return []

    def _verify_scene_attribute_candidate(
        self,
        qwen_client: QwenVL,
        candidate: SceneAttributeCandidate,
        image_paths: List[str]
    ) -> SceneAttributeResult:
        """
        Verify a scene attribute candidate using Qwen binary verification.

        Args:
            qwen_client: Qwen VLM client
            candidate: Scene attribute candidate to verify
            image_paths: List of image file paths

        Returns:
            SceneAttributeResult: Verification result with confidence score
        """
        try:
            # Find the image file path
            image_path = None
            for path in image_paths:
                if candidate.image_id in path:
                    image_path = path
                    break

            if not image_path:
                print(f"Warning: Could not find image path for {candidate.image_id}")
                return None

            # Load image
            image = Image.open(image_path)

            # Create binary verification question
            verification_question = f"""Look at this image. Answer only "Yes" or "No".

Question: {candidate.binary_question}

Answer:"""

            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(image, verification_question)

            # Use proper softmax probability calculation for P(statement is true)
            prob_statement_true = qwen_client.extract_yes_no_probability_with_proper_softmax(logits, response)

            # Validate response format
            is_valid_response = qwen_client.validate_yes_no_response(response)
            if not is_valid_response:
                print(f"Warning: Invalid Yes/No response: '{response}' for scene attribute verification")

            # Create result
            result = SceneAttributeResult(
                image_id=candidate.image_id,
                attribute_class=candidate.attribute_class,
                value=candidate.candidate_value,
                confidence=prob_statement_true,
                binary_question=candidate.binary_question,
                subquery=candidate.required_for_subqueries[0] if candidate.required_for_subqueries else ""
            )

            return result

        except Exception as e:
            print(f"Warning: Failed to verify scene attribute candidate: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Test scene attribute processor
    processor = ContextProcessor()

    # Sample data
    from src.core.types import BinarySubquery, ObjectDetection, ImageData

    scene_subqueries = [
        BinarySubquery(
            question="Do both images show outdoor settings?",
            referenced_objects=[],
            subquery_type="scene_attribute"
        ),
        BinarySubquery(
            question="Is IMAGE_A taken during daytime with blue sky?",
            referenced_objects=[],
            subquery_type="scene_attribute"
        )
    ]

    image_paths = ["test_images/image_a.jpg", "test_images/image_b.jpg"]

    images = {
        "image_a": ImageData(
            objects=[ObjectDetection(0, "person", [10.0, 20.0, 50.0, 60.0], 0.9)],
            attributes={},
            relationships=[],
            scene_context={"caption": "A person standing in a bright outdoor field during daytime with blue sky"}
        ),
        "image_b": ImageData(
            objects=[ObjectDetection(0, "animal", [10.0, 20.0, 50.0, 60.0], 0.9)],
            attributes={},
            relationships=[],
            scene_context={"caption": "An animal grazing in an outdoor meadow"}
        )
    }

    try:
        result = processor.process_scene_attribute_subqueries(
            scene_subqueries, image_paths, images
        )
        print(f"✓ Scene attribute processing result:")
        for image_id, data in result.items():
            print(f"  {image_id}: {len(data['scene_attributes'])} scene attributes")
            for attr in data['scene_attributes']:
                print(f"    - {attr}")
        print("✓ Scene attribute processor ready for pipeline!")
    except Exception as e:
        print(f"✗ Scene attribute processor test failed: {e}")