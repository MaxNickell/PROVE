"""
Scene attribute processor for PROVE pipeline.
Handles scene_attribute subqueries using LLM decomposition + Qwen binary verification.
Follows the established subquery analysis pattern used by attribute and relationship extractors.
"""

from typing import List, Dict, Any, Tuple
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import ObjectDetection, BinarySubquestion, ImageData
from src.core.probability import get_verifier_probability
from src.vision.qwen_vl import QwenVL


class SceneAttributeProcessorError(RuntimeError):
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
    def __init__(self, image_id: str, attribute_class: str, value: str, confidence: float, binary_question: str, subquestion: str):
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


class SceneAttributeProcessor:
    """
    Process scene_attribute subqueries using LLM decomposition + Qwen binary verification.
    Follows the established pattern: Subquery → LLM Analysis → Atomic Binary Questions → Qwen Verification → Results
    """

    def __init__(self):
        """Initialize processor with ModelManager singleton."""
        self.model_manager = ModelManager()

    def process_scene_attribute_subqueries(
        self,
        scene_subquestions: List[BinarySubquestion],
        image_paths: Dict[str, str],
        images: Dict[str, ImageData],
        image_contexts: Dict[str, str] = None
    ) -> Dict[str, int]:
        """
        Process scene_attribute subqueries using proper subquery decomposition.

        Args:
            scene_subquestions: List of scene_attribute binary subqueries
            image_paths: Dict mapping image_id to file path (e.g., {'image_a': './test_images/img0.png'})
            images: ImageData structure containing objects per image
            image_contexts: Optional dict mapping image_id to caption text for processing

        Returns:
            Dict[str, int]: Count of scene attributes extracted per image
            Scene attributes are stored directly in ImageData.scene_attributes field

        Raises:
            ContextProcessorError: If processing fails
        """
        try:
            # Initialize scene_attributes for all images
            for image_id, image_data in images.items():
                if not hasattr(image_data, 'scene_attributes') or image_data.scene_attributes is None:
                    image_data.scene_attributes = {}

            if not scene_subquestions:
                # Return count of scene attributes per image
                return {image_id: len(image_data.scene_attributes) for image_id, image_data in images.items()}

            # Load models
            llm_client = self.model_manager.get_llm_client()
            qwen_client = self.model_manager.get_qwen_vl()

            # Step 1: Analyze all scene subqueries to determine scene attribute candidates
            all_candidates = self._analyze_scene_subquestions_for_candidates(
                llm_client, scene_subquestions, images, image_contexts
            )

            print(f"Generated {len(all_candidates)} scene attribute candidates from {len(scene_subquestions)} subqueries")

            # Step 2: Verify each candidate with Qwen binary verification
            verified_results = []
            for candidate in all_candidates:
                result = self._verify_scene_attribute_candidate(
                    qwen_client, candidate, image_paths
                )
                if result:
                    verified_results.append(result)

            print(f"Verified {len(verified_results)} scene attributes")

            # Step 3: Store results directly in ImageData scene_attributes (matching object attribute structure)
            for result in verified_results:
                if result.image_id in images:
                    # Store using same structure as object attributes: attribute_class -> [AttributeValue, ...]
                    attribute_class = result.attribute_class
                    attribute_value = {"value": result.value, "confidence": result.confidence}

                    if attribute_class not in images[result.image_id].scene_attributes:
                        images[result.image_id].scene_attributes[attribute_class] = []

                    images[result.image_id].scene_attributes[attribute_class].append(attribute_value)

            # Return count of scene attributes per image for logging
            return {image_id: len(image_data.scene_attributes) for image_id, image_data in images.items()}

        except Exception as e:
            raise SceneAttributeProcessorError(f"Failed to process scene_attribute subquestions: {str(e)}")

    def _analyze_scene_subquestions_for_candidates(
        self,
        llm_client,
        scene_subquestions: List[BinarySubquestion],
        images: Dict[str, ImageData],
        image_contexts: Dict[str, str] = None
    ) -> List[SceneAttributeCandidate]:
        """
        Analyze all scene subqueries to determine what scene attributes need verification.
        Decomposes compound subqueries into atomic binary questions.

        Args:
            llm_client: LLM client for analysis
            scene_subquestions: List of scene_attribute subqueries
            images: ImageData structure containing objects per image
            image_contexts: Optional dict mapping image_id to caption text for processing

        Returns:
            List[SceneAttributeCandidate]: All scene attribute candidates that need verification
        """
        all_candidates = []

        for subquery in scene_subquestions:
            if subquestion.subquery_type != "scene_attribute":
                continue

            # Analyze this subquery to determine required scene attributes
            candidates = self._analyze_single_subquery_for_scene_attributes(
                llm_client, subquery, images, image_contexts
            )

            # Add subquery reference to candidates
            for candidate in candidates:
                candidate.required_for_subqueries = [subquestion.question]

            all_candidates.extend(candidates)

        return all_candidates

    def _analyze_single_subquery_for_scene_attributes(
        self,
        llm_client,
        subquestion: BinarySubquestion,
        images: Dict[str, ImageData],
        image_contexts: Dict[str, str] = None
    ) -> List[SceneAttributeCandidate]:
        """
        Analyze a single scene subquery to determine required scene attribute verifications.

        Args:
            llm_client: LLM client
            subquestion: Scene attribute subquery to analyze
            images: ImageData structure
            image_contexts: Optional dict mapping image_id to caption text for processing

        Returns:
            List[SceneAttributeCandidate]: Required scene attribute verifications for this subquery
        """
        try:
            # Build context about available images and their captions
            image_context = image_contexts or {}

            # Format image context without json.dumps
            context_str = "\n".join([f"{img_id}: {desc}" for img_id, desc in image_context.items()])

            prompt = f"""Analyze this scene attribute subquery to determine what atomic scene attributes need verification.

Subquery: "{subquestion.question}"

Available Images and Descriptions:
{context_str}

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

            messages = [{"role": "user", "content": prompt}]

            # Use Pydantic validation for guaranteed structure
            response = llm_client.analyze_scene_attributes(messages, temperature=0.2)

            # Convert Pydantic response to SceneAttributeCandidate objects
            candidates = []
            for item in response.scene_attribute_candidates:
                candidate = SceneAttributeCandidate(
                    image_id=item.image_id,
                    attribute_class=item.attribute_class,
                    candidate_value=item.candidate_value,
                    binary_question=item.binary_question
                )
                candidates.append(candidate)

            return candidates

        except Exception as e:
            print(f"Warning: Failed to analyze scene subquery '{subquestion.question}': {e}")
            return []

    def _verify_scene_attribute_candidate(
        self,
        qwen_client: QwenVL,
        candidate: SceneAttributeCandidate,
        image_paths: Dict[str, str]
    ) -> SceneAttributeResult:
        """
        Verify a scene attribute candidate using Qwen binary verification.

        Args:
            qwen_client: Qwen VLM client
            candidate: Scene attribute candidate to verify
            image_paths: Dict mapping image_id to file path

        Returns:
            SceneAttributeResult: Verification result with confidence score
        """
        try:
            # Get image path directly from mapping
            if candidate.image_id not in image_paths:
                print(f"Warning: Could not find image path for {candidate.image_id}")
                return None

            image_path = image_paths[candidate.image_id]

            # Load image
            image = Image.open(image_path)

            # Create binary verification question
            verification_question = f"""Look at this image. Answer only "Yes" or "No".

Question: {candidate.binary_question}

Answer:"""

            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(image, verification_question)

            # Use unified verifier probability extraction
            prob_statement_true = get_verifier_probability(
                logits,
                response,
                qwen_client.processor.tokenizer
            )

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
    from src.core.types import BinarySubquestion, ObjectDetection, ImageData

    scene_subquestions = [
        BinarySubquestion(
            question="Do both images show outdoor settings?",
            referenced_objects=[],
            subquery_type="scene_attribute"
        ),
        BinarySubquestion(
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
            scene_subquestions, image_paths, images
        )
        print(f"✓ Scene attribute processing result:")
        for image_id, data in result.items():
            print(f"  {image_id}: {len(data['scene_attributes'])} scene attributes")
            for attr in data['scene_attributes']:
                print(f"    - {attr}")
        print("✓ Scene attribute processor ready for pipeline!")
    except Exception as e:
        print(f"✗ Scene attribute processor test failed: {e}")