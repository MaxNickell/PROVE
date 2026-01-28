"""
BLIP-ITM verification module for PROVE pipeline.
Provides well-calibrated probability scores for attribute and relationship verification.
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
from typing import List, Tuple, Union


class BLIPVerifierError(Exception):
    """Custom exception for BLIP Verifier related errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BLIPVerifier:
    """
    BLIP-ITM based verifier for attribute and relationship verification.

    Uses Image-Text Matching (ITM) head to produce well-calibrated probabilities
    for binary verification tasks.
    """

    PADDING_RATIO = 0.15  # 15% padding on each side

    def __init__(self, model_name: str = "Salesforce/blip-itm-large-coco", device: str = "auto"):
        """
        Initialize BLIP-ITM verifier.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self._model_loaded = False

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        try:
            print(f"Loading {model_name}...")

            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForImageTextRetrieval.from_pretrained(
                model_name
            ).to(self.device).eval()

            self._model_loaded = True
            print(f"✓ {model_name} loaded successfully on {self.device}")

        except Exception as e:
            raise BLIPVerifierError(f"Failed to load BLIP-ITM model: {e}")

    def _get_article(self, word: str) -> str:
        """Return 'an' if word starts with vowel sound, else 'a'."""
        vowels = ('a', 'e', 'i', 'o', 'u')
        return "an" if word.lower().startswith(vowels) else "a"

    def _crop_with_padding(
        self,
        image: Image.Image,
        bbox: List[float],
        padding_ratio: float = None
    ) -> Image.Image:
        """
        Crop image to bounding box with relative padding.

        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2] coordinates
            padding_ratio: Padding as fraction of bbox size (default: PADDING_RATIO)

        Returns:
            Cropped PIL Image
        """
        if padding_ratio is None:
            padding_ratio = self.PADDING_RATIO

        x1, y1, x2, y2 = [float(c) for c in bbox]

        # Calculate padding based on bbox dimensions
        width = x2 - x1
        height = y2 - y1
        pad_x = width * padding_ratio
        pad_y = height * padding_ratio

        # Apply padding and clamp to image bounds
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(image.width, int(x2 + pad_x))
        y2 = min(image.height, int(y2 + pad_y))

        return image.crop((x1, y1, x2, y2))

    def _get_itm_score(self, image: Image.Image, prompt: str) -> float:
        """
        Get Image-Text Matching probability score.

        Args:
            image: PIL Image (already cropped)
            prompt: Text prompt to match against image

        Returns:
            float: Probability that text matches image (0.0 to 1.0)
        """
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            itm_scores = self.model(**inputs).itm_score
            # ITM head outputs [not_match, match] logits
            probability = torch.softmax(itm_scores, dim=1)[0, 1].item()

        return probability

    def verify_attribute(
        self,
        image: Union[Image.Image, str],
        bbox: List[float],
        object_class: str,
        attr_value: str
    ) -> float:
        """
        Verify if an entity has a specific attribute.

        Args:
            image: PIL Image or path to image file
            bbox: Bounding box [x1, y1, x2, y2] of the entity
            object_class: Class of the object (e.g., "cat", "car")
            attr_value: Attribute value to verify (e.g., "orange", "metallic")

        Returns:
            float: Probability that the entity has the attribute (0.0 to 1.0)

        Example:
            >>> prob = verifier.verify_attribute(image, [100, 100, 200, 200], "cat", "orange")
            >>> print(f"P(orange cat) = {prob:.3f}")
        """
        if not self.is_available():
            raise BLIPVerifierError("BLIP Verifier model is not loaded")

        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Crop to entity with padding
        cropped = self._crop_with_padding(image, bbox)

        # Build prompt: "a {value} {object}" with proper article
        article = self._get_article(attr_value)
        prompt = f"{article} {attr_value} {object_class}"

        return self._get_itm_score(cropped, prompt)

    def verify_relationship(
        self,
        image: Union[Image.Image, str],
        bbox1: List[float],
        bbox2: List[float],
        obj1_class: str,
        obj2_class: str,
        relation: str
    ) -> float:
        """
        Verify if two entities have a specific relationship.

        Args:
            image: PIL Image or path to image file
            bbox1: Bounding box [x1, y1, x2, y2] of the subject entity
            bbox2: Bounding box [x1, y1, x2, y2] of the object entity
            obj1_class: Class of subject (e.g., "man", "bird")
            obj2_class: Class of object (e.g., "buffalo", "tree")
            relation: Relationship to verify (e.g., "riding", "on_top_of")

        Returns:
            float: Probability that the relationship holds (0.0 to 1.0)

        Example:
            >>> prob = verifier.verify_relationship(
            ...     image, [10, 20, 100, 150], [80, 100, 200, 250],
            ...     "man", "buffalo", "riding"
            ... )
            >>> print(f"P(man riding buffalo) = {prob:.3f}")
        """
        if not self.is_available():
            raise BLIPVerifierError("BLIP Verifier model is not loaded")

        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Compute union bounding box
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        union_bbox = [x1, y1, x2, y2]

        # Crop to union with padding
        cropped = self._crop_with_padding(image, union_bbox)

        # Build prompt: "a {obj1} {relation} a {obj2}"
        # Replace underscores with spaces (e.g., "on_top_of" -> "on top of")
        relation_text = relation.replace("_", " ")

        article1 = self._get_article(obj1_class)
        article2 = self._get_article(obj2_class)
        prompt = f"{article1} {obj1_class} {relation_text} {article2} {obj2_class}"

        return self._get_itm_score(cropped, prompt)

    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return (self._model_loaded and
                hasattr(self, 'model') and
                hasattr(self, 'processor'))

    def get_model_name(self) -> str:
        """Get the name of the BLIP model."""
        return self.model_name

    def cleanup(self):
        """Clean up GPU memory."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._model_loaded = False
            print("✓ BLIP Verifier cleaned up successfully")

        except Exception as e:
            print(f"Warning: Failed to cleanup BLIP Verifier: {e}")
