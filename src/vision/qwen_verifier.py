"""
Qwen 2.5-VL-7B verifier for PROVE pipeline.
Uses True/False prompting for binary verification with probability extraction.
"""

import torch
import math
from PIL import Image
from typing import List, Union, Optional, Tuple
from src.vision.qwen_vl import QwenVL


class QwenVerifier:
    """
    Qwen 2.5-VL-7B based verifier for attribute and relationship verification.

    Uses prompted True/False responses with logit-based probability extraction.
    """

    PADDING_RATIO = 0.15  # 15% padding on each side

    def __init__(
        self,
        qwen_vl: QwenVL = None,
        device: str = "auto"
    ):
        """
        Initialize Qwen verifier.

        Args:
            qwen_vl: Existing QwenVL instance to reuse (avoids redundant loading)
            device: Device to use
        """
        self.device = device

        # Reuse existing QwenVL instance or create new one
        if qwen_vl is not None:
            self._qwen = qwen_vl
            self._owns_model = False
        else:
            self._qwen = None
            self._owns_model = True

    def _get_qwen(self) -> QwenVL:
        """Lazy load Qwen model."""
        if self._qwen is None:
            self._qwen = QwenVL(device=self.device)
        return self._qwen

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
        """Crop image to bounding box with relative padding."""
        if padding_ratio is None:
            padding_ratio = self.PADDING_RATIO

        x1, y1, x2, y2 = [float(c) for c in bbox]

        width = x2 - x1
        height = y2 - y1
        pad_x = width * padding_ratio
        pad_y = height * padding_ratio

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(image.width, int(x2 + pad_x))
        y2 = min(image.height, int(y2 + pad_y))

        return image.crop((x1, y1, x2, y2))

    def _get_vlm_probability(self, image: Image.Image, statement: str) -> Tuple[float, str]:
        """
        Get probability from VLM response via text parsing (fallback).

        Returns:
            Tuple of (probability, raw_response)
        """
        qwen = self._get_qwen()

        prompt = f'Determine if the following statement about this image is true or false.\n\nStatement: "{statement}"\n\nAnswer with ONLY "True" or "False".'
        pos_tokens = ["true", "correct", "yes"]
        neg_tokens = ["false", "incorrect", "no"]

        try:
            response = qwen.run_inference(image, prompt)
            response_lower = response.lower().strip()

            if any(pos in response_lower for pos in pos_tokens):
                return 0.9, response
            elif any(neg in response_lower for neg in neg_tokens):
                return 0.1, response
            else:
                return 0.5, response

        except Exception as e:
            print(f"Qwen inference failed: {e}")
            return 0.5, f"ERROR: {e}"

    def _get_vlm_probability_with_logits(self, image: Image.Image, statement: str) -> Tuple[float, str]:
        """
        Get probability from VLM using token logits for better calibration.

        This method extracts actual probabilities from the model's logits
        over True/False tokens.
        """
        qwen = self._get_qwen()

        prompt = f'Determine if the following statement about this image is true or false.\n\nStatement: "{statement}"\n\nAnswer with ONLY "True" or "False".'

        try:
            # Load image if needed
            if isinstance(image, str):
                image = Image.open(image)

            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = qwen.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = qwen.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            # Move inputs to device
            for key, value in inputs.items():
                if hasattr(value, 'to') and hasattr(value, 'device'):
                    inputs[key] = value.to(qwen.model.device)

            # Get logits for first token
            with torch.no_grad():
                outputs = qwen.model(
                    **inputs,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :]  # Last position logits

                # Get token IDs for True/False
                pos_tokens = ["True", "true", "TRUE"]
                neg_tokens = ["False", "false", "FALSE"]

                pos_ids = []
                neg_ids = []
                for tok in pos_tokens:
                    ids = qwen.processor.tokenizer.encode(tok, add_special_tokens=False)
                    if ids:
                        pos_ids.append(ids[0])
                for tok in neg_tokens:
                    ids = qwen.processor.tokenizer.encode(tok, add_special_tokens=False)
                    if ids:
                        neg_ids.append(ids[0])

                # Get max logit for positive and negative
                if pos_ids and neg_ids:
                    pos_logit = max(logits[0, tid].item() for tid in pos_ids)
                    neg_logit = max(logits[0, tid].item() for tid in neg_ids)

                    # Softmax over just these two options
                    prob = torch.softmax(torch.tensor([pos_logit, neg_logit]), dim=0)[0].item()

                    # Also generate the actual response for logging
                    gen_outputs = qwen.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=qwen.processor.tokenizer.eos_token_id
                    )
                    input_length = inputs['input_ids'].shape[1]
                    response = qwen.processor.decode(gen_outputs[0][input_length:], skip_special_tokens=True)

                    return prob, response.strip()
                else:
                    # Fallback to text parsing
                    return self._get_vlm_probability(image, statement)

        except Exception as e:
            print(f"Logit extraction failed, falling back to text parsing: {e}")
            return self._get_vlm_probability(image, statement)

    def verify_attribute(
        self,
        image: Union[Image.Image, str],
        bbox: List[float],
        object_class: str,
        attr_value: str,
        use_logits: bool = False
    ) -> Tuple[float, str]:
        """
        Verify if an entity has a specific attribute.

        Args:
            image: PIL Image or path
            bbox: Bounding box [x1, y1, x2, y2]
            object_class: Class of the object
            attr_value: Attribute to verify
            use_logits: Whether to use logit-based probability extraction

        Returns:
            Tuple of (probability, raw_response)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        cropped = self._crop_with_padding(image, bbox)

        article = self._get_article(attr_value)
        statement = f"This is {article} {attr_value} {object_class}"

        if use_logits:
            return self._get_vlm_probability_with_logits(cropped, statement)
        else:
            return self._get_vlm_probability(cropped, statement)

    def verify_relationship(
        self,
        image: Union[Image.Image, str],
        bbox1: List[float],
        bbox2: List[float],
        obj1_class: str,
        obj2_class: str,
        relation: str,
        use_logits: bool = False
    ) -> Tuple[float, str]:
        """
        Verify if two entities have a specific relationship.

        Args:
            image: PIL Image or path
            bbox1: Subject bounding box
            bbox2: Object bounding box
            obj1_class: Subject class
            obj2_class: Object class
            relation: Relationship to verify
            use_logits: Whether to use logit-based probability extraction

        Returns:
            Tuple of (probability, raw_response)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Compute union bounding box
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        union_bbox = [x1, y1, x2, y2]

        cropped = self._crop_with_padding(image, union_bbox)

        relation_text = relation.replace("_", " ")
        article1 = self._get_article(obj1_class)
        article2 = self._get_article(obj2_class)
        statement = f"There is {article1} {obj1_class} {relation_text} {article2} {obj2_class}"

        if use_logits:
            return self._get_vlm_probability_with_logits(cropped, statement)
        else:
            return self._get_vlm_probability(cropped, statement)

    def is_available(self) -> bool:
        """Check if model is available."""
        return self._qwen is not None and self._qwen.is_available()

    def cleanup(self):
        """Clean up resources if we own the model."""
        if self._owns_model and self._qwen is not None:
            self._qwen.cleanup()
            self._qwen = None
