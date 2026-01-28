"""
Qwen 2.5-VL-7B implementation for PROVE pipeline.
Provides open-ended visual question answering for perception tasks.
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Tuple, Union


class QwenVLError(Exception):
    """Custom exception for Qwen VL related errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class QwenVL:
    """
    Qwen 2.5-VL-7B Vision-Language Model implementation.

    Used for open-ended perception queries (e.g., "What color is this dog?").
    For binary verification tasks, use BLIPVerifier instead.

    Features:
    - Native bounding box support with <box> tags
    - Unconstrained response generation
    - Memory efficient GPU usage
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        """
        Initialize Qwen VL model.

        Args:
            model_name: Model identifier from HuggingFace
            device: Device allocation strategy (default: "auto" for automatic allocation)

        Raises:
            QwenVLError: If model loading fails
        """
        self.model_name = model_name
        self.device = device
        self._model_loaded = False

        try:
            print(f"Loading {model_name}...")

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True
            )

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self._model_loaded = True
            print(f"✓ {model_name} loaded successfully")

        except Exception as e:
            raise QwenVLError(f"Failed to load Qwen VL model: {e}")

    def run_inference(self, image: Union[Image.Image, str], prompt: str) -> str:
        """
        Run inference and return response text.

        Args:
            image: PIL Image object or path to image file
            prompt: Text prompt for the model

        Returns:
            str: Model response

        Raises:
            QwenVLError: If inference fails
        """
        if not self.is_available():
            raise QwenVLError("Qwen VL model is not loaded")

        try:
            # Handle image input
            if isinstance(image, str):
                image = Image.open(image)

            # Prepare inputs in Qwen 2.5-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template and process
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            # Move inputs to the correct device
            for key, value in inputs.items():
                if hasattr(value, 'to') and hasattr(value, 'device'):
                    inputs[key] = value.to(self.model.device)

            # Generate response (greedy decoding)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode response (excluding input tokens)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            raise QwenVLError(f"Qwen VL inference failed: {e}")

    def format_bbox_prompt(self, bbox: List[float], label: str) -> str:
        """
        Format bounding box coordinates for Qwen's native format.

        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            label: Object label

        Returns:
            str: Formatted bounding box string for Qwen
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        return f"<box>({x1},{y1}),({x2},{y2})</box>{label}"

    def get_model_name(self) -> str:
        """Get the name of the Qwen model."""
        return self.model_name

    def is_available(self) -> bool:
        """Check if Qwen model is loaded and ready."""
        return (self._model_loaded and
                hasattr(self, 'model') and
                hasattr(self, 'processor'))

    def get_memory_info(self) -> dict:
        """
        Get GPU memory usage information.

        Returns:
            dict: Memory usage statistics
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        try:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_cached = torch.cuda.memory_reserved() / (1024**3)
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3)

            return {
                "allocated_gb": round(memory_allocated, 2),
                "cached_gb": round(memory_cached, 2),
                "free_gb": round(memory_free, 2),
                "device": self.device
            }
        except Exception as e:
            return {"error": f"Failed to get memory info: {e}"}

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
            print("✓ Qwen VL model cleaned up successfully")

        except Exception as e:
            print(f"Warning: Failed to cleanup Qwen VL model: {e}")


# Utility functions for bounding box handling
def convert_florence_to_qwen_bbox(florence_bbox: List[float], image_size: Tuple[int, int] = None) -> str:
    """
    Convert Florence-2 bounding box format to Qwen format.

    Args:
        florence_bbox: [x1, y1, x2, y2] coordinates from Florence-2
        image_size: Optional (width, height) for coordinate validation

    Returns:
        str: Qwen format bounding box string
    """
    x1, y1, x2, y2 = [int(coord) for coord in florence_bbox]

    if image_size:
        width, height = image_size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

    return f"<box>({x1},{y1}),({x2},{y2})</box>"


def create_dual_bbox_prompt(obj1_bbox: List[float], obj1_label: str,
                           obj2_bbox: List[float], obj2_label: str) -> str:
    """
    Create prompt with two labeled bounding boxes.

    Args:
        obj1_bbox: First object bounding box
        obj1_label: First object label
        obj2_bbox: Second object bounding box
        obj2_label: Second object label

    Returns:
        str: Formatted prompt with both bounding boxes
    """
    box1 = convert_florence_to_qwen_bbox(obj1_bbox)
    box2 = convert_florence_to_qwen_bbox(obj2_bbox)

    return f"Object 1: {box1}{obj1_label}\nObject 2: {box2}{obj2_label}"
