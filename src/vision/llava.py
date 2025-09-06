import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Union

from src.core.vlm_interface import VLMInterface, VLMError, VLMNotAvailableError, VLMInferenceError


class Llava(VLMInterface):
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        self.device = "cuda:4"
        self._model_loaded = False
        
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map={"": 4}
            )
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self._model_loaded = True
        except Exception as e:
            raise VLMError(f"Failed to load LLaVA model: {e}", "llava")
        
    def run_inference(self, image: Union[Image.Image, str], prompt: str) -> str:
        """
        Run inference on an image with a text prompt using LLaVA.
        
        Args:
            image: PIL Image object or path to image file
            prompt: Text prompt for LLaVA
            
        Returns:
            str: LLaVA response as text
            
        Raises:
            VLMInferenceError: If inference fails
            VLMNotAvailableError: If model is not loaded
        """
        if not self.is_available():
            raise VLMNotAvailableError("LLaVA model is not loaded", "llava")
        
        try:
            # Handle image input
            if isinstance(image, str):
                image = Image.open(image)
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt_text, return_tensors='pt').to(self.device, torch.float16)
            output = self.model.generate(**inputs)
            result = self.processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            
            return result.strip()
            
        except Exception as e:
            raise VLMInferenceError(f"LLaVA inference failed: {e}", "llava")
    
    def get_model_name(self) -> str:
        """Get the name of the LLaVA model."""
        return self.model_name
    
    def is_available(self) -> bool:
        """Check if LLaVA model is loaded and ready."""
        return self._model_loaded and hasattr(self, 'model') and hasattr(self, 'processor')


# Register LLaVA as a VLM provider
from src.core.vlm_interface import VLMRegistry
VLMRegistry.register_provider("llava", Llava)