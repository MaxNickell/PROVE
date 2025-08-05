import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


class Llava:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.device = "cuda:4"
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": 4}
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def run_inference(self, image: Image.Image, query: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device, torch.float16)
        output = self.model.generate(**inputs)
        result = self.processor.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        return result.strip()
    