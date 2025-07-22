from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

class InstructBlip:
    def __init__(self, model_name: str = "Salesforce/instructblip-flan-t5-xl"):
        self.device = "cuda:0"
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name, 
            device_map={"": 0}
        )
        self.processor = InstructBlipProcessor.from_pretrained(model_name)
        
    def run_inference(self, image: Image.Image, query: str) -> str: 
        inputs = self.processor(image, query, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
            
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
