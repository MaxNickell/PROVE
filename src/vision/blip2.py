import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class Blip2:
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl") -> None:
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map={"": 4})

    def caption(self, image: Image.Image, query: str) -> str: 
        inputs = self.processor(image, query, return_tensors="pt").to("cuda:4")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
