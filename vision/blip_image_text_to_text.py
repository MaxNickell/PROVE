import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BlipImageTextToText:
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl") -> None:
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    def caption(self, image: Image.Image, query: str) -> str: 
        inputs = self.processor(image, query, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def caption_and_parse_nouns(self, image: Image.Image, query: str) -> str:
        caption = self.caption(image, query)
        formatted = '. '.join(f"a {item.strip()}" for item in caption.lower().split(', ') if item.strip()) + '.'
        return formatted
