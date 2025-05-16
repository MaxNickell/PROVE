import requests, torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIPCaptioner():
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl"):
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    def caption(self, image_url, query): 
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        inputs = self.processor(image, query, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        
        return self.processor.decode(out[0], skip_special_tokens=True)
