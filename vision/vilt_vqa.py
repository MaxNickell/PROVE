from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


class ViltVqa():
    def __init__(self, model_name="dandelin/vilt-b32-finetuned-vqa"):
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)
    
    def query_image(self, image_url, query):
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        encoding = self.processor(image, query, return_tensors="pt")
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        return self.model.config.id2label[idx]