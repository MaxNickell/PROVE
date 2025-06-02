from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

class ViltVqa:
    def __init__(self, model_name: str = "dandelin/vilt-b32-finetuned-vqa") -> None:
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)
    
    def query_image(self, image: Image.Image, query: str) -> str:
        encoding = self.processor(image, query, return_tensors="pt")
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        return self.model.config.id2label[idx]