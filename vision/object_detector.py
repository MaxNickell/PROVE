import requests, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class GroundingDINODetector():
    def __init__(self, model_name="IDEA-Research/grounding-dino-base"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to("cuda")
    
    def detect(self, image_url, query):
        """
            Query must be in the format - "a lowercaseword1. a lowercaseword2. a loswercaseword3. ..."
            Example: "a glass. a bottle."
        """
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = self.processor(images=image, text=query, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        return results