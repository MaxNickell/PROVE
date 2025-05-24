import requests, torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 


class GroundingDinoZeroShotObjectDetection():
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
        
    def detect_and_save(self, image_url, query, output_path):
        results = self.detect(image_url, query)
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for result in results:
            boxes = result['boxes'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            labels = result['labels']
        
        for box, score, label in zip(boxes, scores, labels):
            box = [int(b) for b in box]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red", font=font)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
    
