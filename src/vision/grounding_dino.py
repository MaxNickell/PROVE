import requests, torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class GroundingDino:
    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base") -> None:
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to("cuda")

    def detect(self, image: Image.Image, query: str) -> list[dict[str, object]]:
        """
            Query must be in the format - "a lowercaseword1. a lowercaseword2. a loswercaseword3. ..."
            Example: "a glass. a bottle."
        """
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

        return self.parse_results(results)
        
    def detect_and_save(self, image: Image.Image, query: str, output_path: str) -> list[dict[str, object]]:
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

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        boxes = results[0]['boxes'].cpu().numpy()
        scores = results[0]['scores'].cpu().numpy()
        labels = results[0]['labels']
        
        for box, score, label in zip(boxes, scores, labels):
            box = [int(b) for b in box]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red", font=font)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        return self.parse_results(results)

    def parse_results(self, results: object) -> list[dict[str, object]]:
        boxes = results[0]['boxes'].cpu().numpy()
        scores = results[0]['scores'].cpu().numpy()
        labels = results[0]['text_labels']
        
        parsed = []
        for i, box in enumerate(boxes):
            confidence = float(scores[i])
            label = labels[i]
            x1, y1, x2, y2 = box.tolist()
            parsed.append({
                "label": label,
                "confidence": confidence,
                "coordinates": (x1, y1, x2, y2)
            })
        
        return parsed
