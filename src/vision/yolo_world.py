from ultralytics import YOLOWorld
from PIL import Image

class YoloWorld:
    def __init__(self, model_name: str = "yolov8l-worldv2.pt") -> None:
        self.model = YOLOWorld(model_name)

    def detect(self, image: Image.Image) -> list[dict[str, object]]:
        results = self.model(image) 
        return self.parse_results(results)

    def detect_and_save(self, image: Image.Image, save_path: str) -> list[dict[str, object]]:
        results = self.model(image) 
        results[0].save(save_path)
        return self.parse_results(results)

    def parse_results(self, results: object) -> list[dict[str, object]]:
        boxes = results[0].boxes
        parsed = []
        for box in boxes:
            class_name = results[0].names[int(box.cls[0])]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            parsed.append({
                "label": class_name,
                "confidence": confidence,
                "coordinates": (x1, y1, x2, y2)
            })
        
        return parsed
        
