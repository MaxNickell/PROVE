from src.vision.florence2 import Florence2
from typing import List, Dict, Any
from PIL import Image


class DetectorError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message

class Detector:

    def __init__(self):
        self.florence2 = Florence2()
    
    def detect(self, image_path: str, visualize: bool = False) -> List[Dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
            output_path = image_path.rsplit(".", 1)[0] + "_annotated." + image_path.rsplit(".", 1)[1]
            if visualize:
                detections = self.florence2.detect_and_visualize(image, output_path)
            else:
                detections = self.florence2.dense_detail_detect(image)
            return detections
        except Exception as err:
            raise DetectorError(f"Florence-2 detection failed: {err}")
        