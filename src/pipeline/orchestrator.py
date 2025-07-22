from src.pipeline.detector import ObjectDetector
from typing import Dict, Any

class Orchestrator:
    def __init__(self, pipeline):
        self.detector = ObjectDetector()
    
    def run_detection(self, image_path_1: str, image_path_2: str) -> Dict[str, Any]:
        image_1_dets = self.detector.detect(image_path_1)
        image_2_dets = self.detector.detect(image_path_2)
    