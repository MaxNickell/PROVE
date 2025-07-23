from src.pipeline.detector import ObjectDetector
from src.language.forge_llm import ForgeLLM
from typing import Dict, Any

class Orchestrator:
    def __init__(self):
        self.detector = ObjectDetector()
        self.llm = ForgeLLM()
    
    def run_pipeline(self, image_path_1: str, image_path_2: str, question: str) -> Dict[str, Any]:
        image_1_dets, image_2_dets = self._run_detection(image_path_1, image_path_2)
        required_relationships = self._infer_required_relationships(image_1_dets, image_2_dets, question)
        return required_relationships, image_1_dets, image_2_dets
    
    def _run_detection(self, image_path_1: str, image_path_2: str) -> Dict[str, Any]:
        image_1_dets = self.detector.detect(image_path_1)
        image_2_dets = self.detector.detect(image_path_2)
        return image_1_dets, image_2_dets
    
    def _infer_required_relationships(self, image_1_dets: Dict[str, Any], image_2_dets: Dict[str, Any], question: str) -> Dict[str, Any]:
        return self.llm.infer_required_relationships(image_1_dets, image_2_dets, question)

    