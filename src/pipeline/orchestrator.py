from typing import Dict, List, Any, Tuple

from src.pipeline.detector import Detector
from src.pipeline.attribute_extractor import AttributeExtractor
from src.pipeline.salient_intra_relationship_extractor import SalientIntraRelationshipExtractor
from src.language.llm_client import LLMClient
import json
class Orchestrator:

    def __init__(self, explainable: bool = False) -> None:
        self.explainable = explainable

        self.llm_client = LLMClient()

        self.detector = Detector()
        self.attribute_extractor = AttributeExtractor()
        self.salient_intra_relationship_extractor = SalientIntraRelationshipExtractor(self.llm_client)

        self.objects: Dict[str, List[Dict[str, Any]]] = {}
        self.attributes: Dict[str, Dict[int, List[str]]] = {}
        self.scene_tags: Dict[str, List[Tuple[str, float]]] = {}
        self.intra_rels: Dict[str, List[Tuple[str, int, int, float]]] = {}
        self.inter_rels: List[Tuple[str, str, int, str, int, float]] = []
        self.kb_facts: List[str] = []
        self.hypotheses: List[Tuple[str, str]] = []
        self.results: List[Tuple[str, float]] = []

    def run(self, image_a_path: str, image_b_path: str, question: str, answer_options: List[str]) -> Dict[str, Any]:
        self._detect_objects(image_a_path, image_id="A")
        self._detect_objects(image_b_path, image_id="B")

        self._extract_attributes(image_a_path, image_id="A")
        self._extract_attributes(image_b_path, image_id="B")

        self._select_salient_intra_relations(question, image_id="A", image_path=image_a_path, objects=self.objects["A"])
        self._select_salient_intra_relations(question, image_id="B", image_path=image_b_path, objects=self.objects["B"])

    def _detect_objects(self, image_path: str, *, image_id: str) -> None:
        if self.explainable:
            self.objects[image_id] = self.detector.detect(image_path, visualize=True)
        else:
            self.objects[image_id] = self.detector.detect(image_path)

    def _extract_attributes(self, image_path: str, *, image_id: str) -> None:
        """Extract attributes from the image using Florence-2 and LLM normalization."""
        if image_id in self.objects:
            # Extract attributes for all detected objects
            self.attributes[image_id] = self.attribute_extractor.extract_attributes(
                image_path, 
                self.objects[image_id],
                save_crops=self.explainable
            )
            
            # Also add attributes to each object for convenience
            for obj in self.objects[image_id]:
                obj_id = obj['id']
                if obj_id in self.attributes[image_id]:
                    obj['attributes'] = self.attributes[image_id][obj_id]


    def _select_salient_intra_relations(self, question: str, *, image_id: str, image_path: str, objects: List[Dict[str, Any]]) -> None:
        self.intra_rels[image_id] = self.salient_intra_relationship_extractor.extract(question, objects, image_path)


