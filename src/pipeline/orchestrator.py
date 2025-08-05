from typing import Dict, List, Any, Tuple

from src.pipeline.detector import Detector
from src.language.tasks.label_canonicalizer import LabelCanonicalizer
from src.language.llm_client import LLMClient
from src.pipeline.attribute_extractor import AttributeExtractor

class Orchestrator:

    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.label_canonicalizer = LabelCanonicalizer(self.llm_client)

        self.detector = Detector(label_canonicalizer=self.label_canonicalizer, explainable=True)
        self.attribute_extractor = AttributeExtractor()

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
        print(self.objects)
        print(self.attributes)

        self._extract_scene_descriptors(image_id="A")
        self._extract_scene_descriptors(image_id="B")

        self._mine_generic_relations(image_id="A")
        self._mine_generic_relations(image_id="B")

        self._select_salient_intra_relations(question) # LLM -> determine whats important -> crop with object pairs to vlm -> answers
        self._select_salient_inter_relations(question) # LLM -> determine whats important -> comparisons (based off attributes of objects) -> larger(weight_a, weight_b) = True

        self._build_knowledge_base()
        # larger(weight_a, weight_b) = True
        # larger(arm_a, arm_b) = False
        # holding(mana, wiehgt_a)
        # leaner(body_a, body_b) = True

        # \phi_1 = stonger(man_a, man_b)
        # \phi_2 = stronger(man_b, man_a)

        # stronger(man_a, man_b) l
        

        self._map_answer_options_to_queries(answer_options, question)
        self._run_problog_inference()
        selected_answer, explanation = self._select_and_explain_answer()

        return {
            "answer": selected_answer,
            "explanation": explanation,
            "kb_facts": self.kb_facts,
            "problog_results": self.results
        }

    def _detect_objects(self, image_path: str, *, image_id: str) -> None:
        self.objects[image_id] = self.detector.detect(image_path)

    def _extract_attributes(self, image_path: str, *, image_id: str) -> None:
        self.attributes[image_id] = self.attribute_extractor.extract_attributes(image_path, self.objects[image_id])

    def _extract_scene_descriptors(self, *, image_id: str) -> None:
        """Run global taggers (Tag2Text, Places-365, VLM caption) to get scene-level facts."""
        # TODO: Implement scene descriptor extraction
        self.scene_tags[image_id] = []

    def _mine_generic_relations(self, *, image_id: str) -> None:
        """Use a scene-graph model to propose spatial/functional relations inside one image."""
        # TODO: Implement relation mining
        self.intra_rels[image_id] = []

    def _select_salient_intra_relations(self, question: str) -> None:
        """Call LLM to pick WHICH intra-image relations are important wrt the question, then
        verify them via DeepSeek-VL2 yes/no queries; store into self.intra_rels."""
        # TODO: Implement salient intra-relation selection
        pass

    def _select_salient_inter_relations(self, question: str) -> None:
        """Similar to above but for cross-image relations that may resolve comparative queries."""
        # TODO: Implement salient inter-relation selection
        pass

    def _build_knowledge_base(self) -> None:
        """Convert objects, attributes, scene tags, and relations into weighted Problog facts
        and populate self.kb_facts."""
        # TODO: Implement knowledge base building
        self.kb_facts = []

    def _map_answer_options_to_queries(self, answer_options: List[str], question: str) -> None:
        """Use an LLM to translate each natural-language answer choice into a formal
        FOL/Prolog query; store as (answer_text, query_str) pairs in self.hypotheses."""
        # TODO: Implement answer option mapping
        self.hypotheses = [(opt, f"query_{i}") for i, opt in enumerate(answer_options)]

    def _run_problog_inference(self) -> None:
        """Send KB facts and each hypothesis query to ProbLog; record probabilities
        into self.results as (query_str, prob)."""
        # TODO: Implement ProbLog inference
        self.results = [(query, 0.5) for _, query in self.hypotheses]

    def _select_and_explain_answer(self) -> Tuple[str, str]:
        """Choose the answer with highest entailment probability, then ask LLM to
        turn the Problog proof trace into a concise explanation.
        
        Returns
        -------
        selected_answer : str
        explanation     : str
        """
        # TODO: Implement proper answer selection and explanation
        if self.hypotheses:
            # For now, just return the first answer option
            selected_answer = self.hypotheses[0][0]
            explanation = f"Selected '{selected_answer}' based on detected objects and attributes."
            return selected_answer, explanation
        else:
            return "Unable to determine", "No answer options provided."
