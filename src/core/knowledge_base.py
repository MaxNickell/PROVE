"""
KnowledgeBase class for PROVE pipeline.
Central data structure with exact JSON schema compliance from build brief.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from src.core.types import (
    ObjectDetection, AttributeData, IntraRelation, InterComparison,
    IntraQuestion, InterQuestion, ProbLogResult, AnswerResult,
    Objects, Attributes, IntraRelations, InterComparisons
)


class KnowledgeBase:
    """
    Central data structure holding all evidence and reasoning results.
    Provides exact JSON schema compliance from build brief.
    """
    
    def __init__(self, question: str):
        """
        Initialize KnowledgeBase with a comparative question.
        
        Args:
            question: The main comparative question to answer
        """
        self.question = question
        
        # Image evidence structures (exact schema from build brief)
        self.image_a: Dict[str, List] = {
            "objects": [],        # List[ObjectDetection]
            "attributes": [],     # List[AttributeData] 
            "intra_relations": [] # List[IntraRelation]
        }
        
        self.image_b: Dict[str, List] = {
            "objects": [],        # List[ObjectDetection]
            "attributes": [],     # List[AttributeData]
            "intra_relations": [] # List[IntraRelation]
        }
        
        # Cross-image comparisons
        self.inter_comparisons: List[InterComparison] = []
        
        # Question generation results
        self._intra_questions_a: List[IntraQuestion] = []
        self._intra_questions_b: List[IntraQuestion] = []
        self._inter_questions: List[InterQuestion] = []
        
        # Reasoning components
        self.subqueries: List[str] = []
        self.problog: Dict[str, Any] = {
            "facts": "",
            "queries": "",
            "results": []  # List[ProbLogResult]
        }
        
        # Final answer
        self.answer: Dict[str, str] = {
            "text": "",
            "explanation": ""
        }
    
    def add_objects(self, image_id: str, objects: List[ObjectDetection]) -> None:
        """
        Add detected objects to the specified image.
        
        Args:
            image_id: 'a' or 'b' for image A or B
            objects: List of ObjectDetection instances
        """
        if image_id not in ['a', 'b']:
            raise ValueError("image_id must be 'a' or 'b'")
        
        if image_id == 'a':
            self.image_a["objects"] = objects
        else:
            self.image_b["objects"] = objects
    
    def add_attributes(self, image_id: str, attributes: List[AttributeData]) -> None:
        """
        Add extracted attributes to the specified image.
        
        Args:
            image_id: 'a' or 'b' for image A or B  
            attributes: List of AttributeData instances
        """
        if image_id not in ['a', 'b']:
            raise ValueError("image_id must be 'a' or 'b'")
            
        if image_id == 'a':
            self.image_a["attributes"] = attributes
        else:
            self.image_b["attributes"] = attributes
    
    def add_intra_relations(self, image_id: str, relations) -> None:
        """
        Add intra-image relations to the specified image.
        
        Args:
            image_id: 'a' or 'b' for image A or B
            relations: List of IntraRelation instances or verification results
        """
        if image_id not in ['a', 'b']:
            raise ValueError("image_id must be 'a' or 'b'")
        
        # Convert verification results to IntraRelation objects if needed
        if relations and isinstance(relations[0], dict):
            intra_relations = []
            for result in relations:
                # Handle new relation-based format
                if "relation" in result and "object_1" in result:
                    intra_relation = IntraRelation(
                        object_1=result["object_1"],
                        object_2=result["object_2"],
                        relation=result["relation"],
                        probability=result["probability"]
                    )
                else:
                    # Handle legacy format during transition
                    intra_relation = IntraRelation(
                        object_1=result.get("subject_object_id", 0),
                        object_2=result.get("object_object_id", 0),
                        relation=result.get("answer", "unknown"),
                        probability=result.get("confidence", 0.5)
                    )
                intra_relations.append(intra_relation)
            relations = intra_relations
            
        if image_id == 'a':
            self.image_a["intra_relations"] = relations
        else:
            self.image_b["intra_relations"] = relations
    
    def add_inter_comparisons(self, comparisons) -> None:
        """
        Add inter-image comparisons.
        
        Args:
            comparisons: List of InterComparison instances or verification results
        """
        # Convert verification results to InterComparison objects if needed
        if comparisons and isinstance(comparisons[0], dict):
            inter_comparisons = []
            for result in comparisons:
                inter_comparison = InterComparison(
                    image_a_object_id=result["image_a_object_id"],
                    image_b_object_id=result["image_b_object_id"],
                    attribute=result.get("attribute", "unknown"),
                    value_a=result.get("value_a", "unknown"),
                    value_b=result.get("value_b", "unknown"),
                    confidence_a=result.get("confidence_a", 1.0),
                    confidence_b=result.get("confidence_b", 1.0)
                )
                inter_comparisons.append(inter_comparison)
            comparisons = inter_comparisons
            
        self.inter_comparisons = comparisons
    
    def set_intra_questions(self, image_id: str, questions: List[IntraQuestion]) -> None:
        """
        Set intra-relationship questions for specified image.
        
        Args:
            image_id: 'a' or 'b' for image A or B
            questions: List of IntraQuestion instances
        """
        if image_id == 'a':
            self._intra_questions_a = questions
        else:
            self._intra_questions_b = questions
    
    def set_inter_questions(self, questions: List[InterQuestion]) -> None:
        """
        Set inter-comparison questions.
        
        Args:
            questions: List of InterQuestion instances
        """
        self._inter_questions = questions
    
    def get_objects(self, image_id: str) -> List[ObjectDetection]:
        """Get objects for specified image."""
        if image_id == 'a':
            return self.image_a["objects"]
        else:
            return self.image_b["objects"]
    
    def get_attributes(self, image_id: str) -> List[AttributeData]:
        """Get attributes for specified image."""  
        if image_id == 'a':
            return self.image_a["attributes"]
        else:
            return self.image_b["attributes"]
    
    def get_intra_relations(self, image_id: str) -> List[IntraRelation]:
        """Get intra-relations for specified image."""
        if image_id == 'a':
            return self.image_a["intra_relations"]
        else:
            return self.image_b["intra_relations"]
    
    def get_intra_questions(self, image_id: str) -> List[IntraQuestion]:
        """Get intra-questions for specified image."""
        if image_id == 'a':
            return self._intra_questions_a
        else:
            return self._intra_questions_b
    
    def get_inter_questions(self) -> List[InterQuestion]:
        """Get inter-comparison questions."""
        return self._inter_questions
    
    def set_subqueries(self, subqueries: List[str]) -> None:
        """Set reasoning subqueries."""
        self.subqueries = subqueries
    
    def set_problog_facts(self, facts: str) -> None:
        """Set ProbLog facts string."""
        self.problog["facts"] = facts
    
    def set_problog_queries(self, queries: str) -> None:
        """Set ProbLog queries string."""
        self.problog["queries"] = queries
    
    def set_problog_results(self, results: List[ProbLogResult]) -> None:
        """Set ProbLog execution results."""
        self.problog["results"] = results
    
    def set_answer(self, text: str, explanation: str) -> None:
        """Set final answer and explanation."""
        self.answer["text"] = text
        self.answer["explanation"] = explanation
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert KnowledgeBase to exact JSON schema from build brief.
        
        Returns:
            Dict[str, Any]: JSON representation matching build brief schema
        """
        # Convert dataclass instances to dictionaries
        def convert_objects_to_dicts(obj_list):
            if not obj_list:
                return []
            if hasattr(obj_list[0], 'to_dict'):
                return [obj.to_dict() for obj in obj_list]
            elif hasattr(obj_list[0], '__dict__'):
                return [asdict(obj) for obj in obj_list]
            else:
                return obj_list
        
        # Build exact schema from build brief
        result = {
            "question": self.question,
            "image_a": {
                "objects": convert_objects_to_dicts(self.image_a["objects"]),
                "attributes": convert_objects_to_dicts(self.image_a["attributes"]),
                "intra_relations": convert_objects_to_dicts(self.image_a["intra_relations"])
            },
            "image_b": {
                "objects": convert_objects_to_dicts(self.image_b["objects"]),
                "attributes": convert_objects_to_dicts(self.image_b["attributes"]), 
                "intra_relations": convert_objects_to_dicts(self.image_b["intra_relations"])
            },
            "inter_comparisons": convert_objects_to_dicts(self.inter_comparisons),
            "subqueries": self.subqueries,
            "problog": {
                "facts": self.problog["facts"],
                "queries": self.problog["queries"],
                "results": convert_objects_to_dicts(self.problog["results"])
            },
            "answer": {
                "text": self.answer["text"],
                "explanation": self.answer["explanation"]
            }
        }
        
        return result
    
    def to_json_string(self, indent: int = 2) -> str:
        """
        Convert KnowledgeBase to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            str: Formatted JSON string
        """
        return json.dumps(self.to_json(), indent=indent)
    
    def save_to_file(self, filepath: str, indent: int = 2) -> None:
        """
        Save KnowledgeBase to JSON file.
        
        Args:
            filepath: Path to output JSON file
            indent: JSON indentation level
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_json(), f, indent=indent)
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'KnowledgeBase':
        """
        Create KnowledgeBase from JSON data.
        
        Args:
            json_data: JSON dictionary
            
        Returns:
            KnowledgeBase: Reconstructed instance
        """
        kb = cls(json_data["question"])
        
        # Reconstruct image data (simplified - would need full type reconstruction in production)
        kb.image_a = json_data.get("image_a", {"objects": [], "attributes": [], "intra_relations": []})
        kb.image_b = json_data.get("image_b", {"objects": [], "attributes": [], "intra_relations": []})
        kb.inter_comparisons = json_data.get("inter_comparisons", [])
        kb.subqueries = json_data.get("subqueries", [])
        kb.problog = json_data.get("problog", {"facts": "", "queries": "", "results": []})
        kb.answer = json_data.get("answer", {"text": "", "explanation": ""})
        
        return kb
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'KnowledgeBase':
        """
        Load KnowledgeBase from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            KnowledgeBase: Loaded instance
        """
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        return cls.from_json(json_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the knowledge base.
        
        Returns:
            Dict[str, Any]: Summary information
        """
        return {
            "question": self.question,
            "image_a_objects": len(self.image_a["objects"]),
            "image_b_objects": len(self.image_b["objects"]),
            "image_a_attributes": len(self.image_a["attributes"]),
            "image_b_attributes": len(self.image_b["attributes"]),
            "image_a_relations": len(self.image_a["intra_relations"]),
            "image_b_relations": len(self.image_b["intra_relations"]),
            "inter_comparisons": len(self.inter_comparisons),
            "subqueries": len(self.subqueries),
            "problog_results": len(self.problog["results"]),
            "has_answer": bool(self.answer["text"])
        }


# Example usage and testing
if __name__ == "__main__":
    # Test KnowledgeBase creation and JSON schema
    kb = KnowledgeBase("What is uniquely similar about these two images?")
    
    print("Empty KB summary:", kb.get_summary())
    print("Empty KB JSON keys:", list(kb.to_json().keys()))
    
    # Test JSON schema compliance
    json_output = kb.to_json()
    required_keys = ['question', 'image_a', 'image_b', 'inter_comparisons', 'subqueries', 'problog', 'answer']
    
    for key in required_keys:
        assert key in json_output, f"Missing required key: {key}"
    
    print("✓ JSON schema validation passed")
    print("KnowledgeBase test completed successfully!")