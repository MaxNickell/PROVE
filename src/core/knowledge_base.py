"""
KnowledgeBase class for PROVE pipeline.
Clean hierarchical data structure using ImageData for organization.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from src.core.types import (
    ObjectDetection, AttributeData, IntraRelation, ImageData,
    BinarySubquery, ProbLogFact, SubqueryResult, AnswerResult
)


class KnowledgeBase:
    """
    Clean hierarchical knowledge base using ImageData structure.
    Each image contains all related data: objects, attributes, relationships, and context.
    """

    def __init__(self, ultimate_question: str):
        """
        Initialize KnowledgeBase with an ultimate question.

        Args:
            ultimate_question: The main comparative question to answer
        """
        self.ultimate_question = ultimate_question

        # Clean image-centric structure - each image contains everything
        self.images: Dict[str, ImageData] = {}  # {"image_a": ImageData, "image_b": ImageData}

        # Pipeline processing results
        self.subqueries: List[BinarySubquery] = []
        self.attribute_requirements: List[Dict[str, Any]] = []
        self.problog_facts: List[ProbLogFact] = []
        self.subquery_results: List[SubqueryResult] = []
        self.answer: Optional[AnswerResult] = None

    def ensure_image_exists(self, image_id: str) -> None:
        """
        Ensure ImageData structure exists for the given image.

        Args:
            image_id: Image identifier (e.g., "image_a")
        """
        if image_id not in self.images:
            self.images[image_id] = ImageData(
                objects=[],
                attributes={},
                relationships=[],
                scene_context={}
            )

    def add_objects(self, image_id: str, objects: List[ObjectDetection]) -> None:
        """
        Add detected objects for an image.

        Args:
            image_id: Image identifier
            objects: List of ObjectDetection instances
        """
        self.ensure_image_exists(image_id)
        self.images[image_id].objects = objects

    def add_image_context(self, image_id: str, context: str) -> None:
        """
        Add detailed image context/caption.

        Args:
            image_id: Image identifier
            context: Detailed image caption
        """
        self.ensure_image_exists(image_id)
        self.images[image_id].scene_context["caption"] = context

    def add_attributes_for_object(self, image_id: str, object_id: int, attributes: AttributeData) -> None:
        """
        Add extracted attributes for a specific object in an image.

        Args:
            image_id: Image identifier
            object_id: Object index within the image
            attributes: AttributeData instance (no object_ref needed!)
        """
        self.ensure_image_exists(image_id)
        self.images[image_id].attributes[object_id] = attributes

    def add_relationships_for_image(self, image_id: str, relationships: List[IntraRelation]) -> None:
        """
        Add extracted relationships for a specific image.

        Args:
            image_id: Image identifier
            relationships: List of IntraRelation instances (using simple object indices)
        """
        self.ensure_image_exists(image_id)
        self.images[image_id].relationships = relationships

    def add_scene_attributes(self, image_id: str, scene_attributes: List[Dict[str, Any]]) -> None:
        """
        Add scene-level attributes for an image.

        Args:
            image_id: Image identifier
            scene_attributes: List of scene attribute dictionaries with attribute_class, value, confidence
        """
        self.ensure_image_exists(image_id)
        if "scene_attributes" not in self.images[image_id].scene_context:
            self.images[image_id].scene_context["scene_attributes"] = []
        self.images[image_id].scene_context["scene_attributes"].extend(scene_attributes)

    def add_subqueries(self, subqueries: List[BinarySubquery]) -> None:
        """Store generated binary subqueries."""
        self.subqueries = subqueries

    def add_problog_facts(self, facts: List[ProbLogFact]) -> None:
        """Store ProbLog knowledge base facts."""
        self.problog_facts = facts

    def add_subquery_results(self, results: List[SubqueryResult]) -> None:
        """Store subquery execution results."""
        self.subquery_results = results

    def set_answer(self, answer: AnswerResult) -> None:
        """Set the final answer."""
        self.answer = answer


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert knowledge base to dictionary for JSON serialization.

        Returns:
            Dict representation of the knowledge base
        """
        return {
            "ultimate_question": self.ultimate_question,
            "images": {image_id: image_data.to_dict() for image_id, image_data in self.images.items()},
            "subqueries": [sq.to_dict() for sq in self.subqueries],
            "problog_facts": [fact.to_dict() for fact in self.problog_facts],
            "subquery_results": [result.to_dict() for result in self.subquery_results],
            "answer": self.answer.to_dict() if self.answer else None
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert knowledge base to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, filepath: str) -> None:
        """Save knowledge base to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


# Example usage and testing
if __name__ == "__main__":
    # Test knowledge base
    kb = KnowledgeBase("What is unique about these images?")

    # Add objects to image_a
    objects = [ObjectDetection(0, "bird", [10, 20, 30, 40], 0.9)]
    kb.add_objects("image_a", objects)

    # Add attributes for object 0
    attributes = AttributeData(attributes={"color": [{"value": "white", "confidence": 0.8}]})
    kb.add_attributes_for_object("image_a", 0, attributes)

    print("✓ Clean KnowledgeBase structure created")
    print(f"✓ Access pattern: kb.images['image_a'].objects[0] = {kb.images['image_a'].objects[0].label}")
    print(f"✓ Access pattern: kb.images['image_a'].attributes[0] = {list(kb.images['image_a'].attributes[0].attributes.keys())}")