"""
KnowledgeBase class for PROVE pipeline.
Clean hierarchical data structure using ImageData for organization.
"""

import json
from typing import List, Dict, Any, Optional

from src.core.types import (
    ObjectDetection, AttributeData, IntraRelation, ImageData, ProbLogFact
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
        self.problog_facts: List[ProbLogFact] = []

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
                counts={}
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

    def _extract_image_id(self, full_object_id: str) -> str:
        """
        Extract image_id from full object ID.

        Args:
            full_object_id: Object ID like 'bird_a_0', 'buffalo_b_2', etc.

        Returns:
            Image ID like 'image_a', 'image_b', etc.

        Raises:
            ValueError: If object ID format is invalid
        """
        if not full_object_id or not isinstance(full_object_id, str):
            raise ValueError(f"Invalid object ID: must be non-empty string, got {type(full_object_id)}")

        try:
            parts = full_object_id.split('_')
            if len(parts) >= 2:
                image_letter = parts[-2]  # Second-to-last part is image letter (a, b, etc.)
                return f"image_{image_letter}"
            raise ValueError(f"Invalid object ID format: '{full_object_id}' (expected format: 'label_imageId_objectId')")
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Cannot parse object ID '{full_object_id}': {e}")

    def _validate_relationship(self, rel: IntraRelation) -> None:
        """
        Validate relationship has required fields.

        Args:
            rel: Relationship to validate

        Raises:
            ValueError: If relationship is invalid
        """
        if not isinstance(rel, IntraRelation):
            raise TypeError(f"Expected IntraRelation, got {type(rel)}")
        if not rel.subject_id or not isinstance(rel.subject_id, str):
            raise ValueError(f"Invalid subject_id in relationship: {rel}")
        if not rel.object_id or not isinstance(rel.object_id, str):
            raise ValueError(f"Invalid object_id in relationship: {rel}")
        if not rel.relation or not isinstance(rel.relation, str):
            raise ValueError(f"Invalid relation in relationship: {rel}")

    def add_relationship(self, relationship: IntraRelation) -> None:
        """
        Add a single relationship (convenience method).
        Automatically determines which image the relationship belongs to.

        Args:
            relationship: Single IntraRelation instance

        Raises:
            ValueError: If relationship is invalid or object ID format is wrong
        """
        self.add_relationships([relationship])

    def add_relationships(self, relationships: List[IntraRelation]) -> None:
        """
        Add multiple relationships, automatically grouping by image.
        Extracts image_id from relationship subject_id and groups accordingly.

        Args:
            relationships: List of IntraRelation instances

        Raises:
            TypeError: If relationships is not a list
            ValueError: If any relationship is invalid
        """
        # Type check
        if not isinstance(relationships, list):
            raise TypeError(f"Expected list of relationships, got {type(relationships).__name__}")

        # Empty list is OK
        if not relationships:
            return

        # Process each relationship
        for i, rel in enumerate(relationships):
            try:
                # Validate relationship
                self._validate_relationship(rel)

                # Extract image ID from subject_id
                image_id = self._extract_image_id(rel.subject_id)

                # Ensure image exists and add relationship
                self.ensure_image_exists(image_id)
                self.images[image_id].relationships.append(rel)

            except (ValueError, TypeError) as e:
                # Log warning but continue processing
                print(f"  ⚠ Warning: Skipping relationship {i}: {e}")
                continue

    def add_relationships_for_image(self, image_id: str, relationships: List[IntraRelation]) -> None:
        """
        Add extracted relationships for a specific image.

        DEPRECATED: Use add_relationships() instead for automatic grouping.
        This method is kept for backward compatibility.

        Args:
            image_id: Image identifier
            relationships: List of IntraRelation instances (using simple object indices)
        """
        self.ensure_image_exists(image_id)
        self.images[image_id].relationships.extend(relationships)

    def add_scene_context(self, image_id: str, context: Dict[str, Any]) -> None:
        """
        Add scene context (processing aids like captions) for an image.

        Args:
            image_id: Image identifier
            context: Context dictionary (e.g., {"caption": "..."})
        """
        self.ensure_image_exists(image_id)
        self.images[image_id].scene_context.update(context)

    def add_problog_facts(self, facts: List[ProbLogFact]) -> None:
        """Store ProbLog knowledge base facts."""
        self.problog_facts = facts

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert knowledge base to dictionary for JSON serialization.

        Returns:
            Dict representation of the knowledge base
        """
        return {
            "ultimate_question": self.ultimate_question,
            "images": {image_id: image_data.to_dict() for image_id, image_data in self.images.items()},
            "problog_facts": [fact.to_dict() for fact in self.problog_facts]
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