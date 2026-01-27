from __future__ import annotations
from typing import List, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator


class SubquestionResponse(BaseModel):
    """Simple list of subquestion strings - no type classification."""
    subquestions: List[str] = Field(..., description="List of binary subquestions")

    @field_validator('subquestions')
    @classmethod
    def validate_subquestions(cls, v):
        if not v:
            raise ValueError("Subquestions list cannot be empty")
        if len(v) > 50:  # Reasonable upper limit
            raise ValueError("Too many subquestions generated")
        for question in v:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Each subquestion must be a non-empty string")
        return [q.strip() for q in v]


class CountRequirementItem(BaseModel):
    """Single count requirement item."""
    image_id: str = Field(..., description="Image identifier (e.g., 'image_a')")
    object_class: str = Field(..., description="Object class to count (e.g., 'cattle', 'bird')")

    @field_validator('image_id')
    @classmethod
    def validate_image_id(cls, v):
        if not v.strip():
            raise ValueError("Image ID cannot be empty")
        return v.strip()

    @field_validator('object_class')
    @classmethod
    def validate_object_class(cls, v):
        if not v.strip():
            raise ValueError("Object class cannot be empty")
        return v.strip()


class CountRequirementResponse(BaseModel):
    """Pydantic model for count requirement analysis output."""
    count_requirements: List[CountRequirementItem] = Field(..., description="List of count requirements extracted from subquery")

    @field_validator('count_requirements')
    @classmethod
    def validate_count_requirements(cls, v):
        if not isinstance(v, list):
            raise ValueError("Count requirements must be a list")
        if len(v) > 10:  # Reasonable upper limit
            raise ValueError("Too many count requirements extracted")
        return v


class EntityExtractionResponse(BaseModel):
    """Pydantic model for entity extraction from questions."""
    entities: List[str] = Field(
        ...,
        description="List of singular noun object classes extracted from question"
    )

    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v):
        if not isinstance(v, list):
            raise ValueError("Entities must be a list")
        # Remove empty strings and strip whitespace
        v = [e.strip() for e in v if e.strip()]
        if not v:
            raise ValueError("Entities list cannot be empty after filtering")
        # Lowercase and deduplicate
        v = list(set(e.lower() for e in v))
        return v


# ==============================================================================
# Unified Agent Action Models
# ==============================================================================

class PerceiveAction(BaseModel):
    """Ask VLM an open-ended question about an entity to gather information."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["perceive"] = Field("perceive", description="Action type")
    image_id: str = Field(..., description="Image containing the entity (e.g., 'image_a')")
    entity_id: str = Field(..., description="Entity to ask about (e.g., 'dog_a_0')")
    question: str = Field(..., description="Open-ended question (e.g., 'What color is this dog?')")

    @field_validator('thought', 'image_id', 'entity_id', 'question')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('image_id')
    @classmethod
    def validate_image_id(cls, v):
        if v not in ['image_a', 'image_b']:
            raise ValueError("image_id must be 'image_a' or 'image_b'")
        return v


class VerifyAttributeAction(BaseModel):
    """Verify if an entity has a specific attribute value (binary Yes/No)."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["verify_attribute"] = Field("verify_attribute", description="Action type")
    image_id: str = Field(..., description="Image containing the entity (e.g., 'image_a')")
    entity_id: str = Field(..., description="Entity to verify (e.g., 'dog_a_0')")
    attribute: str = Field(..., description="Attribute class (e.g., 'color', 'material')")
    value: str = Field(..., description="Value to verify (e.g., 'orange', 'wooden')")

    @field_validator('thought', 'image_id', 'entity_id', 'attribute', 'value')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('image_id')
    @classmethod
    def validate_image_id(cls, v):
        if v not in ['image_a', 'image_b']:
            raise ValueError("image_id must be 'image_a' or 'image_b'")
        return v


class VerifyRelationshipAction(BaseModel):
    """Verify spatial relationship between two entities in the same image."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["verify_relationship"] = Field("verify_relationship", description="Action type")
    image_id: str = Field(..., description="Image containing both entities (e.g., 'image_a')")
    subject_id: str = Field(..., description="Subject entity (e.g., 'bird_a_0')")
    object_id: str = Field(..., description="Object entity (e.g., 'buffalo_a_1')")
    relation: str = Field(..., description="Relationship to verify (e.g., 'on_top_of', 'next_to')")

    @field_validator('thought', 'image_id', 'subject_id', 'object_id', 'relation')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('image_id')
    @classmethod
    def validate_image_id(cls, v):
        if v not in ['image_a', 'image_b']:
            raise ValueError("image_id must be 'image_a' or 'image_b'")
        return v


class VerifyCountAction(BaseModel):
    """Count objects of a specific class in an image."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["verify_count"] = Field("verify_count", description="Action type")
    image_id: str = Field(..., description="Image to count in (e.g., 'image_a')")
    object_class: str = Field(..., description="Object class to count (e.g., 'dog', 'bird')")

    @field_validator('thought', 'image_id', 'object_class')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('image_id')
    @classmethod
    def validate_image_id(cls, v):
        if v not in ['image_a', 'image_b']:
            raise ValueError("image_id must be 'image_a' or 'image_b'")
        return v


class DoneAction(BaseModel):
    """Stop evidence collection when sufficient evidence has been gathered."""

    thought: str = Field(..., description="Why evidence collection is complete")
    action: Literal["done"] = Field("done", description="Action type")

    @field_validator('thought')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# Discriminated union for agent actions
AgentAction = Union[PerceiveAction, VerifyAttributeAction, VerifyRelationshipAction, VerifyCountAction, DoneAction]