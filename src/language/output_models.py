from __future__ import annotations
from typing import List, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator


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
    """Ask VLM an open-ended question about an entity or the whole image."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["perceive"] = Field("perceive", description="Action type")
    image_id: str = Field(..., description="Image to ask about (e.g., 'image_a')")
    entity_id: str | None = Field(None, description="Entity to ask about (e.g., 'dog_a_0'), or None for whole image")
    question: str = Field(..., description="Open-ended question (e.g., 'What color is this dog?')")

    @field_validator('thought', 'image_id', 'question')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('entity_id')
    @classmethod
    def validate_entity_id(cls, v):
        if v is not None and not v.strip():
            raise ValueError("entity_id cannot be empty string, use None for whole image")
        return v.strip() if v else None

    @field_validator('image_id')
    @classmethod
    def validate_image_id(cls, v):
        if v not in ['image_a', 'image_b']:
            raise ValueError("image_id must be 'image_a' or 'image_b'")
        return v


class VerifyAttributeAction(BaseModel):
    """Verify if an entity has a specific attribute (binary Yes/No)."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["verify_attribute"] = Field("verify_attribute", description="Action type")
    image_id: str = Field(..., description="Image containing the entity (e.g., 'image_a')")
    entity_id: str = Field(..., description="Entity to verify (e.g., 'dog_a_0')")
    attribute: str = Field(..., description="Attribute to verify (e.g., 'orange', 'wooden', 'showing teeth')")
    verification: str = Field(..., description="Natural language statement to verify (e.g., 'an orange dog', 'a dog showing its teeth')")

    @field_validator('thought', 'image_id', 'entity_id', 'attribute', 'verification')
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
    """Verify relationship between two entities in the same image (spatial or interaction)."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["verify_relationship"] = Field("verify_relationship", description="Action type")
    image_id: str = Field(..., description="Image containing both entities (e.g., 'image_a')")
    subject_id: str = Field(..., description="Subject entity (e.g., 'bird_a_0')")
    object_id: str = Field(..., description="Object entity (e.g., 'buffalo_a_1')")
    relation: str = Field(..., description="Relationship to verify (e.g., 'on top of', 'next to', 'sitting on', 'wearing', 'holding')")
    verification: str = Field(..., description="Natural language statement to verify (e.g., 'a bird on top of a buffalo', 'a man wearing a coat')")

    @field_validator('thought', 'image_id', 'subject_id', 'object_id', 'relation', 'verification')
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
    """Verify count-related queries about objects in images."""

    thought: str = Field(..., description="Reasoning for this action")
    action: Literal["verify_count"] = Field("verify_count", description="Action type")
    query_type: Literal[
        "at_least", "at_most", "exactly",  # Single image
        "more", "fewer", "equal",  # Cross-image comparison
        "total_exactly", "total_at_least", "total_at_most"  # Total across both
    ] = Field(..., description="Type of count query")
    object_class: str = Field(..., description="Object class to count (e.g., 'dog', 'bird')")

    # For single-image queries (at_least, at_most, exactly)
    image_id: str | None = Field(None, description="Image for single-image queries (e.g., 'image_a')")

    # For cross-image and total queries
    image_id_a: str | None = Field(None, description="First image for comparison/total queries")
    image_id_b: str | None = Field(None, description="Second image for comparison/total queries")

    # For queries with a count value (at_least, at_most, exactly, total_*)
    value: int | None = Field(None, description="Count value N for queries like 'at least N'")

    @field_validator('thought', 'object_class')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('image_id', 'image_id_a', 'image_id_b')
    @classmethod
    def validate_image_id(cls, v):
        if v is not None and v not in ['image_a', 'image_b']:
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