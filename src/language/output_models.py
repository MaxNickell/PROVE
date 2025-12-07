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
    """Pydantic model for entity extraction from image captions."""
    entities: List[str] = Field(
        ...,
        description="List of singular noun object classes extracted from caption"
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
# Unified Agent Models
# ==============================================================================

class PerceiveDecision(BaseModel):
    """Decision to ask VLM an open-ended question to gather information."""

    action: Literal["perceive"] = Field(
        "perceive",
        description="Action type"
    )

    reasoning: str = Field(
        ...,
        description="Why this information is needed"
    )

    target: str = Field(
        ...,
        description="Entity ID to perceive (e.g., 'dog_a_1', 'table_b_3')"
    )

    question: str = Field(
        ...,
        description="Open-ended question for VLM (e.g., 'What color is this dog?')"
    )

    @field_validator('reasoning', 'target', 'question')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class VerifyDecision(BaseModel):
    """Decision to verify facts via binary questions and collect probabilities."""

    action: Literal["verify"] = Field(
        "verify",
        description="Action type"
    )

    reasoning: str = Field(
        ...,
        description="Why verifying these facts"
    )

    verify_type: str = Field(
        ...,
        description="Type of evidence: 'attribute', 'relationship', or 'scene'"
    )

    targets: List[str] = Field(
        ...,
        description="Entity IDs to verify (e.g., ['dog_a_0', 'dog_a_1'])"
    )

    property: str = Field(
        ...,
        description="Property to check (e.g., 'color', 'material', 'on_top_of')"
    )

    value: Union[str, int, None] = Field(
        None,
        description="Expected value (e.g., 'red', 'wooden', 3) or None for any value"
    )

    verification_question: str | None = Field(
        None,
        description="Binary Yes/No question for verification (e.g., 'Is this hat solid-colored?'). "
                    "Must be grammatically correct and answerable with Yes or No. "
                    "Leave None for count verification (system handles those)."
    )

    @field_validator('verify_type')
    @classmethod
    def validate_verify_type(cls, v):
        if v not in ['attribute', 'relationship', 'count']:
            raise ValueError("verify_type must be 'attribute', 'relationship', or 'count'")
        return v

    @field_validator('targets')
    @classmethod
    def validate_targets(cls, v):
        if not v:
            raise ValueError("targets cannot be empty")
        return v

    @field_validator('reasoning', 'property')
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str):
            return v.strip() if v.strip() else None
        # Convert other types to string
        return str(v)

    @field_validator('verification_question')
    @classmethod
    def validate_verification_question(cls, v):
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Basic validation
        if not v.endswith('?'):
            raise ValueError("verification_question must end with '?'")
        v_lower = v.lower()
        # Ensure it's binary, not open-ended
        if v_lower.startswith("what ") or v_lower.startswith("which ") or v_lower.startswith("how many "):
            raise ValueError("verification_question must be binary (Yes/No), not open-ended (What/Which/How)")
        return v


class DoneDecision(BaseModel):
    """Decision that evidence collection is complete."""

    action: Literal["done"] = Field(
        "done",
        description="Action type"
    )

    reasoning: str = Field(
        ...,
        description="Why evidence collection is complete"
    )

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v):
        if not v or not v.strip():
            raise ValueError("Reasoning cannot be empty")
        return v.strip()


# Union type for type hints
UnifiedAgentDecision = PerceiveDecision | VerifyDecision | DoneDecision


# Union type for all possible LLM output models
OutputModel = SubquestionResponse | CountRequirementResponse | EntityExtractionResponse | UnifiedAgentDecision