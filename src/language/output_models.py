from __future__ import annotations
from typing import List, Dict
from pydantic import BaseModel, Field, field_validator


class SubqueryItem(BaseModel):
    """Single subquery item with question, type, and referenced objects."""
    question: str = Field(..., description="The binary question")
    referenced_objects: List[str] = Field(..., description="List of object IDs referenced in the question")
    subquery_type: str = Field(..., description="Type of subquery: attribute, relationship, scene_attribute, or count")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()
    
    @field_validator('subquery_type')
    @classmethod
    def validate_type(cls, v):
        if v not in ['attribute', 'relationship', 'scene_attribute', 'count']:
            raise ValueError("Subquery type must be 'attribute', 'relationship', 'scene_attribute', or 'count'")
        return v


class SubqueryResponse(BaseModel):
    """Pydantic model for subquery generation output."""
    subqueries: List[SubqueryItem] = Field(..., description="List of binary subqueries with metadata")
    
    @field_validator('subqueries')
    @classmethod
    def validate_subqueries(cls, v):
        if not v:
            raise ValueError("Subqueries list cannot be empty")
        if len(v) > 50:  # Reasonable upper limit
            raise ValueError("Too many subqueries generated")
        return v


class AttributeResponse(BaseModel):
    """Pydantic model for attribute extraction output."""
    attributes: Dict[str, str] = Field(..., description="Extracted attributes as key-value pairs")
    
    @field_validator('attributes')
    @classmethod
    def validate_attributes(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Attributes must be a dictionary")
        # Ensure all values are strings
        for key, value in v.items():
            if not isinstance(value, str):
                v[key] = str(value)
        return v


class VerificationResponse(BaseModel):
    """Pydantic model for binary verification output."""
    answer: str = Field(..., description="Binary answer: yes/no")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    
    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v):
        v = v.lower().strip()
        if v not in ['yes', 'no']:
            raise ValueError("Answer must be 'yes' or 'no'")
        return v


class RelationshipItem(BaseModel):
    """Single relationship item."""
    subject_id: str = Field(..., description="Subject object ID")
    relation: str = Field(..., description="Relationship type")
    object_id: str = Field(..., description="Object object ID")


class RelationshipResponse(BaseModel):
    """Pydantic model for relationship extraction output."""
    relationships: List[RelationshipItem] = Field(..., description="List of extracted relationships")
    
    @field_validator('relationships')
    @classmethod
    def validate_relationships(cls, v):
        if len(v) > 100:  # Reasonable upper limit
            raise ValueError("Too many relationships extracted")
        return v


class ContextResponse(BaseModel):
    """Pydantic model for scene context processing output."""
    context: str = Field(..., description="Scene context description")
    key_elements: List[str] = Field(default_factory=list, description="Key contextual elements")
    
    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Context description too short")
        return v.strip()


class AttributePlanningResponse(BaseModel):
    """Pydantic model for attribute planning output."""
    attribute_requirements: Dict[str, List[str]] = Field(..., description="Object ID to required attribute classes mapping")
    
    @field_validator('attribute_requirements')
    @classmethod
    def validate_requirements(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Attribute requirements must be a dictionary")
        for obj_id, attr_classes in v.items():
            if not isinstance(attr_classes, list):
                raise ValueError(f"Attribute classes for {obj_id} must be a list")
            for attr_class in attr_classes:
                if not isinstance(attr_class, str) or not attr_class.strip():
                    raise ValueError(f"Invalid attribute class: {attr_class}")
        return v


class CandidateResponse(BaseModel):
    """Pydantic model for attribute candidate generation output."""
    candidates: List[str] = Field(..., description="List of candidate attribute values")
    
    @field_validator('candidates')
    @classmethod
    def validate_candidates(cls, v):
        if not isinstance(v, list):
            raise ValueError("Candidates must be a list")
        if not v:
            raise ValueError("Candidates list cannot be empty")
        if len(v) > 10:  # Reasonable upper limit
            raise ValueError("Too many candidates generated")
        # Ensure all values are strings
        for i, candidate in enumerate(v):
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError(f"Invalid candidate at index {i}: {candidate}")
            v[i] = candidate.strip()
        return v


# Union type for all possible responses
OutputModel = SubqueryResponse | AttributeResponse | VerificationResponse | RelationshipResponse | ContextResponse | AttributePlanningResponse | CandidateResponse