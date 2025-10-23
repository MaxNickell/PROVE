from __future__ import annotations
from typing import List, Dict
from pydantic import BaseModel, Field, field_validator


class SubquestionItem(BaseModel):
    """Single subquestion item with question, type, and referenced objects."""
    question: str = Field(..., description="The binary question")
    referenced_objects: List[str] = Field(..., description="List of object IDs referenced in the question")
    subquery_type: str = Field(..., description="Type of subquestion: attribute, relationship, scene_attribute, or count")

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
            raise ValueError("Subquestion type must be 'attribute', 'relationship', 'scene_attribute', or 'count'")
        return v


class SubquestionResponse(BaseModel):
    """Pydantic model for subquestion generation output."""
    subquestions: List[SubquestionItem] = Field(..., description="List of binary subquestions with metadata")

    @field_validator('subquestions')
    @classmethod
    def validate_subquestions(cls, v):
        if not v:
            raise ValueError("Subquestions list cannot be empty")
        if len(v) > 50:  # Reasonable upper limit
            raise ValueError("Too many subquestions generated")
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


class SceneAttributeCandidateItem(BaseModel):
    """Single scene attribute candidate."""
    image_id: str = Field(..., description="Image identifier")
    attribute_class: str = Field(..., description="Scene attribute category")
    candidate_value: str = Field(..., description="Specific value to verify")
    binary_question: str = Field(..., description="Binary Yes/No question")

    @field_validator('image_id', 'attribute_class', 'candidate_value', 'binary_question')
    @classmethod
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class SceneAttributeResponse(BaseModel):
    """Pydantic model for scene attribute planning output."""
    scene_attribute_candidates: List[SceneAttributeCandidateItem] = Field(
        ...,
        description="List of scene attribute candidates for verification"
    )

    @field_validator('scene_attribute_candidates')
    @classmethod
    def validate_candidates(cls, v):
        if not isinstance(v, list):
            raise ValueError("Scene attribute candidates must be a list")
        if len(v) > 20:  # Reasonable upper limit
            raise ValueError("Too many scene attribute candidates")
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


class QwenInformationRequest(BaseModel):
    """Request for Qwen VL to gather visual information about an object."""
    object_id: str = Field(..., description="Object ID to query (e.g., 'dog_a_1')")
    question: str = Field(..., description="Open-ended question for Qwen (e.g., 'What color is this dog?')")
    reasoning: str = Field(..., description="Why this information is needed")

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        if not v.endswith("?"):
            raise ValueError("Question must end with '?'")
        return v.strip()


class BinaryAttributeQuestion(BaseModel):
    """Binary question for probability extraction via verification."""
    object_id: str = Field(..., description="Object ID being queried (e.g., 'dog_a_1') - used for bbox grounding")
    attribute_class: str = Field(..., description="Attribute category (e.g., 'color', 'size', 'texture')")
    attribute_value: str = Field(..., description="Specific value to verify (e.g., 'brown', 'large', 'rough')")
    binary_question: str = Field(..., description="Binary Yes/No question using natural language (e.g., 'Is the dog brown?')")

    @field_validator('binary_question')
    @classmethod
    def validate_binary(cls, v):
        if not v.strip():
            raise ValueError("Binary question cannot be empty")
        if "?" not in v:
            raise ValueError("Binary question must be a question (contain '?')")
        v_lower = v.lower().strip()
        if v_lower.startswith("what ") or v_lower.startswith("which ") or v_lower.startswith("how "):
            raise ValueError("Binary question cannot be open-ended (What/Which/How)")
        return v.strip()

    @field_validator('attribute_class', 'attribute_value')
    @classmethod
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class AgentDecision(BaseModel):
    """Agent's decision at each reasoning step in agentic attribute extraction."""
    action: str = Field(..., description="Action to take: 'ask_qwen' or 'generate_binary_questions'")
    reasoning: str = Field(..., description="Chain of thought reasoning for this decision")

    qwen_request: QwenInformationRequest | None = Field(None, description="Qwen information request (if action is 'ask_qwen')")
    binary_questions: List[BinaryAttributeQuestion] | None = Field(None, description="Binary questions (if action is 'generate_binary_questions')")

    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        if v not in ['ask_qwen', 'generate_binary_questions']:
            raise ValueError("Action must be 'ask_qwen' or 'generate_binary_questions'")
        return v

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v):
        if not v.strip() or len(v.strip()) < 10:
            raise ValueError("Reasoning must be substantive (at least 10 characters)")
        return v.strip()

    def model_post_init(self, __context):
        """Validate that action matches provided data."""
        if self.action == "ask_qwen" and self.qwen_request is None:
            raise ValueError("qwen_request required when action is 'ask_qwen'")
        if self.action == "generate_binary_questions" and self.binary_questions is None:
            raise ValueError("binary_questions required when action is 'generate_binary_questions'")
        if self.action == "generate_binary_questions" and not self.binary_questions:
            raise ValueError("binary_questions list cannot be empty when action is 'generate_binary_questions'")


class QwenRelationshipRequest(BaseModel):
    """Request for Qwen VL to describe spatial/interaction relationship between two objects."""
    subject_id: str = Field(..., description="Subject object ID (e.g., 'bird_a_0') - will be marked in RED")
    object_id: str = Field(..., description="Object ID (e.g., 'buffalo_a_1') - will be marked in BLUE")
    question: str = Field(..., description="Open-ended question about their relationship (e.g., 'Describe the spatial relationship between the bird (red) and buffalo (blue)')")
    reasoning: str = Field(..., description="Why this relationship information is needed")

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        if not v.endswith("?"):
            raise ValueError("Question must end with '?'")
        return v.strip()


class BinaryRelationshipQuestion(BaseModel):
    """Binary question for relationship verification with colored bounding boxes."""
    subject_id: str = Field(..., description="Subject object ID (e.g., 'bird_a_0') - will be marked in RED for bbox grounding")
    object_id: str = Field(..., description="Object ID (e.g., 'buffalo_a_1') - will be marked in BLUE for bbox grounding")
    relation: str = Field(..., description="Relationship type to verify (e.g., 'perched_on', 'near', 'touching', 'carrying')")
    binary_question: str = Field(..., description="Natural language Yes/No question (e.g., 'Is the bird perched on the buffalo?')")

    @field_validator('binary_question')
    @classmethod
    def validate_binary(cls, v):
        if not v.strip():
            raise ValueError("Binary question cannot be empty")
        if "?" not in v:
            raise ValueError("Binary question must be a question (contain '?')")
        v_lower = v.lower().strip()
        if v_lower.startswith("what ") or v_lower.startswith("which ") or v_lower.startswith("how "):
            raise ValueError("Binary question cannot be open-ended (What/Which/How)")
        return v.strip()

    @field_validator('relation')
    @classmethod
    def validate_relation(cls, v):
        if not v.strip():
            raise ValueError("Relation cannot be empty")
        return v.strip()


class RelationshipAgentDecision(BaseModel):
    """Agent's decision at each reasoning step in agentic relationship extraction."""
    action: str = Field(..., description="Action to take: 'ask_qwen' or 'generate_binary_questions'")
    reasoning: str = Field(..., description="Chain of thought reasoning for this decision")

    qwen_request: QwenRelationshipRequest | None = Field(None, description="Qwen relationship request (if action is 'ask_qwen')")
    binary_questions: List[BinaryRelationshipQuestion] | None = Field(None, description="Binary relationship questions (if action is 'generate_binary_questions')")

    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        if v not in ['ask_qwen', 'generate_binary_questions']:
            raise ValueError("Action must be 'ask_qwen' or 'generate_binary_questions'")
        return v

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v):
        if not v.strip() or len(v.strip()) < 10:
            raise ValueError("Reasoning must be substantive (at least 10 characters)")
        return v.strip()

    def model_post_init(self, __context):
        """Validate that action matches provided data."""
        if self.action == "ask_qwen" and self.qwen_request is None:
            raise ValueError("qwen_request required when action is 'ask_qwen'")
        if self.action == "generate_binary_questions" and self.binary_questions is None:
            raise ValueError("binary_questions required when action is 'generate_binary_questions'")
        if self.action == "generate_binary_questions" and not self.binary_questions:
            raise ValueError("binary_questions list cannot be empty when action is 'generate_binary_questions'")


# Union type for all possible responses
OutputModel = SubquestionResponse | RelationshipResponse | AttributePlanningResponse | CandidateResponse | CountRequirementResponse | SceneAttributeResponse | EntityExtractionResponse | AgentDecision | RelationshipAgentDecision