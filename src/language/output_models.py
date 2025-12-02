from __future__ import annotations
from typing import List, Dict, Literal
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


class ObjectDiscoveryResponse(BaseModel):
    """Response for discovering relevant object IDs from natural language question."""
    object_ids: List[str] = Field(..., description="List of relevant object IDs")

    @field_validator('object_ids')
    @classmethod
    def validate_object_ids(cls, v):
        if not isinstance(v, list):
            raise ValueError("Object IDs must be a list")
        return v


class ImageDiscoveryResponse(BaseModel):
    """Response for discovering relevant image IDs from natural language question."""
    image_ids: List[str] = Field(..., description="List of relevant image IDs")

    @field_validator('image_ids')
    @classmethod
    def validate_image_ids(cls, v):
        if not isinstance(v, list):
            raise ValueError("Image IDs must be a list")
        return v


class ObjectPair(BaseModel):
    """Object pair for relationship discovery."""
    subject_id: str = Field(..., description="Subject object ID")
    object_id: str = Field(..., description="Object object ID")

    @field_validator('subject_id', 'object_id')
    @classmethod
    def validate_ids(cls, v):
        if not v.strip():
            raise ValueError("Object ID cannot be empty")
        return v.strip()


class ObjectPairDiscoveryResponse(BaseModel):
    """Response for discovering relevant object pairs from natural language question."""
    object_pairs: List[ObjectPair] = Field(..., description="List of relevant object pairs")

    @field_validator('object_pairs')
    @classmethod
    def validate_pairs(cls, v):
        if not isinstance(v, list):
            raise ValueError("Object pairs must be a list")
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


class QwenSceneInformationRequest(BaseModel):
    """Request for Qwen VL to gather visual information about an entire scene."""
    image_id: str = Field(..., description="Image ID to query (e.g., 'image_a')")
    question: str = Field(..., description="Open-ended question for Qwen about the scene (e.g., 'What type of environment is this?')")
    reasoning: str = Field(..., description="Why this information is needed")

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        if not v.endswith("?"):
            raise ValueError("Question must end with '?'")
        return v.strip()


class BinarySceneAttributeQuestion(BaseModel):
    """Binary question for scene attribute verification."""
    image_id: str = Field(..., description="Image ID being queried (e.g., 'image_a')")
    attribute_class: str = Field(..., description="Scene attribute category (e.g., 'environment_type', 'lighting', 'weather', 'vegetation')")
    attribute_value: str = Field(..., description="Specific value to verify (e.g., 'outdoor', 'bright', 'sunny', 'grass')")
    binary_question: str = Field(..., description="Binary Yes/No question using natural language (e.g., 'Is this an outdoor environment?')")

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

    @field_validator('attribute_class', 'attribute_value', 'image_id')
    @classmethod
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class SceneAgentDecision(BaseModel):
    """Agent's decision at each reasoning step in agentic scene attribute extraction."""
    action: str = Field(..., description="Action to take: 'ask_qwen' or 'generate_binary_questions'")
    reasoning: str = Field(..., description="Chain of thought reasoning for this decision")

    qwen_request: QwenSceneInformationRequest | None = Field(None, description="Qwen scene information request (if action is 'ask_qwen')")
    binary_questions: List[BinarySceneAttributeQuestion] | None = Field(None, description="Binary scene questions (if action is 'generate_binary_questions')")

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

    value: str | None = Field(
        None,
        description="Expected value (e.g., 'red', 'wooden') or None for any value"
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


class UnifiedBinaryQuestion(BaseModel):
    """
    Binary question generated by unified agent for verification.

    Used when agent is in 'verify' phase to collect probabilities.
    """

    question_type: str = Field(
        ...,
        description="Type: 'attribute', 'relationship', or 'scene'"
    )

    question_text: str = Field(
        ...,
        description="Binary question for VLM (Yes/No answerable)"
    )

    # For attribute questions
    entity_id: str | None = Field(None, description="Entity being checked")
    attribute_class: str | None = Field(None, description="Attribute category (e.g., 'color')")
    attribute_value: str | None = Field(None, description="Attribute value (e.g., 'orange')")

    # For relationship questions
    subject_id: str | None = Field(None, description="Subject entity")
    object_id: str | None = Field(None, description="Object entity")
    relation: str | None = Field(None, description="Relationship type (e.g., 'on_top_of')")

    # For scene questions
    scene_attribute: str | None = Field(None, description="Scene property")
    scene_value: str | None = Field(None, description="Scene value")

    @field_validator('question_type')
    @classmethod
    def validate_type(cls, v):
        if v not in ['attribute', 'relationship', 'scene']:
            raise ValueError("question_type must be 'attribute', 'relationship', or 'scene'")
        return v

    @field_validator('question_text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("question_text cannot be empty")
        return v.strip()


# Union type for all possible responses
OutputModel = SubquestionResponse | AttributePlanningResponse | CandidateResponse | CountRequirementResponse | EntityExtractionResponse | AgentDecision | RelationshipAgentDecision | SceneAgentDecision | UnifiedAgentDecision | UnifiedBinaryQuestion