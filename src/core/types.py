"""
Type definitions for PROVE pipeline.
Defines all data structures with proper typing for schema compliance.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


@dataclass
class ObjectDetection:
    """Object detection result from Florence-2."""
    object_id: int
    label: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributeValue:
    """Individual attribute value with confidence."""
    value: str
    confidence: float = 1.0  # MVP fixed at 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributeData:
    """Attribute extraction result for an object - clean structure with no redundant references."""
    attributes: Dict[str, List[AttributeValue]]  # attribute categories with individual confidences

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with explicit serialization for nested AttributeValue objects."""
        return {
            'attributes': {
                attr_class: [
                    {'value': av.value, 'confidence': av.confidence}
                    for av in attr_values
                ]
                for attr_class, attr_values in self.attributes.items()
            }
        }


@dataclass
class IntraRelation:
    """Intra-image relationship between objects using string object identifiers."""
    subject_id: str     # Full object ID (e.g., "bird_a_0", "cattle_a_1")
    object_id: str      # Full object ID (e.g., "bird_a_0", "cattle_a_1")
    relation: str       # Specific relationship (perched_on, near, lifting, etc.)
    probability: float  # 0.9 for Yes, 0.1 for No

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageData:
    """Complete data structure for a single image - contains everything related to that image."""
    objects: List[ObjectDetection]
    attributes: Dict[int, AttributeData]  # {object_id: AttributeData}
    relationships: List[IntraRelation]
    scene_context: Dict[str, Any]  # Processing aids like captions
    scene_attributes: Dict[str, Any]  # Extracted scene-level attributes
    counts: Dict[str, Any]  # Probabilistic object counts {"class": {"count": int, "confidence": float}}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with explicit serialization for complex fields."""
        return {
            'objects': [obj.to_dict() for obj in self.objects],
            'attributes': {k: v.to_dict() for k, v in self.attributes.items()},
            'relationships': [rel.to_dict() for rel in self.relationships],
            'scene_context': dict(self.scene_context) if self.scene_context else {},
            'scene_attributes': dict(self.scene_attributes) if self.scene_attributes else {},
            'counts': dict(self.counts) if self.counts else {}
        }


@dataclass
class BinarySubquestion:
    """Binary subquestion - pure natural language, no object IDs."""
    question: str  # Binary question answerable with Yes/No
    subquestion_type: str  # "attribute", "relationship", "scene_attribute", "count"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributeRequirement:
    """Attribute extraction requirement for specific object using simple identifiers."""
    image_id: str             # "image_a", "image_b", etc.
    object_id: int            # Object index within the image (0, 1, 2...)
    attribute_classes: List[str]  # e.g., ["muscle_mass", "body_size"]
    required_for_subquestions: List[str]  # Which subquestions need these attributes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbLogFact:
    """Probabilistic logical fact for knowledge base."""
    probability: float  # 0.0 to 1.0
    predicate: str  # e.g., "attribute", "relation", "object"
    arguments: List[str]  # e.g., ["person_a_0", "muscle_mass", "high"]
    
    def to_prolog_string(self) -> str:
        """Convert to ProbLog fact string."""
        args_str = ", ".join(self.arguments)
        return f"{self.probability}::{self.predicate}({args_str})."
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubquestionResult:
    """Result from ProbLog execution of a subquestion."""
    subquestion: str  # Original binary subquestion
    probability: float  # Computed probability
    supporting_facts: List[str]  # ProbLog facts that contributed
    evidence_trail: List[str]  # Human-readable evidence chain

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbLogResult:
    """Result from ProbLog query execution."""
    query: str
    probability: float
    proof_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerResult:
    """Final answer with explanation."""
    text: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComparisonType(Enum):
    """Types of comparisons for inter-image analysis."""
    GREATER = "a>b"
    LESS = "a<b" 
    EQUAL = "a==b"
    INCOMPARABLE = "incomparable"


class AnswerConfidence(Enum):
    """Confidence levels for LLaVA answers."""
    YES = 0.9
    NO = 0.1
    UNCLEAR = 0.5


# Standard attribute categories from build brief
ATTRIBUTE_CATEGORIES = [
    "color",      # red, blue, multicolored
    "material",   # metal, wood, plastic, fabric
    "texture",    # smooth, rough, glossy, matte  
    "shape",      # round, square, curved
    "size",       # large, small, medium, tall, wide
    "state",      # open, closed, moving, stationary
    "pattern",    # striped, dotted, plaid, solid
    "style",      # modern, vintage, casual, formal
    "condition",  # new, worn, damaged, clean
    "function"    # carrying, supporting, decorative
]


# Type aliases for convenience
Objects = List[ObjectDetection]
Attributes = List[AttributeData]
IntraRelations = List[IntraRelation]
BinarySubquestions = List[BinarySubquestion]
AttributeRequirements = List[AttributeRequirement]
ProbLogFacts = List[ProbLogFact]
SubquestionResults = List[SubquestionResult]
ProbLogResults = List[ProbLogResult]