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
    """Attribute extraction result for an object."""
    object_id: int
    attributes: Dict[str, List[AttributeValue]]  # 10 categories with individual confidences

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntraRelation:
    """Intra-image relationship verification result."""
    object_1: int
    object_2: int
    relation: str  # Specific relationship (eating, near, above, etc.)
    probability: float  # 0.9 for Yes, 0.1 for No

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterComparison:
    """Inter-image comparison verification result."""
    image_a_object_id: int
    image_b_object_id: int
    attribute: str  # The compared attribute (size, color, etc.)
    value_a: str  # Attribute value in image A
    value_b: str  # Attribute value in image B
    confidence_a: float  # Confidence in value_a (1.0 for MVP)
    confidence_b: float  # Confidence in value_b (1.0 for MVP)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntraQuestion:
    """Generated question for intra-image relationship."""
    object_ids: List[int]  # [subject_id, object_id]
    question: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterQuestion:
    """Generated question for inter-image comparison."""
    image_a_object_id: int
    image_b_object_id: int
    question: str

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
InterComparisons = List[InterComparison]
IntraQuestions = List[IntraQuestion]
InterQuestions = List[InterQuestion]
ProbLogResults = List[ProbLogResult]