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
        return asdict(self)


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
    scene_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BinarySubquery:
    """Binary subquery with object references for contextual reasoning."""
    question: str  # Binary question answerable with Yes/No
    referenced_objects: List[str]  # Object IDs referenced in question
    subquery_type: str  # "attribute_comparison", "relationship", "state", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributeRequirement:
    """Attribute extraction requirement for specific object using simple identifiers."""
    image_id: str             # "image_a", "image_b", etc.
    object_id: int            # Object index within the image (0, 1, 2...)
    attribute_classes: List[str]  # e.g., ["muscle_mass", "body_size"]
    required_for_subqueries: List[str]  # Which subqueries need these attributes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipCandidate:
    """Relationship candidate for binary verification using simple object indices."""
    image_id: str             # Image containing the relationship
    subject_id: int           # Subject object index within the image
    object_id: int            # Target object index within the image
    relation: str             # e.g., "lifting"
    required_for_subqueries: List[str]  # Which subqueries need this relationship

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
class SubqueryResult:
    """Result from ProbLog execution of a subquery."""
    subquery: str  # Original binary subquery
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
BinarySubqueries = List[BinarySubquery]
AttributeRequirements = List[AttributeRequirement]
RelationshipCandidates = List[RelationshipCandidate]
ProbLogFacts = List[ProbLogFact]
SubqueryResults = List[SubqueryResult]
ProbLogResults = List[ProbLogResult]