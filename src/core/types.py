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
    counts: Dict[str, Any]  # Probabilistic count distributions {"class": {"distribution": {count: prob}}}
    scene_context: Dict[str, Any] = None  # Processing aids like captions

    def __post_init__(self):
        if self.scene_context is None:
            self.scene_context = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with explicit serialization for complex fields."""
        return {
            'objects': [obj.to_dict() for obj in self.objects],
            'attributes': {k: v.to_dict() for k, v in self.attributes.items()},
            'relationships': [rel.to_dict() for rel in self.relationships],
            'counts': dict(self.counts) if self.counts else {}
        }




@dataclass
class ProbLogFact:
    """Probabilistic logical fact for knowledge base."""
    probability: float  # 0.0 to 1.0
    predicate: str  # e.g., "attribute", "relation", "object"
    arguments: List[str]  # e.g., ["person_a_0", "muscle_mass", "high"]
    
    def to_prolog_string(self) -> str:
        """Convert to ProbLog fact string with proper quoting."""
        def quote_arg(arg: str) -> str:
            """Quote ProbLog arguments that aren't valid bare atoms or numbers."""
            # Handle None/non-string arguments defensively
            if arg is None:
                return "'none'"
            if not isinstance(arg, str):
                arg = str(arg)
            # Already quoted
            if arg.startswith("'") and arg.endswith("'"):
                return arg
            # Don't quote numeric values (integers and floats)
            try:
                float(arg)
                return arg
            except ValueError:
                pass
            # Don't quote valid bare Prolog atoms (lowercase start, only alnum + underscore)
            if arg and arg[0].islower() and arg.replace('_', '').isalnum():
                return arg
            # Quote everything else (hyphens, spaces, dots, uppercase start, etc.)
            escaped = arg.replace("'", "\\'")
            return f"'{escaped}'"

        args_str = ", ".join(quote_arg(arg) for arg in self.arguments)
        return f"{self.probability}::{self.predicate}({args_str})."
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResult:
    """Result from ProbLog execution of a question."""
    question: str  # The question answered
    probability: float  # Computed probability
    supporting_facts: List['ProbLogFact']  # ProbLog facts used
    problog_program: str  # Complete ProbLog program
    evidence_trail: List[str]  # Human-readable evidence chain

    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'probability': self.probability,
            'supporting_facts': [f.to_dict() for f in self.supporting_facts],
            'problog_program': self.problog_program,
            'evidence_trail': self.evidence_trail
        }


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


@dataclass
class ModeResult:
    """Results for a single execution mode (probabilistic or deterministic)."""
    probability: float  # Computed probability for the question
    final_answer: str  # "True" or "False"
    problog_program: str  # Complete ProbLog program for this mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            'probability': self.probability,
            'final_answer': self.final_answer,
            'problog_program': self.problog_program
        }


@dataclass
class SharedEvidence:
    """Evidence shared between probabilistic and deterministic modes."""
    question: str  # The question being answered
    evidence_collection: Any  # EvidenceCollection (avoid circular import)
    detected_objects: Dict[str, List['ObjectDetection']]  # {image_id: [objects]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'detected_objects': {
                img_id: [obj.to_dict() for obj in objs]
                for img_id, objs in self.detected_objects.items()
            }
        }


@dataclass
class UnifiedResult:
    """
    Combined result from unified pipeline execution.

    Contains shared evidence (same for both modes) and separate results
    for probabilistic and deterministic execution.
    """
    threshold: float  # Threshold used for deterministic mapping
    shared: SharedEvidence
    probabilistic: ModeResult
    deterministic: ModeResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            'threshold': self.threshold,
            'shared': self.shared.to_dict(),
            'probabilistic': self.probabilistic.to_dict(),
            'deterministic': self.deterministic.to_dict()
        }


# Type aliases for convenience
Objects = List[ObjectDetection]
Attributes = List[AttributeData]
IntraRelations = List[IntraRelation]
ProbLogFacts = List[ProbLogFact]