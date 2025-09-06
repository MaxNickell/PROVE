# PROVE Pipeline Implementation Documentation

## Overview

PROVE (Programmatic Reasoning Over Visual Evidence) is a sophisticated pipeline for comparative visual reasoning that combines Large Language Models (LLMs) and Vision-Language Models (VLMs) to answer complex questions about pairs of images. The pipeline is designed with a **LLM-first approach** where contextual reasoning drives candidate generation, and VLMs provide visual verification.

## Current Implementation Status

### ✅ Completed Phases (Steps 1-5)

| Phase | Component | Status | Description |
|-------|-----------|---------|-------------|
| **Step 1** | Core Infrastructure | ✅ Complete | ModelManager singleton, KnowledgeBase, Type definitions |
| **Step 2** | Object Detection & Attributes | ✅ Complete | Florence-2 detection + LLM-driven attribute extraction |
| **Step 3** | Question Generation | ✅ Complete | LLM-driven candidate generation for relations & attributes |
| **Step 4** | Image Processing | ✅ Complete | Union crops, bounding boxes, image utilities |
| **Step 5** | VLM Verification | ✅ Complete | Binary VLM verification with VLM abstraction layer |

### 🚧 Pending Phases (Steps 6-10)

| Phase | Component | Status | Description |
|-------|-----------|---------|-------------|
| **Step 6** | ProbLog Integration | ❌ Pending | Convert evidence to probabilistic facts |
| **Step 7** | Subquery Planning | ❌ Pending | Break main question into sub-questions |
| **Step 8** | Probabilistic Reasoning | ❌ Pending | Execute ProbLog inference |
| **Step 9** | Answer Generation | ❌ Pending | LLM synthesis of final answer |
| **Step 10** | Testing & Validation | ❌ Pending | End-to-end pipeline validation |

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROVE Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  Input: Ultimate Question + Image A + Image B                  │
│                                ↓                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Step 1: Core Infrastructure                │   │
│  │  ModelManager ← → Florence-2, VLM, LLM                 │   │
│  │  KnowledgeBase ← → Structured Evidence Collection      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                ↓                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │        Step 2: Visual Evidence Extraction              │   │
│  │  Object Detection (Florence-2) → Objects               │   │
│  │  LLM Candidates → VLM Verification → Attributes        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                ↓                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │       Step 3: LLM-Driven Candidate Generation          │   │
│  │  Intra-Relations: LLM → Contextual Relation Candidates │   │
│  │  Inter-Comparisons: LLM → Relevant Attribute Lists    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                ↓                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Step 4: Image Processing Utilities             │   │
│  │  Union Crops, Bounding Boxes, Blackout Masks          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                ↓                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Step 5: VLM Verification Engine              │   │
│  │  Binary VLM Questions → Specific Relations (0.9/0.1)   │   │
│  │  Attribute Extraction → Individual Confidence Values   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                ↓                               │
│  Output: Structured Evidence (JSON) → Ready for Reasoning      │
└─────────────────────────────────────────────────────────────────┘
```

### VLM Abstraction Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  VLM Abstraction Layer                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  VLMInterface   │    │      VLMRegistry            │ │
│  │  (Abstract)     │◄───┤  • register_provider()      │ │
│  │                 │    │  • get_provider()           │ │
│  │  • run_inference│    │  • create_provider()        │ │
│  │  • get_model_name│   └─────────────────────────────┘ │
│  │  • is_available │                                   │ │
│  └─────────────────┘                                   │
│         ▲                                              │
│         │ implements                                    │
│  ┌──────┴──────┬──────────────┬─────────────────┐      │
│  │             │              │                 │      │
│  │   LLaVA     │   GPT-4V     │  Claude Vision  │ ...  │
│  │             │   (Future)   │    (Future)     │      │
│  └─────────────┴──────────────┴─────────────────┘      │
└─────────────────────────────────────────────────────────┘

ModelManager.get_vlm() → Returns current VLM provider
ModelManager.set_vlm_provider(name) → Switches VLM provider
```

## Component Details

### 1. Core Infrastructure

#### ModelManager Singleton
- **Purpose**: Memory-efficient model management with single instances
- **Location**: `src/core/model_manager.py`
- **Key Features**:
  - Thread-safe singleton pattern
  - Lazy loading of models
  - VLM provider abstraction
  - Memory usage monitoring

```python
# Usage Example
manager = ModelManager()
vlm = manager.get_vlm()  # Current VLM (default: LLaVA)
manager.set_vlm_provider("claude")  # Switch to Claude Vision
florence = manager.get_florence2()  # Object detection model
llm = manager.get_llm_client()  # Language model client
```

#### KnowledgeBase Structure
- **Purpose**: Structured evidence collection with exact JSON schema
- **Location**: `src/core/knowledge_base.py`
- **Schema Compliance**: Matches build brief exactly

```json
{
  "question": "What is uniquely similar about these images?",
  "image_a": {
    "objects": [{"object_id": 0, "label": "carnivore", "bbox": [...], "confidence": 0.95}],
    "attributes": [{"object_id": 0, "attributes": {"color": [{"value": "brown", "confidence": 1.0}]}}],
    "intra_relations": [{"object_1": 0, "object_2": 1, "relation": "hunting", "probability": 0.9}]
  },
  "image_b": {...},
  "inter_comparisons": [
    {"attribute": "state", "value_a": "alive", "value_b": "dead", "confidence_a": 1.0, "confidence_b": 1.0}
  ]
}
```

#### Type Definitions
- **Purpose**: Comprehensive type safety with dataclasses
- **Location**: `src/core/types.py`
- **Key Types**: `ObjectDetection`, `AttributeData`, `AttributeValue`, `IntraRelation`, `InterComparison`

### 2. VLM Abstraction Layer

#### VLMInterface
- **Purpose**: Abstract base class for all Vision-Language Models
- **Location**: `src/core/vlm_interface.py`
- **Supported VLMs**: LLaVA (current), GPT-4V (planned), Claude Vision (planned)

#### VLM Provider System
```python
# Register new VLM providers
VLMRegistry.register_provider("gpt4v", GPT4VisionProvider)

# Easy switching between VLMs
manager.set_vlm_provider("gpt4v")
vlm = manager.get_vlm()  # Now returns GPT-4V instance
```

### 3. LLM-VLM Interaction Patterns

#### Pattern 1: Contextual Candidate Generation
```
Ultimate Question + Context → LLM Reasoning → Candidates → VLM Verification
```

**Example - Intra Relationship Candidates**:
```python
# Input
ultimate_question = "What is uniquely similar about these images?"
objects = [carnivore, zebra]

# LLM Generation
llm_prompt = f"Given '{ultimate_question}', suggest relations for carnivore-zebra"
candidates = ["hunting", "chasing", "near", "looking at"]

# VLM Verification 
for relation in candidates:
    prompt = f"Is the carnivore {relation} the zebra?"
    response = vlm.run_inference(union_crop, prompt)  # "Yes" → 0.9, "No" → 0.1
```

#### Pattern 2: Binary VLM Verification
- **Advantage**: Clear, reliable responses from VLMs
- **Mapping**: Yes → 0.9 probability, No → 0.1 probability
- **Input**: Binary questions with visual context

#### Pattern 3: Contextual Attribute Extraction
```python
# LLM suggests relevant attributes based on context
attributes = llm.generate_candidates(ultimate_question, object_label, description)
# → ["state", "condition", "size"] (not generic attributes)

# VLM verifies and extracts values
for attribute in attributes:
    prompt = f"What is the {attribute} of this {object_label}?"
    value = vlm.run_inference(object_crop, prompt)
```

### 4. Pipeline Flow Detail

#### Step 2: Enhanced Attribute Extraction
1. **Florence-2 Dense Caption**: Get detailed object description
2. **LLM Candidate Generation**: Context-aware attribute suggestions
3. **VLM Verification**: Extract specific values for candidates
4. **Individual Confidence**: Each attribute value has separate confidence

#### Step 3: Question Generation
1. **Intra-Relations**: 
   - LLM analyzes object pairs and ultimate question
   - Generates contextual relation candidates (not generic spatial relations)
   - Example: carnivore-zebra → ["hunting", "chasing"] (not "above", "below")

2. **Inter-Comparisons**:
   - LLM determines discriminating attributes for object pairs
   - Same-class objects get fine-grained discrimination attributes
   - Cross-class objects get contrasting attribute suggestions

#### Step 5: VLM Verification
1. **Image Preparation**: Union crops with colored bounding boxes
2. **Binary Questions**: "Is object 1 [relation] object 2?"
3. **Probability Mapping**: Clear confidence values (0.9/0.1, not 0.6)
4. **Structured Results**: Specific relations, not generic answers

## Key Improvements Implemented

### 1. ✅ VLM Abstraction Layer
**What**: Abstract interface for swapping between VLMs
**Impact**: Easy switching between LLaVA, GPT-4V, Claude Vision
**Usage**: 
```python
manager.set_vlm_provider("claude")  # Switch VLM
vlm = manager.get_vlm()  # Get current VLM
```

### 2. ✅ LLM-Driven Candidate Generation
**What**: LLM generates contextual candidates instead of hardcoded lists
**Impact**: Contextually relevant relations and attributes
**Before**: Generic "near, far, above, below" for all object pairs
**After**: Contextual "hunting, chasing, stalking" for carnivore-zebra

### 3. ✅ Binary VLM Verification
**What**: Binary "Yes/No" questions instead of open-ended queries
**Impact**: More reliable VLM responses with clear probabilities
**Before**: "Describe the relationship" → ambiguous parsing
**After**: "Is the carnivore hunting the zebra?" → "Yes" → 0.9 probability

### 4. ✅ Individual Confidence Tracking
**What**: Each attribute value has individual confidence instead of object-level
**Impact**: Fine-grained uncertainty quantification
**Before**: Object-level confidence aggregation
**After**: `{"color": [{"value": "brown", "confidence": 0.9}]}`

### 5. ✅ Memory Efficiency
**What**: ModelManager singleton ensures single model instances
**Impact**: Supports multiple model types without memory overflow
**Feature**: Thread-safe lazy loading with cleanup capabilities

## Testing & Validation

### Current Test Framework
- **File**: `test_pipeline.py`
- **Scope**: Steps 1-5 (Evidence Collection)
- **Features**: Real model testing, VLM abstraction validation

### Test Execution
```bash
python test_pipeline.py
```

### Expected Output
```
=== LLM-Driven Pipeline Summary ===
✓ Step 1: Core infrastructure initialized with ModelManager singleton
✓ Step 2: 2 images processed with contextual attribute extraction
✓ Step 3: LLM-generated relation & attribute candidates  
✓ Step 4: Image processing utilities ready for VLM verification
✓ Step 5: Binary VLM verification of LLM candidates (0.9/0.1 probabilities)

🎯 Results: Specific relations (eating, near, etc.) with binary confidence
🎯 Enhanced: LLM contextual reasoning drives all candidate generation
```

### Result Validation
- ✅ **Specific Relations**: "hunting", "near", "chasing" (not "Yes"/"No")
- ✅ **Binary Probabilities**: 0.9/0.1 mapping (not 0.6 hardcoded values)
- ✅ **Contextual Attributes**: Relevant to ultimate question
- ✅ **VLM Abstraction**: Easy provider switching

## Configuration & Usage

### VLM Provider Configuration
```python
# Default LLaVA provider
manager = ModelManager()
vlm = manager.get_vlm()  # Returns LLaVA instance

# Switch to different VLM (when implemented)
manager.set_vlm_provider("gpt4v")  # Switch to GPT-4V
manager.set_vlm_provider("claude")  # Switch to Claude Vision
```

### Pipeline Execution
```python
from src.pipeline.vlm_verifier import VLMVerifier
from src.pipeline.attribute_extractor import AttributeExtractor
from src.core.knowledge_base import KnowledgeBase

# Initialize components
verifier = VLMVerifier()  # Uses current VLM provider
extractor = AttributeExtractor()
kb = KnowledgeBase("What is uniquely similar?")

# Run contextual attribute extraction
attributes = extractor.extract_attributes_with_candidates(
    image_path, objects, ultimate_question
)

# Generate and verify relation candidates
relation_candidates = generator.generate_relation_candidates(question, objects)
relations = verifier.verify_intra_relations(image_path, objects, relation_candidates)
```

## Future Enhancements

### Immediate (Steps 6-10)
1. **ProbLog Integration**: Convert evidence to probabilistic facts
2. **Subquery Planning**: LLM-driven question decomposition  
3. **Probabilistic Reasoning**: Execute inference over evidence
4. **Answer Generation**: LLM synthesis with evidence citations
5. **End-to-End Validation**: Complete pipeline testing

### Extended Capabilities
1. **Multi-VLM Ensemble**: Combine multiple VLM responses
2. **Dynamic Prompt Engineering**: Adaptive prompts per VLM provider
3. **Advanced Uncertainty**: Bayesian confidence updating
4. **Real-time Inference**: Optimized model serving
5. **Evaluation Framework**: Comprehensive benchmarking

## Performance Characteristics

### Memory Efficiency
- **Singleton Pattern**: Single model instances across pipeline
- **Lazy Loading**: Models loaded only when needed
- **Memory Monitoring**: GPU usage tracking and cleanup

### Processing Speed
- **Batch Operations**: Efficient processing of multiple candidates
- **Parallel Verification**: Concurrent VLM calls where possible
- **Caching**: Florence descriptions and LLM responses cached

### Accuracy Improvements
- **Contextual Reasoning**: LLM-driven candidate generation
- **Binary Verification**: More reliable VLM responses
- **Individual Confidence**: Fine-grained uncertainty tracking

## Directory Structure

```
PROVE/
├── src/
│   ├── core/
│   │   ├── model_manager.py      # Singleton model management
│   │   ├── vlm_interface.py      # VLM abstraction layer
│   │   ├── knowledge_base.py     # Evidence collection
│   │   └── types.py              # Type definitions
│   ├── pipeline/
│   │   ├── detector.py           # Object detection (Florence-2)
│   │   ├── attribute_extractor.py # LLM-driven attributes
│   │   ├── intra_question_generator.py # Relation candidates
│   │   ├── inter_question_generator.py # Attribute candidates  
│   │   └── vlm_verifier.py       # VLM verification engine
│   ├── vision/
│   │   ├── florence2.py          # Object detection model
│   │   ├── llava.py              # VLM implementation
│   │   └── image_utils.py        # Image processing utilities
│   └── language/
│       └── llm_client.py         # Language model interface
├── test_pipeline.py              # Main testing script
├── IMPLEMENTATION.md             # This documentation
└── improvements.md               # Enhancement tracking
```

## Error Handling

### VLM-Specific Errors
- `VLMNotAvailableError`: Model not loaded or unavailable
- `VLMInferenceError`: Inference failure with specific model context
- `VLMConfigurationError`: Invalid provider or configuration

### Pipeline Robustness
- **Graceful Degradation**: Fallback to generic candidates if LLM fails
- **Error Context**: All errors include model names and specific context
- **Recovery Mechanisms**: Retry logic and alternative processing paths

---

This implementation provides a robust, extensible foundation for comparative visual reasoning with full VLM abstraction and LLM-driven contextual intelligence. The pipeline is ready for completion with Steps 6-10 and future enhancements.