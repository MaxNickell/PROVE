# PROVE Pipeline Architecture

**Programmatic Reasoning Over Visual Evidence**  
*Unified Binary Verification with Qwen 2.5-VL-7B*

## Current Architecture (Steps 1-5 Complete)

### Core Pattern: Florence-2 → LLM → Binary VLM

All evidence extraction follows this unified flow:

1. **Florence-2**: Dense captions of object regions
2. **LLM**: Generates contextual candidates based on visual descriptions  
3. **Qwen 2.5-VL**: Binary verification ("Is this object X?" → Yes/No + logit confidence)
4. **Output**: Single-word attributes with reliable confidence scores

### Pipeline Components

#### Step 1: Core Infrastructure ✅
- **ModelManager**: Singleton pattern for memory-efficient model loading
- **KnowledgeBase**: Structured evidence storage with JSON serialization
- **Types**: Standardized data structures (ObjectDetection, AttributeValue, etc.)

#### Step 2: Visual Evidence Extraction ✅
- **Detector**: Florence-2 object detection with confidence scores
- **AttributeExtractor**: Florence → LLM → Binary Qwen verification
  ```python
  florence_desc = "A spotted brown carnivore with visible fangs"
  candidates = llm.generate_candidates(florence_desc, "color")  # ["brown", "spotted", "tan"]
  for candidate in candidates:
      result = qwen.verify("Is this carnivore brown?")  # "Yes" (0.87 confidence)
  ```

#### Step 3: LLM Candidate Generation ✅  
- **IntraQuestionGenerator**: Contextual relationship candidates for same-image objects
- **InterQuestionGenerator**: Cross-image attribute comparison candidates
- All generators use ultimate question context for relevance

#### Step 4: Image Processing ✅
- **Native Bounding Boxes**: `<box>(x1,y1),(x2,y2)</box>` format with Qwen
- **Crop Utilities**: Florence-2 object cropping for focused analysis
- **Device Management**: CUDA-optimized tensor processing

#### Step 5: Binary VLM Verification ✅
- **QwenVerifier**: Unified binary verification across all evidence types
- **Direct Logit Extraction**: Real probabilities from model output scores
- **Confidence Calculation**: `confidence if response.startswith('yes') else (1.0 - confidence)`

### Current Output Format

The pipeline produces structured evidence with real confidence scores:

```json
{
  "objects": [{"label": "carnivore", "confidence": 0.95}],
  "attributes": [{
    "object_id": 0,
    "attributes": {
      "color": [{"value": "brown", "confidence": 0.88}],
      "pattern": [{"value": "spotted", "confidence": 0.85}]
    }
  }],
  "intra_relations": [
    {"relation": "near", "probability": 0.90},
    {"relation": "chasing", "probability": 0.17}
  ],
  "inter_comparisons": [
    {"attribute": "color", "value_a": "brown", "value_b": "brown", 
     "confidence_a": 0.88, "confidence_b": 0.89}
  ]
}
```

### Key Technical Innovations

#### 1. Unified Binary Verification
- **Before**: Mixed open-ended and binary VLM calls with text-parsed confidence
- **After**: ALL VLM calls are "Yes/No" with direct logit extraction

#### 2. Real Confidence Scores
```python
outputs = model.generate(..., output_scores=True)
confidence = extract_response_probability(outputs.scores)
```

#### 3. Native Bounding Box Integration
```python
prompt = f"Look at this object: <box>({x1},{y1}),({x2},{y2})</box>{label}\nIs this {obj} {value}?"
```

#### 4. Single-Word Extraction
- Inter-comparisons return specific values: "brown", "large", "metal"
- No verbose sentences or explanations in evidence extraction

## Next Pipeline Stages (Steps 6-10)

### Step 6: ProbLog Integration
- Convert structured evidence to probabilistic facts
- Map confidence scores to ProbLog probabilities  
- Create fact templates:
  ```prolog
  0.88::attribute(carnivore_0, color, brown).
  0.90::relation(carnivore_0, zebra_1, near).
  ```

### Step 7: Subquery Planning
- Break complex questions into smaller sub-questions
- Plan inference strategy based on available evidence
- Generate ProbLog queries for reasoning chains

### Step 8: Probabilistic Reasoning  
- Execute ProbLog inference over evidence facts
- Handle uncertainty propagation through reasoning
- Generate probability distributions over possible answers

### Step 9: Answer Generation
- LLM synthesis of final answers based on reasoning results
- Generate explanations tracing evidence → reasoning → conclusion
- Format responses with confidence levels and supporting evidence

### Step 10: Validation & Testing
- End-to-end pipeline testing with diverse question types
- Performance benchmarking and optimization
- Error analysis and robustness validation

## Potential Improvements

### Performance Optimizations
- **Batch Processing**: Group similar VLM calls for parallel execution
- **Caching**: Store Florence descriptions and LLM candidates across runs
- **Model Quantization**: Reduce memory usage for deployment scenarios

### Accuracy Enhancements
- **Multi-Round Verification**: Ask follow-up questions for low-confidence results
- **Context Expansion**: Use image-level context for object-level decisions
- **Ensemble Methods**: Combine multiple VLM responses for robust confidence

### Architecture Extensions
- **Video Support**: Extend pipeline for temporal reasoning over video sequences
- **Multi-Modal Input**: Support text+image composite questions
- **Interactive Refinement**: Allow user feedback to improve evidence extraction

### Robustness Improvements
- **Error Recovery**: Graceful fallbacks when individual components fail
- **Confidence Calibration**: Adjust confidence scores based on validation performance
- **Edge Case Handling**: Special processing for unusual objects or relationships

---

**Current Status**: Steps 1-5 complete with unified binary verification architecture. Pipeline produces reliable structured evidence ready for probabilistic reasoning phases 6-10.