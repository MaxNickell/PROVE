# PROVE Pipeline Improvements & Enhancements

## Recent Improvements Implemented

### 1. LLM-Driven Candidate Generation System
**Status**: ✅ **COMPLETED**

**What was improved**:
- Replaced hardcoded relation lists with contextual LLM-generated candidates
- Each object pair gets custom relation candidates based on object types and ultimate question
- Candidates are generated using object-aware prompting for better relevance

**Impact**:
- More contextually relevant relationships tested
- Better alignment with comparative questions
- Reduced noise from irrelevant spatial relations

**Implementation**:
- `IntraQuestionGenerator.generate_relation_candidates()` method
- Contextual prompts considering object types and ultimate question
- Fallback to generic spatial relations if LLM generation fails

### 2. Binary VLM Verification with Structured Relations
**Status**: ✅ **COMPLETED**

**What was improved**:
- Changed from open-ended questions to binary "Is object 1 [relation] object 2?" format
- Structured IntraRelation type with explicit relation field instead of generic answer
- Clear probability mapping: Yes → 0.9, No → 0.1

**Impact**:
- More reliable VLM responses with binary questions
- Clearer relationship structure in knowledge base
- Improved confidence calibration

**Implementation**:
- Updated `IntraRelation` type with `relation` field
- Binary verification prompts in `LLaVAVerifier`
- Structured probability mapping

### 3. Individual Attribute Confidence Tracking
**Status**: ✅ **COMPLETED**

**What was improved**:
- Replaced aggregate confidence per object with individual confidence per attribute value
- Updated `AttributeData` to use `AttributeValue` objects with individual confidences
- Updated `InterComparison` to use `confidence_a` and `confidence_b` fields

**Impact**:
- Fine-grained uncertainty tracking
- Better probabilistic reasoning in downstream ProbLog
- More accurate confidence propagation

**Implementation**:
- New `AttributeValue` dataclass with value and confidence
- Updated type definitions and knowledge base handling
- Individual confidence tracking in all pipelines

### 4. Enhanced KnowledgeBase Format Handling
**Status**: ✅ **COMPLETED**

**What was improved**:
- Updated `add_intra_relations()` to handle both legacy and new relation formats
- Automatic conversion from verification results to structured types
- Backward compatibility during transition period

**Impact**:
- Smooth transition between old and new formats
- Robust handling of different result structures
- Future-proof knowledge base management

## Future Enhancement Ideas

### 1. Advanced Attribute Extraction with LLM Candidates
**Priority**: 🔥 **HIGH** - Currently in progress

**Proposed Enhancement**:
- Generate contextual attribute candidates using LLM reasoning
- Instead of generic attribute extraction, ask LLM what specific attributes to look for
- Use VLM to verify and extract values for LLM-suggested attributes

**Implementation Plan**:
```python
def generate_attribute_candidates(self, ultimate_question: str, object_label: str, 
                                 florence_description: str) -> Dict[str, List[str]]:
    # Use LLM to suggest relevant attributes and candidate values
    # Based on object type, question context, and Florence description
```

**Benefits**:
- More relevant attribute extraction
- Context-aware attribute selection
- Better alignment with comparative questions

### 2. Enhanced Inter-Question Generation for Same-Class Objects
**Priority**: 🔥 **HIGH** - Currently in progress

**Proposed Enhancement**:
- Improve inter-comparison question generation for same-class objects
- Generate more nuanced attribute comparison questions
- Use LLM to suggest discriminating attributes between similar objects

**Implementation Plan**:
- Update `InterQuestionGenerator` with same-class detection
- Enhanced prompts for fine-grained attribute differences
- Prioritize attributes that can distinguish between similar objects

### 3. Dynamic Prompt Engineering
**Priority**: 🔶 **MEDIUM**

**Proposed Enhancement**:
- Adaptive prompts based on object types and question complexity
- Different prompt strategies for different object categories
- Context-aware prompt selection

**Implementation Ideas**:
```python
class AdaptivePromptManager:
    def select_prompt_strategy(self, objects: List[ObjectDetection], 
                              question_type: str) -> str:
        # Choose optimal prompt based on context
```

### 4. Multi-Modal Evidence Fusion
**Priority**: 🔶 **MEDIUM**

**Proposed Enhancement**:
- Combine multiple VLM responses for higher confidence
- Use ensemble methods for critical decisions
- Cross-validation between different model outputs

**Implementation Ideas**:
- Multiple verification rounds for high-stakes relations
- Confidence-weighted voting between model outputs
- Uncertainty quantification improvements

### 5. Hierarchical Object Relationships
**Priority**: 🟡 **LOW**

**Proposed Enhancement**:
- Support for complex multi-object relationships
- Hierarchical spatial reasoning (inside → on top of → near)
- Compositional relationship understanding

**Implementation Ideas**:
- Extended relationship vocabulary
- Multi-step spatial reasoning
- Relationship dependency modeling

### 6. Performance Optimization
**Priority**: 🔶 **MEDIUM**

**Proposed Enhancement**:
- Batch processing for similar operations
- Caching of repeated computations
- GPU memory optimization strategies

**Implementation Ideas**:
```python
class BatchProcessor:
    def batch_verify_relations(self, image_path: str, 
                              relation_batches: List[List[tuple]]):
        # Process multiple relations in single VLM call
```

### 7. Advanced Probabilistic Reasoning
**Priority**: 🟡 **LOW**

**Proposed Enhancement**:
- Incorporate uncertainty propagation through reasoning chain
- Bayesian updating of confidences
- Evidence combination strategies

**Implementation Ideas**:
- Uncertainty-aware ProbLog facts
- Confidence interval tracking
- Multi-hypothesis reasoning

### 8. Evaluation and Benchmarking
**Priority**: 🔶 **MEDIUM**

**Proposed Enhancement**:
- Comprehensive evaluation framework
- Performance benchmarking against baselines
- Error analysis and debugging tools

**Implementation Ideas**:
```python
class PROVEEvaluator:
    def evaluate_pipeline(self, test_cases: List[TestCase]) -> EvaluationReport:
        # Comprehensive pipeline evaluation
```

### 9. Explanatory Interface
**Priority**: 🟡 **LOW**

**Proposed Enhancement**:
- Interactive explanation of reasoning steps
- Visualization of evidence and reasoning chain
- Debugging interface for pipeline inspection

### 10. Multi-Language Support
**Priority**: 🟡 **LOW**

**Proposed Enhancement**:
- Support for non-English questions and responses
- Multilingual reasoning capabilities
- Cross-linguistic evaluation

## Implementation Priority Guidelines

### High Priority (🔥)
- Directly improves core reasoning capabilities
- Addresses current limitations in candidate generation
- Essential for publication-quality results

### Medium Priority (🔶)
- Performance and robustness improvements
- Evaluation and benchmarking capabilities
- Quality of life improvements for development

### Low Priority (🟡)
- Advanced features for future research
- Nice-to-have enhancements
- Experimental capabilities

## Development Guidelines

### When Implementing Improvements:
1. **Maintain Backward Compatibility**: Ensure existing functionality still works
2. **Add Comprehensive Tests**: Every improvement should be thoroughly tested
3. **Update Documentation**: Keep plan.txt and code comments current
4. **Performance Monitoring**: Track memory usage and processing time
5. **Schema Compliance**: Maintain exact JSON schema from build brief

### Code Quality Standards:
- Type hints for all new functions
- Comprehensive error handling
- Memory-efficient implementations
- Clear, self-documenting code
- Modular design for easy testing

## Research Directions

### Potential Research Papers:
1. **"LLM-Guided Visual Reasoning"**: Focus on candidate generation approach
2. **"Binary VLM Verification"**: Analysis of binary vs. open-ended VLM questions
3. **"Probabilistic Visual Evidence"**: Uncertainty quantification in visual reasoning
4. **"Comparative Visual Question Answering"**: End-to-end pipeline evaluation

### Experimental Extensions:
- Video reasoning capabilities
- Multi-image comparative reasoning (>2 images)
- Temporal relationship understanding
- Causal relationship inference

---

*This document tracks ongoing improvements to the PROVE pipeline. Update regularly as new enhancements are implemented or proposed.*