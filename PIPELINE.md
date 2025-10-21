# PROVE Pipeline: Complete Implementation Guide

**PROVE (Probabilistic Reasoning Over Visual Evidence)** - A subquery-driven architecture that transforms ambiguous visual questions into structured evidence extraction, probabilistic reasoning, and confident answers with complete provenance.

---

## Overview

**Core Philosophy**: Break complex comparative questions into specific binary subquestions, extract evidence using multi-modal verification, and synthesize probabilistic answers.

**Architecture**: 11-step pipeline with agentic attribute extraction, caption-based object detection, and probabilistic logic reasoning.

---

## 5-Tier Knowledge Framework

- **Objects**: Spatial entities with bounding boxes and confidence scores
- **Attributes**: Object characteristics verified through agentic binary VLM extraction
- **Relationships**: Spatial/interaction relationships between objects
- **Scene Attributes**: Environmental and contextual facts
- **Count Distributions**: Probabilistic object counts using Poisson-Binomial distributions

**Data Structure**: Clean `ImageData` hierarchy - `kb.images[image_id].{objects, attributes, relationships, scene_attributes, counts}`

---

## Models & Quantization

### Primary Models:
- **Florence-2-large**: Caption-based open vocabulary detection, image captions, region descriptions
- **Llama-3.3-70B-Instruct**: Subquery generation, entity extraction, agentic orchestration, candidate generation (8-bit quantization via BitsAndBytes)
- **Qwen-2.5-VL-7B-Instruct**: Binary verification for attributes, relationships, and scene attributes; open-ended visual Q&A for agentic information gathering

### Model Loading:
- **Device Allocation**: Auto device mapping for optimal GPU distribution
- **Memory Optimization**: 8-bit quantization for Llama-3.3-70B to fit memory constraints
- **Lazy Loading**: Models loaded on-demand via ModelManager singleton

---

## Phase-by-Phase Implementation

### Step 1: Image Context Generation
**Goal**: Capture rich scene-level information upfront for efficient reuse

**Implementation**:
- **Model**: Florence-2-large detailed captioning
- **Task**: `<MORE_DETAILED_CAPTION>`
- **Method**: `detector.generate_detailed_captions()` → `Dict[str, str]`
- **Data Flow**: Image → Comprehensive scene description
- **Example**: "The image shows a white egret perched on a black buffalo in a grassy field"

**Usage**:
- Used for entity extraction in Step 2 (object detection)
- Used for contextual reasoning in subquery generation
- Processing aid only - NOT stored in final knowledge base

**Output**: `Dict[image_id, caption_string]`

---

### Step 2: Object Extraction (Caption-Based Open Vocabulary Detection)
**Goal**: Identify all visual entities with spatial grounding using caption-based approach

**2-Step Pipeline**:

**Step 2a: Extract Entity Classes**
- **Model**: Llama-3.3-70B-Instruct with Pydantic validation
- **Method**: `llm_client.extract_entities(messages)` → `EntityExtractionResponse`
- **Input**: Pre-generated caption from Step 1
- **Pydantic Processing**:
  - Automatically lowercases all entities
  - Deduplicates using `set()`
  - Validates non-empty list
- **Example**: Caption → `["egret", "buffalo", "field"]`

**Step 2b: Open Vocabulary Detection Per Entity**
- **Model**: Florence-2-large
- **Task**: `<OPEN_VOCABULARY_DETECTION>` + text prompt
- **Method**: `detector.detect_from_caption()` → `florence2.detect_open_vocabulary(image, entity_class)` per entity
- **Data Flow**: For each entity → Bounding boxes + labels + raw scores

**Probability Calculation**:
- **Source**: Florence-2 sequence-level confidence using **geometric mean**
- **Method**: `exp(mean(log_probs))` - length-normalized likelihood (P(sequence)^(1/L))
- **Why Geometric Mean**: Standard in language modeling, more conservative than arithmetic mean, penalizes low-confidence tokens
- **Raw Extraction**: `compute_transition_scores()` with `normalize_logits=True`
- **Calibration**: Anchored sigmoid mapping transforms raw scores to operational probabilities
- **Formula**: `p' = 1 / (1 + ((1-p)/p)^a * e^(-c))`
- **Anchor Points**: `0.1 → 0.7`, `0.5 → 0.9` (hardcoded)
- **Parameters**: `a ≈ 0.6144`, `c ≈ 2.1972` (pre-computed)
- **Function**: `calibrate_detector_confidence(raw_score)` in `src/core/probability.py`
- **Range**: Transforms 0.1-0.6 raw scores → 0.7-0.95 operational probabilities

**Benefits**:
- **Efficient**: Reuses caption from Step 1
- **Comprehensive**: LLM finds ALL entities in caption
- **Open Vocabulary**: Can detect any object (not limited to pre-defined classes)
- **Attribute-Free**: Extracts base nouns only
- **Deduplicated**: Automatic deduplication prevents redundancy

**Output**: `ObjectDetection(object_id, label, bbox, confidence)` with calibrated confidence

---

### Step 3: Subquery Generation
**Goal**: Break ambiguous questions into specific binary subquestions

**Implementation**:
- **Model**: Llama-3.3-70B-Instruct with Pydantic validation
- **Method**: `subquery_generator.generate_binary_subqueries(question, images)`
- **Categories**:
  - **attribute**: Object characteristics (color, size, position, shape)
  - **relationship**: Spatial/interaction relations between objects
  - **scene_attribute**: Scene-level characteristics
  - **count**: Questions about quantity

**LLM-Driven Approach**:
- **Intelligence**: LLM handles object reference extraction and type classification
- **Validation**: Pydantic `SubqueryResponse` with field validation
- **Trust LLM**: No manual pattern matching
- **Structured Output**: System message enforces strict JSON format

**Output**: `List[BinarySubquery(question, subquery_type, referenced_objects)]`

---

### Step 4: Subquery Type Routing
**Goal**: Organize subqueries by type for specialized processing

**Routing**:
- `attribute` → AgenticAttributeProcessor
- `relationship` → RelationshipExtractor
- `count` → CountProcessor
- `scene_attribute` → SceneAttributeProcessor

---

### Step 5: Agentic Attribute Extraction (NEW)
**Goal**: Extract attributes through LLM-orchestrated iterative information gathering

**Architecture**: Agent loop with Qwen VL for visual information gathering

**Agentic Loop Flow**:
```
Initialize AgentState
    ↓
Agent Decides: Need more info?
    ├─ YES → Ask Qwen VL open-ended question
    │        ↓
    │   Store answer in state
    │        ↓
    │   Loop back (max 15 iterations)
    │
    └─ NO → Generate binary questions
            ↓
       Verify with Qwen + logits
            ↓
       Extract probabilities
```

**Key Components**:

**1. AgentState** (Conversation Memory):
- `original_question`: Attribute subquery to answer
- `referenced_objects`: Objects mentioned in subquery
- `qwen_qa_history`: List of Q&A interactions with Qwen
- `information_gathered`: Dict mapping object_id to descriptions
- `binary_questions`: Final questions for verification
- `reasoning_trace`: Agent's chain of thought

**2. Pydantic Models**:
- `QwenInformationRequest`: Agent's request for visual info
  - `object_id`: Object to query
  - `question`: Open-ended question (e.g., "What color is this dog?")
  - `reasoning`: Why agent needs this info

- `BinaryAttributeQuestion`: Final binary question for verification
  - `object_id`: Object being queried
  - `attribute_class`: Category (e.g., "color", "size")
  - `attribute_value`: Specific value (e.g., "brown")
  - `binary_question`: Yes/No question (e.g., "Is dog_a_1 brown?")

- `AgentDecision`: Agent's decision at each step
  - `action`: "ask_qwen" or "generate_binary_questions"
  - `reasoning`: Chain of thought explanation
  - `qwen_request` or `binary_questions`: Based on action

**3. Processing Flow**:

**Information Gathering Phase** (Iterative):
```python
# Agent analyzes current knowledge
decision = agent_decide_next_action(state)

if decision.action == "ask_qwen":
    # Ask Qwen VL open-ended question with bbox grounding
    answer = qwen_vl.run_inference_with_logits(image, question)
    state.add_qwen_interaction(request, answer)
    # Loop continues...
```

**Binary Question Generation**:
```python
if decision.action == "generate_binary_questions":
    # Agent has enough info, generates final questions
    state.binary_questions = decision.binary_questions
    # Exit loop, proceed to verification
```

**Verification Phase**:
```python
for bq in binary_questions:
    # Verify with Qwen using logits
    response, logits = qwen_vl.run_inference_with_logits(image, bq.binary_question)

    # Extract probability using verbalizer summing
    probability = get_verifier_probability(logits, response, tokenizer)

    # Store result
    results.append(AttributeData(
        object_id=bq.object_id,
        attribute_class=bq.attribute_class,
        value=bq.attribute_value,
        confidence=probability
    ))
```

**Example Execution**:

**Input**: "Do the dogs in image_1 have the same color as the dogs in image_2?"
**Referenced Objects**: `["dog_a_1", "dog_a_2", "dog_b_1", "dog_b_2"]`

**Iteration 1-4** (Information Gathering):
- Agent: "I need to know colors of all 4 dogs"
- Ask Qwen: "What color is dog_a_1?" → "brown with white patches"
- Ask Qwen: "What color is dog_a_2?" → "tan"
- Ask Qwen: "What color is dog_b_1?" → "brown"
- Ask Qwen: "What color is dog_b_2?" → "white with grey spots"

**Iteration 5** (Binary Question Generation):
- Agent: "I now have all colors. Generating binary questions."
- Binary Questions:
  - "Is dog_a_1 brown?"
  - "Is dog_a_2 tan?"
  - "Is dog_b_1 brown?"
  - "Is dog_b_2 white?"
  - "Is dog_a_2 brown?" (for comparison)
  - "Is dog_b_2 brown?" (for comparison)

**Verification**: Each question → Qwen with logits → P(Yes)

**Key Features**:
- ✅ **Fully General**: Works with ANY attribute category
- ✅ **Adaptive**: Agent decides how much info needed
- ✅ **Explainable**: Full reasoning trace saved
- ✅ **Probabilistic**: Extracts P(Yes) from verbalizer logits
- ✅ **Safety**: Max 15 iterations prevents infinite loops
- ✅ **Validated**: All LLM outputs use Pydantic

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification via `get_verifier_probability(logits, response, tokenizer)`
- **Process**: 2-token softmax over Yes/No verbalizers
- **Formula**: `P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
- **Error Handling**: Return 0.5 (neutral) for failed extractions
- **No Filtering**: All results preserved for ProbLog inference

**Implementation**:
- **File**: `src/pipeline/agentic_attribute_processor.py` (463 lines)
- **Method**: `process_attribute_subqueries(attribute_subqueries, image_paths, images)`

**Output**: `List[AttributeData(object_id, attribute_class, value, confidence)]`

---

### Step 6: Relationship Extraction
**Goal**: Extract spatial and interaction relationships

**Implementation**:
- **Pipeline**: Compound subquery analysis → Llama-3.3-70B relationship candidates → Qwen-2.5-VL binary verification
- **Method**: `relationship_extractor.extract_relationships(subqueries, image_paths, images)`
- **Compound Handling**: LLM analyzes subqueries for cross-image and multi-relationship requirements

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification with bounding box context
- **Process**: 2-token softmax over Yes/No verbalizers
- **Bounding Box Context**: `<box>(x1,y1),(x2,y2)</box>label` format

**Output**: `List[IntraRelation(subject_id, object_id, relation, probability)]`

---

### Step 7: Count Processing
**Goal**: Determine probabilistic counts using Poisson-Binomial distributions

**Implementation**:
- **Model**: Poisson-Binomial distribution from object detection confidences
- **Method**: `count_processor.process_count_subqueries(count_subqueries, images)`
- **Distribution**: Combines individual object existence probabilities

**Output**: Count distributions stored in knowledge base

---

### Step 8: Scene Attribute Processing
**Goal**: Extract scene-level attributes using compound subquery decomposition

**Implementation**:
- **Pipeline**: Compound subquery analysis → LLM decomposition into atomic binary questions → Qwen verification
- **Method**: `scene_attribute_processor.process_scene_attribute_subqueries(subqueries, image_paths, images)`
- **Decomposition**: Single subquery can generate multiple atomic questions for different images/attributes

**Example**:
- Input: "Do both images show outdoor settings with grass?"
- Atomic Questions:
  - "Is IMAGE_A an outdoor environment?" (environment_type=outdoor)
  - "Does IMAGE_A show grass?" (vegetation=grass)
  - "Is IMAGE_B an outdoor environment?" (environment_type=outdoor)
  - "Does IMAGE_B show grass?" (vegetation=grass)

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification per atomic question
- **Process**: 2-token softmax for each question independently

**Output**: Multiple `SceneAttributeResult` per subquery with individual confidences

---

### Step 9: ProbLog Knowledge Base Construction
**Goal**: Convert all extracted evidence into probabilistic logical facts

**Implementation**:
- **Method**: `problog_builder.build_knowledge_base(images)`
- **Fact Types**:
  - Object facts: `prob::object(id, label, image)`
  - Attribute facts: `prob::attribute(obj_id, class, value)`
  - Relation facts: `prob::relation(subj_id, obj_id, relation)`
  - Scene facts: `prob::scene_attr(image, value)`
  - Count facts: Poisson-Binomial distributions

**Probability Preservation**:
- Preserves original confidence scores from extraction phases
- No filtering - all results included (even low-probability)
- Complete evidence for probabilistic inference

**Output**: `List[ProbLogFact(probability, predicate, arguments)]`

---

### Step 10: Subquery Decomposition to ProbLog
**Goal**: Convert contextual subqueries into executable logical queries

**Implementation**:
- **Model**: LLM decomposes subquestions into ProbLog queries
- **Method**: Generate ProbLog queries over extracted facts
- **Structure**: Query logic that references probabilistic facts from KB

**Output**: ProbLog queries ready for execution

---

### Step 11: ProbLog Execution and Evidence Tracing
**Goal**: Execute probabilistic reasoning to answer subqueries

**Implementation**:
- **Engine**: ProbLog probabilistic logic programming engine
- **Method**: `problog_executor.execute_subqueries(subqueries, problog_facts)`
- **Algorithm**: Weighted model counting over probabilistic facts

**Probability Calculation**:
- **Source**: ProbLog inference engine
- **Method**: Compute marginal probability of query being true
- **Output**: Query probability + supporting evidence trail

**Output**: `List[SubqueryResult(subquery, probability, supporting_facts, evidence_trail)]`

---

### Step 12: Final Answer Generation
**Goal**: Synthesize ultimate answer using subquery results

**Implementation**:
- **Model**: Llama-3.3-70B-Instruct
- **Method**: `answer_generator.generate_final_answer(question, subquery_results, image_contexts)`
- **Synthesis**: LLM combines probabilistic evidence into coherent reasoning

**Probability Calculation**:
- **Source**: Aggregation of subquery result probabilities
- **Process**: Weight evidence based on confidence and relevance
- **Confidence Categories**: High (>0.8), Medium (0.5-0.8), Low (<0.5)

**Output**: `AnswerResult(text, explanation, confidence, supporting_evidence)`

---

## Probability Flow Architecture

```
Florence-2 Object Detection
  ├─ Geometric mean: exp(mean(log_probs))
  └─ Anchored sigmoid calibration → 0.7-0.95 range
      ↓
Agentic Attribute Extraction
  ├─ Qwen VL open-ended Q&A (information gathering)
  └─ Qwen VL binary verification (2-token softmax) → P(Yes)
      ↓
Relationship Extraction
  └─ Qwen VL binary verification + bbox context → P(relation_true)
      ↓
Scene Attribute Extraction
  └─ Qwen VL binary verification per atomic question → P(attribute_true)
      ↓
ProbLog Probabilistic Facts
  └─ Preserve ALL confidences (no filtering)
      ↓
ProbLog Inference Engine
  └─ Weighted model counting → Query marginal probabilities
      ↓
Final Answer
  └─ Evidence-weighted aggregation → Overall confidence
```

---

## Key Technical Details

### Binary Verification Strategy:
- **Method**: All verification via binary Yes/No questions
- **Verbalizer Summing**: Sum logits for ["Yes", "yes", "YES"] and ["No", "no", "NO"]
- **2-Token Softmax**: `P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
- **Avoids Inflation**: No full-vocabulary renormalization
- **Error Handling**: Return 0.5 (neutral) for failed extractions
- **No Filtering**: Preserve ALL results for complete ProbLog inference
- **Unified Function**: `get_verifier_probability()` in `src/core/probability.py`

### Confidence Calibration:
- **Object Detection**: Anchored sigmoid mapping
  - Formula: `p' = 1 / (1 + ((1-p)/p)^a * e^(-c))`
  - Transforms 0.1-0.6 → 0.7-0.95
- **Open Vocabulary Detection**: Geometric mean of log-probs
  - Formula: `exp(mean(log_probs))`
  - Length-normalized likelihood
- **Binary Verification**: 2-token softmax (no calibration needed)

### Agentic Attribute Extraction:
- **Agent Loop**: Max 15 iterations
- **Decision Points**: Ask Qwen vs Generate binary questions
- **State Tracking**: Full conversation history and reasoning trace
- **Pydantic Validation**: All agent decisions validated
- **General**: Works with ANY attribute category without hardcoding

### Memory & Performance:
- **8-bit Quantization**: Llama-3.3-70B via BitsAndBytesConfig
- **Auto Device Mapping**: Optimal GPU distribution
- **Lazy Loading**: Models loaded on-demand via singleton
- **Efficient Caption Reuse**: Step 1 captions reused throughout pipeline

---

## Model Summary Table

| Step | Task | Model | Probability Method |
|------|------|-------|-------------------|
| 1 | Image Captioning | Florence-2-large | N/A (deterministic) |
| 2a | Entity Extraction | Llama-3.3-70B | N/A (deterministic) |
| 2b | Object Detection | Florence-2-large | Geometric mean + Anchored sigmoid |
| 3 | Subquery Generation | Llama-3.3-70B | N/A (deterministic) |
| 5 | Agentic Attribute (Info Gathering) | Qwen-2.5-VL | Open-ended inference |
| 5 | Agentic Attribute (Orchestration) | Llama-3.3-70B | N/A (decision logic) |
| 5 | Agentic Attribute (Verification) | Qwen-2.5-VL | 2-token softmax |
| 6 | Relationship Extraction | Qwen-2.5-VL | 2-token softmax + bbox |
| 7 | Count Processing | Poisson-Binomial | Distribution from object confidences |
| 8 | Scene Attributes | Qwen-2.5-VL | 2-token softmax |
| 9 | KB Construction | N/A | Preserve original confidences |
| 10 | ProbLog Queries | Llama-3.3-70B | N/A (query generation) |
| 11 | ProbLog Execution | ProbLog Engine | Weighted model counting |
| 12 | Final Answer | Llama-3.3-70B | Evidence-weighted aggregation |

---

## File Structure

```
src/
├── core/
│   ├── image_utils.py          # Image loading utilities
│   ├── knowledge_base.py       # Knowledge base management
│   ├── model_manager.py        # Singleton model loading
│   ├── probability.py          # Probability functions (calibration, verifier)
│   └── types.py                # Core data types
├── language/
│   ├── llm_client.py           # Llama client with Pydantic validation
│   └── output_models.py        # Pydantic models (including agentic)
├── pipeline/
│   ├── detector.py             # Caption-based object detection
│   ├── subquery_generator.py  # Subquery generation
│   ├── agentic_attribute_processor.py  # Agentic attribute extraction (NEW)
│   ├── relationship_extractor.py  # Relationship extraction
│   ├── count_processor.py      # Count processing
│   ├── scene_attribute_processor.py  # Scene attribute extraction
│   ├── problog_builder.py      # KB → ProbLog facts
│   ├── problog_executor.py     # ProbLog inference
│   └── answer_generator.py     # Final answer synthesis
└── vision/
    ├── florence2.py            # Florence-2 wrapper
    └── qwen_vl.py              # Qwen VL wrapper
```

---

This pipeline transforms questions like "What is uniquely similar about these images?" into structured probabilistic reasoning with complete evidence provenance, agentic attribute extraction, and confidence quantification through every step.
