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
- **Florence-2-large**: Caption-based open vocabulary detection, image captions
- **GPT-4o** (via Forge API): Subquery generation, entity extraction, agentic orchestration (LLM Reasoner & Planner)
- **Qwen-2.5-VL-7B-Instruct**: Binary verification (VLM Verifier) and open-ended visual Q&A (VLM Perceiver) for agentic loops

### Model Loading:
- **GPT-4o API**: OpenAI-compatible API via Forge (no local loading)
- **Device Allocation**: Auto device mapping for optimal GPU distribution (Florence-2, Qwen VL)
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
- **Model**: GPT-4o with Pydantic validation
- **Method**: `llm_client.chat_with_validation(messages, EntityExtractionResponse)`
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
- **Model**: GPT-4o with Pydantic validation
- **Method**: `subquery_generator.generate_binary_subquestions(question, images)`
- **Categories**:
  - **attribute**: Object characteristics (color, size, position, shape)
  - **relationship**: Spatial/interaction relations between objects
  - **scene_attribute**: Scene-level characteristics
  - **count**: Questions about quantity

**LLM-Driven Approach**:
- **Intelligence**: GPT-4o handles object reference extraction and type classification
- **Validation**: Pydantic `SubquestionResponse` with field validation
- **Trust LLM**: No manual pattern matching
- **Structured Output**: System message enforces strict JSON format

**Output**: `List[BinarySubquestion(question, subquery_type, referenced_objects)]`

---

### Step 4: Subquery Type Routing
**Goal**: Organize subqueries by type for specialized processing

**Routing**:
- `attribute` → AttributeAgent (agentic LLM-VLM loop)
- `relationship` → RelationshipAgent (agentic LLM-VLM loop)
- `count` → CountProcessor
- `scene_attribute` → SceneAttributeAgent (agentic LLM-VLM loop)

---

### Step 5: Agentic Attribute Extraction
**Goal**: Extract attributes through LLM-orchestrated iterative information gathering with VLM

**4-Role Architecture**:
1. **LLM as Reasoner**: Analyzes attribute subquestions and determines what information is needed
2. **LLM as Planner**: Decides whether to ask VLM for more info or generate binary questions
3. **VLM as Perceiver**: Answers open-ended visual questions to gather information
4. **VLM as Verifier**: Provides binary Yes/No answers with probability extraction

**Agentic Loop**:
```
Initialize AgentState
    ↓
LLM Reasoner: Analyze current knowledge
    ↓
LLM Planner: Need more info?
    ├─ YES → VLM Perceiver: Answer open-ended question
    │        ↓
    │   Store answer, loop back (max 15 iterations)
    │
    └─ NO → LLM generates binary questions
            ↓
       VLM Verifier: Answer binary questions with probabilities
```

**AgentState** (Conversation Memory):
- `original_question`: Attribute subquery
- `referenced_objects`: Objects mentioned
- `qwen_qa_history`: Q&A interactions with VLM
- `information_gathered`: Visual descriptions per object
- `binary_questions`: Final verification questions
- `reasoning_trace`: Agent's chain of thought

**Pydantic Models**:
- `AgentDecision`: "ask_qwen" or "generate_binary_questions" with reasoning
- `QwenInformationRequest`: Open-ended question for VLM (e.g., "What color is this dog?")
- `BinaryAttributeQuestion`: Yes/No question with object_id, attribute_class, attribute_value

**Verification with Direct Cropping** ⭐:
```python
# Crop to object with 15% margin - removes distracting context
cropped_image = crop_with_margin(image, obj.bbox, margin=0.15)

# Simple prompt (NO bbox coordinates in text!)
prompt = f"Is the buffalo large? Answer Yes or No.\n\nAnswer:"

# Extract probability via verbalizer summing
response, logits = qwen_vl.run_inference_with_logits(cropped_image, prompt)
probability = get_verifier_probability(logits, response, tokenizer)
```

**Key Features**:
- ✅ **Fully General**: Works with ANY attribute category (color, size, texture, shape, etc.)
- ✅ **Adaptive**: Agent decides how much information to gather
- ✅ **Explainable**: Full reasoning trace and Q&A history preserved
- ✅ **Direct Cropping**: Focuses VLM attention by cropping to object bbox with margin
- ✅ **No Text Coordinates**: Natural language prompts only, no confusing bbox text
- ✅ **Safety**: Max 15 iterations prevents infinite loops

**Implementation**: `src/pipeline/attribute_agent.py`

**Output**: Attributes stored in `kb.images[image_id].attributes[object_index]`

---

### Step 6: Agentic Relationship Extraction
**Goal**: Extract spatial and interaction relationships through LLM-orchestrated iterative gathering

**4-Role Architecture** (Mirrors Step 5):
1. **LLM as Reasoner**: Analyzes relationship subquestions and object pairs
2. **LLM as Planner**: Decides whether to ask VLM about relationships or generate binary questions
3. **VLM as Perceiver**: Describes spatial/interaction relationships between object pairs
4. **VLM as Verifier**: Provides binary Yes/No answers with probability extraction

**RelationshipAgentState**:
- `original_question`: Relationship subquery
- `object_pairs`: Pairs to investigate (e.g., [(bird, buffalo), (buffalo, bird)])
- `relationship_descriptions`: Visual descriptions per object pair
- `qwen_qa_history`: Q&A interactions with VLM
- `binary_questions`: Final verification questions

**Pydantic Models**:
- `RelationshipAgentDecision`: "ask_qwen" or "generate_binary_questions" with reasoning
- `QwenRelationshipRequest`: Question about object pair (subject_id marked RED, object_id marked BLUE)
- `BinaryRelationshipQuestion`: Yes/No question with subject_id, object_id, relation

**Verification with Union Crop + Colored Boxes** ⭐:
```python
# Crop to union of both objects with 15% margin
cropped_image, adj_subj_bbox, adj_obj_bbox = crop_to_union_bbox(
    image, subject_bbox, object_bbox, margin=0.15
)

# Draw thick colored boxes on CROPPED image (easier to see!)
annotated = draw_colored_boxes(cropped_image, adj_subj_bbox, adj_obj_bbox)
# RED box for subject (width=10), BLUE box for object (width=10)

# Clear prompt with color references (NO bbox coordinates in text!)
prompt = f"The bird is marked in RED and the buffalo is marked in BLUE.\n\nIs the bird perched on the buffalo?\n\nAnswer Yes or No."

# Extract probability via verbalizer summing
response, logits = qwen_vl.run_inference_with_logits(annotated, prompt)
probability = get_verifier_probability(logits, response, tokenizer)
```

**Key Features**:
- ✅ **Agentic Planning**: LLM decides which object pairs need visual investigation
- ✅ **Colored Markers**: RED/BLUE boxes provide clear visual grounding
- ✅ **Union Cropping**: Removes distracting objects while showing spatial relationship
- ✅ **Thick Lines**: 10-pixel width boxes for better visibility
- ✅ **No Text Coordinates**: Visual markers only, no confusing bbox text

**Implementation**: `src/pipeline/relationship_agent.py`

**Output**: Relationships stored in `kb.images[image_id].relationships`

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
- **Method**: All verification via binary Yes/No questions with direct cropping
- **Cropping**: Crop to object bbox (attributes) or union bbox (relationships) with 15% margin
- **Visual Grounding**: Colored boxes (relationships) or simple crops (attributes)
- **No Text Coordinates**: Natural language prompts only, no bbox parsing required
- **Verbalizer Summing**: Sum logits for ["Yes", "yes", "YES"] and ["No", "no", "NO"]
- **2-Token Softmax**: `P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
- **Error Handling**: Return 0.5 (neutral) for failed extractions
- **Unified Function**: `get_verifier_probability()` in `src/core/probability.py`

### Confidence Calibration:
- **Object Detection**: Anchored sigmoid mapping
  - Formula: `p' = 1 / (1 + ((1-p)/p)^a * e^(-c))`
  - Transforms 0.1-0.6 → 0.7-0.95
- **Open Vocabulary Detection**: Geometric mean of log-probs
  - Formula: `exp(mean(log_probs))`
  - Length-normalized likelihood
- **Binary Verification**: 2-token softmax (no calibration needed)

### Agentic Extraction (Attributes & Relationships):
- **4-Role Architecture**: LLM Reasoner → LLM Planner → VLM Perceiver → VLM Verifier
- **Agent Loop**: Max 15 iterations per subquestion
- **State Tracking**: Full Q&A history and reasoning trace
- **Pydantic Validation**: All agent decisions validated
- **Direct Cropping**: Focuses VLM attention on relevant objects
- **General**: Works with ANY attribute/relationship type without hardcoding

### Memory & Performance:
- **GPT-4o API**: Cloud-hosted, no local memory requirements
- **Auto Device Mapping**: Optimal GPU distribution (Florence-2, Qwen VL)
- **Lazy Loading**: Models loaded on-demand via singleton
- **Efficient Caption Reuse**: Step 1 captions reused throughout pipeline

---

## Model Summary Table

| Step | Task | Model | Probability Method |
|------|------|-------|-------------------|
| 1 | Image Captioning | Florence-2-large | N/A (deterministic) |
| 2a | Entity Extraction | GPT-4o | N/A (deterministic) |
| 2b | Object Detection | Florence-2-large | Geometric mean + Anchored sigmoid |
| 3 | Subquery Generation | GPT-4o | N/A (deterministic) |
| 5 | Attribute Agent (Perceiver) | Qwen-2.5-VL | Open-ended inference |
| 5 | Attribute Agent (Reasoner/Planner) | GPT-4o | N/A (decision logic) |
| 5 | Attribute Agent (Verifier) | Qwen-2.5-VL | 2-token softmax + direct crop |
| 6 | Relationship Agent (Perceiver) | Qwen-2.5-VL | Open-ended inference |
| 6 | Relationship Agent (Reasoner/Planner) | GPT-4o | N/A (decision logic) |
| 6 | Relationship Agent (Verifier) | Qwen-2.5-VL | 2-token softmax + union crop + colored boxes |
| 7 | Count Processing | Poisson-Binomial | Distribution from object confidences |
| 8 | Scene Attributes | Qwen-2.5-VL | 2-token softmax |
| 9 | KB Construction | N/A | Preserve original confidences |
| 10 | ProbLog Queries | GPT-4o | N/A (query generation) |
| 11 | ProbLog Execution | ProbLog Engine | Weighted model counting |
| 12 | Final Answer | GPT-4o | Evidence-weighted aggregation |

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
│   ├── llm_client.py           # GPT-4o client with Pydantic validation
│   └── output_models.py        # Pydantic models (agentic decisions, questions)
├── pipeline/
│   ├── detector.py             # Caption-based object detection
│   ├── subquestion_generator.py  # Subquestion generation
│   ├── attribute_agent.py      # Agentic attribute extraction (4-role architecture)
│   ├── relationship_agent.py   # Agentic relationship extraction (4-role architecture)
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
