# PROVE - Probabilistic Reasoning Over Visual Evidence

**A subquery-driven architecture that transforms ambiguous visual questions into structured evidence extraction, probabilistic reasoning, and confident answers with complete provenance.**

---

## Core Philosophy

**Goal**: Break complex comparative questions into specific binary subquestions, extract evidence using multi-modal verification, and compose probabilistic answers through logical reasoning.

**Architectural Principle**:
```
Subquestions → Agents Collect Evidence → ProbLog Reasoning → Probabilities
```

**Key Insight**: The ultimate question is answered by **ProbLog composition**, not LLM synthesis. Agents collect evidence to answer subquestions, then ProbLog logically composes those answers to determine the ultimate probability.

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
- **GPT-4o** (via Forge API): Subquestion generation, entity extraction, agentic orchestration (LLM Reasoner & Planner)
- **Qwen-2.5-VL-7B-Instruct**: Binary verification (VLM Verifier) and open-ended visual Q&A (VLM Perceiver) for agentic loops

### Model Loading:
- **GPT-4o API**: OpenAI-compatible API via Forge (no local loading)
- **Device Allocation**: Auto device mapping for optimal GPU distribution (Florence-2, Qwen VL)
- **Lazy Loading**: Models loaded on-demand via ModelManager singleton

---

## Pipeline Architecture (10 Steps)

### Step 1: Image Context Generation

**Goal**: Capture rich scene-level information upfront for efficient reuse

**Intuition**: Generate comprehensive scene descriptions once, reuse throughout pipeline for object detection and contextual reasoning

**Implementation**:
- **Model**: Florence-2-large detailed captioning
- **Task**: `<MORE_DETAILED_CAPTION>`
- **Method**: `detector.generate_detailed_captions()` → `Dict[str, str]`
- **Example**: "The image shows a white egret perched on a black buffalo in a grassy field"

**Usage**:
- Used for entity extraction in Step 2 (object detection)
- Used for contextual reasoning in subquestion generation
- Processing aid only - NOT stored in final knowledge base

**Output**: `Dict[image_id, caption_string]`

---

### Step 2: Object Detection (Caption-Based Open Vocabulary)

**Goal**: Identify all visual entities with spatial grounding using caption-based approach

**Intuition**: Extract entity classes from captions, then detect each entity separately for comprehensive coverage

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
- **Method**: `exp(mean(log_probs))` - length-normalized likelihood
- **Calibration**: Anchored sigmoid mapping transforms raw scores to operational probabilities
- **Formula**: `p' = 1 / (1 + ((1-p)/p)^a * e^(-c))`
- **Anchor Points**: `0.1 → 0.7`, `0.5 → 0.9`
- **Range**: Transforms 0.1-0.6 raw scores → 0.7-0.95 operational probabilities

**Benefits**:
- Efficient: Reuses caption from Step 1
- Comprehensive: LLM finds ALL entities in caption
- Open Vocabulary: Can detect any object (not limited to pre-defined classes)
- Attribute-Free: Extracts base nouns only

**Output**: `ObjectDetection(object_id, label, bbox, confidence)` with calibrated confidence

---

### Step 3: Subquestion Generation

**Goal**: Break ambiguous questions into specific binary subquestions that compositionally answer the ultimate question

**Intuition**: Complex questions can't be answered directly - decompose into verifiable yes/no questions that ProbLog can compose

**Implementation**:
- **Model**: GPT-4o with Pydantic validation
- **Method**: `subquestion_generator.generate_binary_subquestions(question, images)`
- **Categories**:
  - **attribute**: Object characteristics (color, size, position, shape)
  - **relationship**: Spatial/interaction relations between objects
  - **scene_attribute**: Scene-level characteristics
  - **count**: Questions about quantity

**LLM-Driven Approach**:
- Intelligence: GPT-4o handles object reference extraction and type classification
- Validation: Pydantic `SubquestionResponse` with field validation
- Trust LLM: No manual pattern matching
- Structured Output: System message enforces strict JSON format

**Output**: `List[BinarySubquestion(question, subquestion_type, referenced_objects)]`

---

### Step 4: Route Subquestions by Type

**Goal**: Organize subquestions by type for specialized processing

**Intuition**: Different subquestion types require different evidence collection strategies

**Routing**:
- `attribute` → AttributeAgent (agentic LLM-VLM loop)
- `relationship` → RelationshipAgent (agentic LLM-VLM loop)
- `count` → CountProcessor (Poisson-Binomial distributions)
- `scene_attribute` → SceneAttributeAgent (agentic LLM-VLM loop)

---

### Step 5: Attribute Agent (Evidence Collection)

**Goal**: Extract attributes through LLM-orchestrated iterative information gathering with VLM

**Intuition**: Agents don't just observe - they actively verify what subquestions ask about, even testing contradictory hypotheses

**4-Role Agentic Architecture**:
1. **LLM as Reasoner**: Analyzes attribute subquestions and determines what information is needed
2. **LLM as Planner**: Decides whether to ask VLM for more info or generate binary questions
3. **VLM as Perceiver**: Answers open-ended visual questions to gather information
4. **VLM as Verifier**: Provides binary Yes/No answers with probability extraction

**Agentic Loop**:
```
Initialize AgentState (with target claims from subquestion)
    ↓
LLM Reasoner: Analyze current knowledge
    ↓
LLM Planner: Need more info?
    ├─ YES → VLM Perceiver: Answer open-ended question
    │        ↓
    │   Store answer, loop back (max 15 iterations)
    │
    └─ NO → LLM generates binary questions (MUST include target claims)
            ↓
       VLM Verifier: Answer binary questions with probabilities
```

**Subquestion-Aware Verification** ⭐:
- Extracts **target claims** from subquestion (e.g., "Is the shirt black?" → target: "black")
- Agent MUST generate binary questions that directly verify target claims
- Even if visual evidence suggests different values, verifies the hypothesis
- Ensures knowledge base has probabilities for what was ASKED, not just what was observed

**AgentState** (Conversation Memory):
- `original_question`: Attribute subquestion
- `referenced_objects`: Objects mentioned
- `target_claims`: Specific attribute values to verify (NEW)
- `qwen_qa_history`: Q&A interactions with VLM
- `information_gathered`: Visual descriptions per object
- `binary_questions`: Final verification questions
- `reasoning_trace`: Agent's chain of thought

**Verification with Direct Cropping**:
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
- ✅ **Fully General**: Works with ANY attribute category
- ✅ **Adaptive**: Agent decides how much information to gather
- ✅ **Subquestion-Aligned**: Verifies target claims, not just observations
- ✅ **Direct Cropping**: Focuses VLM attention on relevant objects
- ✅ **No Text Coordinates**: Natural language prompts only
- ✅ **Safety**: Max 15 iterations prevents infinite loops

**Implementation**: `src/pipeline/attribute_agent.py`

**Output**: Attributes stored in `kb.images[image_id].attributes[object_index]`

---

### Step 6: Relationship Agent (Evidence Collection)

**Goal**: Extract spatial and interaction relationships through LLM-orchestrated iterative gathering

**Intuition**: Relationships require understanding spatial arrangement between multiple objects - agent iteratively refines understanding

**4-Role Architecture** (Mirrors Step 5):
1. **LLM as Reasoner**: Analyzes relationship subquestions and object pairs
2. **LLM as Planner**: Decides whether to ask VLM about relationships or generate binary questions
3. **VLM as Perceiver**: Describes spatial/interaction relationships between object pairs
4. **VLM as Verifier**: Provides binary Yes/No answers with probability extraction

**Subquestion-Aware Verification**:
- Extracts **target relationship types** from subquestion
- Agent MUST verify the claimed relationship, even if evidence suggests otherwise
- Example: "Is candy on top of table?" → Verifies "on_top_of" even if candy is below

**RelationshipAgentState**:
- `original_question`: Relationship subquestion
- `object_pairs`: Pairs to investigate
- `target_claims`: Specific relationship types to verify (NEW)
- `relationship_descriptions`: Visual descriptions per object pair
- `qwen_qa_history`: Q&A interactions with VLM
- `binary_questions`: Final verification questions

**Verification with Union Crop + Colored Boxes**:
```python
# Crop to union of both objects with 15% margin
cropped_image, adj_subj_bbox, adj_obj_bbox = crop_to_union_bbox(
    image, subject_bbox, object_bbox, margin=0.15
)

# Draw thick colored boxes on CROPPED image
annotated = draw_colored_boxes(cropped_image, adj_subj_bbox, adj_obj_bbox)
# RED box for subject (width=3), BLUE box for object (width=3)

# Clear prompt with color references
prompt = f"The bird is marked in RED and the buffalo is marked in BLUE.\n\nIs the bird perched on the buffalo?\n\nAnswer Yes or No."

# Extract probability via verbalizer summing
response, logits = qwen_vl.run_inference_with_logits(annotated, prompt)
probability = get_verifier_probability(logits, response, tokenizer)
```

**Key Features**:
- ✅ **Agentic Planning**: LLM decides which object pairs need visual investigation
- ✅ **Colored Markers**: RED/BLUE boxes provide clear visual grounding
- ✅ **Union Cropping**: Removes distracting objects while showing spatial relationship
- ✅ **Subquestion-Aligned**: Verifies target relationship claims directly
- ✅ **No Text Coordinates**: Visual markers only

**Implementation**: `src/pipeline/relationship_agent.py`

**Output**: Relationships stored in `kb.images[image_id].relationships`

---

### Step 7: Count Processor (Evidence Collection)

**Goal**: Determine probabilistic counts using Poisson-Binomial distributions

**Intuition**: Object detection gives existence probabilities - combine them to get count distributions

**Implementation**:
- **Model**: Poisson-Binomial distribution from object detection confidences
- **Method**: `count_processor.process_count_subquestions(count_subquestions, images)`
- **Distribution**: Combines individual object existence probabilities

**Output**: Count distributions stored in knowledge base

---

### Step 8: Scene Attribute Agent (Evidence Collection)

**Goal**: Extract scene-level attributes using agentic approach

**Intuition**: Scene attributes (lighting, weather, environment) require analyzing full images, not individual objects

**Implementation**:
- **4-Role Architecture**: Same as attribute/relationship agents
- **Difference**: Works with FULL IMAGES (no bounding boxes)
- **Subquestion-Aware**: Verifies target scene attribute claims
- **Method**: `scene_attribute_agent.process_scene_attribute_subquestions()`

**Example**:
- Input: "Is the sky purple?"
- Target Claim: "purple"
- Visual Evidence: "The sky is blue"
- **Generated Questions**:
  - ✓ "Is the sky purple?" (verifies target claim - REQUIRED!)
  - ✓ "Is the sky blue?" (confirms observation - OPTIONAL)

**Output**: Multiple `SceneAttributeResult` with individual confidences

---

### Step 9: ProbLog Knowledge Base Construction

**Goal**: Convert all extracted evidence into probabilistic logical facts

**Intuition**: Transform evidence from agents into a format that ProbLog can reason over

**Implementation**:
- **Method**: `problog_builder.build_knowledge_base(images)`
- **Fact Types**:
  - Object facts: `prob::entity(image, obj_id, label, x1, y1, x2, y2)`
  - Attribute facts: `prob::attribute(image, obj_id, attr_class, value)`
  - Relation facts: `prob::relation(image, subj_id, obj_id, relation)`
  - Scene facts: `prob::scene_attr(image, value)`
  - Count facts: Poisson-Binomial distributions

**Probability Preservation**:
- Preserves original confidence scores from extraction phases
- No filtering - all results included (even low-probability)
- Complete evidence for probabilistic inference

**Output**: `List[ProbLogFact(probability, predicate, arguments)]`

---

### Step 10: ProbLog Execution with Ultimate Composition

**Goal**: Execute probabilistic reasoning to answer subquestions AND compose them to answer the ultimate question

**Intuition**: This is where the magic happens - ProbLog mathematically composes subquestion answers to determine the ultimate probability

**Implementation**:
- **Engine**: ProbLog probabilistic logic programming engine
- **Method**: `problog_executor.execute_subquestions(subquestions, facts, ultimate_question)`
- **Algorithm**: Weighted model counting over probabilistic facts

**Two-Phase Execution**:

**Phase 1: Generate Rules for Subquestions**
```python
# For each subquestion, LLM generates ProbLog rules
# Example: "Is there a dog wearing a collar in image A?"

dog_wearing_collar_a :-
    is_category(image_a, D, dog),
    is_category(image_a, C, collar),
    has_relationship(image_a, D, C, wearing).

query(dog_wearing_collar_a).
```

**Phase 2: Generate Ultimate Composition Rule** ⭐ (NEW)
```python
# LLM analyzes ultimate question and composes subquestions
# Example: "Are there more birds in A than B AND are all birds orange?"

ultimate_answer :-
    more_birds_in_a_than_b,    # From count subquestion
    all_birds_orange_in_a,     # From attribute subquestions
    all_birds_orange_in_b.

query(ultimate_answer).  # This answers the ULTIMATE question!
```

**Execution Flow**:
1. Generate rules for each subquestion
2. Generate ultimate composition rule
3. Build unified ProbLog program (facts + sugar + all rules + all queries)
4. Execute once
5. Extract subquestion probabilities
6. Extract **ultimate probability** from composition query

**Why This Works**:
- Subquestions decompose the ultimate question
- Agents collect evidence to answer each subquestion
- ProbLog verifies each subquestion against evidence
- ProbLog **logically composes** subquestion results
- Ultimate probability computed mathematically, not guessed by LLM

**Output**:
- `List[SubquestionResult]`: Probability for each subquestion
- `float`: **Ultimate probability** from composition

---

## Architectural Alignment

### ❌ What Was Removed

**Old Step 11: Answer Generator**
- Used LLM to "synthesize" final answers
- Re-analyzed subquestion probabilities with narrative generation
- Added subjective confidence levels ("high", "medium", "low")
- **Problem**: Redundant - ProbLog already computed the answer!

### ✅ Why ProbLog Composes the Answer

**The Problem with LLM Synthesis**:
```
ProbLog: P(ultimate_question) = 0.782  [computed mathematically]
    ↓
LLM: "Based on the evidence, Person A appears more powerful (high confidence)"
    ↑
  [guessing based on subquestion probabilities]
```

**The Aligned Approach**:
```
Subquestions → Agents → Evidence → ProbLog Facts
    ↓
ProbLog Rules (per subquestion)
    ↓
ProbLog Composition Rule (ultimate question)
    ↓
P(ultimate_question) = 0.782  [mathematically composed]
```

### ✅ Benefits of Alignment

1. **Mathematical Purity**: Probabilities computed through logical reasoning, not LLM guessing
2. **Compositional Reasoning**: Ultimate question broken into subquestions, composed logically
3. **Complete Provenance**: Full ProbLog program shows exact reasoning chain
4. **No Black Box**: Every probability traceable to evidence
5. **Alignment with Goal**: Exactly matches "subquestions → evidence → ProbLog → probabilities"

---

## Output Format

### PipelineResult Structure

```python
@dataclass
class PipelineResult:
    ultimate_question: str           # Original question
    ultimate_probability: float      # From ProbLog composition (0.0 to 1.0)
    subquestion_results: List[SubquestionResult]  # Evidence trail
    problog_program: str            # Full program for debugging
```

### Example Output

```
================================================================================
PIPELINE RESULT
================================================================================

🎯 ULTIMATE QUESTION:
   Are there more birds in image A than image B and are all birds orange?

📊 PROBABILITY: 0.7820

📋 SUBQUESTION EVIDENCE (6 total):
   1. Are there more birds in image A than image B?
      → 0.8910
   2. Are all birds in image A orange?
      → 0.9230
   3. Are all birds in image B orange?
      → 0.8540
   4. Is bird_a_0 orange?
      → 0.9150
   5. Is bird_a_1 orange?
      → 0.9310
   6. Is bird_b_0 orange?
      → 0.8540

📄 PROBLOG PROGRAM: knowledge_base.pl
   15234 characters

📊 EVIDENCE COLLECTED:
   Objects: 8
   Attributes: 12
   Relationships: 3
   Scene Attributes: 4
   Count Distributions: 2

================================================================================
✓ PIPELINE COMPLETE
================================================================================
```

### Interpreting Results

- **Ultimate Probability**: Direct answer to your question (0-1 scale)
  - > 0.8: Very likely
  - 0.6-0.8: Likely
  - 0.4-0.6: Uncertain
  - 0.2-0.4: Unlikely
  - < 0.2: Very unlikely

- **Subquestion Evidence**: Shows which sub-answers contributed to ultimate probability

- **ProbLog Program**: Inspect `knowledge_base.pl` to see:
  - All facts extracted from images
  - Rules generated for each subquestion
  - Ultimate composition rule
  - Queries executed

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
  └─ Qwen VL binary verification + colored boxes → P(relation_true)
      ↓
Scene Attribute Extraction
  └─ Qwen VL binary verification on full image → P(attribute_true)
      ↓
ProbLog Probabilistic Facts
  └─ Preserve ALL confidences (no filtering)
      ↓
ProbLog Inference Engine
  ├─ Query each subquestion → subquestion probabilities
  └─ Query ultimate composition → ULTIMATE PROBABILITY
```

---

## Key Technical Details

### Binary Verification Strategy:
- **Method**: All verification via binary Yes/No questions with direct cropping
- **Cropping**: Crop to object bbox (attributes) or union bbox (relationships) with 15% margin
- **Visual Grounding**: Colored boxes (relationships) or simple crops (attributes)
- **No Text Coordinates**: Natural language prompts only
- **Verbalizer Summing**: Sum logits for ["Yes", "yes", "YES"] and ["No", "no", "NO"]
- **2-Token Softmax**: `P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
- **Error Handling**: Return 0.5 (neutral) for failed extractions
- **Unified Function**: `get_verifier_probability()` in `src/core/probability.py`

### Confidence Calibration:
- **Object Detection**: Anchored sigmoid mapping
  - Transforms 0.1-0.6 → 0.7-0.95
- **Open Vocabulary Detection**: Geometric mean of log-probs
- **Binary Verification**: 2-token softmax (no calibration needed)

### Agentic Extraction (Attributes & Relationships):
- **4-Role Architecture**: LLM Reasoner → LLM Planner → VLM Perceiver → VLM Verifier
- **Agent Loop**: Max 15 iterations per subquestion
- **State Tracking**: Full Q&A history and reasoning trace
- **Pydantic Validation**: All agent decisions validated
- **Direct Cropping**: Focuses VLM attention on relevant objects
- **General**: Works with ANY attribute/relationship type without hardcoding
- **Subquestion-Aware**: Verifies target claims, not just observations

---

## File Structure

```
src/
├── core/
│   ├── image_utils.py          # Image loading utilities
│   ├── knowledge_base.py       # Knowledge base management
│   ├── model_manager.py        # Singleton model loading
│   ├── probability.py          # Probability functions (calibration, verifier)
│   └── types.py                # Core data types (includes PipelineResult)
├── language/
│   ├── llm_client.py           # GPT-4o client with Pydantic validation
│   └── output_models.py        # Pydantic models (agentic decisions, questions)
├── pipeline/
│   ├── detector.py             # Caption-based object detection
│   ├── subquestion_generator.py  # Subquestion generation
│   ├── attribute_agent.py      # Agentic attribute extraction
│   ├── relationship_agent.py   # Agentic relationship extraction
│   ├── count_processor.py      # Count processing
│   ├── scene_attribute_agent.py  # Scene attribute extraction
│   ├── problog_builder.py      # KB → ProbLog facts
│   └── problog_executor.py     # ProbLog inference + ultimate composition
└── vision/
    ├── florence2.py            # Florence-2 wrapper
    └── qwen_vl.py              # Qwen VL wrapper

test_pipeline.py                # Main pipeline script (10 steps, aligned)
```

---

## Usage

### Run Pipeline

```bash
python test_pipeline.py
```

### Access Results

```python
# Result structure
result = PipelineResult(
    ultimate_question="Are there more birds...",
    ultimate_probability=0.782,
    subquestion_results=[...],
    problog_program="..."
)

# Ultimate answer
print(f"Probability: {result.ultimate_probability:.4f}")

# Evidence trail
for sq in result.subquestion_results:
    print(f"{sq.subquestion} → {sq.probability:.4f}")

# Inspect reasoning
with open('knowledge_base.pl', 'r') as f:
    print(f.read())
```

---

## Model Summary

| Step | Task | Model | Probability Method |
|------|------|-------|-------------------|
| 1 | Image Captioning | Florence-2-large | N/A (deterministic) |
| 2a | Entity Extraction | GPT-4o | N/A (deterministic) |
| 2b | Object Detection | Florence-2-large | Geometric mean + Anchored sigmoid |
| 3 | Subquestion Generation | GPT-4o | N/A (deterministic) |
| 5 | Attribute Agent (Perceiver) | Qwen-2.5-VL | Open-ended inference |
| 5 | Attribute Agent (Reasoner/Planner) | GPT-4o | N/A (decision logic) |
| 5 | Attribute Agent (Verifier) | Qwen-2.5-VL | 2-token softmax + crop |
| 6 | Relationship Agent (Perceiver) | Qwen-2.5-VL | Open-ended inference |
| 6 | Relationship Agent (Reasoner/Planner) | GPT-4o | N/A (decision logic) |
| 6 | Relationship Agent (Verifier) | Qwen-2.5-VL | 2-token softmax + colored boxes |
| 7 | Count Processing | Poisson-Binomial | Distribution from confidences |
| 8 | Scene Attributes | Qwen-2.5-VL | 2-token softmax + full image |
| 9 | KB Construction | N/A | Preserve original confidences |
| 10 | ProbLog Queries | GPT-4o | N/A (query generation) |
| 10 | **Ultimate Composition** | GPT-4o | N/A (composition rule) |
| 10 | ProbLog Execution | ProbLog Engine | Weighted model counting |

---

## Summary

This pipeline transforms questions like "Are there more birds in A than B and are all birds orange?" into:

1. **Decomposition**: Binary subquestions that break down the ultimate question
2. **Evidence Collection**: Agents iteratively gather visual evidence through VLM interactions
3. **Knowledge Base**: Probabilistic facts encoding all extracted evidence
4. **Logical Reasoning**: ProbLog rules for each subquestion + composition rule for ultimate question
5. **Probabilistic Answer**: Ultimate probability computed through mathematical composition

**Key Principle**: The ultimate answer comes from **ProbLog composition**, not LLM guessing. Agents collect evidence, ProbLog reasons logically, probabilities flow mathematically from evidence to conclusion.
