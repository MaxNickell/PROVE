# PROVE: Probabilistic Reasoning Over Visual Evidence

Neuro-symbolic visual question answering using subquestion decomposition, agentic evidence collection, and probabilistic logic programming.

---

## Installation

### Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- AWS Bedrock access for Llama 3.3 70B

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/PROVE.git
cd PROVE

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials for Bedrock
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

---

## Quick Start

```python
from src import PROVE

# Initialize model (probabilistic mode by default)
model = PROVE(mode="probabilistic")

# Run inference
answer = model.predict(
    "image_a.jpg",
    "image_b.jpg",
    "Is there a white bird on top of another animal in both images?"
)
print(answer)  # "True" or "False"
```

---

## Core Architecture

```
Question → Subquestions → Agent (Perceive/Verify) → ProbLog → LLM → True/False
              ↓              ↓ (investigation)        ↓          ↓
        Binary Qs      Probabilities (logits)    Per-subQ   Binary
```

**Key Principle**: Break complex questions into binary subquestions, collect visual evidence through agentic VLM interaction (investigation + verification), compose results through probabilistic logic, and synthesize binary answer via LLM.

---

## Pipeline (5 Steps)

### Step 1: Image Captioning

**Purpose**: Generate detailed scene descriptions for downstream processing

**Model**: Florence-2-large (`<MORE_DETAILED_CAPTION>`)

**Output**: Caption per image stored in knowledge base

**Example**: `"The image shows a black buffalo with a white egret perched on its back in a grassy field"`

---

### Step 2: Object Detection

**Purpose**: Detect only entities mentioned in the ultimate question (eliminates noise + synonym mismatches)

**Process**:
1. **Entity Extraction**: Llama 3.3 70B extracts nouns from **ultimate question** → `["bird", "buffalo"]`
   - Uses question-specific prompt with 5 diverse in-context examples from dev set
   - Avoids synonyms: question says "mitten" → detects "mitten" (not "glove")
2. **Open Vocabulary Detection**: Florence-2 detects each entity → bounding boxes + confidences
3. **Calibration**: Anchored sigmoid transforms raw scores (0.1-0.6) → operational probabilities (0.7-0.95)

**Key Innovation**: Uses question instead of caption → only relevant objects detected

**Output**: `ObjectDetection(object_id, label, bbox, confidence)` per entity

---

### Step 3: Subquestion Generation

**Purpose**: Decompose complex question into verifiable binary subquestions

**Model**: Llama 3.3 70B with structured output

**Input**: Ultimate question + captions + detected object lists

**Example**:
```
Question: "Is there a white bird on top of another animal in both images?"

Subquestions:
1. "In image A, is there a white bird on top of another animal?"
2. "In image B, is there a white bird on top of another animal?"
```

**Output**: `List[BinarySubquestion(question)]` (pure natural language)

---

### Step 4: Unified Evidence Collection

**Purpose**: Collect probabilistic evidence through agentic VLM reasoning

**Architecture**: ReAct agent loop (max 20 iterations per subquestion)

**Agent Actions**:
- **Perceive**: Ask VLM open-ended investigation questions (e.g., "What color is this hat?")
  - Stores response in QA history for context
  - NO probability extraction
- **Verify**: Generate binary Yes/No verification questions (e.g., "Is this hat solid-colored?")
  - LLM generates grammatically correct questions (no f-string templates)
  - Always adds "Answer Yes or No" instruction
  - Extracts probability from Yes/No token logits
- **Done**: Evidence collection complete

**Agent Decision Flow**:
```
┌─────────────────────────────────────────┐
│ LLM Reasoner & Planner                  │
│ Analyzes: question, candidates,         │
│ evidence so far, what's missing         │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────┐      ┌──────────┐      ┌──────┐
│ Perceive │      │  Verify  │      │ Done │
│ (gather) │      │ (check)  │      └──────┘
└──────────┘      └──────────┘
      │                 │
      └────────┬────────┘
               ▼
         Update State
               │
         ┌─────┴─────┐
         │ Continue? │
         └───────────┘
```

**Evidence Types**:

1. **Attributes** (object properties):
   - VLM verification on cropped image (15% margin)
   - **LLM-generated verification question**: `"Is this hat solid-colored?"`
   - Natural language, grammatically correct
   - Probability via logit summing: `P(yes) = softmax([z_yes, z_no])[0]`

2. **Relationships** (spatial/interaction):
   - Crop to union of both objects + colored boxes (RED=subject, BLUE=object)
   - **LLM-generated verification question**: `"Is the mitten pointing towards the face?"`
   - Natural phrasing, probability via logit summing

3. **Counts** (quantity distributions):
   - Poisson-Binomial distribution from detection confidences
   - Dynamic programming: convolve [1-p, p] for each detection
   - Output: full distribution `{0: p0, 1: p1, 2: p2, ...}`

**Key Features**:
- Checks ALL candidates (no early filtering)
- Stores all probabilities (even low ones)
- General: works with ANY subquestion type
- Efficient: reuses captions, lazy model loading

**Output**: `EvidenceCollection(attributes, relationships, counts)` per subquestion

---

### Step 5: ProbLog Reasoning + LLM Composition

**Purpose**: Execute probabilistic logic reasoning and compose natural language answer

**Two-Phase Process**:

#### Phase A: ProbLog Execution (per subquestion)

**Model**: Llama 3.3 70B generates ProbLog rules from evidence

**Process**:
1. Build scoped facts (only entities referenced in evidence)
2. LLM generates rule matching subquestion
3. Execute ProbLog to get probability

**Example**:
```prolog
% Facts (scoped to this subquestion)
0.874::entity(image_a, buffalo_a_0, buffalo, 93,182,402,597).
0.938::entity(image_a, bird_a_7, bird, 196,96,270,202).
0.906::relation(image_a, bird_a_7, buffalo_a_0, on_top_of).
0.787::attribute(image_a, bird_a_7, white).

% Rule (generated by LLM)
white_bird_on_animal(I) :-
    is_category(I, B, bird),
    is_category(I, A, buffalo),
    has_relationship(I, B, A, on_top_of),
    has_attribute(I, B, white).

% Query
query(white_bird_on_animal(image_a)).
% Result: P=0.5847
```

**Output**: `List[SubquestionResult(subquestion, probability)]`

#### Phase B: LLM Ultimate Composition

**Model**: Llama 3.3 70B synthesizes final answer

**Process**:
1. Convert subquestion probabilities to binary (≥0.5 = TRUE, <0.5 = FALSE)
2. Show LLM binary answers to subquestions
3. Ask ultimate question with instruction to output only "True" or "False"
4. Return binary answer

**Minimal Prompt**:
```
Given the following subquestion answers:

1. In image A, is there a white bird on top of another animal? → TRUE
2. In image B, is there a white bird on top of another animal? → FALSE

Is there a white bird on top of another animal in both images?

Answer with ONLY 'True' or 'False', nothing else.
```

**Output**: Binary answer ("False")

**Why LLM Composition?**:
- Handles logical structure ("both", "either", "equal count")
- Better than probabilistic product (overly pessimistic, assumes independence)
- Clean binary output format

---

## Execution Modes

PROVE supports two execution modes for handling probabilistic evidence:

### Probabilistic Mode (Default)
Uses actual probabilities throughout the pipeline. Object detection confidences and verification probabilities are preserved as continuous values (e.g., 0.874, 0.623).

```python
model = PROVE(mode="probabilistic")
```

### Deterministic Mode
Maps all probabilities to binary values (0% or 100%) for symbolic reasoning:
- Object detection confidences: All detected objects mapped to 100%
- Verification probabilities: <50% maps to 0%, >=50% maps to 100%

```python
model = PROVE(mode="deterministic")
```

Both modes achieve similar accuracy (~64% on NLVR2), with deterministic mode showing slightly higher recall.

---

## Models & Quantization

| Model | Purpose | Loading | Quantization |
|-------|---------|---------|--------------|
| **Florence-2-large** | Captioning, object detection | Auto device map | BF16 |
| **Llama 3.3 70B Instruct** (via AWS Bedrock) | Subquestion generation, agent reasoning, rule generation, ultimate composition | API call | N/A |
| **Qwen-2.5-VL-7B-Instruct** | Binary verification, open-ended VQA | Auto device map | BF16 |

**Memory Efficiency**: ModelManager singleton with lazy loading

---

## Data Structures

**Knowledge Base Hierarchy**:
```python
KnowledgeBase
└── images: Dict[str, ImageData]
    ├── objects: List[ObjectDetection]
    ├── attributes: Dict[int, AttributeData]
    ├── relationships: List[IntraRelation]
    ├── scene_context: Dict[str, Any]
    └── counts: Dict[str, Dict[int, float]]
```

**Evidence Collection** (per subquestion):
```python
EvidenceCollection
├── attributes: List[(entity_id, attr_class, value, prob)]
├── relationships: List[(subj_id, obj_id, relation, prob)]
├── counts: Dict[str, Dict[int, float]]
├── reasoning_trace: List[str]
└── verifications_completed: Dict[Tuple, bool]
```

---

## Probability Flow

```
Florence-2 Detection
  │ Geometric mean of log-probs
  │ Anchored sigmoid calibration
  ↓ 0.7-0.95 range

Qwen VL Verification
  │ Binary Yes/No question
  │ Logit summing: sum(logits["Yes","yes","YES"]), sum(logits["No","no","NO"])
  │ 2-token softmax
  ↓ P(statement_true)

ProbLog Facts
  │ Preserve ALL confidences
  ↓ No filtering

ProbLog Inference
  │ Weighted model counting
  │ Per-subquestion probability
  ↓

LLM Composition
  │ Binary conversion (≥0.5 = TRUE)
  │ Natural language reasoning
  ↓ Final answer
```

---

## Repository Structure

```
src/
├── prove.py                    # Main PROVE model class
├── __init__.py                 # Package exports
├── core/
│   ├── knowledge_base.py       # KB management
│   ├── model_manager.py        # Singleton model loading
│   ├── types.py                # Data structures
│   ├── probability.py          # Calibration, verifier functions
│   └── image_utils.py          # Image loading utilities
├── language/
│   ├── llm_client.py           # Llama 3.3 client (AWS Bedrock)
│   └── output_models.py        # Pydantic models
├── pipeline/
│   ├── detector.py             # Caption-based detection
│   ├── subquestion_generator.py  # Subquestion generation
│   ├── unified_agent.py        # Agentic evidence collection
│   ├── count_processor.py      # Poisson-Binomial counting
│   ├── problog_builder.py      # Evidence to ProbLog facts
│   └── problog_executor.py     # ProbLog execution and composition
└── vision/
    ├── florence2.py            # Florence-2 wrapper
    └── qwen_vl.py              # Qwen VL wrapper

eval/                           # Evaluation results (gitignored)
requirements.txt                # Python dependencies
```

---

## Usage

### Basic Usage

```python
from src import PROVE

# Initialize model
model = PROVE(mode="probabilistic")  # or mode="deterministic"

# Run inference
answer = model.predict(
    image_a_path="img1.jpg",
    image_b_path="img2.jpg",
    question="Are there more birds in image A than image B?"
)
print(answer)  # "True" or "False"
```

### Detailed Results with Logging

```python
from src import PROVE

model = PROVE(mode="probabilistic")

# Get detailed results with intermediate outputs
result = model.predict_with_details(
    image_a_path="img1.jpg",
    image_b_path="img2.jpg",
    question="Are there more birds in image A than image B?",
    save_logs=True,
    log_dir="logs"
)

# Access results
print(result['answer'])           # Final binary answer
print(result['subquestions'])     # Subquestions with probabilities
print(result['problog_program'])  # Generated ProbLog program
print(result['metadata'])         # Evidence statistics
print(result['log_path'])         # Path to saved logs
```

**Log Directory Structure**:
```
logs/20250112_143022_abc123/
├── images/
│   ├── image_a.jpg
│   └── image_b.jpg
├── knowledge_base.pl          # Full ProbLog program
└── results.json              # Structured results
```

---

## Key Technical Details

### Binary Verification Strategy

**Method**: All verification via binary Yes/No questions with direct cropping

**Probability Extraction**:
```python
# Sum logits for all variants of Yes/No
z_yes = sum(logits["Yes"], logits["yes"], logits["YES"])
z_no = sum(logits["No"], logits["no"], logits["NO"])

# 2-token softmax
P(statement_true) = exp(z_yes) / (exp(z_yes) + exp(z_no))
```

**Error Handling**: Return 0.5 (neutral) for failed extractions

### Poisson-Binomial Counting

**Algorithm**: Dynamic programming convolution

**Process**:
1. Filter detections by target class
2. Extract probabilities `[p1, p2, ..., pn]`
3. Initialize: `P = [1.0]` (0 objects with prob 1)
4. For each `p`: convolve `P` with `[1-p, p]`
5. Result: `P[k]` = probability of exactly k objects

**Example**:
```
Detections: [0.9, 0.8, 0.7]
Distribution: {0: 0.006, 1: 0.092, 2: 0.398, 3: 0.504}
Expected count: 2.4
```

### Unified Agent Loop

**ReAct Pattern**: Reasoning + Acting

**State Tracking**:
- Candidate entities (from detection)
- Evidence collected so far
- QA history with VLM
- Reasoning trace
- Verifications completed (prevents duplicates)

**Decision Logic** (every iteration):
1. Analyze what evidence is missing
2. Choose action: perceive, verify, or done
3. Execute action
4. Update state
5. Continue or stop (max 20 iterations)

**Efficiency**:
- Tracks `verifications_completed` to prevent duplicate checks
- Property-granular: `(entity_id, property)` for attributes
- Relation-specific: `(subj_id, obj_id, relation)` for relationships

---

## Design Rationale

### Why Subquestion Decomposition?

Complex questions contain multiple requirements that can't be verified atomically. Example:

```
"Is there a white bird on top of another animal in both images?"

Contains:
- Existence check: white bird
- Spatial relationship: on top of another animal
- Comparison: in BOTH images

Decompose to:
1. "In image A, is there a white bird on top of another animal?"
2. "In image B, is there a white bird on top of another animal?"
```

### Why Agentic Evidence Collection?

VLMs don't directly output probabilities. Agent:
1. Identifies what needs verification
2. Gathers context through open-ended questions
3. Generates specific binary questions
4. Extracts probabilities via logits
5. Handles any subquestion type without hardcoding

### Why ProbLog?

Probabilistic logic enables:
- Mathematical composition of evidence
- Handling uncertainty throughout reasoning
- Complete provenance (trace every probability)
- Logical structure (conjunctions, disjunctions, counts)

### Why LLM Ultimate Composition?

ProbLog returns subquestion probabilities, but ultimate question requires:
- Logical structure understanding ("both", "equal count")
- Binary output ("True" or "False")
- Semantic reasoning beyond probability multiplication

LLM sees binary subquestion answers → reasons about ultimate question → binary response

---

## Example Output

```
Step 1: Image Context Generation...
  [DONE] Captions generated
Step 2: Object Detection...
  [DONE] Detected 18 objects
Step 3: Subquestion Generation...
  [DONE] Generated 2 subquestions
Step 4: Evidence Collection...
  [DONE] Collected 8 attributes, 4 relationships, 0 counts
Step 5: ProbLog Reasoning...
  [DONE] Reasoning complete

Answer: False

Subquestions:
1. In image A, is there a white bird on top of another animal? (P=0.5847)
2. In image B, is there a white bird on top of another animal? (P=0.2463)

Logs saved to: logs/20250112_143022_abc123/
```

---

## Summary

PROVE transforms complex visual questions into probabilistic answers through:

1. **Decomposition**: Binary subquestions that break down complexity
2. **Agentic Collection**: Autonomous evidence gathering through VLM interaction
3. **Probabilistic Logic**: ProbLog composes evidence mathematically
4. **LLM Synthesis**: Binary answer from subquestion results

**Key Innovation**: Neuro-symbolic architecture combining neural perception (VLMs) with symbolic reasoning (ProbLog) via agentic orchestration.

---

## Performance

Evaluated on NLVR2 test set (1,388 examples with complete predictions):

| Mode | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Probabilistic | 63.47% | 0.678 | 0.504 | 0.579 |
| Deterministic | 63.83% | 0.669 | 0.541 | 0.598 |

Both modes show high agreement (87.8%) with no statistically significant difference (McNemar's test, p >= 0.05).
