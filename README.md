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

# Initialize model (threshold=0.5 by default)
model = PROVE(threshold=0.5)

# Run inference - returns both probabilistic and deterministic results
result = model.predict(
    "image_a.jpg",
    "image_b.jpg",
    "Is there a white bird on top of another animal in both images?"
)

# Access results
print(result.probabilistic.final_answer)  # "True" or "False"
print(result.deterministic.final_answer)  # "True" or "False"
print(f"Modes agree: {result.probabilistic.final_answer == result.deterministic.final_answer}")
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

**Subquestion Types**:
- **Count (single image)**: "Are there exactly 2 dogs in image A?"
- **Count comparison**: "Are there more dogs in image A than in image B?"
- **Count equality**: "Are there the same number of dogs in image A as in image B?"
- **Attribute**: "Is the dog in image A orange?"
- **Universal attribute**: "Is every dog in image A orange?"
- **Relationship**: "Is the bird sitting on the buffalo in image A?"
- **Universal relationship**: "Is every bird sitting on a buffalo in image A?"
- **Cross-image attribute**: "Are the dogs in image A the same color as the dogs in image B?"
- **Cross-image relationship**: "Is there a person wearing a hat in both images?"

**Example**:
```
Ultimate: "Is there a white bird sitting on a buffalo?"

Subquestions:
1. "Is there a white bird sitting on a buffalo in image A?"
2. "Is there a white bird sitting on a buffalo in image B?"
```

**Output**: `List[BinarySubquestion(question)]` (pure natural language)

---

### Step 4: Unified Evidence Collection

**Purpose**: Collect probabilistic evidence through agentic VLM reasoning

**Architecture**: ReAct agent loop (max 10 iterations per subquestion)

**Agent Actions** (Pydantic-validated):
- **perceive**: Ask VLM open-ended question about an entity
  - Requires: `image_id`, `entity_id`, `question`
  - Stores response in perceive history for context
  - NO probability extraction
- **verify_attribute**: Check if entity has specific attribute (Yes/No)
  - Requires: `image_id`, `entity_id`, `attribute`, `value`
  - Extracts probability from Yes/No token logits
- **verify_relationship**: Check spatial relationship between two entities
  - Requires: `image_id`, `subject_id`, `object_id`, `relation`
  - Both entities must be in the same image
- **verify_count**: Count objects of a class in an image
  - Requires: `image_id`, `object_class`
  - Returns Poisson-Binomial distribution from detection confidences
- **done**: Evidence collection complete

**Agent Prompt Structure**:
```
IMAGES:
- Image A, image_id: image_a
- Image B, image_id: image_b

DETECTED OBJECTS:
Image A, image_id: image_a
  - object_id: dog_a_0, object_class: dog
  - object_id: cat_a_1, object_class: cat

Image B, image_id: image_b
  - object_id: bird_b_0, object_class: bird
```

**Evidence Types**:

1. **Attributes** (object properties):
   - VLM verification on cropped image
   - Binary question: `"Is this dog orange? Answer with only Yes or No."`
   - Probability via logit summing: `P(yes) = softmax([z_yes, z_no])[0]`

2. **Relationships** (spatial/interaction):
   - Crop to union of both objects with padding
   - Binary question: `"Is the bird on top of the buffalo? Answer with only Yes or No."`

3. **Counts** (quantity distributions):
   - Poisson-Binomial distribution from detection confidences
   - Dynamic programming: convolve [1-p, p] for each detection
   - Output: full distribution `{0: p0, 1: p1, 2: p2, ...}`

**Output**: `EvidenceCollection(attributes, relationships, counts, perceive_history)` per subquestion

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
0.874::entity(image_a, buffalo_a_0, buffalo).
0.938::entity(image_a, bird_a_7, bird).
0.906::relation(image_a, bird_a_7, buffalo_a_0, on_top_of).
0.787::attribute(image_a, bird_a_7, white).

% Sugar rules (always available)
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C).
has_relationship(I,A,B,R) :- relation(I,A,B,R).

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

## Unified Execution Mode

PROVE uses a unified pipeline that runs **both** probabilistic and deterministic modes with shared evidence to isolate the effect of perception uncertainty.

### How It Works

1. **Shared Evidence Collection**: Object detection and verification run ONCE with probabilistic confidences
2. **Dual Fact Generation**: Same evidence generates two fact sets:
   - **Probabilistic facts**: Original probabilities preserved (e.g., 0.874, 0.623)
   - **Deterministic facts**: Thresholded (p < t → 0.0, p >= t → 1.0)
3. **Dual ProbLog Execution**: Same queries run against both fact sets
4. **Two Answers**: Returns both probabilistic and deterministic final answers

### Threshold Parameter

```python
model = PROVE(threshold=0.5)  # Default threshold
model = PROVE(threshold=0.7)  # Higher threshold = more conservative
```

The threshold determines how probabilities map to binary values in deterministic mode:
- `p < threshold` → 0.0 (false)
- `p >= threshold` → 1.0 (true)

### Why Unified Pipeline?

The previous separate-mode approach had a flaw: different modes could collect different evidence, making it impossible to isolate uncertainty's effect. The unified pipeline ensures:
- **Same objects detected** in both modes
- **Same evidence collected** in both modes
- **Same ProbLog queries** in both modes
- **Only probability values differ** between modes

This allows true isolation of perception uncertainty's impact on reasoning.

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
    ├── scene_context: Dict[str, Any]  # Contains caption
    └── counts: Dict[str, Dict[int, float]]
```

**Evidence Collection** (per subquestion):
```python
EvidenceCollection
├── subquestion: str
├── attributes: List[(entity_id, attr_class, value, prob)]
├── relationships: List[(subj_id, obj_id, relation, prob)]
├── counts: Dict[str, Dict[int, float]]
├── reasoning_trace: List[str]
└── perceive_history: List[Dict[str, str]]
```

**ProbLog Predicates**:
```prolog
entity(image_id, entity_id, category)
attribute(image_id, entity_id, value)
relation(image_id, subject_id, object_id, relation_type)
count(image_id, category, count_value)
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
model = PROVE(threshold=0.5)

# Run inference - returns UnifiedResult with both modes
result = model.predict(
    image_a_path="img1.jpg",
    image_b_path="img2.jpg",
    question="Are there more birds in image A than image B?"
)

# Access probabilistic results
print(result.probabilistic.final_answer)  # "True" or "False"
for sq in result.probabilistic.subquestion_results:
    print(f"  {sq.subquestion}: p={sq.probability:.3f}")

# Access deterministic results
print(result.deterministic.final_answer)  # "True" or "False"
for sq in result.deterministic.subquestion_results:
    print(f"  {sq.subquestion}: p={sq.probability:.3f}")

# Check agreement
if result.probabilistic.final_answer == result.deterministic.final_answer:
    print("Modes agree!")
else:
    print("Modes disagree - perception uncertainty affected the outcome")
```

### Detailed Results with Logging

```python
from src import PROVE

model = PROVE(threshold=0.5)

# Get detailed results with logging
result = model.predict_with_details(
    image_a_path="img1.jpg",
    image_b_path="img2.jpg",
    question="Are there more birds in image A than image B?",
    save_logs=True,
    log_dir="logs"
)

# Access shared evidence (same for both modes)
print(f"Subquestions: {len(result.shared.subquestions)}")
print(f"Detected objects: {sum(len(objs) for objs in result.shared.detected_objects.values())}")

# Access mode-specific results
print(f"Probabilistic answer: {result.probabilistic.final_answer}")
print(f"Deterministic answer: {result.deterministic.final_answer}")

# Access ProbLog programs
print(result.probabilistic.problog_program)  # Probabilistic facts + rules
print(result.deterministic.problog_program)  # Deterministic facts + same rules
```

**Log Directory Structure**:
```
logs/20250112_143022_abc123/
├── images/
│   ├── image_a.jpg
│   └── image_b.jpg
├── probabilistic.pl          # ProbLog program with probabilistic facts
├── deterministic.pl          # ProbLog program with deterministic facts
└── results.json              # Unified results (both modes)
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

**ReAct Pattern**: Think → Act → Observe

**State Tracking**:
- Entity candidates (from detection, grouped by image)
- Evidence collected so far (attributes, relationships, counts)
- Perceive history (QA pairs from investigation)
- Reasoning trace

**Decision Logic** (every iteration):
1. LLM sees: subquestion, candidates, evidence, perceive history
2. LLM outputs: thought + action (Pydantic-validated)
3. Execute action (perceive/verify_attribute/verify_relationship/verify_count)
4. Update evidence
5. Continue or stop (max 10 iterations or "done" action)

**Action Validation**:
- All actions require explicit `image_id` (image_a or image_b)
- Entity IDs must match candidates from detection
- Pydantic validates all fields before execution

---

## Design Rationale

### Why Subquestion Decomposition?

Complex questions contain multiple requirements that can't be verified atomically. Example:

```
"Is there a white bird on top of another animal in both images?"

Contains:
- Attribute check: white bird
- Spatial relationship: on top of another animal
- Scope: in BOTH images

Decompose to:
1. "Is there a white bird on top of another animal in image A?"
2. "Is there a white bird on top of another animal in image B?"
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
Step 1: Caption Generation...
  Image A: "A black buffalo standing in a grassy field with a white bird on its back"
  Image B: "Two cows grazing in a meadow"

Step 2: Object Detection...
  Image A: buffalo (1), bird (1)
  Image B: cow (2)

Step 3: Subquestion Generation...
  1. "Is there a white bird on top of another animal in image A?"
  2. "Is there a white bird on top of another animal in image B?"

Step 4: Evidence Collection...
  [1/2] Is there a white bird on top of another animal in image A?
    [Verify Attribute] bird_a_0.color=white → Yes (p=0.787)
    [Verify Relationship] bird_a_0 on_top_of buffalo_a_0 → Yes (p=0.906)
  [2/2] Is there a white bird on top of another animal in image B?
    [Verify Attribute] No birds detected in image B

Step 5: ProbLog Reasoning (dual mode)...
  [1/2] Is there a white bird on top of another animal in image A?
    Probabilistic: 0.5847
    Deterministic: 1.0000
  [2/2] Is there a white bird on top of another animal in image B?
    Probabilistic: 0.0000
    Deterministic: 0.0000

Ultimate Composition:
  Probabilistic: False
  Deterministic: False
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

**Note**: With the unified pipeline, these metrics can now be computed on identical evidence, providing a cleaner ablation study on the effect of perception uncertainty.
