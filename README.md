# PROVE: Probabilistic Reasoning Over Visual Evidence

Neuro-symbolic visual question answering using agentic evidence collection and probabilistic logic programming.

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

### Run on NLVR2 Example

```bash
# Random example
python run_example.py

# Specific example
python run_example.py --identifier test1-366-0-0

# With logging
python run_example.py --save-logs
```

### Programmatic Usage

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

## Architecture

```
Question + Images → Detection → Agent (Perceive/Verify) → ProbLog → Answer
                        ↓              ↓                      ↓
                   Entities    Probabilistic Evidence    True/False
```

**Key Principle**: Pass the question directly to a ReAct agent that collects visual evidence through investigation and verification, then compose results through probabilistic logic programming.

---

## Pipeline (3 Steps)

### Step 1: Object Detection

**Purpose**: Detect entities mentioned in the question

**Process**:
1. **Entity Extraction**: Llama 3.3 70B extracts nouns from the question (e.g., `["bird", "buffalo"]`)
2. **Open Vocabulary Detection**: Florence-2 detects each entity with bounding boxes
3. **Calibration**: Anchored sigmoid transforms raw scores to operational probabilities

**Output**: `ObjectDetection(object_id, label, bbox, confidence)` per entity

---

### Step 2: Evidence Collection

**Purpose**: Collect probabilistic evidence through agentic VLM reasoning

**Architecture**: ReAct agent loop (max 15 iterations)

**Agent Actions** (Pydantic-validated):

| Action | Purpose | Returns |
|--------|---------|---------|
| `perceive` | Ask open-ended question about entity | Text answer (context gathering) |
| `verify_attribute` | Check if entity has specific attribute | Probability from BLIP-ITM |
| `verify_relationship` | Check spatial relationship between entities | Probability from BLIP-ITM |
| `verify_count` | Count objects of a class | Poisson-Binomial distribution |
| `done` | Evidence collection complete | - |

**Agent Prompt Structure**:
```
QUESTION: "Is there a white bird on top of another animal in both images?"

DETECTED OBJECTS:
Image A, image_id: image_a
  - object_id: buffalo_a_0, object_class: buffalo
  - object_id: bird_a_1, object_class: bird

Image B, image_id: image_b
  - object_id: cow_b_0, object_class: cow

ACTION HISTORY:
[Turn 1]
Thought: I need to check if the bird in image A is white
Action: verify_attribute(image_id=image_a, entity_id=bird_a_1, attribute=color, value=white)
Result: p=0.787
```

**Evidence Types**:

1. **Attributes**: BLIP-ITM verification on cropped entity (e.g., "an orange dog")
2. **Relationships**: BLIP-ITM on union bbox (e.g., "a bird on top of a buffalo")
3. **Counts**: Poisson-Binomial distribution from detection confidences

**Output**: `EvidenceCollection(attributes, relationships, counts, action_history)`

---

### Step 3: ProbLog Reasoning

**Purpose**: Execute probabilistic logic to compute answer probability

**Process**:
1. Build ProbLog facts from collected evidence
2. LLM generates rules and query matching the question
3. Execute ProbLog program
4. Return probability and convert to True/False

**Dual Mode Execution**:
- **Probabilistic**: Original probabilities preserved (e.g., 0.874, 0.623)
- **Deterministic**: Thresholded (p < threshold → 0.0, p >= threshold → 1.0)

**Example ProbLog Program**:
```prolog
% Facts
0.874::entity(image_a, buffalo_a_0, buffalo).
0.938::entity(image_a, bird_a_1, bird).
0.906::relation(image_a, bird_a_1, buffalo_a_0, on_top_of).
0.787::attribute(image_a, bird_a_1, white).

% Sugar rules
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C).
has_relationship(I,A,B,R) :- relation(I,A,B,R).

% Generated rule
white_bird_on_animal(I) :-
    is_category(I, B, bird),
    is_category(I, A, buffalo),
    has_relationship(I, B, A, on_top_of),
    has_attribute(I, B, white).

query(white_bird_on_animal(image_a)).
% Result: P=0.5847
```

**Output**: `ModeResult(probability, final_answer, problog_program)`

---

## Unified Execution Mode

PROVE runs **both** probabilistic and deterministic modes with shared evidence to isolate the effect of perception uncertainty.

### How It Works

1. **Shared Evidence Collection**: Detection and verification run ONCE with probabilistic confidences
2. **Dual Fact Generation**: Same evidence generates two fact sets
3. **Dual ProbLog Execution**: Same queries run against both fact sets
4. **Two Answers**: Returns both probabilistic and deterministic final answers

### Threshold Parameter

```python
model = PROVE(threshold=0.5)  # Default
model = PROVE(threshold=0.7)  # More conservative
```

The threshold determines how probabilities map to binary values in deterministic mode:
- `p < threshold` → 0.0 (false)
- `p >= threshold` → 1.0 (true)

---

## Models

| Model | Purpose | Notes |
|-------|---------|-------|
| **Florence-2-large** | Object detection | Open vocabulary, BF16 |
| **Llama 3.3 70B** (AWS Bedrock) | Entity extraction, agent reasoning, rule generation | API call |
| **BLIP-ITM-large** | Attribute & relationship verification | Well-calibrated ITM head |
| **Qwen-2.5-VL-7B** | Open-ended perception | For `perceive` action |

---

## Data Structures

**Evidence Collection**:
```python
EvidenceCollection
├── question: str
├── attributes: List[(entity_id, attr_class, value, prob)]
├── relationships: List[(subj_id, obj_id, relation, prob)]
├── counts: Dict[str, Dict[int, float]]
└── action_history: List[{thought, action, result}]
```

**ProbLog Predicates**:
```prolog
entity(image_id, entity_id, category)
attribute(image_id, entity_id, value)
relation(image_id, subject_id, object_id, relation_type)
count(image_id, category, count_value)
```

**Unified Result**:
```python
UnifiedResult
├── threshold: float
├── shared: SharedEvidence
├── probabilistic: ModeResult
└── deterministic: ModeResult
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
│   ├── probability.py          # Detector confidence calibration
│   └── image_utils.py          # Image loading utilities
├── language/
│   ├── llm_client.py           # Llama 3.3 client (AWS Bedrock)
│   └── output_models.py        # Pydantic models for agent actions
├── pipeline/
│   ├── detector.py             # Question-based detection
│   ├── unified_agent.py        # ReAct evidence collection agent
│   ├── problog_builder.py      # Evidence to ProbLog facts
│   └── problog_executor.py     # ProbLog execution
└── vision/
    ├── florence2.py            # Florence-2 wrapper
    ├── blip_verifier.py        # BLIP-ITM verification
    ├── qwen_vl.py              # Qwen VL for perception
    └── spatial_reasoning.py    # Spatial relationship utilities

run_example.py                  # Run on NLVR2 examples
spatial_test.py                 # Spatial reasoning tests
```

---

## Usage

### Basic Usage

```python
from src import PROVE

model = PROVE(threshold=0.5)

result = model.predict(
    image_a_path="img1.jpg",
    image_b_path="img2.jpg",
    question="Are there more birds in image A than image B?"
)

print(f"Probabilistic: {result.probabilistic.final_answer}")
print(f"Deterministic: {result.deterministic.final_answer}")
```

### With Logging

```python
result = model.predict_with_details(
    image_a_path="img1.jpg",
    image_b_path="img2.jpg",
    question="Are there more birds in image A than image B?",
    save_logs=True,
    log_dir="logs"
)

# Access ProbLog programs
print(result.probabilistic.problog_program)
print(result.deterministic.problog_program)
```

**Log Directory Structure**:
```
logs/20250112_143022_abc123/
├── images/
│   ├── image_a.jpg
│   └── image_b.jpg
├── probabilistic.pl
├── deterministic.pl
└── results.json
```

---

## Example Output

```
Question: "Is there a white bird on top of another animal in both images?"
Threshold: 0.5

Step 1: Object Detection...
  image_a: 2 objects detected
  image_b: 2 objects detected

Step 2: Evidence Collection...
  [Verify Attribute] bird_a_1.color=white
    → p=0.787
  [Verify Relationship] bird_a_1 on_top_of buffalo_a_0
    → p=0.906
  [Verify Attribute] bird_b_0.color=white
    → p=0.234

Step 3: ProbLog Reasoning (dual mode)...

============================================================
RESULTS SUMMARY
============================================================

Probabilistic Mode:
  Probability: 0.167
  → Final Answer: False

Deterministic Mode (threshold=0.5):
  Probability: 0.000
  → Final Answer: False

Modes AGREE
============================================================
```

---

## Key Technical Details

### BLIP-ITM Verification

**Attribute Verification**:
```python
cropped = crop_with_padding(image, bbox, padding=0.15)
prompt = f"a {attr_value} {object_class}"  # "an orange cat"
probability = softmax(model(cropped, prompt).itm_score)[1]
```

**Relationship Verification**:
```python
union_bbox = union(bbox1, bbox2)
cropped = crop_with_padding(image, union_bbox, padding=0.15)
prompt = f"a {obj1} {relation} a {obj2}"  # "a bird on top of a buffalo"
probability = softmax(model(cropped, prompt).itm_score)[1]
```

### Poisson-Binomial Counting

Computes probability distribution over counts from detection confidences:

```
Detections: [0.9, 0.8, 0.7]
Distribution: {0: 0.006, 1: 0.092, 2: 0.398, 3: 0.504}
```

### ReAct Agent Loop

**Pattern**: Think → Act → Observe

1. Agent sees: question, detected objects, action history
2. Agent outputs: thought + action (Pydantic-validated)
3. Execute action and record result
4. Repeat until `done` or max iterations

---

## Summary

PROVE transforms visual questions into probabilistic answers through:

1. **Detection**: Question-guided object detection
2. **Agentic Evidence**: ReAct agent collects verification evidence
3. **Probabilistic Logic**: ProbLog composes evidence mathematically

**Key Innovation**: Neuro-symbolic architecture combining neural perception (BLIP-ITM, Qwen VL) with symbolic reasoning (ProbLog) via agentic orchestration.
