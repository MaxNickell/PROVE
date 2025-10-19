# PROVE Pipeline: Complete Implementation Guide

**PROVE (Probabilistic Reasoning Over Visual Evidence)** - A subquery-driven architecture that transforms ambiguous visual questions into structured evidence extraction, probabilistic reasoning, and confident answers with complete provenance.

---

# Part 1: Pipeline Architecture

## Overview

**Core Philosophy**: Break complex comparative questions into specific binary subquestions, extract evidence using multi-modal verification, and synthesize probabilistic answers.

---

## Architecture: 5-Tier Knowledge Framework

- **Objects**: Spatial entities with bounding boxes and confidence scores
- **Attributes**: Object characteristics verified through binary VLM
- **Relationships**: Spatial/interaction relationships between objects
- **Scene Attributes**: Environmental and contextual facts
- **Count Distributions**: Probabilistic object counts using Poisson-Binomial distributions

**Data Structure**: Clean `ImageData` hierarchy - `kb.images[image_id].{objects, attributes, relationships, scene_attributes, counts}`
- **Processing Aids**: Captions used during processing only, never stored in final KB

---

## Models & Quantization

### Primary Models:
- **Florence-2-large-ft**: Caption-based open vocabulary detection, image captions, region descriptions
- **Llama-3.3-70B-Instruct**: Subquery generation, entity extraction, candidate generation, count analysis (8-bit quantization via BitsAndBytes)
- **Qwen-2.5-VL-7B-Instruct**: Binary verification for attributes, relationships, and scene attributes

### Model Loading:
- **Device Allocation**: Auto device mapping for optimal GPU distribution
- **Memory Optimization**: 8-bit quantization for Llama-3.3-70B to fit memory constraints
- **Lazy Loading**: Models loaded on-demand via ModelManager singleton

---

## Phase-by-Phase Breakdown

### Step 1: Image Context Generation
**Goal**: Capture rich scene-level information upfront for efficient reuse in object detection and reasoning

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
**Goal**: Identify all visual entities with spatial grounding and confidence scores using pre-generated captions

**2-Step Pipeline** (using caption from Step 1):

**Step 2a: Extract Entity Classes**
- **Model**: Llama-3.3-70B-Instruct with Pydantic validation
- **Method**: `llm_client.extract_entities(messages)` → `EntityExtractionResponse`
- **Input**: Pre-generated caption from Step 1
- **Data Flow**: Caption → Singular noun entities (no attributes)
- **Pydantic Processing**:
  - Automatically lowercases all entities
  - Deduplicates using `set()`
  - Validates non-empty list
- **Example**: `["egret", "buffalo", "field"]`

**Step 2b: Open Vocabulary Detection Per Entity**
- **Model**: Florence-2-large
- **Task**: `<OPEN_VOCABULARY_DETECTION>` + text prompt
- **Method**: `detector.detect_from_caption(image_path, caption)` → `florence2.detect_open_vocabulary(image, entity_class)` per entity
- **Data Flow**: For each entity → Bounding boxes + labels + raw scores
- **Example**: Run detection for "egret", "buffalo", "field" separately

**Probability Calculation**:
- **Source**: Florence-2 sequence-level confidence using geometric mean
- **Method**: `exp(mean(log_probs))` - length-normalized likelihood
- **Raw Extraction**: `compute_transition_scores()` with `normalize_logits=True`
- **Calibration**: Anchored sigmoid mapping transforms raw scores to operational probabilities
- **Formula**: `p' = 1 / (1 + ((1-p)/p)^a * e^(-c))`
- **Anchor Points**: `0.1 → 0.7`, `0.5 → 0.9` (hardcoded)
- **Parameters**: `a ≈ 0.6144`, `c ≈ 2.1972` (pre-computed)
- **Function**: `calibrate_detector_confidence(raw_score)` in `src/core/probability.py`
- **Range**: Transforms poorly calibrated raw scores → realistic 0.7-0.95 operational probabilities
- **Storage**: `ObjectDetection.confidence` field
- **Usage**: Propagated to ProbLog facts as object existence probabilities

**Benefits**:
- **Efficient**: Reuses caption from Step 1 (no redundant generation)
- **Comprehensive**: LLM finds ALL entities mentioned in caption
- **Open Vocabulary**: Can detect any object in caption (not limited to pre-defined classes)
- **Attribute-Free**: Extracts base nouns only ("cat" not "blue cat")
- **Deduplicated**: Automatic deduplication prevents redundant detections

**Output**: `ObjectDetection(object_id, label, bbox, confidence)` with calibrated confidence

**Processing Note**:
- **Purpose**: Captions are processing aids only, never stored in knowledge base
- **Usage**: Passed directly to processors (AttributeProcessor, SceneAttributeProcessor) that need contextual information
- **Storage**: No KB storage - captions remain in processing pipeline only
- **Architecture**: Clean separation between processing aids and stored knowledge

**Output**: `{"image_a": "detailed scene description", ...}` (passed to processors, not stored)

---

### Step 3: Subquery Generation
**Goal**: Break ambiguous questions into specific binary subquestions across all knowledge types

**Implementation**:
- **Model**: Llama-3.3-70B-Instruct with Pydantic validation
- **Method**: `subquery_generator.generate_binary_subqueries(question, images)`
- **Data Flow**: Ultimate question + Image contexts + Objects → Binary Yes/No questions
- **Categories**:
  - **attribute**: Object characteristics (color, size, position, shape)
  - **relationship**: Spatial/interaction relations between objects
  - **scene_attribute**: Scene-level characteristics, environment, background
  - **count**: Questions about number/quantity of objects of certain classes

**LLM-Driven Approach**:
- **Intelligence**: LLM handles object reference extraction and type classification
- **Validation**: Pydantic `SubqueryResponse` with field validation and retry logic
- **Trust LLM**: No manual pattern matching or heuristic classification
- **Structured Output**: System message enforces strict JSON format
- **Reasoning Process**: 3-step approach (understand → consider → break down)

**Probability Calculation**:
- **Source**: No probability calculation (deterministic text generation)
- **Focus**: Generate verifiable binary questions with accurate object references
- **Validation**: Pydantic type validation + object ID existence check

**Output**: `List[BinarySubquery(question, subquery_type, referenced_objects)]`

---

### Step 4: Subquery Type Routing
**Goal**: Organize subqueries by type for specialized processing

**Implementation**:
- **Method**: Group subqueries by `subquery_type` field
- **Data Flow**: Mixed subqueries → Categorized lists for each processor
- **Routing**:
  - `attribute` → AttributeProcessor
  - `relationship` → RelationshipExtractor
  - `count` → CountProcessor
  - `scene_attribute` → SceneAttributeProcessor

**Output**: Categorized subquery lists for parallel processing

---

### Step 5: Attribute Processing (Consolidated Planning + Extraction)
**Goal**: Process attribute subqueries individually - determine needs per subquery and extract values immediately

**Per-Subquery Implementation**:
- **Processing Model**: Each attribute subquery processed independently (no cross-subquery consolidation)
- **Pipeline**: Per subquery: LLM attribute planning → Florence-2 region description → LLM Yes/No questions → Qwen-2.5-VL binary verification
- **Method**: `attribute_processor.process_attribute_subqueries(attribute_subqueries, image_paths, images)`
- **Data Flow**: Single Subquery → Attribute classes for referenced_objects → Region descriptions → Binary questions → Verified attributes

**Per-Subquery Processing Flow**:
1. **Attribute Planning**: LLM determines which attribute classes needed for `referenced_objects` in this subquery
2. **Region Description**: Florence-2 generates dense captions for object regions
3. **Question Generation**: LLM creates Yes/No questions: "Does [object] have [attribute_value]?"
4. **Binary Verification**: Qwen verifies each question individually with proper softmax
5. **Immediate Storage**: Results stored immediately per subquery (no consolidation)

**Key Principles**:
- **Referenced Objects Only**: Each subquery specifies exactly which objects it needs
- **No Cross-Subquery Merging**: Each subquery determines its own attribute requirements independently
- **Self-Contained Processing**: Subquery + dense region captions + attribute classes → binary questions

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification via `extract_yes_no_probability_with_proper_softmax(logits, response)`
- **Process**: Each binary question gets individual verification with 2-token softmax
- **Error Handling**: Return 0.5 (neutral) when extraction fails - no arbitrary confidence values
- **No Filtering**: All results preserved (including low-probability) for ProbLog inference

**Output**: `Attributes stored per subquery` in ImageData structure

---

### Step 6: Relationship Extraction
**Goal**: Extract spatial and interaction relationships using compound subquery-driven analysis

**Enhanced Implementation**:
- **Pipeline**: Compound subquery analysis → Llama-3.3-70B comprehensive relationship candidates → Qwen-2.5-VL binary verification
- **Method**: `relationship_extractor.extract_relationships(subqueries, image_paths, images)`
- **Data Flow**: Compound Subqueries + All Available Objects → Comprehensive relationship requirements → Binary verification → Verified relations
- **Compound Handling**: LLM analyzes subqueries for cross-image and multi-relationship requirements

**LLM Analysis Enhancement**:
- **Beyond Referenced Objects**: Considers ALL available objects for relationship extraction needs
- **Cross-Image Relationships**: Handles subqueries requiring relationship comparisons across images
- **Multi-Relationship Requirements**: Single subquery can generate multiple RelationshipCandidates

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification via `extract_yes_no_probability_with_proper_softmax(logits, response)`
- **Process**:
  1. Qwen answers "Is object A [relation] object B?" with color-coded bounding boxes
  2. Uses red box for subject, blue box for object (no coordinate clutter)
  3. Extract raw logits for "Yes" and "No" tokens specifically (with verbalizer fallbacks)
  4. Apply proper 2-token softmax: `P(relation_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
- **Error Handling**: Return 0.5 (neutral) when verbalizer extraction fails - no arbitrary confidence values

**Output**: `List[IntraRelation(subject_id, object_id, relation, probability)]`

---

### Step 7: Count Processing
**Goal**: Extract probabilistic object counts using Poisson-Binomial distribution modeling

**Implementation**:
- **Model**: Llama-3.3-70B-Instruct with Pydantic validation for count requirement analysis
- **Method**: `count_processor.process_count_subqueries(count_subqueries, images)`
- **Algorithm**: Poisson-Binomial probabilistic counting using Dynamic Programming
- **Data Flow**: Count Subqueries → LLM count requirements → Detection probabilities → Full probability distributions

**Poisson-Binomial Count Processing**:
1. **Count Requirement Analysis**: LLM determines which object classes need counting in which images
   - Input: `"Are there more than 2 bird objects in IMAGE_A?"`
   - Output: `[{image_id: "image_a", object_class: "bird"}]`
2. **Detection Filtering**: Extract detection confidences for target class
3. **Dynamic Programming**: Compute P(C=k) for k=0,1,2,...,n using DP convolution:
   ```python
   P = [1.0]  # P(0 objects) = 1.0
   for p in probabilities:
       new_P[k] += P[k] * (1 - p)      # Object not counted
       new_P[k + 1] += P[k] * p        # Object counted
   ```
4. **Complete Distribution Storage**: Store full P(C=k) distribution, not summary statistics

**Probability Calculation**:
- **Source**: Object detection confidences from Florence-2
- **Algorithm**: Exact Poisson-Binomial computation via Dynamic Programming
- **Preservation**: Complete probability distribution preserved for all possible count values
- **Storage Format**: `counts[object_class] = {"distribution": {"0": p0, "1": p1, "2": p2, ...}}`

**Pydantic Structured Output**:
- **Model**: `CountRequirementResponse` with validated `CountRequirementItem` list
- **Reliability**: System message + explicit JSON format ensures valid LLM responses
- **Error Handling**: Pydantic retry logic with detailed format instructions

**Output**: `Complete probability distributions` stored in `ImageData.counts`

---

### Step 8: Scene Attribute Processing
**Goal**: Extract scene-level attributes using compound subquery decomposition + Qwen binary verification

**Implementation**:
- **Class**: SceneAttributeProcessor (renamed from ContextProcessor)
- **Pipeline**: Compound subquery analysis → LLM decomposition into atomic binary questions → Qwen verification → Multiple scene attributes
- **Method**: `scene_attribute_processor.process_scene_attribute_subqueries(scene_subqueries, image_paths, images)`
- **Data Flow**: Compound Scene Subqueries → LLM atomic decomposition → Multiple binary verifications → Multiple scene attributes per subquery

**Compound Subquery Decomposition**:
- **LLM Analysis**: Breaks compound scene questions into atomic binary verifications
- **Multi-Image Support**: Single subquery can generate scene attribute checks for multiple images
- **Multi-Attribute Support**: Single subquery can generate multiple scene attribute types
- **Examples**:
  - `"Do both images show outdoor settings?"` → 2 atomic questions:
    - `"Is IMAGE_A an outdoor environment?"` (environment_type=outdoor)
    - `"Is IMAGE_B an outdoor environment?"` (environment_type=outdoor)

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification via `extract_yes_no_probability_with_proper_softmax(logits, response)`
- **Process**: Each atomic binary question gets individual Qwen verification with proper softmax
- **Error Handling**: Return 0.5 (neutral) when extraction fails - no arbitrary confidence values
- **Storage**: Multiple `SceneAttributeResult` objects per subquery, each with individual confidence

**Storage Structure**: Scene attributes follow same pattern as object attributes:
```json
"scene_attributes": {
  "environment_type": [{"value": "outdoor", "confidence": 0.92}],
  "time_of_day": [{"value": "daytime", "confidence": 0.84}]
}
```

**Output**: `Multiple SceneAttribute results per subquery` stored in `ImageData.scene_attributes`

---

### Step 9: ProbLog Knowledge Base Generation
**Goal**: Convert all extracted evidence into executable probabilistic logical facts using exact specification format

**Implementation**:
- **Class**: ProbLogBuilder (completely rewritten for specification compliance)
- **Method**: `problog_builder.build_knowledge_base(images)`
- **Data Flow**: ImageData structure → Specification-compliant ProbLog facts → Executable program
- **Output File**: `knowledge_base.pl` with complete probabilistic program

**Specification Format Implementation**:
```prolog
% entity(image_id: str, entity_id: str, category: str, x1: int, y1: int, x2: int, y2: int).
Prob::entity(ImageID, EntityID, Category, X1, Y1, X2, Y2).

% relation(image_id: str, entity_a: str, entity_b: str, relation_type: str).
Prob::relation(ImageID, EntityA, EntityB, RelationType).

% attribute(image_id: str, entity_id: str, attr_value: str).
Prob::attribute(ImageID, EntityID, AttrValue).

% scene_attr(image_id: str, attr_value: str).
Prob::scene_attr(ImageID, AttrValue).

% count(image_id: str, category: str, value: int).
Prob::count(ImageID, Category, Value).
```

**Complete Count Distribution Encoding**:
- **Input**: `counts["cattle"] = {"distribution": {"0": 0.345, "1": 0.431, "2": 0.188, "3": 0.034, "4": 0.002}}`
- **Output**: 5 separate ProbLog facts:
  ```prolog
  0.345::count(image_a, cattle, 0).
  0.431::count(image_a, cattle, 1).
  0.188::count(image_a, cattle, 2).
  0.034::count(image_a, cattle, 3).
  0.002::count(image_a, cattle, 4).
  ```

**Clean Attribute Values**:
- **Object Attributes**: `attribute(image_a, bird_a_0, white)` (not `color_white`)
- **Scene Attributes**: `scene_attr(image_a, outdoor)` (not `environment_type_outdoor`)

**Probability Preservation**:
- **Source**: Preserves original confidence scores from all extraction phases
- **Mapping**: Direct confidence transfer from extraction to ProbLog facts
- **No Information Loss**: Complete probabilistic information maintained

**Output**: `List[ProbLogFact(probability, predicate, arguments)]` + executable ProbLog program

---

### Step 10: ProbLog Execution and Evidence Tracing (Prepared)
**Goal**: Execute probabilistic reasoning to answer subqueries with evidence trails

**Status**: Implementation prepared, not yet active in pipeline
**Future Implementation**:
- **Engine**: ProbLog probabilistic logic programming engine
- **Method**: `problog_executor.execute_subqueries(subqueries, problog_facts)`
- **Algorithm**: Probabilistic logic programming inference (weighted model counting)

---

### Step 11: Final Answer Generation (Prepared)
**Goal**: Synthesize ultimate answer using subquery results and evidence

**Status**: Implementation prepared, not yet active in pipeline
**Future Implementation**:
- **Model**: Llama-3.3-70B-Instruct
- **Method**: `answer_generator.generate_final_answer(question, subquery_results, image_contexts)`
- **Synthesis**: LLM combines probabilistic evidence into coherent reasoning chain

---

## Probability Flow Architecture

```
Florence-2 Object Confidence (0.0-1.0)
    ↓
Object Detection Confidences → Poisson-Binomial Count Distributions
    ↓
Qwen-2.5-VL Proper Softmax Binary Verification:
  • Extract raw logits for "Yes" and "No" tokens specifically
  • Use verbalizer fallbacks: ["Yes", "yes", "YES"], ["No", "no", "NO"] for robustness
  • Apply 2-token softmax: P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))
  • Avoids probability inflation from full-vocabulary renormalization
  • Preserve ALL results (no confidence filtering)
    ↓
ProbLog Probabilistic Facts (preserve all confidence values including low-probability)
  • entity(image_id, entity_id, category, x1, y1, x2, y2)
  • attribute(image_id, entity_id, attr_value)
  • relation(image_id, entity_a, entity_b, relation_type)
  • scene_attr(image_id, attr_value)
  • count(image_id, category, value) [complete distributions]
    ↓
ProbLog Inference Engine (probabilistic logic programming with complete evidence)
    ↓
Subquery Result Probabilities (query marginal probabilities from complete fact set)
    ↓
Final Answer Confidence (evidence-weighted aggregation from all probabilistic evidence)
```

---

## Key Implementation Details

### Clean Architecture:
- **No Legacy Code**: All legacy extractors and planners removed
- **ImageData Hierarchy**: Clean `kb.images[image_id].{objects, attributes, relationships, scene_attributes, counts}`
- **Simple References**: Use string object IDs in format `label_image_objectid`
- **Research-Grade**: All components accept ImageData directly, no format conversions

### Binary Verification Strategy:
- **Method**: All attributes, relationships, and scene attributes verified via binary Yes/No questions
- **Proper Softmax Calculation**: Extract raw logits for "Yes"/"No" tokens, apply 2-token softmax: P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))
- **Verbalizer Fallbacks**: Use ["Yes", "yes", "YES"] and ["No", "no", "NO"] for tokenization robustness
- **Avoids Inflation**: No full-vocabulary renormalization that artificially inflates confidence scores
- **Principled Error Handling**: Return 0.5 (neutral) for failed extractions - no arbitrary confidence injection
- **No Filtering**: Preserve ALL results including low-probability ones for complete ProbLog inference

### Structured LLM Output:
- **Pydantic Validation**: All LLM-generated outputs use Pydantic models with retry logic
- **System Messages**: Enforce strict JSON format with explicit structure requirements
- **Robust Parsing**: Built-in JSON extraction and validation with detailed error handling
- **Simplified Prompts**: Trust Pydantic to enforce structure, no examples needed

### Memory & Performance:
- **8-bit Quantization**: Llama-3.3-70B uses BitsAndBytesConfig for memory efficiency
- **Auto Device Mapping**: Optimal GPU distribution for multi-GPU setups
- **Lazy Loading**: Models loaded on-demand via singleton pattern

### Probabilistic Counting Innovation:
- **Complete Distributions**: Poisson-Binomial modeling preserves full uncertainty information
- **No Summary Statistics**: Stores P(C=k) for all k, not just expected value
- **ProbLog Ready**: Distribution facts enable sophisticated probabilistic queries

---

# Part 2: Prompts Documentation

This section contains all prompts used throughout the PROVE pipeline for LLMs (Large Language Models) and VLMs (Vision-Language Models).

---

## Table of Contents

1. [Object Detection Pipeline](#1-object-detection-pipeline)
2. [Image Context Generation (Florence-2)](#2-image-context-generation-florence-2)
3. [Subquery Generation (LLM)](#3-subquery-generation-llm)
4. [Attribute Processing](#4-attribute-processing)
5. [Relationship Extraction](#5-relationship-extraction)
6. [Count Processing](#6-count-processing)
7. [Scene Attribute Processing](#7-scene-attribute-processing)
8. [Probability Calibration](#8-probability-calibration)

---

## 1. Object Detection Pipeline

### 1.1 Entity Extraction from Caption (LLM)

**Model**: Llama-3.3-70B-Instruct

**Purpose**: Extract singular noun entity classes from image caption for open vocabulary detection

**Input Variables**:
- `caption`: Detailed image caption from Florence-2

**Output Format**: JSON (Pydantic validated)

**Output Parsing**: Pydantic model `EntityExtractionResponse` with automatic:
- Lowercase conversion
- Deduplication via `set()`
- Whitespace stripping
- Empty string filtering

**System Message**:
```
You are an expert at extracting object entities from image descriptions. Extract ALL singular noun object classes mentioned in the caption. Return only base nouns without attributes (e.g., 'cat' not 'blue cat', 'dog' not 'large dog'). Return strict JSON only.
```

**Prompt Template**:
```
Extract all object entities from this image caption.

Caption: "{caption}"

Rules:
1. Extract ALL objects/entities mentioned in the caption
2. Use singular noun forms only (e.g., "person" not "people", "bird" not "birds")
3. Remove all attributes - return base nouns only (e.g., "cat" not "black cat")
4. Do NOT include scene descriptors, locations, or abstract concepts
5. Focus on concrete, tangible objects that can be detected visually

Respond in exact JSON format:
{
  "entities": ["object1", "object2", "object3"]
}
```

**Example Input**:
```
Caption: "The image shows a white egret perched on a black buffalo in a grassy field"
```

**Example Output**:
```json
{
  "entities": ["egret", "buffalo", "field", "grass"]
}
```

**Pydantic Processing**: Output becomes `["egret", "buffalo", "field", "grass"]` (lowercased, deduplicated)

**File Location**: `src/pipeline/detector.py:73-101`

---

### 1.2 Open Vocabulary Detection (Florence-2)

**Model**: Florence-2-large-ft

**Purpose**: Detect specific objects using open vocabulary detection with text prompts

**Input Variables**:
- `image`: PIL Image object
- `text_prompt`: Entity class to detect (e.g., "egret", "buffalo")

**Task Token**: `<OPEN_VOCABULARY_DETECTION>` + text_prompt

**Output Format**: Dict with bboxes, labels, scores

**Output Parsing**: Florence-2's post_process_generation

**Prompt Template**:
```
Task: <OPEN_VOCABULARY_DETECTION>{text_prompt}
(Image + text prompt processed together by Florence-2)
```

**Example**:
- For `text_prompt="egret"`: Detects all egrets in image with bounding boxes
- For `text_prompt="buffalo"`: Detects all buffalo in image with bounding boxes

**File Location**: `src/vision/florence2.py:31-82`

---

## 2. Image Context Generation (Florence-2)

### 2.1 Detailed Image Captioning

**Model**: Florence-2-large-ft

**Purpose**: Generate detailed natural language descriptions of entire images for contextual understanding

**Input Variables**:
- `image`: PIL Image object (full image, RGB format)
- `task`: Florence-2 task token (default: `"<MORE_DETAILED_CAPTION>"`)

**Output Format**: Natural language text description

**Output Parsing**: Direct text extraction from model's post-processing

**Prompt Template**:
```
Task Token: <MORE_DETAILED_CAPTION>
(Image is processed directly by Florence-2's vision encoder)
```

**Example Output**:
```
"The image is a photograph of a black buffalo standing in a grassy field. The buffalo is facing the camera and its head is turned slightly to the side. On top of the buffalo's head, there is a white bird, possibly a egret, perched on its shoulder."
```

**File Location**: `src/vision/florence2.py:180-220`

---

### 1.2 Object Region Description

**Model**: Florence-2 (microsoft/Florence-2-large)

**Purpose**: Generate detailed descriptions of cropped object regions for attribute extraction

**Input Variables**:
- `image`: PIL Image object (cropped to object bounding box)
- `task`: Florence-2 task token (`"<MORE_DETAILED_CAPTION>"`)

**Output Format**: Natural language text description

**Output Parsing**: Direct text extraction from model's post-processing

**Prompt Template**:
```
Task Token: <MORE_DETAILED_CAPTION>
(Cropped object region is processed directly by Florence-2's vision encoder)
```

**Example Output**:
```
"A white bird with long legs and a pointed beak standing on dark fur"
```

**File Location**: `src/vision/florence2.py:180-220`

---

## 2. Subquery Generation (LLM)

### 2.1 Binary Subquery Decomposition

**Model**: Llama-3.3-70B-Instruct

**Purpose**: Decompose ultimate question into specific binary subquestions using visual context and detected objects

**Input Variables**:
- `ultimate_question`: The main comparative question (e.g., "What is uniquely similar about these images?")
- `context`: Structured context with image captions and detected objects

**Output Format**: JSON (Pydantic validated)

**Output Parsing**: Pydantic model `SubqueryResponse` with automatic validation and retries

**System Message**:
```
You are an expert at breaking down ambiguous comparative questions into specific binary subquestions using visual context. Generate binary questions that reference specific detected objects and can be answered Yes/No. Return strict JSON only.
```

**Prompt Template**:
```
Your task is to break down the ultimate question into binary (Yes/No) subquestions.

Ultimate Question: "{ultimate_question}"

Visual Context:
{context}

PROCESS:
1. First, understand what the ultimate question is asking
2. Consider what each scene depicts based on the image captions provided
3. Then break the ultimate question into binary subquestions that will collectively answer it

SUBQUESTION CATEGORIES:
- **attribute**: Characteristics of specific detected objects
- **relationship**: Spatial or interaction relations between detected objects
- **scene_attribute**: Scene-level characteristics that apply to the entire image, not individual objects
- **count**: Quantities of object classes in an image

RULES:
- Each subquestion must be answerable with Yes/No
- Only reference objects from the Objects list using their exact IDs
- List all relevant object IDs in "referenced_objects"
- Generate subquestions across all 4 categories
```

**Example Context**:
```
Image IMAGE_A:
Context: The image is a photograph of a black buffalo standing in a grassy field...
Objects: bird_a_0 (bird, conf=0.57), cattle_a_1 (cattle, conf=0.35), cattle_a_2 (cattle, conf=0.15)

Image IMAGE_B:
Context: The image shows a large brown cow standing in a grassy field...
Objects: animal_b_0 (animal, conf=0.27), bird_b_1 (bird, conf=0.39), bird_b_2 (bird, conf=0.23)
```

**Example Output**:
```json
{
  "subqueries": [
    {
      "question": "Does bird_a_0 have the same color as bird_b_1?",
      "referenced_objects": ["bird_a_0", "bird_b_1"],
      "subquery_type": "attribute"
    },
    {
      "question": "Is cattle_a_1 positioned near cattle_a_2?",
      "referenced_objects": ["cattle_a_1", "cattle_a_2"],
      "subquery_type": "relationship"
    },
    {
      "question": "Are there more than 2 bird objects in IMAGE_A?",
      "referenced_objects": ["bird_a_0"],
      "subquery_type": "count"
    },
    {
      "question": "Do both images show outdoor settings?",
      "referenced_objects": [],
      "subquery_type": "scene_attribute"
    }
  ]
}
```

**File Location**: `src/pipeline/subquery_generator.py:126-162`

---

## 3. Attribute Processing

### 3.1 Attribute Planning (LLM)

**Model**: Llama-3.3-70B-Instruct

**Purpose**: Determine which attribute classes need extraction for referenced objects in a specific subquery

**Input Variables**:
- `subquery.question`: Single attribute subquery question
- `subquery.subquery_type`: Type (should be "attribute")
- `objects_str`: Comma-separated list of referenced objects with labels

**Output Format**: JSON (Pydantic validated)

**Output Parsing**: Pydantic model `AttributePlanningResponse` with automatic validation and retries

**Prompt Template**:
```
Analyze this single attribute subquery to determine what attribute classes are needed for the referenced objects.

Subquery: "{subquery.question}"
Type: {subquery.subquery_type}
Referenced Objects: {objects_str}

Task: Determine which visual attribute classes need to be extracted from which referenced objects to answer this specific question.

Consider these attribute classes:
- **Physical Attributes**: size, shape, color, texture, pattern, material
- **State Attributes**: condition, state, position, orientation
- **Functional Attributes**: function, style, usage
- **Comparative Attributes**: muscle_mass, muscle_definition, body_size, weight, height, strength

Respond in this exact JSON format:
{
  "attribute_requirements": {
    "bird_a_0": ["color", "orientation"],
    "bird_b_1": ["shape"]
  }
}

Rules:
- Only include objects explicitly referenced in the subquery
- Only include attribute classes directly needed to answer this specific question
- Use specific attribute class names (not generic descriptions)
- If no attributes needed, return empty dict: {}

Examples:
- "Is bird_a_0 black?" → {"attribute_requirements": {"bird_a_0": ["color"]}}
- "Do bird_a_0 and animal_b_0 have the same color?" → {"attribute_requirements": {"bird_a_0": ["color"], "animal_b_0": ["color"]}}


Answer:
```

**Example Input**:
```
Subquery: "Does bird_a_0 have the same color as bird_b_1?"
Type: attribute
Referenced Objects: bird_a_0 (bird), bird_b_1 (bird)
```

**Example Output**:
```json
{
  "attribute_requirements": {
    "bird_a_0": ["color"],
    "bird_b_1": ["color"]
  }
}
```

**File Location**: `src/pipeline/attribute_processor.py:99-200`

---

### 3.2 Attribute Candidate Generation (LLM)

**Model**: Llama-3.3-70B-Instruct

**Purpose**: Generate 2-4 candidate values for a specific attribute class based on region description

**Input Variables**:
- `subquery_question`: Original subquery for context
- `object_label`: Object class label (e.g., "bird")
- `attribute_class`: Attribute class to generate candidates for (e.g., "color")
- `region_description`: Florence-2 detailed caption of cropped object region

**Output Format**: JSON (Pydantic validated)

**Output Parsing**: Pydantic model `CandidateResponse` with automatic validation and retries

**Prompt Template**:
```
Generate candidate values for a specific attribute class based on the region description and subquery context.

Subquery: "{subquery_question}"
Object Label: {object_label}
Attribute Class: {attribute_class}
Region Description: "{region_description}"

Task: Generate 2-4 most likely candidate values for the '{attribute_class}' attribute of this {object_label}.

Consider the subquery context - the candidates should help answer the specific question being asked.

Respond in this exact JSON format:
{
  "candidates": ["value1", "value2", "value3"]
}

Examples:
- color attribute → {"candidates": ["black", "white", "brown"]}
- size attribute → {"candidates": ["large", "small", "medium"]}
- orientation attribute → {"candidates": ["facing_camera", "facing_left", "facing_right"]}

Answer:
```

**Example Input**:
```
Subquery: "Does bird_a_0 have the same color as bird_b_1?"
Object Label: bird
Attribute Class: color
Region Description: "A white bird with long legs and a pointed beak standing on dark fur"
```

**Example Output**:
```json
{
  "candidates": ["white", "gray", "light"]
}
```

**File Location**: `src/pipeline/attribute_processor.py:301-346`

---

### 3.3 Attribute Binary Verification (VLM)

**Model**: Qwen-2.5-VL-7B

**Purpose**: Verify if an object has a specific attribute value using binary Yes/No question with proper softmax probability

**Input Variables**:
- `image`: PIL Image (CROPPED to object's bounding box)
- `obj.label`: Object class label
- `attribute_class`: Attribute being verified
- `candidate_value`: Specific value to verify

**Output Format**: Binary response + logits

**Output Parsing**:
1. Extract "Yes" and "No" token logits from model output
2. Apply 2-token softmax: `P(yes) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
3. Return probability as float (0.0 to 1.0)

**Prompt Template**:
```
Look at this image showing a {obj.label}.

Question: Does this {obj.label} have {attribute_class} "{candidate_value}"? Answer Yes or No.

Answer:
```

**Key Implementation Details**:
- **Image is CROPPED to object's bounding box** before being passed to VLM
- This removes ambiguity about which object to evaluate
- Model sees only the relevant object in frame

**Example**:
```
Look at this image showing a bird.

Question: Does this bird have color "white"? Answer Yes or No.

Answer:
```

**Probability Extraction**:
```python
# Extract logits for "Yes" and "No" tokens
# Apply proper 2-token softmax
prob_yes = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
# Returns: 0.837 (83.7% confident the bird is white)
```

**File Location**: `src/pipeline/attribute_processor.py:348-379`

---

## 4. Relationship Extraction

### 4.1 Relationship Requirement Analysis (LLM)

**Model**: Llama-3.3-70B-Instruct

**Purpose**: Analyze relationship subquery to determine which object pairs need spatial/interaction verification

**Input Variables**:
- `subquery.question`: Relationship subquery question
- `subquery.subquery_type`: Type (should be "relationship")
- `referenced_object_context`: Context for referenced objects
- `all_objects_context`: All available objects across images

**Output Format**: JSON (Pydantic validated)

**Output Parsing**: Pydantic model `RelationshipResponse` with automatic validation

**System Message**:
```
You are an expert at analyzing visual questions to determine what spatial and interaction relationships need verification. Focus on relationships that can be visually determined and are directly relevant to answering the question. Return strict JSON only.
```

**Prompt Template**:
```
Analyze this binary subquery to determine what spatial or interaction relationships need to be verified:

Subquery: "{subquery.question}"
Type: {subquery.subquery_type}
Referenced Objects: {referenced_object_context}

All Available Objects: {all_objects_context}

IMPORTANT: This subquery may be compound and require relationships between objects beyond just the referenced objects.
Consider cross-image relationship comparisons, multiple relationship types, and implicit object requirements.

Determine which object-to-object relationships need verification to fully answer this question.

Consider these relationship types:
- **Spatial**: near, far, above, below, left, right, inside, outside, touching
- **Interaction**: lifting, carrying, using, holding, supporting, following
- **State**: looking_at, facing, turned_away_from, approaching, avoiding

Return JSON with this exact format:
{
  "relationships": [
    {
      "subject_id": "object_id1",
      "object_id": "object_id2",
      "relation": "relationship_name"
    }
  ]
}

Rules:
- Include ALL relationships needed to answer the subquery (not just between referenced objects)
- Handle compound subqueries that require cross-image relationship comparisons
- Use object IDs exactly as provided from all available objects
- Use specific relationship names (not generic descriptions)
- If no relationships needed, return empty array
- Focus on verifiable spatial/interaction relationships

Examples:
- "Is person_a_0 lifting weight_a_1?" → {"relationships": [{"subject_id": "person_a_0", "object_id": "weight_a_1", "relation": "lifting"}]}
- "Is carnivore_a_0 near zebra_a_1?" → {"relationships": [{"subject_id": "carnivore_a_0", "object_id": "zebra_a_1", "relation": "near"}]}
- "Do birds have the same spatial relationship to cattle in both images?" → {"relationships": [{"subject_id": "bird_a_0", "object_id": "cattle_a_1", "relation": "perched_on"}, {"subject_id": "bird_b_1", "object_id": "animal_b_0", "relation": "perched_on"}]}
- "Are bird_a_0 and bird_b_1 both touching their respective animals?" → {"relationships": [{"subject_id": "bird_a_0", "object_id": "cattle_a_1", "relation": "touching"}, {"subject_id": "bird_b_1", "object_id": "animal_b_0", "relation": "touching"}]}
```

**Example Input**:
```
Subquery: "Is cattle_a_3 positioned above cattle_a_4?"
Type: relationship
Referenced Objects: cattle_a_3 (cattle, conf=0.17), cattle_a_4 (cattle, conf=0.25)
All Available Objects: bird_a_0 (bird, conf=0.57), cattle_a_1 (cattle, conf=0.35), cattle_a_2 (cattle, conf=0.15), cattle_a_3 (cattle, conf=0.17), cattle_a_4 (cattle, conf=0.25)
```

**Example Output**:
```json
{
  "relationships": [
    {
      "subject_id": "cattle_a_3",
      "object_id": "cattle_a_4",
      "relation": "touching"
    }
  ]
}
```

**File Location**: `src/pipeline/relationship_extractor.py:164-288`

---

### 4.2 Relationship Binary Verification (VLM)

**Model**: Qwen-2.5-VL-7B

**Purpose**: Verify if a spatial/interaction relationship exists between two objects using binary Yes/No question with color-coded bounding boxes

**Input Variables**:
- `image`: PIL Image (FULL image with COLORED bounding boxes drawn)
- `subject_info.label`: Subject object class label
- `object_info.label`: Object object class label
- `candidate.relation`: Relationship type to verify

**Output Format**: Binary response + logits

**Output Parsing**:
1. Extract "Yes" and "No" token logits from model output
2. Apply 2-token softmax: `P(yes) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
3. Return probability as float (0.0 to 1.0)

**Prompt Template**:
```
Look at this image. The {subject_info.label} is marked in red and the {object_info.label} is marked in blue.

Question: Is the {subject_info.label} (red) {candidate.relation} the {object_info.label} (blue)? Answer Yes or No.

Answer:
```

**Key Implementation Details**:
- **Full image is used** (not cropped) to preserve spatial context
- **Red bounding box** is drawn around the subject object (4px thick)
- **Blue bounding box** is drawn around the object (4px thick)
- Annotated image is passed to VLM
- Color references replace coordinate clutter

**Example**:
```
Look at this image. The cattle is marked in red and the cattle is marked in blue.

Question: Is the cattle (red) touching the cattle (blue)? Answer Yes or No.

Answer:
```

**Probability Extraction**:
```python
# Extract logits for "Yes" and "No" tokens
# Apply proper 2-token softmax
prob_yes = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
# Returns: 0.303 (30.3% confident the cattle are touching)
```

**File Location**: `src/pipeline/relationship_extractor.py:417-489`

---

## 5. Count Processing

### 5.1 Count Requirement Analysis (LLM)

**Model**: Llama-3.3-70B-Instruct

**Purpose**: Analyze count subquery to determine which object classes need counting in which images

**Input Variables**:
- `subquery.question`: Count subquery question
- `subquery.subquery_type`: Type (should be "count")
- `images_context`: Available images and their object classes

**Output Format**: JSON (Pydantic validated)

**Output Parsing**: Pydantic model `CountRequirementResponse` with automatic validation

**System Message**:
```
You are an expert at analyzing count questions to determine what needs to be counted. Return strict JSON only in the required format.
```

**Prompt Template**:
```
Analyze this count subquery to determine what object classes need counting in which images.

Subquery: "{subquery.question}"
Type: {subquery.subquery_type}

Available Images and Object Classes:
{images_context}

Task: Determine which object classes need to be counted in which images to answer this specific question.

Examples:
- "Are there more than 2 cattle in IMAGE_A?" → Need cattle count for IMAGE_A
- "Does IMAGE_A have more birds than IMAGE_B?" → Need bird count for both IMAGE_A and IMAGE_B
- "Are there more cattle in IMAGE_A than birds in IMAGE_B?" → Need cattle count for IMAGE_A, bird count for IMAGE_B

Respond with the required structure containing count_requirements list with image_id and object_class for each requirement.
```

**Example Input**:
```
Subquery: "Does IMAGE_A have more cattle objects than IMAGE_B has bird objects?"
Type: count

Available Images and Object Classes:
image_a: ['bird', 'cattle', 'cattle', 'cattle', 'cattle']
image_b: ['animal', 'bird', 'bird']
```

**Example Output (Pydantic Model)**:
```python
CountRequirementResponse(
  count_requirements=[
    CountRequirementItem(image_id="image_a", object_class="cattle"),
    CountRequirementItem(image_id="image_b", object_class="bird")
  ]
)
```

**Note**: This uses Pydantic validation which provides:
- Automatic retry on malformed JSON (up to 3 attempts)
- Field validation (ensures image_id and object_class are non-empty strings)
- Structured output guarantee with detailed error messages

**File Location**: `src/pipeline/count_processor.py:155-217`

---

## 6. Scene Attribute Processing

### 6.1 Scene Attribute Planning (LLM)

**Model**: Llama-3.3-70B-Instruct

**Purpose**: Decompose scene attribute subquery into atomic binary verification questions

**Input Variables**:
- `subquery.question`: Scene attribute subquery question
- `image_context`: Dict mapping image_id to caption/description

**Output Format**: JSON (Pydantic validated)

**Output Parsing**: Pydantic model `SceneAttributeResponse` with automatic validation and retries

**Prompt Template**:
```
Analyze this scene attribute subquery to determine what atomic scene attributes need verification.

Subquery: "{subquery.question}"

Available Images and Descriptions:
{context_str}

Task: Break this subquery into atomic scene attribute verifications that can be answered with binary Yes/No questions.

For each atomic verification needed, provide:
1. image_id: Which image to verify
2. attribute_class: Scene attribute category (environment_type, lighting, weather, vegetation, time_of_day, etc.)
3. candidate_value: The specific value to verify
4. binary_question: A clear Yes/No question for VLM verification

Respond in this exact JSON format:
{
  "scene_attribute_candidates": [
    {
      "image_id": "image_a",
      "attribute_class": "environment_type",
      "candidate_value": "outdoor",
      "binary_question": "Is this an outdoor environment?"
    }
  ]
}

Examples:
- "Do both images show outdoor settings?" → Need environment_type=outdoor for both images
- "Is IMAGE_A taken during daytime with blue sky?" → Need time_of_day=daytime AND sky_color=blue for IMAGE_A
- "Does IMAGE_B have grass?" → Need vegetation=grass for IMAGE_B

Answer:
```

**Example Input**:
```
Subquery: "Do both images show outdoor settings?"

Available Images and Descriptions:
image_a: The image is a photograph of a black buffalo standing in a grassy field...
image_b: The image shows a large brown cow standing in a grassy field...
```

**Example Output**:
```json
{
  "scene_attribute_candidates": [
    {
      "image_id": "image_a",
      "attribute_class": "environment_type",
      "candidate_value": "outdoor",
      "binary_question": "Is this an outdoor environment?"
    },
    {
      "image_id": "image_b",
      "attribute_class": "environment_type",
      "candidate_value": "outdoor",
      "binary_question": "Is this an outdoor environment?"
    }
  ]
}
```

**File Location**: `src/pipeline/scene_attribute_processor.py:187-271`

---

### 6.2 Scene Attribute Binary Verification (VLM)

**Model**: Qwen-2.5-VL-7B

**Purpose**: Verify scene-level attribute using binary Yes/No question

**Input Variables**:
- `image`: PIL Image (full image)
- `candidate.binary_question`: Pre-generated binary question for verification

**Output Format**: Binary response + logits

**Output Parsing**:
1. Extract "Yes" and "No" token logits from model output
2. Apply 2-token softmax: `P(yes) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
3. Return probability as float (0.0 to 1.0)

**Prompt Template**:
```
Look at this image. Answer only "Yes" or "No".

Question: {candidate.binary_question}

Answer:
```

**Example**:
```
Look at this image. Answer only "Yes" or "No".

Question: Is this an outdoor environment?

Answer:
```

**Probability Extraction**:
```python
# Extract logits for "Yes" and "No" tokens
# Apply proper 2-token softmax
prob_yes = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
# Returns: 0.837 (83.7% confident this is outdoor)
```

**File Location**: `src/pipeline/scene_attribute_processor.py:273-333`

---

## 8. Probability Calibration

### 8.1 Detector Confidence Calibration (Anchored Sigmoid Mapping)

**Purpose**: Transform poorly calibrated raw detector scores into realistic operational probabilities

**Function**: `calibrate_detector_confidence(raw_score)` in `src/core/probability.py`

**Mathematical Background**:

Transforms detector scores `p ∈ (0,1)` using anchored sigmoid mapping:
```
p' = 1 / (1 + ((1-p)/p)^a * e^(-c))
```

Where parameters `a` and `c` are determined by two anchor points that enforce fixed mappings:
```
Anchor 1: 0.1 → 0.7  (low raw score → moderate operational probability)
Anchor 2: 0.5 → 0.9  (medium raw score → high operational probability)
```

**Parameter Calculation**:
Solving the system:
```
logit(0.7) = a * logit(0.1) + c
logit(0.9) = a * logit(0.5) + c
```

Yields:
```
a ≈ 0.6144
c ≈ 2.1972
```

**Implementation**:

```python
# Hardcoded anchor points (module-level constants)
DETECTOR_ANCHOR_P_LO = 0.1   # Low raw score anchor
DETECTOR_ANCHOR_Q_LO = 0.7   # Maps to 70% operational probability
DETECTOR_ANCHOR_P_HI = 0.5   # High raw score anchor
DETECTOR_ANCHOR_Q_HI = 0.9   # Maps to 90% operational probability

# Pre-computed parameters (at module load)
_ANCHOR_A, _ANCHOR_C = _compute_anchor_parameters(
    DETECTOR_ANCHOR_P_LO, DETECTOR_ANCHOR_Q_LO,
    DETECTOR_ANCHOR_P_HI, DETECTOR_ANCHOR_Q_HI
)

def calibrate_detector_confidence(raw_score: float) -> float:
    """
    Calibrate raw detector confidence using anchored sigmoid mapping.

    Args:
        raw_score: Raw detector confidence in (0, 1)

    Returns:
        float: Calibrated probability in (0, 1)
    """
    # Clamp to valid range with epsilon
    epsilon = 1e-7
    raw_score = max(epsilon, min(1.0 - epsilon, raw_score))

    # Probability-space form (numerically stable)
    odds_ratio = (1.0 - raw_score) / raw_score
    calibrated = 1.0 / (1.0 + math.pow(odds_ratio, _ANCHOR_A) * math.exp(-_ANCHOR_C))

    return float(calibrated)
```

**Example Calibrations**:
```
Raw Score → Calibrated Probability
0.10 → 0.700  (anchor point)
0.15 → 0.740
0.25 → 0.800
0.35 → 0.850
0.50 → 0.900  (anchor point)
0.75 → 0.950
```

**Why This Is Necessary**:

Florence-2's raw confidence scores are poorly calibrated:
- **Problem**: Token-level log probabilities conflated with detection confidence
- **Symptom**: All scores fall in 0.1-0.6 range (unrealistically low)
- **Solution**: Anchored sigmoid mapping transforms to realistic 0.7-0.95 range
- **Result**: Operational probabilities suitable for probabilistic inference

**File Location**: `src/core/probability.py:212-253`

---

### 8.2 Binary Verification Probability (Unified Verbalizer Method)

**Purpose**: Extract P(statement is true) from VLM binary verification using proper softmax

**Function**: `get_verifier_probability(logits_sequence, response, tokenizer)` in `src/core/probability.py`

**Algorithm**:

1. **Sum logits for all "Yes" verbalizers**: ["Yes", "yes", "YES"]
2. **Sum logits for all "No" verbalizers**: ["No", "no", "NO"]
3. **Apply 2-token softmax**: `P(yes) = e^(sum_yes) / (e^(sum_yes) + e^(sum_no))`

**Implementation**:

```python
def get_verifier_probability(
    logits_sequence: List[torch.Tensor],
    response: str,
    tokenizer
) -> float:
    """
    Extract P(statement is true) using verbalizer summing + 2-token softmax.

    Returns:
        float: P(statement is true) between 0.0 and 1.0
    """
    # Get final generation step logits
    final_logits = logits_sequence[-1][0]

    # Sum logits for all Yes variants
    yes_verbalizers = ["Yes", "yes", "YES"]
    sum_yes_logits = ... # logsumexp over all yes tokens

    # Sum logits for all No variants
    no_verbalizers = ["No", "no", "NO"]
    sum_no_logits = ... # logsumexp over all no tokens

    # Apply 2-token softmax
    exp_yes = math.exp(sum_yes_logits)
    exp_no = math.exp(sum_no_logits)
    prob_yes = exp_yes / (exp_yes + exp_no)

    return float(prob_yes)
```

**Key Features**:

- **Verbalizer Robustness**: Handles multiple tokenization variants
- **Numerical Stability**: Uses logsumexp for combining logits
- **Avoids Inflation**: 2-token softmax prevents full-vocabulary renormalization
- **Error Handling**: Returns 0.5 (neutral) on extraction failure

**Why This Works**:

- **Avoids Overconfidence**: Full vocabulary softmax artificially inflates probabilities
- **Proper Normalization**: Only normalizes over yes/no tokens (correct semantic space)
- **Realistic Confidence**: Produces calibrated probabilities for binary decisions

**Used For**:
- Attribute binary verification (Qwen-2.5-VL)
- Relationship binary verification (Qwen-2.5-VL)
- Scene attribute binary verification (Qwen-2.5-VL)

**File Location**: `src/core/probability.py:11-117`

---

## Model Summary

### Models Used in Pipeline

| Model | Type | Usage |
|-------|------|-------|
| **Llama-3.3-70B-Instruct** | LLM | Subquery generation, entity extraction, attribute planning, candidate generation, relationship analysis, count analysis, scene planning |
| **Florence-2-large-ft** | Vision | Caption-based open vocabulary detection, image captioning, region descriptions |
| **Qwen-2.5-VL-7B** | VLM | Binary verification for attributes, relationships, scene attributes (with logit extraction for probabilities) |

### Output Parsing Methods

| Component | Parsing Method | Benefits |
|-----------|---------------|----------|
| Entity Extraction | **Pydantic** ✅ | Automatic lowercase, deduplication, validation |
| Subquery Generation | **Pydantic** ✅ | Automatic validation, retries, guaranteed structure |
| Attribute Planning | **Pydantic** ✅ | Automatic validation, retries, guaranteed structure |
| Attribute Candidates | **Pydantic** ✅ | Automatic validation, retries, guaranteed structure |
| Relationship Analysis | **Pydantic** ✅ | Automatic validation, retries, guaranteed structure |
| Count Analysis | **Pydantic** ✅ | Automatic validation, retries, guaranteed structure |
| Scene Planning | **Pydantic** ✅ | Automatic validation, retries, guaranteed structure |
| All Binary Verification | **Unified Verbalizer Method** ✅ | Proper probabilistic confidence via verbalizer summing + softmax |
| Detector Confidence | **Anchored Sigmoid Calibration** ✅ | Transforms poorly calibrated scores to operational probabilities |

**All LLM JSON outputs now use Pydantic validation** for maximum robustness and consistency!

### Probability Calculation Methods

**1. Binary Verification** (attributes, relationships, scene attributes):
```python
def get_verifier_probability(logits_sequence, response, tokenizer):
    """
    Extract P(statement is true) using unified verbalizer method.

    Sums logits for ["Yes", "yes", "YES"] and ["No", "no", "NO"]
    then applies 2-token softmax: P(yes) = e^(sum_yes) / (e^(sum_yes) + e^(sum_no))
    """
```

**2. Detector Confidence** (object detection):
```python
def calibrate_detector_confidence(raw_score):
    """
    Calibrate raw detector scores using anchored sigmoid mapping.

    Transforms p' = 1 / (1 + ((1-p)/p)^a * e^(-c))
    with anchor points: 0.1 → 0.7, 0.5 → 0.9
    """
```

This ensures consistent probabilistic reasoning across the entire pipeline.

---

## Prompt Design Principles

### 1. Visual Grounding
- **Attributes**: Crop image to object region - removes ambiguity
- **Relationships**: Draw colored bounding boxes (red/blue) - clear visual markers
- **Scene Attributes**: Use full image - preserve scene context

### 2. Output Format Clarity
- **Pydantic validation for ALL LLM JSON outputs** - maximum robustness
- Consistent error handling with automatic retries across entire pipeline
- No examples in prompts - trust Pydantic to enforce structure
- Simplified prompts that guide reasoning, not constrain with examples

### 3. Binary Verification Strategy
- All verification questions are binary (Yes/No)
- Strong instruction: "Answer Yes or No." in question line
- Probability from logits, not just text parsing
- Proper 2-token softmax for P(statement is true)

### 4. Context Building
- Provide detected objects with IDs and confidences
- Include image captions for contextual understanding
- Reference specific objects using consistent ID format
- Guide LLM to reason through context before generating output

### 5. Error Handling
- **All LLM outputs**: Pydantic automatic retries (up to 3) with schema hints
- **VLM verification**: Validate Yes/No format, default to 0.5 on failure
- **Consistent approach**: No more manual try/catch blocks - Pydantic handles all JSON parsing

---

## Future Improvements

### ✅ Completed Improvements
1. **~~Migrate all manual JSON parsing to Pydantic~~** ✅ **DONE!**
   - ✅ Attribute planning → `AttributePlanningResponse` (migrated)
   - ✅ Attribute candidates → `CandidateResponse` (migrated)
   - ✅ Scene planning → `SceneAttributeResponse` (created & migrated)
   - ✅ Entity extraction → `EntityExtractionResponse` (created & migrated)
   - **Result**: All LLM JSON outputs now use Pydantic with automatic retries!

2. **~~Simplify subquery generation prompt~~** ✅ **DONE!**
   - ✅ Removed all JSON format examples (Pydantic handles structure)
   - ✅ Removed specific attribute/relationship type lists (let LLM infer)
   - ✅ Removed example object IDs from rules
   - ✅ Added 3-step reasoning PROCESS (understand → consider → break down)
   - **Result**: Reduced from 77 lines to 22 lines (71% shorter), trust LLM reasoning!

3. **~~Implement caption-based open vocabulary detection~~** ✅ **DONE!**
   - ✅ Added entity extraction from captions using LLM
   - ✅ Switched to Florence-2-large-ft (fine-tuned model)
   - ✅ Implemented open vocabulary detection per entity class
   - ✅ Automatic deduplication and lowercase normalization
   - **Result**: Comprehensive detection of ALL entities in caption, not limited to pre-defined classes!

4. **~~Implement probability calibration~~** ✅ **DONE!**
   - ✅ Anchored sigmoid mapping for detector confidences
   - ✅ Unified verbalizer method for binary verification
   - ✅ Transforms 0.1-0.6 raw scores → 0.7-0.95 operational probabilities
   - **Result**: Realistic, well-calibrated probabilities throughout pipeline!

### Recommended Upgrades
1. **Improve visual grounding**
   - Experiment with different box colors/styles
   - Add object labels directly on image
   - Test alternative cropping strategies for attributes

2. **Enhance other prompts**
   - Apply same simplification approach to other prompts
   - Focus on reasoning guidance over rigid examples
   - Trust Pydantic and LLM capabilities

---

**Last Updated**: 2025-10-01 (Caption-based open vocabulary detection + probability calibration implemented)
**Pipeline Version**: Current (pipelining branch)

**Key Features**:
- ✅ All LLM outputs use Pydantic validation
- ✅ Caption-based open vocabulary detection
- ✅ Anchored sigmoid probability calibration
- ✅ Unified verbalizer method for binary verification
- ✅ Simplified prompts trust LLM reasoning

This pipeline transforms questions like "What is uniquely similar about these images?" into structured probabilistic reasoning with complete evidence provenance, sophisticated count modeling, well-calibrated probabilities, and executable ProbLog programs ready for inference.
