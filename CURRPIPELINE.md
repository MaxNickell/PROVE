# PROVE Pipeline: Current Implementation Guide

## Overview
**PROVE (Probabilistic Reasoning Over Visual Evidence)** is an 11-step subquery-driven architecture that transforms ambiguous visual questions into structured evidence extraction, probabilistic reasoning, and confident answers with complete provenance.

**Core Philosophy**: Break complex comparative questions into specific binary subquestions, extract evidence using multi-modal verification, and synthesize probabilistic answers.

---

## Architecture: 4-Tier Knowledge Framework
- **Objects**: Spatial entities with bounding boxes and confidence scores
- **Attributes**: Object characteristics verified through binary VLM
- **Relationships**: Spatial/interaction relationships between objects
- **Scene Context**: Environmental and counting facts

**Data Structure**: Clean `ImageData` hierarchy - `kb.images[image_id].{objects, attributes, relationships, scene_context}`

---

## Models & Quantization

### Primary Models:
- **Florence-2-large**: Object detection, image captions, region descriptions
- **Llama-3.3-70B-Instruct**: Subquery generation, candidate generation, final synthesis (8-bit quantization via BitsAndBytes)
- **Qwen-2.5-VL-7B-Instruct**: Binary verification for attributes and relationships

### Model Loading:
- **Device Allocation**: Auto device mapping for optimal GPU distribution
- **Memory Optimization**: 8-bit quantization for Llama-3.3-70B to fit memory constraints
- **Lazy Loading**: Models loaded on-demand via ModelManager singleton

---

## Phase-by-Phase Breakdown

### Step 1: Object Extraction
**Goal**: Identify all visual entities with spatial grounding and confidence scores

**Implementation**:
- **Model**: Florence-2-large object detection
- **Method**: `detector.detect(image_path)` → `List[ObjectDetection]`
- **Data Flow**: Image → Bounding boxes + labels + object IDs

**Probability Calculation**:
- **Source**: Florence-2 `compute_transition_scores()` from model logits
- **Range**: 0.0-1.0 object existence confidence
- **Storage**: `ObjectDetection.confidence` field
- **Usage**: Propagated to ProbLog facts as object existence probabilities

**Output**: `ObjectDetection(object_id, label, bbox, confidence)`

---

### Step 2: Image Context Generation
**Goal**: Capture rich scene-level information for contextual reasoning

**Implementation**:
- **Model**: Florence-2-large detailed captioning
- **Method**: `detector.generate_detailed_captions()` → `Dict[str, str]`
- **Data Flow**: Image → Comprehensive scene description

**Probability Calculation**:
- **Source**: Fixed at 1.0 (deterministic caption generation)
- **Rationale**: Scene descriptions treated as ground truth context
- **Storage**: `ImageData.scene_context["caption"]`

**Output**: `{"image_a": "detailed scene description", ...}`

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
- **Validation**: Only essential constraints - object ID existence and JSON structure
- **Trust LLM**: No manual pattern matching or heuristic classification
- **Generalized Examples**: Multiple example patterns per category to guide generation

**Probability Calculation**:
- **Source**: No probability calculation (deterministic text generation)
- **Focus**: Generate verifiable binary questions with accurate object references
- **Validation**: Pydantic type validation + object ID existence check

**Output**: `List[BinarySubquery(question, subquery_type, referenced_objects)]`

---

### Step 4: Attribute Planning
**Goal**: Analyze compound subqueries to determine comprehensive attribute extraction requirements

**Enhanced Implementation**:
- **Model**: Llama-3.3-70B-Instruct with compound subquery analysis
- **Method**: `attribute_planner.determine_required_attributes(subqueries, objects)`
- **Data Flow**: Compound Subqueries + All Available Objects → Comprehensive attribute requirements
- **Compound Handling**: LLM analyzes subqueries for cross-object and cross-image attribute comparisons

**LLM Analysis Enhancement**:
- **Beyond Referenced Objects**: Considers ALL available objects, not just `referenced_objects`
- **Cross-Image Comparisons**: Handles subqueries requiring attributes from multiple images
- **Multi-Object Requirements**: Single subquery can generate multiple AttributeRequirements
- **Examples**:
  - `"Do bird_a_0 and animal_b_0 have the same color?"` → Requires `color` for both objects
  - `"Is cattle_a_1 facing same direction as animal_b_0?"` → Requires `orientation` for both objects

**Probability Calculation**:
- **Source**: No probability calculation (requirement determination)
- **Focus**: Comprehensive mapping of compound subqueries to all required attribute extractions
- **Consolidation**: Merge requirements across multiple compound subqueries

**Output**: `List[AttributeRequirement(image_id, object_id, attribute_classes)]` with complete coverage

---

### Step 5: Attribute Extraction
**Goal**: Extract specific attribute values using 3-stage verification pipeline

**Implementation**:
- **Pipeline**: Florence-2 region description → Llama-3.3-70B candidates → Qwen-2.5-VL binary verification
- **Method**: `attribute_extractor.extract_attributes(image_paths, images, requirements)`
- **Data Flow**: Object regions → Descriptions → Candidate values → Verified attributes

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification via `extract_yes_no_probability_with_proper_softmax(logits, response)`
- **Process**:
  1. Qwen answers "Does this object have X attribute?" → Yes/No + logits
  2. Extract raw logits for "Yes" and "No" tokens specifically (with verbalizer fallbacks)
  3. Apply proper 2-token softmax: `P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
  4. This avoids probability inflation from renormalizing across full vocabulary
- **Error Handling**: Return 0.5 (neutral) when verbalizer extraction fails - no arbitrary confidence values
- **No Filtering**: All results preserved (including low-probability) for ProbLog inference
- **Storage**: `AttributeValue.confidence` field contains P(statement_true) or 0.5 for failed extractions

**Output**: `List[AttributeData(attributes: Dict[str, List[AttributeValue]])]`

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
- **Examples**:
  - `"Do birds have the same spatial relationship to cattle in both images?"` → Extracts bird→cattle relationships in both images
  - `"Are bird_a_0 and bird_b_1 both touching their respective animals?"` → Multiple touching relationships

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification via `extract_yes_no_probability_with_proper_softmax(logits, response)`
- **Process**:
  1. Qwen answers "Is object A [relation] object B?" with bounding box context
  2. Uses bounding boxes: `<box>(x1,y1),(x2,y2)</box>label`
  3. Extract raw logits for "Yes" and "No" tokens specifically (with verbalizer fallbacks)
  4. Apply proper 2-token softmax: `P(relation_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))`
- **Error Handling**: Return 0.5 (neutral) when verbalizer extraction fails - no arbitrary confidence values
- **No Filtering**: All results preserved (including low-probability) for ProbLog inference
- **Storage**: `IntraRelation.probability` field contains P(relation_true) or 0.5 for failed extractions

**Output**: `List[IntraRelation(subject_id, object_id, relation, probability)]`

---

### Step 7: Scene Attribute Processing
**Goal**: Extract scene-level attributes using compound subquery decomposition + Qwen binary verification

**Completely Rewritten Implementation**:
- **New Architecture**: Follows established subquery analysis pattern (like attribute/relationship extractors)
- **Pipeline**: Compound subquery analysis → LLM decomposition into atomic binary questions → Qwen verification → Multiple scene attributes
- **Method**: `context_processor.process_scene_attribute_subqueries(scene_subqueries, image_paths, images)`
- **Data Flow**: Compound Scene Subqueries → LLM atomic decomposition → Multiple binary verifications → Multiple scene attributes per subquery

**Compound Subquery Decomposition**:
- **LLM Analysis**: Breaks compound scene questions into atomic binary verifications
- **Multi-Image Support**: Single subquery can generate scene attribute checks for multiple images
- **Multi-Attribute Support**: Single subquery can generate multiple scene attribute types
- **Examples**:
  - `"Do both images show outdoor settings with grass?"` → 4 atomic questions:
    - `"Is IMAGE_A an outdoor environment?"` (environment_type=outdoor)
    - `"Does IMAGE_A show grass?"` (vegetation=grass)
    - `"Is IMAGE_B an outdoor environment?"` (environment_type=outdoor)
    - `"Does IMAGE_B show grass?"` (vegetation=grass)
  - `"Is IMAGE_A taken during daytime with blue sky?"` → 2 atomic questions:
    - `"Is IMAGE_A taken during daytime?"` (time_of_day=daytime)
    - `"Does IMAGE_A show blue sky?"` (sky_color=blue)

**Probability Calculation**:
- **Source**: Qwen-2.5-VL binary verification via `extract_yes_no_probability_with_proper_softmax(logits, response)`
- **Process**: Each atomic binary question gets individual Qwen verification with proper softmax
- **Error Handling**: Return 0.5 (neutral) when extraction fails - no arbitrary confidence values
- **Storage**: Multiple `SceneAttributeResult` objects per subquery, each with individual confidence

**Note**: Count subqueries are NOT processed in this step - they are skipped for future implementation.

**Output**: `Multiple SceneAttribute results per subquery` stored in `ImageData.scene_context["scene_attributes"]`

---

### Step 8: ProbLog Knowledge Base Construction
**Goal**: Convert all extracted evidence into probabilistic logical facts

**Implementation**:
- **Method**: `problog_builder.build_knowledge_base(images)`
- **Data Flow**: ImageData structure → ProbLog probabilistic facts
- **Fact Types**: Object existence, attribute facts, relationship facts, location facts

**Probability Calculation**:
- **Source**: Preserves original confidence scores from extraction phases
- **Mapping**:
  - Object facts: `prob::object(id, label, image) :- confidence`
  - Attribute facts: `prob::attribute(obj_id, class, value) :- qwen_confidence`
  - Relation facts: `prob::relation(subj_id, obj_id, relation) :- qwen_confidence`
- **Format**: ProbLog probabilistic fact syntax with confidence values

**Output**: `List[ProbLogFact(probability, predicate, arguments)]`

---

### Step 9: Subquery Decomposition to ProbLog
**Goal**: Convert contextual subqueries into executable logical queries

**Implementation**:
- **Model**: LLM decomposes subquestions into ProbLog queries
- **Method**: Generate ProbLog queries over extracted facts
- **Data Flow**: Binary subqueries → ProbLog query syntax

**Probability Calculation**:
- **Source**: No probability calculation (query generation phase)
- **Focus**: Generate syntactically correct ProbLog queries
- **Structure**: Query logic that references probabilistic facts from KB

**Output**: ProbLog queries ready for execution

---

### Step 10: ProbLog Execution and Evidence Tracing
**Goal**: Execute probabilistic reasoning to answer subqueries with evidence trails

**Implementation**:
- **Engine**: ProbLog probabilistic logic programming engine
- **Method**: `problog_executor.execute_subqueries(subqueries, problog_facts)`
- **Data Flow**: ProbLog queries + Probabilistic facts → Query results with probabilities

**Probability Calculation**:
- **Source**: ProbLog inference engine probabilistic computation
- **Method**:
  1. Parse queries and probabilistic facts
  2. Build probabilistic logic program
  3. Compute marginal probability of query being true
  4. Generate supporting fact traces
- **Algorithm**: Probabilistic logic programming inference (weighted model counting)
- **Output**: Query probability + supporting evidence trail

**Output**: `List[SubqueryResult(subquery, probability, supporting_facts, evidence_trail)]`

---

### Step 11: Final Answer Generation
**Goal**: Synthesize ultimate answer using subquery results and evidence

**Implementation**:
- **Model**: Llama-3.3-70B-Instruct
- **Method**: `answer_generator.generate_final_answer(question, subquery_results, image_contexts)`
- **Data Flow**: Ultimate question + Subquery probabilities + Evidence → Final answer with reasoning

**Probability Calculation**:
- **Source**: Aggregation of subquery result probabilities weighted by evidence strength
- **Process**:
  1. Analyze subquery result probabilities and supporting evidence
  2. Weight evidence based on confidence and relevance to ultimate question
  3. Generate overall confidence based on evidence consistency and strength
  4. Map to confidence categories: High (>0.8), Medium (0.5-0.8), Low (<0.5)
- **Synthesis**: LLM combines probabilistic evidence into coherent reasoning chain

**Output**: `AnswerResult(text, explanation, confidence, supporting_evidence)`

---

## Probability Flow Architecture

```
Florence-2 Object Confidence (0.0-1.0)
    ↓
Qwen-2.5-VL Proper Softmax Binary Verification:
  • Extract raw logits for "Yes" and "No" tokens specifically
  • Use verbalizer fallbacks: ["Yes", "yes", "YES"], ["No", "no", "NO"] for robustness
  • Apply 2-token softmax: P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))
  • Avoids probability inflation from full-vocabulary renormalization
  • Preserve ALL results (no confidence filtering)
    ↓
ProbLog Probabilistic Facts (preserve all confidence values including low-probability)
    ↓
ProbLog Inference Engine (probabilistic logic programming with complete evidence)
    ↓
Subquery Result Probabilities (query marginal probabilities from complete fact set)
    ↓
Final Answer Confidence (evidence-weighted aggregation from all probabilistic evidence)
```

---

## Key Implementation Details

### Clean Architecture (Recent Refactor):
- **No Legacy Code**: Eliminated all ObjectReference compatibility methods
- **ImageData Hierarchy**: Clean `kb.images[image_id].{objects, attributes, relationships, scene_context}`
- **Simple References**: Use integer object indices instead of complex reference objects
- **Research-Grade**: All components accept ImageData directly, no format conversions

### Binary Verification Strategy:
- **Method**: All attributes, relationships, and scene attributes verified via binary Yes/No questions with explicit format enforcement
- **Proper Softmax Calculation**: Extract raw logits for "Yes"/"No" tokens, apply 2-token softmax: P(statement_true) = e^(z_yes) / (e^(z_yes) + e^(z_no))
- **Verbalizer Fallbacks**: Use ["Yes", "yes", "YES"] and ["No", "no", "NO"] for tokenization robustness, prefer single-token variants
- **Avoids Inflation**: No full-vocabulary renormalization that artificially inflates confidence scores
- **Principled Error Handling**: Return 0.5 (neutral) for failed extractions - no arbitrary confidence injection
- **No Filtering**: Preserve ALL results including low-probability ones for complete ProbLog inference
- **Bounding Box Context**: Include spatial context in relationship verification questions
- **Format Validation**: Enforce strict "Yes" or "No" responses with warning for invalid formats

### Memory & Performance:
- **8-bit Quantization**: Llama-3.3-70B uses BitsAndBytesConfig for memory efficiency
- **Auto Device Mapping**: Optimal GPU distribution for multi-GPU setups
- **Lazy Loading**: Models loaded on-demand via singleton pattern
- **Pipeline Warnings**: Handle temperature parameter warnings gracefully

This pipeline transforms questions like "What is uniquely similar about these images?" into structured probabilistic reasoning with complete evidence provenance and confidence quantification.