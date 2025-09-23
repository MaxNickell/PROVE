# PROVE - Probabilistic Reasoning Over Visual Evidence

## Pipeline Overview
**4-Tier Knowledge Framework**: Objects + Attributes + Relationships + Scene Context
*(Note: Attributes encompass all object characteristics - no separate "properties" category)*

Extract objects -> Generate subqueries -> Extract attributes & relationships & context -> Build knowledge base -> Answer questions

---

## Step 1: Object Extraction
**Intuition**: Identify all visual entities across images with spatial grounding  
**Process**: Use Florence-2 to detect objects and assign unique identifiers  
**Implementation**: Florence-2 object detection with bounding boxes, assign IDs like `person_a_0`, `weight_a_1`

---

## Step 2: Image Context Generation
**Intuition**: Capture rich scene-level information for contextual reasoning  
**Process**: Generate comprehensive image descriptions using Florence-2  
**Implementation**: Use Florence-2 detailed captions to describe complete images

---

## Step 3: Subquery Generation
**Intuition**: Break ambiguous questions into specific binary subquestions  
**Process**: LLM generates subquestions across 3 types: attributes, relationships, scene context  
**Implementation**: LLM uses image captions + objects to create Yes/No questions covering all knowledge types

---

## Step 4: Attribute Planning
**Intuition**: Identify which attribute classes need extraction for specific objects  
**Process**: LLM analyzes subquestions to determine required attributes per object  
**Implementation**: Map attribute requirements to objects, consolidate across all subquestions

---

## Step 5: Attribute Extraction
**Intuition**: Extract specific attribute values using visual descriptions and VLM verification  
**Process**: Florence region description → LLM candidates → Binary VLM verification  
**Implementation**: Crop object regions, generate Florence captions, LLM creates candidates, Qwen verifies with binary questions

---

## Step 6: Relationship Extraction
**Intuition**: Extract spatial and interaction relationships needed to answer subquestions  
**Process**: Subquestions + objects → LLM relationship candidates → Binary VLM verification  
**Implementation**: LLM determines required relationships, generates candidates, Qwen verifies with binary questions using object bounding boxes

---

## Step 7: Scene Context Processing
**Intuition**: Extract scene-level facts like object counts and environmental properties
**Process**: Context subqueries → Scene analysis → Context facts with confidence
**Implementation**: Process context-type subqueries to generate scene facts (counts, environment, properties)

---

## Step 8: ProbLog Knowledge Base Construction
**Intuition**: Convert all extracted evidence into probabilistic logical facts  
**Process**: Transform extracted data into structured ProbLog facts with per-image indexing  
**Implementation**: Convert objects, attributes, relationships, and context into probabilistic facts

---

## Step 9: Subquery Decomposition to ProbLog
**Intuition**: Convert contextual subqueries into executable logical queries  
**Process**: LLM decomposes subquestions into ProbLog queries over extracted facts  
**Implementation**: Generate ProbLog queries that use attribute/relationship facts to answer each subquestion

---

## Step 10: ProbLog Execution and Evidence Tracing
**Intuition**: Execute probabilistic reasoning to answer subqueries with evidence trails  
**Process**: Run ProbLog inference and capture supporting facts  
**Implementation**: Execute each ProbLog query, capture probability results and supporting facts for explanations

---

## Step 11: Final Answer Generation
**Intuition**: Synthesize ultimate answer using subquery results and evidence  
**Process**: LLM combines subquery answers with evidence trails for final reasoning  
**Implementation**: Generate final answer with confidence and clear reasoning chain from evidence to conclusion

---

## Three-Model Integration
- **Florence-2**: Object detection, image captions, region descriptions
- **LLM**: Subquery generation, candidate generation, final synthesis
- **Qwen-2.5-VL**: Binary verification for all attributes and relationships

This pipeline transforms "Who is more powerful?" into structured evidence extraction -> logical reasoning -> confident answers with complete provenance.
