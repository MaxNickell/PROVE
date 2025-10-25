# PROVE Pipeline - Simple Test

## Overview

`test_pipeline_simple.py` is a streamlined version of the complete PROVE pipeline that runs all 11 steps from start to finish and generates a final answer.

## Usage

### Basic Run
```bash
python test_pipeline_simple.py
```

### Debug Mode (Detailed Logging)
```bash
DEBUG=true python test_pipeline_simple.py
```

## Pipeline Steps

1. **Image Context Generation** - Generate detailed captions for both images
2. **Object Detection** - Detect objects using Florence-2 open-vocabulary detection
3. **Subquestion Generation** - Generate binary subquestions from ultimate question
4. **Attribute Processing** - Extract visual attributes (color, size, etc.)
5. **Relationship Extraction** - Extract spatial relationships between objects
6. **Count Processing** - Compute probabilistic object counts using Poisson-Binomial
7. **Scene Attribute Processing** - Extract scene-level attributes
8. **ProbLog KB Generation** - Build probabilistic knowledge base
9. **ProbLog Execution** - Execute subquestions against knowledge base
10. **Answer Generation** - Synthesize final answer with explanation
11. **Save Results** - Export to JSON

## Output Files

- `knowledge_base.pl` - ProbLog program with all facts
- `pipeline_results.json` - Complete results in JSON format
- `test_images/*_annotated.png` - Annotated images with bounding boxes (in debug mode)

## Example Output

```
=== PROVE: Visual Reasoning Pipeline ===

Question: Are there a total of 3 birds across both images and is there a bird perched on a large neon green animal?

[1/11] Generating image captions...
  ✓ Captions generated for 2 images

[2/11] Detecting objects...
  ✓ Detected 19 objects across 2 images

[3/11] Generating subquestions...
  ✓ Generated 7 binary subquestions

[4/11] Processing attributes...
  ✓ Extracted 2 attributes

[5/11] Processing relationships...
  ✓ Extracted 6 relationships

[6/11] Processing counts...
  ✓ Computed 2 probabilistic counts

[7/11] Processing scene attributes...
  ✓ No scene attributes to process

[8/11] Building ProbLog knowledge base...
  ✓ Built 27 facts (avg confidence: 0.749)

[9/11] Executing ProbLog reasoning...
  ✓ Executed 7 subquestions

[10/11] Generating final answer...
  ✓ Answer generated

[11/11] Saving results...
  ✓ Results saved to pipeline_results.json

====================================
FINAL ANSWER
====================================

Question:
  Are there a total of 3 birds across both images and is there a bird perched on a large neon green animal?

Answer:
  [Final synthesized answer]

Explanation:
  [Detailed explanation with evidence from subquestion results]

====================================

✓ Pipeline completed successfully!
```

## Differences from Original Test

### Removed:
- Verbose step-by-step debugging output
- Import validation tests
- Intermediate validation checks
- Commented-out code sections

### Added:
- **Complete Steps 9-11** (ProbLog execution and answer generation)
- Clean progress indicators [N/11]
- Simplified output format
- DEBUG environment variable for detailed logging
- Final answer display

### Kept:
- All 11 pipeline steps
- Cache clearing for count_processor
- Error handling
- JSON export

## Requirements

Same as main pipeline:
- Python 3.10+
- All dependencies in requirements.txt
- FORGE_API_KEY in .env for GPT-4o
- Test images in ./test_images/
