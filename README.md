# PROVE

## STEP 1 - Get Image Content
YOLO -> list 1 of entities
BLIP -> Captioner of nouns -> Grounding DINO -> list 2 of entities



Need a solution to open vocabulary instance-segmentation
- Thinking SAM
- Pass results of SAM to some model to predict class and bounding boxe
- Store segment, bounding boxes, and class

1. **Captioner**
   - BLIP

2. **Open Vocabulary Object Detection**
   - Grounding DINO or OWL-ViT

3. **Bounding Boxes**
   - Grounding DINO

4. **Masks**
   - SAM

5. **Spacial Relations**
   - Python image cropping

6. **Descriptive Attributes**
   - *To be determined*

7. **Text & Brand Recognition**
   - *To be determined* (Some sort of OCR model)

## STEP 2 - LLM Program Composer
- Few-shot prompting
- ProbLog/Prolog?

## STEP 3 - ProbLog Executor
- Error --> **STEP 2** + error
- Output response

## STEP 4 - Post-Processor
- Output response (+ confidence level if problog)
- Output rationale (log outputs)

# Models
- https://huggingface.co/IDEA-Research/grounding-dino-base
- https://huggingface.co/Salesforce/blip2-flan-t5-xl
- https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
- https://huggingface.co/facebook/sam-vit-base
- 