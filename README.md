# PROVE
Programmatic Reasoning Over Visual Evidence

## Getting Started
Set up conda environment for project
- `conda create -n PROVE python=3.10`

Activate conda environment
- `conda activate PROVE`

Download requirements
- `pip install -r requirements.txt`

Deactivate conda environment
- `conda deactivate`

Set up conda environment for deepseek vl2
- `conda create -n DEEPSEEK_VL2_ENV python=3.10`

Activate conda environment
- `conda activate DEEPSEEK_VL2_ENV`

Download deepseek vl2
- `git clone git@github.com:deepseek-ai/DeepSeek-VL2.git`

Downlad deepseek vl2 requirements
- `cd DeepSeek-VL2`
- `pip install .`
- `cd ..`
- `rm -Rf DeepSeek-VL2`
- `pip install "numpy<2.0.0"`

Confirm download
- `pip show deepseek_vl2`

Switch back to main conda environment
- `conda deactivate`
- `conda activate PROVE`


## Problems
- What granualarity of object detection should we choose?
- Not grabbing all the relationships we may need because it does not see the image
    - Maybe we can use the blip and union detection first pass where objects are overlapping (doesn't consider relationships that are far away, so we may need geometry)


## Hyperparameters
- model checkpoints

## Models
- GroundingDino - https://huggingface.co/IDEA-Research/grounding-dino-base
- Blip - https://huggingface.co/Salesforce/blip2-flan-t5-xl
- Vilt - https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
- Sam - https://huggingface.co/facebook/sam-vit-base
- Yolo - https://github.com/ultralytics/ultralytics
- Deepseek-VL2 - https://github.com/deepseek-ai/DeepSeek-VL2


# Pipeline Plan
1. Object Detection
    - Florence-2 -> bbox, class, confidence
2. Attribute Recognition
    - Crop -> Florence-2 -> Description -> LLM -> attributes list
3. Salient Intra-Relationship Questions 
    - List of objects -> LLM -> List of questions for relationships
4. Salient Inter-Comparison Questions
    - List of objects -> LLM -> list of questions for comparisons
5. Intra-Relationships Answers
    - List of questions for relationships from step 3 -> Union Crop + Bounding boxes + Questions -> VLM -> answer
6. Inter-Comparision Answers
    - list of questions for comparisons from step 4 -> Crop on each image + Bounding box + Questions -> VLM -> attributes for each object involved in each image
7. Subqueries
    - Knowledge base + Main Query (context) -> Sub queries
8. Convert knowledge base to ProbLog
    - Objects, Attributes, Inter Relations -> LLM -> Problog Generation Engine -> Problog facts
9. Problog query 
    - Subqueries -> LLM -> Problog Generation Engine -> Problog queries
10. Reasoned Output
    - Problog queries -> Problog Execution Engine -> Outputs + Problog Trace -> LLM -> answer + reasoned explanation
    


