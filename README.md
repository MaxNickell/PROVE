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
- YOLO is mislabeling objects and missing other key objects
- Grounding Dino is missing objects passed in, giving low confidence, and not using labels properly
- Grounding DINO is broken
- Bunch of random warnings possibly related to transformers
- DeepSeek-VL2 requires an old version of Transformers to run (transformers==4.38.2) but Grounding Dino needs a newer version(transformers>=4.40.0)

## Hyperparameters
- model checkpoints

## Models
- GroundingDino - https://huggingface.co/IDEA-Research/grounding-dino-base
- Blip - https://huggingface.co/Salesforce/blip2-flan-t5-xl
- Vilt - https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
- Sam - https://huggingface.co/facebook/sam-vit-base
- Yolo - https://github.com/ultralytics/ultralytics
- Deepseek-VL2 - https://github.com/deepseek-ai/DeepSeek-VL2


## Ideads
- instruct blip
- grounding dino 1.5
- choosing bounding box from IOU


Step 1: Yolo World -> boxes
Step 2: Blip2 Instruct -> Nouns -> Grounding Dino -> boxes
Step 3: Deepseek vl2 -> boxes
Step 4: Normalize w/ IOU
Step 5: Classify individual boxes with Deepseek vl2