# PROVE
Programmatic Reasoning Over Visual Evidence

## Getting Started
Set up virtual environment (If you haven't already)
- `conda create -n PROVE python=3.10`

Activate virtual environment
- `conda activate PROVE`

Download deepseek
- `git clone git@github.com:deepseek-ai/DeepSeek-VL2.git`

Downlad deepseek requirements
- `cd DeepSeek-VL2`
- `pip install .`
- `cd ..`
- `rm -Rf DeepSeek-VL2`

Download requirements
- `pip install -r requirements.txt`

## Problems
- YOLO is mislabeling objects and missing other key objects
- Grounding Dino is missing objects passed in, giving low confidence, and not using labels properly
- Grounding DINO is broken
- Bunch of random warnings possibly related to transformers

## Hyperparameters
- model checkpoints

## Models
- GroundingDino - https://huggingface.co/IDEA-Research/grounding-dino-base
- Blip - https://huggingface.co/Salesforce/blip2-flan-t5-xl
- Vilt - https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
- Sam - https://huggingface.co/facebook/sam-vit-base
- Yolo - https://github.com/ultralytics/ultralytics
- Deepseek - https://github.com/deepseek-ai/DeepSeek-VL2
