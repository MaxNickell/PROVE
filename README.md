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
- Sometimes grabbing the entire image and need to remove those
- Deepseek proving not reliable labeling all the images


## Hyperparameters
- model checkpoints

## Models
- GroundingDino - https://huggingface.co/IDEA-Research/grounding-dino-base
- Blip - https://huggingface.co/Salesforce/blip2-flan-t5-xl
- Vilt - https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
- Sam - https://huggingface.co/facebook/sam-vit-base
- Yolo - https://github.com/ultralytics/ultralytics
- Deepseek-VL2 - https://github.com/deepseek-ai/DeepSeek-VL2




use llm to parse deepseek output

use llm to help determine what relationships we should get

determing query(s) -> finding what we still need -> prolog (python)


LLAVA-SG
https://arxiv.org/html/2408.16224v1?utm_source=chatgpt.com