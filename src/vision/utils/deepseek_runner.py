import sys
import json
import argparse
from pathlib import Path

from src.vision.deepseek_vl2 import DeepSeekVL2

def run_deepseek(image_path: str) -> str:
    try:
        model = DeepSeekVL2()
        result = model.list_and_bound_objects(image_path)
        return result
    except Exception as e:
        print(f"[START_DEEPSEEK_VL2]FAILURE[END_DEEPSEEK_VL2]")

def main():
    parser = argparse.ArgumentParser(description="Run DeepseekVL2 object detection")
    parser.add_argument("image_path", type=str, help="Path to the image to process")
    args = parser.parse_args()
    
    result = run_deepseek(args.image_path)
    print(f"[START_DEEPSEEK_VL2]{result}[END_DEEPSEEK_VL2]")   

if __name__ == "__main__":
    main()