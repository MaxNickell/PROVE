import os
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.vision.blip2 import Blip2
from src.vision.grounding_dino import GroundingDino
from src.vision.yolo_world import YoloWorld
from PIL import Image, ImageDraw, ImageFont


class Orchestrator:
    def __init__(self, deepseek_env_name: str = "DEEPSEEK_VL2_ENV"):
        self.deepseek_env = deepseek_env_name
    
    def parse_deepseek_output(self,output: str, image_size: Optional[Tuple[int, int]] = None) -> List[Dict[str, object]]:
        TOKEN_RE = re.compile(r"([A-Za-z ]+?)\s*\[\[([0-9\s,]+?)\]\]")
        detections: List[Dict[str, object]] = []
        seen = set()

        for label, coord_str in TOKEN_RE.findall(output):
            coords = [int(v) for v in coord_str.split(",")[:4]]
            if len(coords) != 4:
                continue

            if image_size:
                w, h = image_size
                x1, y1, x2, y2 = coords
                coords = [
                    round(x1 / 999 * w),
                    round(y1 / 999 * h),
                    round(x2 / 999 * w),
                    round(y2 / 999 * h),
                ]

            label = label.split()[-1]
            key = (label.lower(), tuple(coords))
            if key in seen:
                continue
            seen.add(key)

            detections.append(
                {
                    "label": label,
                    "confidence": 1.0,
                    "coordinates": tuple(coords),
                }
            )

        return detections


    def run_deepseek_subprocess(self, image_path: str) -> List[Dict[str, object]]:
        try:
            result = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    self.deepseek_env,
                    "python",
                    "-m",
                    "src.vision.utils.deepseek_runner",
                    image_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            match = re.search(
                r"\[START_DEEPSEEK_VL2\](.*?)\[END_DEEPSEEK_VL2\]",
                result.stdout,
                re.DOTALL,
            )
            if not match:
                raise RuntimeError("Could not find output markers in subprocess response")

            vl2_text = match.group(1)

            with Image.open(image_path) as im:
                w, h = im.size

            return self.parse_deepseek_output(vl2_text, image_size=(w, h))

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to run DeepSeek-VL2: {exc.stderr}") from exc
        except Exception as exc:
            raise RuntimeError(f"Error processing DeepSeek-VL2 output: {exc}") from exc


    def run_captioner_and_grounding(self, image_path: str) -> Dict:
        prompt = (
            "Look at the image and list every distinct object you see.\n"
            "FORMAT RULES – follow all four EXACTLY:\n"
            "1. Output only common-noun words in lowercase (no adjectives, verbs, numbers, or proper nouns).\n"
            "2. Separate nouns with a single comma and one space.\n"
            "3. List each noun only once (no duplicates).\n"
            "4. Do not add any other words, punctuation, or explanations.\n"
            "Example (not related to the image): cat, glove, lamp, car\n"
            "List the nouns now:"
        )

        image = Image.open(image_path).convert("RGB")

        blip2 = Blip2()
        nouns = blip2.caption_and_parse_nouns(image, prompt)
        print(f"Nouns: {nouns}")

        grounding_dino = GroundingDino()
        result = grounding_dino.detect(image, nouns)

        return result
    
    def run_yolo_world(self, image_path: str) -> List[Dict[str, object]]:
        image = Image.open(image_path).convert("RGB")
        yolo_world = YoloWorld()
        result = yolo_world.detect(image)
        return result

    def visualize_detections(self, image_path: str, all_detections: List[Dict[str, List[Dict[str, object]]]], output_path: str = "./out.png"):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        colors = {
            "deepseek": (255, 0, 0),
            "grounding": (0, 255, 0),
            "yolo": (0, 0, 255)
        }

        for model_name, detections in zip(["deepseek", "grounding", "yolo"], all_detections):
            color = colors[model_name]
            
            for det in detections:
                if isinstance(det["coordinates"], (list, tuple)):
                    x1, y1, x2, y2 = det["coordinates"]
                else:
                    continue

                box_coords = [int(x1), int(y1), int(x2), int(y2)]
                
                draw.rectangle(box_coords, outline=color, width=3)
                
                label_text = f"{det['label']} ({det['confidence']:.2f})"
                
                text_width = font.getlength(label_text) if hasattr(font, 'getlength') else len(label_text) * 10
                text_height = 24 
                
                text_box_coords = [box_coords[0], box_coords[1] - text_height, 
                                 box_coords[0] + text_width, box_coords[1]]
                draw.rectangle(text_box_coords, fill=color)
                
                draw.text((box_coords[0], box_coords[1] - text_height), 
                         label_text, fill=(255, 255, 255), font=font)

        image.save(output_path)

    def process_image(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print("\n--------------------------------Running Deepseek--------------------------------")
        deepseek_result = self.run_deepseek_subprocess(image_path)
        print("Deepseek result:")
        print(json.dumps(deepseek_result, indent=2))

        print("\n--------------------------------Running Captioner and Grounding--------------------------------")
        captioner_and_grounding_result = self.run_captioner_and_grounding(image_path)
        print("Captioner and grounding result:")
        print(json.dumps(captioner_and_grounding_result, indent=2))

        print("\n--------------------------------Running Yolo World--------------------------------")
        yolo_world_result = self.run_yolo_world(image_path)
        print("Yolo world result:")
        print(json.dumps(yolo_world_result, indent=2))

        self.visualize_detections(
            image_path,
            [deepseek_result, captioner_and_grounding_result, yolo_world_result]
        )

        return {
            "deepseek": deepseek_result,
            "grounding": captioner_and_grounding_result,
            "yolo": yolo_world_result
        }


        