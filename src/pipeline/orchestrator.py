import os
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.vision.yolo_world import YoloWorld
from src.vision.yoloe import YoloE
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil

class Orchestrator:
    def __init__(self, deepseek_env_name: str = "DEEPSEEK_VL2_ENV"):
        self.deepseek_env = deepseek_env_name
        self.iou_threshold_pass_1 = 0.7
        self.iou_threshold_pass_2 = 0.6
    
    def parse_deepseek_list_and_bound_output(self,output: str, image_size: Optional[Tuple[int, int]] = None) -> List[Dict[str, object]]:
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


    def run_deepseek_list_and_bound_subprocess(self, image_path: str) -> List[Dict[str, object]]:
        try:
            result = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    self.deepseek_env,
                    "python",
                    "-m",
                    "src.vision.utils.deepseek_list_and_bound",
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

            return self.parse_deepseek_list_and_bound_output(vl2_text, image_size=(w, h))

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to run DeepSeek-VL2: {exc.stderr}") from exc
        except Exception as exc:
            raise RuntimeError(f"Error processing DeepSeek-VL2 output: {exc}") from exc
    
    def run_deepseek_identify_subprocess(self, crop_list_json: str) -> list[str]:
        try:
            result = subprocess.run(
                [
                    "conda", "run", "-n", self.deepseek_env,
                    "python", "-m", "src.vision.utils.deepseek_identify",
                    crop_list_json,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            m = re.search(r"\[START_DEEPSEEK_VL2](.*?)\[END_DEEPSEEK_VL2]", result.stdout, re.DOTALL)
            if not m:
                raise RuntimeError("No identify markers in subprocess output")
            return json.loads(m.group(1))   # -> list[str]

        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"DeepSeek identify failed: {exc.stderr}") from exc
    
    def run_yolo_world(self, image_path: str) -> List[Dict[str, object]]:
        image = Image.open(image_path).convert("RGB")
        yolo_world = YoloWorld()
        result = yolo_world.detect(image)
        return result
    
    def run_yolo_e(self, image_path: str) -> List[Dict[str, object]]:
        image = Image.open(image_path).convert("RGB")
        yolo_e = YoloE()
        result = yolo_e.detect(image)
        return result

    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict[str, object]],
        output_path: str = "./out.png",
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        """Draw every det in `detections` onto `image_path` and save to `output_path`."""
        img   = Image.open(image_path).convert("RGB")
        draw  = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            if not isinstance(det.get("coordinates"), (list, tuple)) or len(det["coordinates"]) != 4:
                continue  # skip malformed

            x1, y1, x2, y2 = map(int, det["coordinates"])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label = f"{det['label']} ({det['confidence']:.2f})"
            text_w = font.getlength(label) if hasattr(font, "getlength") else len(label) * 10
            text_h = 24

            # filled background for readability
            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
            draw.text((x1, y1 - text_h), label, fill=(255, 255, 255), font=font)

        img.save(output_path)
    
    def iou(self, box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2

        inter_w = max(0.0, min(x2a, x2b) - max(x1a, x1b))
        inter_h = max(0.0, min(y2a, y2b) - max(y1a, y1b))
        inter   = inter_w * inter_h
        if inter == 0:
            return 0.0

        area1 = (x2a - x1a) * (y2a - y1a)
        area2 = (x2b - x1b) * (y2b - y1b)
        return inter / (area1 + area2 - inter)
    
    def refine_labels_with_deepseek(self, image_path, detections):
        base = Image.open(image_path).convert("RGB")
        tmp_dir = Path(tempfile.mkdtemp())

        crop_paths = []
        for idx, det in enumerate(detections):
            # draw red rectangle
            x1, y1, x2, y2 = map(int, det["coordinates"])
            tmp_img = base.copy()
            ImageDraw.Draw(tmp_img).rectangle([x1, y1, x2, y2], outline=(255,0,0), width=4)

            p = tmp_dir / f"crop_{idx}.png"
            tmp_img.save(p)
            crop_paths.append(str(p))

        # write list of crops to disk for the subprocess
        list_file = tmp_dir / "crops.json"
        list_file.write_text(json.dumps(crop_paths))

        # ⬇ one subprocess, one model load
        labels = self.run_deepseek_identify_subprocess(str(list_file))  # modify script to accept JSON list

        # fill in results
        for det, lab in zip(detections, labels):
            det["label"] = lab
            det["confidence"] = 1.0

        shutil.rmtree(tmp_dir)
        return detections


    def process_image(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        deepseek_dets: List[Dict[str, object]] = self.run_deepseek_list_and_bound_subprocess(image_path)
        print(f"Deepseek detections: {deepseek_dets}")
        self.visualize_detections(image_path, deepseek_dets, output_path="./out/deepseek_out.png", color=(255, 0, 0))
        
        yolo_world_dets: List[Dict[str, object]] = self.run_yolo_world(image_path)
        print(f"Yolo World detections: {yolo_world_dets}")
        self.visualize_detections(image_path, yolo_world_dets, output_path="./out/yolo_world_out.png", color=(255, 0, 0))
        
        yolo_e_dets: List[Dict[str, object]] = self.run_yolo_e(image_path)
        print(f"Yolo E detections: {yolo_e_dets}")
        self.visualize_detections(image_path, yolo_e_dets, output_path="./out/yolo_e_out.png", color=(255, 0, 0))

        all_dets: List[Dict[str, object]] = (
            deepseek_dets
            + yolo_world_dets
            + yolo_e_dets
        )

        filtered: List[Dict[str, object]] = []

        for det in sorted(all_dets, key=lambda d: -d["confidence"]):
            if not any(
                det["label"] == kept["label"]
                and self.iou(det["coordinates"], kept["coordinates"]) >= self.iou_threshold_pass_1
                for kept in filtered
            ):
                filtered.append(det)
        self.visualize_detections(image_path, filtered, output_path="./out/filtered_out.png", color=(255, 0, 0))
        
        refined_dets = self.refine_labels_with_deepseek(image_path, filtered)
        
        self.visualize_detections(image_path, refined_dets, output_path="./out/refined_out.png", color=(255, 0, 0))
        
        final: List[Dict[str, object]] = []
        for det in sorted(refined_dets, key=lambda d: -d["confidence"]):
            if not any(
                det["label"] == kept["label"]
                and self.iou(det["coordinates"], kept["coordinates"]) >= self.iou_threshold_pass_2
                for kept in final
            ):
                final.append(det)
        

        self.visualize_detections(image_path, final, output_path="./out/final.png", color=(255, 0, 0))
