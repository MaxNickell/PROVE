import re, subprocess, json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from src.vision.yolo_world import YoloWorld
from src.vision.yoloe import YoloE
from src.pipeline.utils.iou import IOUHelper
from src.language.forge_llm import ForgeLLM


class ObjectDetector:
    def __init__(
        self, 
        deepseek_env: str = "DEEPSEEK_VL2_ENV",
        confidence_threshold: float = 0.7,
        nms_iou_threshold: float = 0.6,
        
    ):
        self.deepseek_env = deepseek_env
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.deepseek_confidence = 0.90
        
        self.yolo_world = YoloWorld()
        self.yolo_e = YoloE()
        self.llm = ForgeLLM()

    def detect(self, image_path: str) -> Dict[str, Any]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_name = Path(image_path).stem
        output_dir = Path(f"./out/{image_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            deepseek_dets = self._get_deepseek_detections(image_path)
            yolo_world_dets = self._get_yolo_world_detections(image_path)
            yolo_e_dets = self._get_yolo_e_detections(image_path)
            
            self._save_visualization(image_path, deepseek_dets, output_dir / "deepseek_detections.png", (255, 0, 0))
            self._save_visualization(image_path, yolo_world_dets, output_dir / "yolo_world_detections.png", (0, 255, 0))
            self._save_visualization(image_path, yolo_e_dets, output_dir / "yolo_e_detections.png", (0, 128, 255))
            
            all_detections = deepseek_dets + yolo_world_dets + yolo_e_dets
            final_detections = IOUHelper.non_max_suppression(all_detections, self.nms_iou_threshold)
            
            self._save_visualization(image_path, final_detections, output_dir / "final_detections.png", (255, 255, 0))
            
            return {
                "objects": final_detections,
                "output_dir": str(output_dir)
            }
            
        except Exception as e:
            return {
                "objects": [],
                "output_dir": str(output_dir),
                "error": str(e)
            }

    def _get_deepseek_detections(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            proc = subprocess.run(
                ["conda", "run", "-n", self.deepseek_env,
                 "python", "-m", "src.vision.utils.deepseek_list_and_bound", image_path],
                capture_output=True, text=True, check=True, timeout=60
            )
            
            match = re.search(r"\[START_DEEPSEEK_VL2\](.*?)\[END_DEEPSEEK_VL2\]", proc.stdout, re.DOTALL)
            if not match:
                return []
            
            caption = match.group(1).strip()
            print(f"DeepSeek raw caption: {caption}")
            
            parsed_items = self.llm._parse_caption(caption)
            print(f"Parsed items: {parsed_items}")
            
            w, h = Image.open(image_path).size
            detections = []
            
            for item in parsed_items:
                x1, y1, x2, y2 = item["bbox"]
                object_name = item["object"]
                
                x1_px = max(0, min(x1 * w // 999, w - 1))
                y1_px = max(0, min(y1 * h // 999, h - 1))
                x2_px = max(x1_px + 1, min(x2 * w // 999, w))
                y2_px = max(y1_px + 1, min(y2 * h // 999, h))
                
                detections.append({
                    "label": object_name,
                    "confidence": self.deepseek_confidence,
                    "coordinates": (x1_px, y1_px, x2_px, y2_px)
                })
            
            return detections
            
        except Exception as e:
            print(f"Error in DeepSeek detection: {e}")
            return []

    def _get_yolo_world_detections(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
            detections = self.yolo_world.detect(image)
            return [d for d in detections if d.get("confidence", 0) >= self.confidence_threshold]
        except Exception:
            return []

    def _get_yolo_e_detections(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
            detections = self.yolo_e.detect(image)
            return [d for d in detections if d.get("confidence", 0) >= self.confidence_threshold]
        except Exception:
            return []

    def _save_visualization(self, image_path: str, detections: List[Dict[str, Any]], output_path: Path, color: tuple):
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except Exception:
                font = ImageFont.load_default()
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det["coordinates"])
                label = f"{det['label']} ({det['confidence']:.2f})"
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                text_width = font.getlength(label) if hasattr(font, "getlength") else len(label) * 10
                text_height = 24
                draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill=color)
                draw.text((x1, y1 - text_height), label, fill=(255, 255, 255), font=font)
            
            img.save(output_path)
            
        except Exception as e:
            print(f"Warning: Could not save visualization to {output_path}: {e}")
