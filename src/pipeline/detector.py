from __future__ import annotations
import json, re, subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
from src.language.tasks.label_canonicalizer import LabelCanonicalizer
from src.vision.yolo_world import YoloWorld
from src.vision.yoloe import YoloE

class DetectorError(RuntimeError):
    pass

class Detector:
    def __init__(
        self,
        *,
        label_canonicalizer: LabelCanonicalizer,
        deepseek_env: str = "DEEPSEEK_VL2_ENV",
        deepseek_confidence: float = 0.95,
        yolo_world_conf: float = 0.70,
        yolo_e_conf: float = 0.70,
        nms_iou_threshold: float = 0.60,
        explainable: bool = False,
    ) -> None:
        self.label_canonicalizer = label_canonicalizer

        self.deepseek_env = deepseek_env
        self.deepseek_confidence = deepseek_confidence
        self.yolo_world_conf = yolo_world_conf
        self.yolo_e_conf = yolo_e_conf
        self.nms_iou_threshold = nms_iou_threshold
        self.explainable = explainable

        self.yolo_world = YoloWorld()
        self.yolo_e = YoloE()

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        img_path = Path(image_path)
        if not img_path.exists():
            raise DetectorError(f"Image not found: {image_path}")

        try:
            deepseek = self._deepseek_detections(image_path)
            yolo_w = self._yolo_world_detections(image_path)
            yolo_e = self._yolo_e_detections(image_path)
        except DetectorError:
            raise
        except Exception as err:
            raise DetectorError(f"Unexpected detection error: {err}") from err

        fused = self._non_max_suppression(
            deepseek + yolo_w + yolo_e, self.nms_iou_threshold
        )

        for idx, det in enumerate(fused, start=1):
            det["object_id"] = idx

        if self.explainable:
            self._draw_detections(image_path, fused)

        return fused

    def _deepseek_detections(self, image_path: str) -> List[Dict[str, Any]]:
        cmd = [
            "conda", "run", "-n", self.deepseek_env,
            "python", "-m", "src.vision.utils.deepseek_list_and_bound",
            image_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        except subprocess.SubprocessError as err:
            raise DetectorError(f"DeepSeek subprocess failed: {err}") from err

        m = re.search(r"\[START_DEEPSEEK_VL2\](.*?)\[END_DEEPSEEK_VL2\]",
                    proc.stdout, re.DOTALL)
        if m is None:
            raise DetectorError("DeepSeek output markers not found.")
        caption = m.group(1).strip()

        try:
            obj_list = self.label_canonicalizer.run(caption)
        except Exception as err:
            raise DetectorError(f"Grounding canonicalisation failed: {err}") from err
        w, h = Image.open(image_path).size
        dets: List[Dict[str, Any]] = []
        for obj in obj_list:
            x1, y1, x2, y2 = obj["bbox"]
            dets.append(
                {
                    "label": obj["label"].lower(),
                    "confidence": self.deepseek_confidence,
                    "coordinates": (
                        self._clamp(x1 * w // 999, 0, w - 1),
                        self._clamp(y1 * h // 999, 0, h - 1),
                        self._clamp(x2 * w // 999, 1, w),
                        self._clamp(y2 * h // 999, 1, h),
                    ),
                }
            )
        return dets

    def _yolo_world_detections(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
            dets = self.yolo_world.detect(image)
            return [d for d in dets if d["confidence"] >= self.yolo_world_conf]
        except Exception as err:
            raise DetectorError(f"YOLO-World failed: {err}") from err

    def _yolo_e_detections(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
            dets = self.yolo_e.detect(image)
            return [d for d in dets if d["confidence"] >= self.yolo_e_conf]
        except Exception as err:
            raise DetectorError(f"YOLO-E failed: {err}") from err

    @staticmethod
    def _non_max_suppression(
        dets: List[Dict[str, Any]], iou_thresh: float
    ) -> List[Dict[str, Any]]:
        if not dets:
            return []

        boxes = [
            (*det["coordinates"], det["confidence"], det["label"], idx)
            for idx, det in enumerate(dets)
        ]
        boxes.sort(key=lambda b: b[4], reverse=True)

        keep: List[int] = []
        while boxes:
            *_, idx = boxes.pop(0)
            keep.append(idx)
            boxes = [
                b
                for b in boxes
                if Detector._iou(b[0:4], dets[idx]["coordinates"]) < iou_thresh
            ]
        return [dets[i] for i in keep]

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
        inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
        inter = inter_w * inter_h
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter
        return 0.0 if union == 0 else inter / union

    @staticmethod
    def _clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(val, hi))

    def _draw_detections(self, image_path: str, detections: List[Dict[str, Any]]) -> None:
        try:
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except (OSError, IOError):
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            for det in detections:
                x1, y1, x2, y2 = det["coordinates"]
                label = det["label"]
                confidence = det.get("confidence", 0.0)
                
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                label_text = f"{label} ({confidence:.2f})"
                
                if font:
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width = len(label_text) * 8
                    text_height = 12
                
                text_x = x1
                text_y = max(0, y1 - text_height - 5)
                draw.rectangle(
                    [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
                    fill="red"
                )
                
                draw.text(
                    (text_x + 2, text_y + 2), 
                    label_text, 
                    fill="white", 
                    font=font
                )
            
            img_path = Path(image_path)
            output_path = img_path.parent / f"{img_path.stem}_annotated{img_path.suffix}"
            image.save(output_path)
            print(f"Annotated image saved to: {output_path}")
            
        except Exception as err:
            print(f"Warning: Failed to draw detections: {err}")
