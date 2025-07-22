from typing import List, Dict, Any, Tuple

class IOUHelper:
    @staticmethod
    def iou(
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
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

    @staticmethod
    def non_max_suppression(
        dets: List[Dict[str, Any]], iou_thresh: float
    ) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for det in sorted(dets, key=lambda d: -d["confidence"]):
            if not any(
                IOUHelper.iou(det["coordinates"], k["coordinates"]) >= iou_thresh
                for k in kept
            ):
                kept.append(det)
        return kept
