"""
Geometric spatial reasoning using DepthAnything + bounding boxes.
Provides soft probabilistic spatial verification based on NAVER's approach.

References:
- NAVER (Cai et al., ICCV 2025): Neuro-Symbolic Compositional Automaton for Visual Grounding
  https://github.com/ControlNet/NAVER
- Uses sigmoid-scaled coordinate differences with ALPHA parameter
"""

"""
Add
if self.spatial_reasoner.can_verify(action.relation):
            probability, debug = self.spatial_reasoner.verify(
                image=image,
                subject=subject,
                obj=obj,
                relation=action.relation,
                image_id=action.image_id
            )

in the pipeline in unified_agent.py to use this tool
            
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from src.pipeline.unified_agent import EntityCandidate


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


# =============================================================================
# Bbox helper functions
# =============================================================================

def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get (cx, cy) from bbox [x1, y1, x2, y2]."""
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def bbox_size(bbox: List[float]) -> Tuple[float, float]:
    """Get (width, height) from bbox."""
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def bbox_area(bbox: List[float]) -> float:
    """Get area from bbox."""
    w, h = bbox_size(bbox)
    return w * h


class SpatialReasoner:
    """
    Geometric spatial verification using NAVER-style soft probabilistic formulas.

    Based on NAVER (Cai et al., ICCV 2025):
    - Uses ALPHA-scaled sigmoid for directional relations
    - Normalizes coordinates by image dimensions
    - Uses 3D distance with depth for proximity relations
    - Uses bbox containment ratios for topological relations

    Probability model:
        P(relation) = P(geometric_criteria) × P(subject_exists) × P(object_exists)
    """

    SUPPORTED_RELATIONS = {
        "above", "below", "left_of", "right_of",
        "on_top_of", "near", "next_to", "far", "overlaps",
        "in_front_of", "behind", "contains", "inside"
    }

    DEPTH_RELATIONS = {"on_top_of", "in_front_of", "behind", "near", "next_to", "far"}

    def __init__(self, alpha: float = 5.0):
        """
        Initialize spatial reasoner with NAVER-style parameters.

        Args:
            alpha: Scaling factor for sigmoid (NAVER uses 5.0)
        """
        self.alpha = alpha
        self._depth_model = None
        self._depth_cache: Dict[str, np.ndarray] = {}  # image_id -> depth_map

    def _get_depth_model(self):
        """Lazy load DepthAnything V2."""
        if self._depth_model is None:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Loading DepthAnything V2...")
            self._depth_model = pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Base-hf",
                device=device
            )
            print(f"DepthAnything loaded on {device}")
        return self._depth_model

    def can_verify(self, relation: str) -> bool:
        """Check if this relation can be verified geometrically."""
        return relation.lower().replace(" ", "_") in self.SUPPORTED_RELATIONS

    def clear_depth_cache(self):
        """Clear cached depth maps. Call between different image pairs."""
        self._depth_cache.clear()

    def verify(
        self,
        image: Image.Image,
        subject: 'EntityCandidate',
        obj: 'EntityCandidate',
        relation: str,
        image_id: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Verify spatial relation with soft probabilistic output.

        Args:
            image: PIL Image for depth estimation
            subject: Subject EntityCandidate
            obj: Object EntityCandidate
            relation: Relation property (e.g., "left_of")
            image_id: Optional cache key for depth map (e.g., "image_a")
                      If None, depth is computed but not cached.

        Returns:
            Tuple[probability, debug_info]
        """
        relation = relation.lower().replace(" ", "_")

        # Compute depth only if needed (with caching)
        if relation in self.DEPTH_RELATIONS:
            depth_map = self._get_depth_map(image, image_id)
            depth_a = self._get_entity_depth(subject.bbox, depth_map)
            depth_b = self._get_entity_depth(obj.bbox, depth_map)
        else:
            depth_a = depth_b = 0.0

        # Compute soft geometric probability
        geo_prob, debug = self._compute_relation_probability(
            subject.bbox, obj.bbox,
            depth_a, depth_b,
            image.width, image.height,
            relation
        )

        # Final probability = geometric × detection confidences
        prob = geo_prob * subject.confidence * obj.confidence

        debug.update({
            "relation": relation,
            "geometric_prob": round(geo_prob, 4),
            "probability": round(prob, 4),
            "holds": prob >= 0.5,
            "subject_id": subject.entity_id,
            "object_id": obj.entity_id,
            "subject_conf": subject.confidence,
            "object_conf": obj.confidence,
            "depth_cached": image_id is not None and image_id in self._depth_cache
        })

        return prob, debug

    def _get_depth_map(self, image: Image.Image, image_id: Optional[str]) -> np.ndarray:
        """
        Get depth map, using cache if available.

        Args:
            image: PIL Image
            image_id: Cache key (e.g., "image_a"). If None, compute without caching.

        Returns:
            Normalized depth map (0 = close, 1 = far)
        """
        # Check cache first
        if image_id and image_id in self._depth_cache:
            return self._depth_cache[image_id]

        # Compute depth
        result = self._get_depth_model()(image)
        depth = np.array(result["depth"])
        d_min, d_max = depth.min(), depth.max()
        depth_normalized = (depth - d_min) / (d_max - d_min + 1e-6)

        # Cache if image_id provided
        if image_id:
            self._depth_cache[image_id] = depth_normalized

        return depth_normalized

    def _get_entity_depth(self, bbox: List[float], depth_map: np.ndarray) -> float:
        """Get mean depth for bbox region."""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = depth_map.shape
        region = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return float(np.mean(region)) if region.size > 0 else 0.5

    def _compute_relation_probability(
        self,
        bbox_a: List[float],
        bbox_b: List[float],
        depth_a: float,
        depth_b: float,
        img_w: int,
        img_h: int,
        relation: str
    ) -> Tuple[float, Dict]:
        """
        Compute soft probability for spatial relation using NAVER formulas.
        """
        # Get centers and normalize by image dimensions
        a_cx, a_cy = bbox_center(bbox_a)
        b_cx, b_cy = bbox_center(bbox_b)
        a_cx_norm, a_cy_norm = a_cx / img_w, a_cy / img_h
        b_cx_norm, b_cy_norm = b_cx / img_w, b_cy / img_h

        a_w, a_h = bbox_size(bbox_a)
        b_w, b_h = bbox_size(bbox_b)

        debug = {
            "a_center_normalized": (round(a_cx_norm, 4), round(a_cy_norm, 4)),
            "b_center_normalized": (round(b_cx_norm, 4), round(b_cy_norm, 4)),
            "a_size": (round(a_w, 1), round(a_h, 1)),
            "b_size": (round(b_w, 1), round(b_h, 1)),
            "alpha": self.alpha
        }

        if depth_a != 0.0 or depth_b != 0.0:
            debug["a_depth"] = round(depth_a, 4)
            debug["b_depth"] = round(depth_b, 4)

        # =====================================================================
        # DIRECTIONAL RELATIONS
        # NAVER formula: sigmoid(ALPHA * (target_coord - source_coord))
        #
        # Note on y-axis convention:
        # NAVER uses inverted y-axis (y increases upward, Cartesian convention)
        # We use standard image coordinates (y increases downward)
        # Formulas are adjusted accordingly
        # =====================================================================

        if relation == "left_of":
            prob = sigmoid(self.alpha * (b_cx_norm - a_cx_norm))
            debug["x_diff"] = round(b_cx_norm - a_cx_norm, 4)

        elif relation == "right_of":
            prob = sigmoid(self.alpha * (a_cx_norm - b_cx_norm))
            debug["x_diff"] = round(a_cx_norm - b_cx_norm, 4)

        elif relation == "above":
            prob = sigmoid(self.alpha * (b_cy_norm - a_cy_norm))
            debug["y_diff"] = round(b_cy_norm - a_cy_norm, 4)

        elif relation == "below":
            prob = sigmoid(self.alpha * (a_cy_norm - b_cy_norm))
            debug["y_diff"] = round(a_cy_norm - b_cy_norm, 4)

        # =====================================================================
        # DEPTH-BASED RELATIONS
        # =====================================================================

        elif relation == "in_front_of":
            prob = sigmoid(self.alpha * (depth_b - depth_a))
            debug["depth_diff"] = round(depth_b - depth_a, 4)

        elif relation == "behind":
            prob = sigmoid(self.alpha * (depth_a - depth_b))
            debug["depth_diff"] = round(depth_a - depth_b, 4)

        # =====================================================================
        # PROXIMITY RELATIONS
        # NAVER: exp(-ALPHA * distance) * (1 - iou)
        # =====================================================================

        elif relation in {"near", "next_to"}:
            iou = self._iou(bbox_a, bbox_b)

            center_a_3d = np.array([a_cx_norm, a_cy_norm, depth_a])
            center_b_3d = np.array([b_cx_norm, b_cy_norm, depth_b])
            distance = np.linalg.norm(center_a_3d - center_b_3d) / np.sqrt(3)

            # Original NAVER formula
            # prob = math.exp(-self.alpha * distance) * (1 - iou)

            # Adjusted formula so that objects with overlapping hitboxes are counted as near rather than punished
            prob_by_distance = math.exp(-self.alpha * distance)
            prob = max(prob_by_distance, iou)

            debug.update({
                "distance_3d": round(float(distance), 4),
                "iou": round(iou, 4),
                "exp_term": round(math.exp(-self.alpha * distance), 4)
            })

        elif relation == "far":
            center_a_3d = np.array([a_cx_norm, a_cy_norm, depth_a])
            center_b_3d = np.array([b_cx_norm, b_cy_norm, depth_b])
            distance = np.linalg.norm(center_a_3d - center_b_3d) / np.sqrt(3)

            prob = 1.0 - math.exp(-self.alpha * distance)
            debug["distance_3d"] = round(float(distance), 4)

        # =====================================================================
        # TOPOLOGICAL RELATIONS
        # =====================================================================

        elif relation == "contains":
            prob = self._containment_ratio(bbox_a, bbox_b)
            debug["containment_ratio"] = round(prob, 4)

        elif relation == "inside":
            prob = self._containment_ratio(bbox_b, bbox_a)
            debug["containment_ratio"] = round(prob, 4)

        elif relation == "overlaps":
            iou = self._iou(bbox_a, bbox_b)
            containment_a_in_b = self._containment_ratio(bbox_b, bbox_a)
            containment_b_in_a = self._containment_ratio(bbox_a, bbox_b)

            has_intersection = min(1.0, iou * 10)
            not_contained = (1 - containment_a_in_b) * (1 - containment_b_in_a)
            prob = has_intersection * not_contained

            debug.update({
                "iou": round(iou, 4),
                "containment_a_in_b": round(containment_a_in_b, 4),
                "containment_b_in_a": round(containment_b_in_a, 4)
            })

        # =====================================================================
        # COMPOSITE: ON_TOP_OF
        # =====================================================================

        elif relation == "on_top_of":
            above_prob = sigmoid(self.alpha * (b_cy_norm - a_cy_norm))

            depth_diff = abs(depth_a - depth_b)
            same_depth_prob = sigmoid(self.alpha * (0.1 - depth_diff))

            h_overlap = self._horizontal_overlap_ratio(bbox_a, bbox_b)
            overlap_prob = sigmoid(self.alpha * (h_overlap - 0.2))

            prob = above_prob * same_depth_prob * overlap_prob

            debug.update({
                "above_prob": round(above_prob, 4),
                "same_depth_prob": round(same_depth_prob, 4),
                "overlap_prob": round(overlap_prob, 4),
                "h_overlap": round(h_overlap, 4),
                "depth_diff": round(depth_diff, 4)
            })

        else:
            prob = 0.0
            debug["error"] = f"Unsupported relation: {relation}"

        return prob, debug

    def _horizontal_overlap_ratio(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        """Compute horizontal overlap ratio."""
        x_left = max(bbox_a[0], bbox_b[0])
        x_right = min(bbox_a[2], bbox_b[2])

        if x_right <= x_left:
            return 0.0

        overlap = x_right - x_left
        min_width = min(bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0])

        return overlap / min_width if min_width > 0 else 0.0

    def _iou(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        """Compute Intersection over Union."""
        ix1 = max(bbox_a[0], bbox_b[0])
        iy1 = max(bbox_a[1], bbox_b[1])
        ix2 = min(bbox_a[2], bbox_b[2])
        iy2 = min(bbox_a[3], bbox_b[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = bbox_area(bbox_a)
        area_b = bbox_area(bbox_b)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def _containment_ratio(self, outer: List[float], inner: List[float]) -> float:
        """Compute what fraction of inner's area is contained in outer."""
        ix1 = max(outer[0], inner[0])
        iy1 = max(outer[1], inner[1])
        ix2 = min(outer[2], inner[2])
        iy2 = min(outer[3], inner[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        inner_area = bbox_area(inner)

        return intersection / inner_area if inner_area > 0 else 0.0