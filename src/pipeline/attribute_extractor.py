# src/pipeline/attribute_extractor.py
from __future__ import annotations

import re
from typing import List, Dict, Any
from PIL import Image

from src.vision.llava import Llava


class AttributeExtractor:
    _SPLIT_RE = re.compile(r"[,\n]")

    _PROMPT_TPL = (
        "Describe the {noun} in terms of its observable attributes: "
        "Respond with a comma-separated list of single words or short phrases. "
        "Do not use full sentences."
    )

    def __init__(self) -> None:
        self.vlm = Llava()

    def extract_attributes(
        self, image_path: str, objects: List[Dict[str, Any]]
    ) -> Dict[int, List[str]]:
        attr_map: Dict[int, List[str]] = {}

        for obj in objects:
            oid = obj["object_id"]
            try:
                attrs = self._extract_single(image_path, obj)
            except Exception as err:
                attrs = []
            attr_map[oid] = attrs

        return attr_map

    def _extract_single(self, image_path: str, obj: Dict[str, Any]) -> List[str]:
        x1, y1, x2, y2 = map(int, obj["coordinates"])
        with Image.open(image_path).convert("RGB") as im:
            crop = im.crop((x1, y1, x2, y2))

        noun = obj["label"]
        prompt = self._PROMPT_TPL.format(noun=noun)
        raw = self.vlm.run_inference(crop, prompt)
        tokens = [
            tok.strip().lower()
            for tok in self._SPLIT_RE.split(raw)
            if tok.strip()
        ]

        cleaned: List[str] = []
        for tok in tokens:
            words = tok.split()[:2]
            if words and words[0] not in {"the", "a", "an"}:
                cleaned.append(" ".join(words))

        return list(dict.fromkeys(cleaned))[:4]
