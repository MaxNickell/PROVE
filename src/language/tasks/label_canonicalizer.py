from __future__ import annotations
import json
import re
from typing import List, Dict, Any
from ..llm_client import LLMClient
from ..prompt_templates import (
    GROUNDING_CANONICALISE_SYSTEM,
)

class LabelCanonicalizer:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._md_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

    def run(self, caption: str) -> List[Dict[str, Any]]:
        if not caption or not caption.strip():
            return []
            
        resp = self.llm.chat(
            [
                {"role": "system", "content": GROUNDING_CANONICALISE_SYSTEM},
                {"role": "user",   "content": caption},
            ],
            temperature=0,
        )
        clean = self._strip_md(resp)
        
        if not clean or not clean.strip():
            return []
            
        try:
            objs = json.loads(clean)
        except Exception as err:
            raise RuntimeError(f"Grounding JSON parse error: {err}") from err

        if not (
            isinstance(objs, list)
            and all(
                isinstance(o, dict)
                and isinstance(o.get("label"), str)
                and isinstance(o.get("bbox"), list)
                and len(o["bbox"]) == 4
                and all(isinstance(n, int) for n in o["bbox"])
                for o in objs
            )
        ):
            raise RuntimeError("Grounding canonicaliser returned invalid schema")

        for o in objs:
            o["label"] = o["label"].lower().strip()
        return objs

    def _strip_md(self, text: str) -> str:
        m = self._md_re.search(text.strip())
        return m.group(1).strip() if m else text.strip()
