from __future__ import annotations
from typing import Any, Dict, List

from ..llm_client import LLMClient
from ..prompt_templates import INTRA_RELATION_SELECT_SYSTEM


class IntraRelationSelector:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def select_intra(self, img_objects: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        user_payload = {
            "question": question,
            "objects": img_objects,
        }
        messages = [
            {"role": "system", "content": INTRA_RELATION_SELECT_SYSTEM},
            {"role": "user", "content": json.dumps(user_payload)},
        ]
        raw = self.llm.chat(messages, temperature=0)
        try:
            return json.loads(raw)["image_questions"]
        except Exception as err:
            raise RuntimeError(f"Relation-selector JSON error: {err}") from err
