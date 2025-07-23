import json, os, re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI


class ForgeLLM:
    DEFAULT_MODEL = "OpenAI/gpt-4o"

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        base_url = os.environ["FORGE_BASE_URL"]
        api_key = os.environ["FORGE_KEY"]

        self.model = model or self.DEFAULT_MODEL
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content

    def parse_caption(self, caption: str) -> List[Dict[str, Any]]:
        system_prompt = """
            You are an expert at parsing image descriptions that contain object locations. Extract objects and their bounding boxes from descriptions where objects are followed by [[x1,y1,x2,y2]] coordinate tags.
            IMPORTANT RULES:
            1. Extract the main object/noun that the coordinates refer to
            2. Keep compound objects together (like "coffee table", "traffic light", "kitchen sink")
            3. Remove unnecessary adjectives but keep descriptive parts that identify the object
            4. Return clean, specific object names
            5. Return ONLY valid JSON - no markdown, no explanations

            Return format: {"items":[{"object":"name","bbox":[x1,y1,x2,y2]}]}

            EXAMPLES:

            Input: "A red sports car[[120,300,450,600]] is parked next to a tall oak tree[[500,50,700,580]] while a small dog[[300,520,380,600]] runs nearby."
            Output: {"items":[{"object":"sports car","bbox":[120,300,450,600]},{"object":"oak tree","bbox":[500,50,700,580]},{"object":"dog","bbox":[300,520,380,600]}]}

            Input: "The woman is sitting on a wooden bench[[200,400,800,650]] holding a coffee cup[[350,250,420,350]] with her purse[[150,480,250,580]] beside her."
            Output: {"items":[{"object":"wooden bench","bbox":[200,400,800,650]},{"object":"coffee cup","bbox":[350,250,420,350]},{"object":"purse","bbox":[150,480,250,580]}]}

            Input: "A blue bicycle[[100,200,400,700]] leans against a brick wall[[0,0,999,500]] next to a street lamp[[450,80,550,500]] and a trash can[[600,450,700,650]]."
            Output: {"items":[{"object":"bicycle","bbox":[100,200,400,700]},{"object":"brick wall","bbox":[0,0,999,500]},{"object":"street lamp","bbox":[450,80,550,500]},{"object":"trash can","bbox":[600,450,700,650]}]}

            Input: "The chef is preparing food on a stainless steel counter[[0,600,999,999]] using a sharp knife[[300,400,380,450]] while ingredients sit in a glass bowl[[200,350,350,450]]."
            Output: {"items":[{"object":"counter","bbox":[0,600,999,999]},{"object":"knife","bbox":[300,400,380,450]},{"object":"glass bowl","bbox":[200,350,350,450]}]}"""

        user_prompt = f"Parse this image description:\n\n{caption}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self._chat(messages, temperature=0)
            raw_response = self._strip_md_fences(raw_response)
            
            parsed = json.loads(raw_response)
            return parsed.get("items", [])
            
        except Exception as e:
            print(f"Error parsing with LLM: {e}")
            return self._parse_caption_regex_fallback(caption)

    def _strip_md_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text
    
    def _parse_caption_regex_fallback(self, caption: str) -> List[Dict[str, Any]]:
        pattern = r'([a-zA-Z\s]+)\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
        matches = re.findall(pattern, caption)
        
        items = []
        for text, x1, y1, x2, y2 in matches:
            words = text.strip().split()
            if not words:
                object_name = "object"
            elif len(words) == 1:
                object_name = words[0]
            else:
                object_name = " ".join(words[-2:])
            
            items.append({
                "object": object_name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        return items
    
    def infer_required_relationships(self, image_1_dets: Dict[str, Any], image_2_dets: Dict[str, Any], question: str) -> Dict[str, Any]:
        system_prompt = """
            You are an intelligent visual reasoning assistant.

            You will receive:
            - An **ultimate question** that compares two images (e.g., "Who is wealthier?" or "Which room is messier?")
            - Two lists of **objects**—one per image—where each object has a label, bounding box, confidence, and ID

            Your task is to generate a set of **intra-image relational questions** for each image. These are natural-language questions that relate **two or more objects within the same image**. They should capture **meaningful relationships** that could help an AI reason about the ultimate question.

            You do **not** have access to the images—only the object metadata. Focus only on **within-image relationships**. Do **not** generate questions that compare objects across the two images.

            ---

            Input Format (You Will Receive):

            question: str

            image_1_objects: [
                {
                    "object_id": int,
                    "label": str,
                    "coordinates": [x1, y1, x2, y2],
                    "confidence": float
                },
                ...
            ]

            image_2_objects: [
                {
                    "object_id": int,
                    "label": str,
                    "coordinates": [x1, y1, x2, y2],
                    "confidence": float
                },
                ...
            ]

            ---

            Output Format (You Must Return):

            {
            "image_1_questions": [
                {
                "question": str,
                "object_ids": [int, int, ...]
                },
                ...
            ],
            "image_2_questions": [
                {
                "question": str,
                "object_ids": [int, int, ...]
                },
                ...
            ]
            }

            ---

            Example 1:

            Input:

            question: "Which person is more powerful?"

            image_1_objects: [
                {"label": "woman", "object_id": 1, "coordinates": [...], "confidence": 0.95},
                {"label": "throne", "object_id": 2, "coordinates": [...], "confidence": 0.92},
                {"label": "crown", "object_id": 3, "coordinates": [...], "confidence": 0.90}
            ]

            image_2_objects: [
                {"label": "man", "object_id": 1, "coordinates": [...], "confidence": 0.94},
                {"label": "suit", "object_id": 2, "coordinates": [...], "confidence": 0.91},
                {"label": "briefcase", "object_id": 3, "coordinates": [...], "confidence": 0.89}
            ]

            Output:

            {
            "image_1_questions": [
                {
                "question": "Is the woman wearing a crown?",
                "object_ids": [1, 3]
                },
                {
                "question": "Is the woman sitting on a throne?",
                "object_ids": [1, 2]
                }
            ],
            "image_2_questions": [
                {
                "question": "Is the man wearing a suit?",
                "object_ids": [1, 2]
                },
                {
                "question": "Is the man holding a briefcase?",
                "object_ids": [1, 3]
                }
            ]
            }

            ---

            Example 2:

            Input:

            question: "Which place is cleaner?"

            image_1_objects: [
                {"label": "countertop", "object_id": 1, "coordinates": [...], "confidence": 0.96},
                {"label": "trash bin", "object_id": 2, "coordinates": [...], "confidence": 0.94},
                {"label": "sink", "object_id": 3, "coordinates": [...], "confidence": 0.92}
            ]

            image_2_objects: [
                {"label": "floor", "object_id": 1, "coordinates": [...], "confidence": 0.93},
                {"label": "toys", "object_id": 2, "coordinates": [...], "confidence": 0.90},
                {"label": "laundry basket", "object_id": 3, "coordinates": [...], "confidence": 0.88}
            ]

            Output:

            {
            "image_1_questions": [
                {
                "question": "Is the trash bin full near the sink?",
                "object_ids": [2, 3]
                },
                {
                "question": "Is the countertop clean near the sink?",
                "object_ids": [1, 3]
                }
            ],
            "image_2_questions": [
                {
                "question": "Are toys scattered on the floor?",
                "object_ids": [1, 2]
                },
                {
                "question": "Is the laundry basket overflowing onto the floor?",
                "object_ids": [1, 3]
                }
            ]
            }

            ---

            Instructions:

            - Use the **labels** and the **ultimate question** to infer relationships that might help answer it.
            - Each question must reference **two or more object IDs** from the **same image**.
            - Do **not** refer to objects across different images.
            - Be plausible—base reasoning **only on object labels**, not assumed image content.
            - Output must strictly follow the required **JSON format** above.
        """

        user_prompt = f"""
            question: {question}
            image_1_objects: {image_1_dets}
            image_2_objects: {image_2_dets}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self._chat(messages, temperature=0)
            raw_response = self._strip_md_fences(raw_response)
            
            parsed = json.loads(raw_response)
            return parsed
        
        except Exception as e:
            print(f"Error inferring required relationships with LLM: {e}")
            return {}
