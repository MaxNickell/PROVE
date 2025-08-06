from typing import List, Dict, Any
from src.language.llm_client import LLMClient
from src.language.prompt_templates import SALIENT_INTRA_RELATION_SELECT_SYSTEM
from src.vision.blip2 import Blip2
from PIL import Image, ImageDraw
import numpy as np
import os
import json


class SalientIntraRelationshipExtractor:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.system_prompt = SALIENT_INTRA_RELATION_SELECT_SYSTEM
        self.blip2 = Blip2()
    
    def extract(self, question: str, objects: List[Dict[str, Any]], image_path: str) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        relationships = self.infer_salient_intra_relationships(question, objects)
        return self.answer_relationships(relationships, image, objects)
        
    
    def infer_salient_intra_relationships(self, question: str, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        object_list = []
        for object in objects:
            object_list.append({"object_id": object["id"], "label": object["label"]})
        
        user_prompt = f"""
        ultimate_question: {question}
        objects: 
        {object_list}
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm_client.chat(messages)
            if isinstance(response, str):
                try:
                    parsed_response = json.loads(response)
                    return parsed_response
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse JSON response")
            return response
        except Exception as e:
            raise ValueError(f"Error in infer_salient_intra_relationships: {e}")
    
    def answer_relationships(self, relationships: List[Dict[str, Any]], image: Image.Image, objects: List[Dict[str, Any]], output_dir: str = "crops") -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, relationship in enumerate(relationships):
            answer, colored_crop = self.crop_image(image, relationship, objects)
            
            object_ids = relationship["object_ids"]
            
            crop_filename = f"relationship_crop_{i}_objects_{object_ids[0]}_{object_ids[1]}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            colored_crop.save(crop_path)
            
            result = {
                "object_ids": object_ids,
                "question": relationship["question"],
                "answer": answer,
                "crop_path": crop_path
            }
            results.append(result)
        
        return results
    def crop_image(self, image: Image.Image, relationship: Dict[str, Any], objects: List[Dict[str, Any]]) -> tuple[str, Image.Image]:
        object_ids = relationship["object_ids"]
        object_1 = objects[object_ids[0]]
        object_2 = objects[object_ids[1]]
        
        bbox_1 = object_1["bbox"]
        bbox_2 = object_2["bbox"]
        
        union_x1 = min(bbox_1[0], bbox_2[0])
        union_y1 = min(bbox_1[1], bbox_2[1])
        union_x2 = max(bbox_1[2], bbox_2[2])
        union_y2 = max(bbox_1[3], bbox_2[3])
        
        cropped_image = image.crop((int(union_x1), int(union_y1), int(union_x2), int(union_y2)))
        
        draw = ImageDraw.Draw(cropped_image)
        
        rel_bbox_1 = [
            bbox_1[0] - union_x1,
            bbox_1[1] - union_y1, 
            bbox_1[2] - union_x1,
            bbox_1[3] - union_y1
        ]
        rel_bbox_2 = [
            bbox_2[0] - union_x1,
            bbox_2[1] - union_y1,
            bbox_2[2] - union_x1, 
            bbox_2[3] - union_y1
        ]
        
        draw.rectangle(rel_bbox_1, outline="red", width=3)
        
        draw.rectangle(rel_bbox_2, outline="green", width=3)
        
        object_1_label = object_1["label"]
        object_2_label = object_2["label"]
        
        enhanced_question = f"Focus on the {object_1_label} (marked with RED box) and the {object_2_label} (marked with GREEN box). {relationship['question']}"
        
        answer = self.blip2.caption(cropped_image, enhanced_question)
        
        return answer, cropped_image