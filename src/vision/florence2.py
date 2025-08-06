from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image, ImageDraw
import requests
import copy
import torch


class Florence2:
    def __init__(self, model_name: str = "microsoft/Florence-2-large") -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype='auto').eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def detect(self, image: Image.Image):
        results = self.run_task("<OD>", image)
        return self.parse_results(results, "<OD>")
    
    def detect_and_visualize(self, image: Image.Image, output_path: str):
        detections = self.detect(image)
        annotated_image = self.visualize_detections(image, detections)
        annotated_image.save(output_path)
        return detections
    
    def dense_detect(self, image: Image.Image):
        results = self.run_task("<DENSE_REGION_CAPTION>", image)
        return self.parse_results(results, "<DENSE_REGION_CAPTION>")
    
    def dense_detect_and_visualize(self, image: Image.Image, output_path: str):
        detections = self.dense_detect(image)
        annotated_image = self.visualize_detections(image, detections)
        annotated_image.save(output_path)
        return detections
    
    def dense_detail_detect_and_visualize(self, image: Image.Image, output_path: str):
        detections = self.dense_detail_detect(image)
        annotated_image = self.visualize_detections(image, detections)
        annotated_image.save(output_path)
        return detections
    
    def dense_detail_detect(self, image: Image.Image):
        results = self.run_task("<MORE_DETAILED_CAPTION>", image)
        text_input = results["<MORE_DETAILED_CAPTION>"]
        results = self.run_task("<CAPTION_TO_PHRASE_GROUNDING>", image, text_input)
        return self.parse_results(results, "<CAPTION_TO_PHRASE_GROUNDING>")
    
    def parse_results(self, results: dict, task: str) -> list:
        detections = []
        if task in results:
            od_results = results[task]
            
            if 'bboxes' in od_results and 'labels' in od_results:
                bboxes = od_results['bboxes']
                labels = od_results['labels']
                i = 0
                for bbox, label in zip(bboxes, labels):
                    detection = {
                        'id': i,
                        'bbox': bbox,
                        'label': label,
                        'confidence': 1.0
                    }
                    detections.append(detection)
                    i += 1
        
        return detections
    
    def visualize_detections(self, image: Image.Image, detections: list):
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1]), label, fill="red")
        
        return annotated_image
    
    def run_task(self, task_prompt: str, image: Image.Image, text_input: str = None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,  
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,     
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
    
        return parsed_answer
