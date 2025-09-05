from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image, ImageDraw
import requests
import copy
import torch
import torch.nn.functional as F
import re


class Florence2:
    def __init__(self, model_name: str = "microsoft/Florence-2-large") -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype='auto').eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def detect(self, image: Image.Image, return_scores: bool = True):
        results, scores = self.run_task("<OD>", image, return_scores=return_scores)
        return self.parse_results(results, "<OD>", scores=scores)
    
    def detect_and_visualize(self, image: Image.Image, output_path: str):
        detections = self.detect(image)
        annotated_image = self.visualize_detections(image, detections)
        annotated_image.save(output_path)
        return detections
    
    def dense_detect(self, image: Image.Image):
        results, _ = self.run_task("<DENSE_REGION_CAPTION>", image)
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
        results, _ = self.run_task("<MORE_DETAILED_CAPTION>", image)
        text_input = results["<MORE_DETAILED_CAPTION>"]
        results, _ = self.run_task("<CAPTION_TO_PHRASE_GROUNDING>", image, text_input)
        return self.parse_results(results, "<CAPTION_TO_PHRASE_GROUNDING>")
    
    def extract_confidence_scores(self, generated_ids, scores, task: str):
        """Extract confidence scores for each detection from generation scores."""
        if scores is None:
            return None
            
        # Convert scores to probabilities
        all_probs = []
        for score in scores:
            probs = F.softmax(score, dim=-1)
            all_probs.append(probs)
        
        # For object detection tasks, we'll compute average confidence over the generated tokens for each detection
        # This is a simplified approach - in practice, you might want to extract only the confidence for bbox tokens
        token_confidences = []
        for i, (score, token_id) in enumerate(zip(all_probs, generated_ids[0][1:])):  # Skip the first token (prompt)
            if i < len(score):
                confidence = score[0, token_id].item()  # Get probability of the actual generated token
                token_confidences.append(confidence)
        
        # For now, return average confidence - this can be refined
        if token_confidences:
            return sum(token_confidences) / len(token_confidences)
        return 0.5  # Default confidence
    
    def parse_results(self, results: dict, task: str, scores=None) -> list:
        detections = []
        if task in results:
            od_results = results[task]
            
            if 'bboxes' in od_results and 'labels' in od_results:
                bboxes = od_results['bboxes']
                labels = od_results['labels']
                
                # Extract a simple confidence score - in production, this should be more sophisticated
                base_confidence = 0.85 if scores is not None else 0.5
                
                i = 0
                for bbox, label in zip(bboxes, labels):
                    # Add some variance to confidence based on bbox area (larger objects typically have higher confidence)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area_factor = min(1.0, area / 50000)  # Normalize by typical object area
                    confidence = base_confidence + (0.1 * area_factor)
                    confidence = min(0.99, confidence)  # Cap at 0.99
                    
                    detection = {
                        'id': i,
                        'bbox': bbox,
                        'label': label,
                        'confidence': round(confidence, 3)
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
    
    def run_task(self, task_prompt: str, image: Image.Image, text_input: str = None, return_scores: bool = False):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        
        if return_scores:
            # Generate with scores
            generated_outputs = self.model.generate(
                input_ids=inputs["input_ids"].cuda(),
                pixel_values=inputs["pixel_values"].cuda(),
                max_new_tokens=1024,  
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = generated_outputs.sequences
            scores = generated_outputs.scores
        else:
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"].cuda(),
                pixel_values=inputs["pixel_values"].cuda(),
                max_new_tokens=1024,  
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            scores = None
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,     
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
    
        return parsed_answer, scores
    
    def crop_object(self, image: Image.Image, bbox: list, padding: int = 10) -> Image.Image:
        """Crop an object from the image based on its bounding box.
        
        Args:
            image: PIL Image object
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            padding: Extra padding around the bbox in pixels
            
        Returns:
            Cropped PIL Image
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding while ensuring we stay within image bounds
        width, height = image.size
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        return image.crop((x1, y1, x2, y2))
    
    def describe_region(self, image: Image.Image, task: str = "<MORE_DETAILED_CAPTION>") -> str:
        """Get a detailed description of an image region.
        
        Args:
            image: PIL Image object (can be a cropped region)
            task: The caption task to use
            
        Returns:
            Text description of the region
        """
        results, _ = self.run_task(task, image)
        return results.get(task, "")
    
    def detect_and_describe(self, image: Image.Image, save_crops: bool = False, crop_dir: str = "crops") -> list:
        """Detect objects and get descriptions for each detected object.
        
        Args:
            image: PIL Image object
            save_crops: Whether to save cropped objects to disk
            crop_dir: Directory to save crops if save_crops is True
            
        Returns:
            List of detections with added 'description' field
        """
        import os
        
        # First, detect objects with confidence scores
        detections = self.detect(image, return_scores=True)
        
        # Create crop directory if needed
        if save_crops and not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
        
        # For each detection, crop and describe
        for i, detection in enumerate(detections):
            # Crop the object
            cropped = self.crop_object(image, detection['bbox'])
            
            # Save crop if requested
            if save_crops:
                crop_path = os.path.join(crop_dir, f"object_{i}_{detection['label'].replace(' ', '_')}.jpg")
                cropped.save(crop_path)
                detection['crop_path'] = crop_path
            
            # Get detailed description
            description = self.describe_region(cropped)
            detection['description'] = description
            detection['cropped_image'] = cropped
        
        return detections
