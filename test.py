#!/usr/bin/env python3
"""
Florence-2 Object Detection Script
This script uses Microsoft's Florence-2 model to detect objects in a local image.
"""

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re
import os
import argparse

class Florence2ObjectDetector:
    def __init__(self, model_name="microsoft/Florence-2-large", device=None):
        """
        Initialize the Florence-2 object detector.
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to run on ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading Florence-2 model on {self.device}...")
        
        # Load model and processor
        # Use float32 to avoid mixed precision issues
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
    
    def detect_objects(self, image_path, task_prompt="<OD>"):
        """
        Detect objects in an image.
        
        Args:
            image_path (str): Path to the input image
            task_prompt (str): Task prompt for Florence-2
                - "<OD>" for object detection
                - "<DENSE_REGION_CAPTION>" for dense captioning
                - "<REGION_PROPOSAL>" for region proposals
        
        Returns:
            dict: Detection results containing bboxes and labels
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs with proper tensor types
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Ensure all inputs are on the correct device with proper dtype
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        # Decode results
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        # Parse the output
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        # Debug: print raw output
        print(f"Raw generated text: {generated_text}")
        print(f"Parsed answer keys: {parsed_answer.keys() if parsed_answer else 'None'}")
        if parsed_answer:
            print(f"Parsed answer content: {parsed_answer}")
        
        return parsed_answer, image
    
    def parse_detection_results(self, parsed_answer):
        """
        Parse detection results into a more readable format.
        
        Args:
            parsed_answer (dict): Raw detection results from Florence-2
            
        Returns:
            list: List of detection dictionaries with bbox and label
        """
        detections = []
        
        # Florence-2 returns results under the task key (e.g., '<OD>')
        if '<OD>' in parsed_answer:
            od_results = parsed_answer['<OD>']
            print(f"OD results: {od_results}")
            
            if 'bboxes' in od_results and 'labels' in od_results:
                bboxes = od_results['bboxes']
                labels = od_results['labels']
                
                for bbox, label in zip(bboxes, labels):
                    detection = {
                        'bbox': bbox,  # [x1, y1, x2, y2]
                        'label': label,
                        'confidence': 1.0  # Florence-2 doesn't provide confidence scores
                    }
                    detections.append(detection)
        
        # Also check if results are directly in parsed_answer (different format)
        elif 'bboxes' in parsed_answer and 'labels' in parsed_answer:
            bboxes = parsed_answer['bboxes']
            labels = parsed_answer['labels']
            
            for bbox, label in zip(bboxes, labels):
                detection = {
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'label': label,
                    'confidence': 1.0  # Florence-2 doesn't provide confidence scores
                }
                detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image, detections, output_path=None, show_plot=True):
        """
        Visualize detection results on the image using matplotlib.
        
        Args:
            image (PIL.Image): Input image
            detections (list): List of detection dictionaries
            output_path (str): Path to save the visualization
            show_plot (bool): Whether to display the plot
        """
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        if len(detections) > 0:
            # Color map for different labels
            unique_labels = list(set([d['label'] for d in detections]))
            colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_labels), 1)))
            label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            for detection in detections:
                bbox = detection['bbox']  # [x1, y1, x2, y2]
                label = detection['label']
                
                # Create rectangle
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2,
                    edgecolor=label_to_color[label],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(
                    x1, y1 - 5,
                    f"{label}",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=label_to_color[label], alpha=0.7),
                    fontsize=10,
                    color='black'
                )
        
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')
        ax.set_title(f"Object Detection Results - {len(detections)} objects detected")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Matplotlib visualization saved to: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def draw_detections_on_image(self, image, detections, output_path=None):
        """
        Draw detection results directly on the PIL image and save it.
        
        Args:
            image (PIL.Image): Input image
            detections (list): List of detection dictionaries
            output_path (str): Path to save the annotated image
            
        Returns:
            PIL.Image: Annotated image
        """
        # Create a copy of the image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Define colors for different labels
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
            "#FFA500", "#800080", "#008000", "#FFC0CB", "#A52A2A", "#808080"
        ]
        
        if len(detections) > 0:
            unique_labels = list(set([d['label'] for d in detections]))
            label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            for detection in detections:
                bbox = detection['bbox']  # [x1, y1, x2, y2]
                label = detection['label']
                color = label_to_color[label]
                
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label background
                label_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(label_bbox, fill=color)
                
                # Draw label text
                draw.text((x1, y1 - 25), label, fill="white", font=font)
        
        # Save the annotated image
        if output_path:
            annotated_image.save(output_path)
            print(f"Annotated image saved to: {output_path}")
        
        return annotated_image
    
    def print_detection_summary(self, detections):
        """
        Print a summary of detected objects.
        
        Args:
            detections (list): List of detection dictionaries
        """
        print(f"\n{'='*50}")
        print(f"OBJECT DETECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total objects detected: {len(detections)}")
        
        # Group by label
        label_counts = {}
        for detection in detections:
            label = detection['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nObject counts:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        
        print(f"\nDetailed results:")
        for i, detection in enumerate(detections, 1):
            bbox = detection['bbox']
            label = detection['label']
            print(f"  {i}. {label} - BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")


def main():
    parser = argparse.ArgumentParser(description="Florence-2 Object Detection")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Path to save the visualization")
    parser.add_argument("--model", default="microsoft/Florence-2-large", 
                       help="Model name (default: microsoft/Florence-2-large)")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto",
                       help="Device to run on")
    parser.add_argument("--task", default="<OD>", 
                       choices=["<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>"],
                       help="Detection task type")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        return
    
    # Initialize detector
    device = None if args.device == "auto" else args.device
    detector = Florence2ObjectDetector(model_name=args.model, device=device)
    
    # Perform detection
    print(f"\nAnalyzing image: {args.image_path}")
    parsed_answer, image = detector.detect_objects(args.image_path, task_prompt=args.task)
    
    # Parse results
    detections = detector.parse_detection_results(parsed_answer)
    
    # Print summary
    detector.print_detection_summary(detections)
    
    # Visualize results
    if detections:
        detector.visualize_detections(image, detections, output_path=args.output)
    else:
        print("No objects detected in the image.")


if __name__ == "__main__":
    # Example usage if run directly without command line arguments
    if len(os.sys.argv) == 1:
        print("Florence-2 Object Detection Script")
        print("=" * 40)
        print("Usage examples:")
        print("  python florence2_detector.py image.jpg")
        print("  python florence2_detector.py image.jpg --output results.png")
        print("  python florence2_detector.py image.jpg --model microsoft/Florence-2-base")
        print("  python florence2_detector.py image.jpg --device cuda")
        print("\nFor more help: python florence2_detector.py --help")
        
        # If you want to test with a hardcoded image path, uncomment below:
        # test_image = "path/to/your/image.jpg"
        # if os.path.exists(test_image):
        #     detector = Florence2ObjectDetector()
        #     parsed_answer, image = detector.detect_objects(test_image)
        #     detections = detector.parse_detection_results(parsed_answer)
        #     detector.print_detection_summary(detections)
        #     detector.visualize_detections(image, detections)
    else:
        main()