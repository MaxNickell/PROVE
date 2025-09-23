from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import torch


class Florence2:
    """Simplified Florence-2 implementation based on HuggingFace best practices."""

    def __init__(self, model_name: str = "microsoft/Florence-2-large", device: str = "auto") -> None:
        # Simple device allocation - avoid device_map="auto" which causes meta-device issues
        self.has_cuda = torch.cuda.is_available()
        if device == "auto":
            self.device = torch.device("cuda:0" if self.has_cuda else "cpu")
        else:
            self.device = torch.device(device)

        # Use appropriate dtype for device
        self.dtype = torch.float16 if self.has_cuda else torch.float32

        # Load processor and model with safe settings
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,  # Keep safe for Florence-2
        )
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: Image.Image, return_scores: bool = True):
        """
        Detect objects in image with confidence scores.

        Args:
            image: PIL Image to process
            return_scores: Whether to return confidence scores

        Returns:
            List of detection dictionaries with bbox, label, confidence
        """
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Object detection task
        task = "<OD>"
        inputs = self.processor(text=task, images=image, return_tensors="pt")

        # Move tensors to device properly
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
                return_dict_in_generate=True,
                output_scores=return_scores,  # Only compute scores if needed
            )

            # Compute confidence scores if requested
            if return_scores and hasattr(generated, 'scores'):
                transition = self.model.compute_transition_scores(
                    sequences=generated.sequences,
                    scores=generated.scores,
                    beam_indices=generated.beam_indices,
                )
                transition_score = transition[0]
            else:
                transition_score = None

        # Parse results using processor's built-in method
        parsed = self.processor.post_process_generation(
            sequence=generated.sequences[0],
            transition_beam_score=transition_score,
            task=task,
            image_size=(image.width, image.height),
        )

        # Extract results
        results = parsed.get(task, {})
        bboxes = results.get("bboxes", [])
        labels = results.get("labels", [])
        scores = results.get("scores", [])

        # Format as detection objects
        detections = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            detection = {
                'bbox': bbox,
                'label': label,
                'confidence': scores[i] if scores and i < len(scores) else 0.9  # Default confidence
            }
            detections.append(detection)

        return detections

    def detect_and_visualize(self, image: Image.Image, output_path: str):
        """
        Detect objects and save annotated image.

        Args:
            image: PIL Image to process
            output_path: Path to save annotated image

        Returns:
            List of detections
        """
        detections = self.detect(image)
        annotated_image = self.visualize_detections(image, detections)
        annotated_image.save(output_path)
        return detections

    def visualize_detections(self, image: Image.Image, detections: list) -> Image.Image:
        """
        Draw bounding boxes and labels on image.

        Args:
            image: Original PIL Image
            detections: List of detection dictionaries

        Returns:
            Annotated PIL Image
        """
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection.get('confidence', 0.0)

            # Draw bounding box
            draw.rectangle(bbox, outline="red", width=2)

            # Draw label with confidence
            text = f"{label} ({confidence:.2f})" if confidence > 0 else label
            draw.text((bbox[0], bbox[1] - 10), text, fill="red")

        return annotated_image

    def detect_and_describe(self, image: Image.Image, save_crops: bool = False, crop_dir: str = None):
        """
        Detect objects and optionally save cropped regions with descriptions.

        Args:
            image: PIL Image to process
            save_crops: Whether to save object crops
            crop_dir: Directory to save crops

        Returns:
            List of detection dictionaries with optional crop paths
        """
        detections = self.detect(image)

        if save_crops and crop_dir:
            import os
            os.makedirs(crop_dir, exist_ok=True)

            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                label = detection['label']

                # Crop the object region
                cropped = image.crop(bbox)

                # Save the crop
                crop_filename = f"{label}_{i}.jpg"
                crop_path = os.path.join(crop_dir, crop_filename)
                cropped.save(crop_path)

                # Add crop path to detection
                detection['crop_path'] = crop_path

        return detections

    def describe_region(self, image: Image.Image, task: str = "<MORE_DETAILED_CAPTION>") -> str:
        """
        Get detailed description of image region.

        Args:
            image: PIL Image (typically cropped region)
            task: Florence-2 task string (default: detailed caption)

        Returns:
            Description string
        """
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        inputs = self.processor(text=task, images=image, return_tensors="pt")

        # Move tensors to device
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
            )

        # Decode the generated text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse the result
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        # Extract the description
        description = parsed.get(task, "No description available")
        return description


# Example usage and testing
if __name__ == "__main__":
    # Test Florence-2 simplified implementation
    florence = Florence2()

    # Test object detection (commented out to avoid loading during import)
    # image = Image.open("test_image.jpg")
    # detections = florence.detect(image)
    # print(f"Detected {len(detections)} objects")

    print("✓ Florence-2 simplified implementation ready")