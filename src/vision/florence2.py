from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import torch

from src.core.probability import calibrate_detector_confidence

class Florence2:
    """Simplified Florence-2 implementation based on HuggingFace best practices."""

    def __init__(self, model_name: str = "microsoft/Florence-2-large", device: str = "auto") -> None:
        # Device allocation
        self.has_cuda = torch.cuda.is_available()
        if device == "auto":
            self.device = torch.device("cuda:0" if self.has_cuda else "cpu")
        else:
            self.device = torch.device(device)

        # Use appropriate dtype
        self.dtype = torch.float16 if self.has_cuda else torch.float32

        # Load processor + model
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
        )
        self.model.to(self.device).eval()

    def detect_open_vocabulary(self, image: Image.Image, text_prompt: str) -> dict:
        """
        Detect objects using open vocabulary detection with text prompt.

        Args:
            image: PIL Image
            text_prompt: Text description of object to detect (e.g., "cat")

        Returns:
            dict: Detection result with bboxes, labels, scores
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Open vocabulary detection task with text prompt
        task = "<OPEN_VOCABULARY_DETECTION>"
        prompt = f"{task}{text_prompt}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode generated tokens to text (required by post_process_generation)
        generated_text = self.processor.batch_decode(generated.sequences, skip_special_tokens=False)[0]

        # Parse results from decoded text
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        # Compute sequence-level confidence using geometric mean
        result = parsed.get(task, {})
        raw_confidence = None

        if hasattr(generated, "scores") and hasattr(generated, "beam_indices"):
            transition = self.model.compute_transition_scores(
                sequences=generated.sequences,
                scores=generated.scores,
                beam_indices=generated.beam_indices,
                normalize_logits=True,
            )
            # Geometric mean: exp(mean of log-probs) = length-normalized likelihood
            # This is P(sequence)^(1/L), the standard measure in language modeling
            log_probs = transition[0]
            raw_confidence = torch.exp(log_probs.mean()).item()

        # Inject confidence scores into result
        if raw_confidence is not None:
            num_detections = len(result.get("bboxes", []))
            # All detections share same sequence-level confidence
            result["scores"] = [raw_confidence] * num_detections

        return result

    def detect(self, image: Image.Image, return_scores: bool = True):
        """
        Detect objects in an image with bounding boxes, labels, and confidence scores.

        Args:
            image: PIL Image
            return_scores: Whether to return detection confidences

        Returns:
            List[dict]: detection dictionaries with bbox, label, confidence
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        task = "<OD>"
        inputs = self.processor(text=task, images=image, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        # Decode generated tokens to text (required by post_process_generation)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse results from decoded text
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        results = parsed.get(task, {})
        bboxes = results.get("bboxes", [])
        labels = results.get("labels", [])
        scores = results.get("scores", [])  # now filled if model/processor is updated

        detections = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Get raw confidence score
            raw_conf = scores[i] if scores and i < len(scores) else None

            # Calibrate confidence using anchored sigmoid mapping
            calibrated_conf = calibrate_detector_confidence(raw_conf) if raw_conf is not None else None

            detections.append({
                "bbox": bbox,
                "label": label,
                "confidence": calibrated_conf
            })
        return detections


    def visualize_detections(self, image: Image.Image, detections: list) -> Image.Image:
        """
        Draw bounding boxes and labels with confidence scores on an image.
        """
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for det in detections:
            bbox = det["bbox"]
            label = det["label"]
            conf = det.get("confidence")
            draw.rectangle(bbox, outline="red", width=2)
            text = f"{label} ({conf:.2f})" if conf is not None else label
            draw.text((bbox[0], bbox[1] - 10), text, fill="red")

        return annotated_image

    def detect_and_describe(self, image: Image.Image, save_crops: bool = False, crop_dir: str = None):
        """
        Detect objects and optionally save cropped regions.
        """
        detections = self.detect(image)

        if save_crops and crop_dir:
            import os
            os.makedirs(crop_dir, exist_ok=True)

            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                label = detection["label"]

                cropped = image.crop(bbox)
                crop_filename = f"{label}_{i}.jpg"
                crop_path = os.path.join(crop_dir, crop_filename)
                cropped.save(crop_path)
                detection["crop_path"] = crop_path

        return detections

    def describe_region(self, image: Image.Image, task: str = "<MORE_DETAILED_CAPTION>") -> str:
        """
        Get detailed description of an image region.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(text=task, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        description = parsed.get(task, "No description available")
        return description
