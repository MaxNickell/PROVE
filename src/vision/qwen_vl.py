"""
Qwen 2.5-VL-7B implementation for PROVE pipeline.
Provides unconstrained visual question answering with direct logit probability extraction.
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Tuple, Union

from src.core.probability import get_verifier_probability


class QwenVLError(Exception):
    """Custom exception for Qwen VL related errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class QwenVL:
    """
    Qwen 2.5-VL-7B Vision-Language Model implementation.
    
    Features:
    - Native bounding box support with <box> tags
    - Direct logit probability extraction 
    - Unconstrained response generation
    - Memory efficient GPU usage
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        """
        Initialize Qwen VL model.

        Args:
            model_name: Model identifier from HuggingFace
            device: Device allocation strategy (default: "auto" for automatic allocation)

        Raises:
            QwenVLError: If model loading fails
        """
        self.model_name = model_name
        self.device = device  # Keep for compatibility, but model uses device_map="auto"
        self._model_loaded = False
        
        try:
            print(f"Loading {model_name}...")
            
            # Load Qwen 2.5-VL model using the correct class
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Use eager attention to avoid flash attention issues
                low_cpu_mem_usage=True
            )
            
            # Load processor (handles both text and images)
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self._model_loaded = True
            print(f"✓ {model_name} loaded successfully")
            
        except Exception as e:
            raise QwenVLError(f"Failed to load Qwen VL model: {e}")
    
    def run_inference_with_logits(self, image: Union[Image.Image, str], prompt: str) -> Tuple[str, List[torch.Tensor]]:
        """
        Run inference and return both response text and generation logits.
        
        Args:
            image: PIL Image object or path to image file
            prompt: Text prompt for the model
            
        Returns:
            Tuple[str, List[torch.Tensor]]: (response_text, generation_logits)
            
        Raises:
            QwenVLError: If inference fails
        """
        if not self.is_available():
            raise QwenVLError("Qwen VL model is not loaded")
        
        try:
            # Handle image input
            if isinstance(image, str):
                image = Image.open(image)
            
            # Prepare inputs in Qwen 2.5-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template and process
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            
            # Move inputs to the correct device (handle both tensor and non-tensor values)
            for key, value in inputs.items():
                if hasattr(value, 'to') and hasattr(value, 'device'):
                    inputs[key] = value.to(self.model.device)
            
            # Generate with logit tracking
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,  # Deterministic for consistent probabilities
                    return_dict_in_generate=True,
                    output_scores=True,  # This gives us the logits
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response (excluding input tokens)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[0][input_length:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Return response and logits for probability extraction
            return response.strip(), outputs.scores
            
        except Exception as e:
            raise QwenVLError(f"Qwen VL inference failed: {e}")
    
    def extract_response_probability(self, logits_sequence: List[torch.Tensor]) -> float:
        """
        Extract probability of the generated response from logits.
        
        Args:
            logits_sequence: List of logit tensors from generation
            
        Returns:
            float: Average probability of generated tokens (0.0 to 1.0)
        """
        if not logits_sequence:
            return 1.0
        
        try:
            token_probs = []
            
            for step_logits in logits_sequence:
                # Convert logits to probabilities
                probs = torch.softmax(step_logits[0], dim=-1)
                
                # Get probability of the most likely token (the one that was chosen)
                max_prob = torch.max(probs).item()
                token_probs.append(max_prob)
            
            # Return average probability across all generated tokens
            avg_prob = sum(token_probs) / len(token_probs)
            return float(avg_prob)
            
        except Exception as e:
            print(f"Warning: Failed to extract probability from logits: {e}")
            return 1.0

    def extract_yes_no_probability_with_verbalizers(
        self,
        logits_sequence: List[torch.Tensor],
        response: str
    ) -> float:
        """
        Extract P(statement is true) using verbalizer sets for robustness.

        Uses verbalizer sets ["yes", "Yes", "YES"] and ["no", "No", "NO"] to handle
        tokenization variants and improve probability estimation stability.

        Args:
            logits_sequence: List of logit tensors from generation
            response: The actual response text (for validation)

        Returns:
            float: P(statement is true) = P(Yes_total) / (P(Yes_total) + P(No_total))
        """
        if not logits_sequence:
            return 0.5  # Default neutral probability

        try:
            # Get final generation step logits (where Yes/No token is produced)
            final_logits = logits_sequence[-1][0]  # Shape: [vocab_size]

            # Define verbalizer sets for robustness
            yes_verbalizers = ["yes", "Yes", "YES"]
            no_verbalizers = ["no", "No", "NO"]

            # Convert logits to probabilities for all tokens
            all_probs = torch.softmax(final_logits, dim=-1)

            # Sum probabilities for all Yes verbalizers
            yes_prob_total = 0.0
            for verbalizer in yes_verbalizers:
                try:
                    # Get token IDs for this verbalizer (handle multi-token cases)
                    token_ids = self.processor.tokenizer.encode(
                        verbalizer,
                        add_special_tokens=False
                    )
                    # Sum probabilities for all tokens of this verbalizer
                    for token_id in token_ids:
                        if token_id < len(all_probs):
                            yes_prob_total += all_probs[token_id].item()
                except Exception as e:
                    # Skip verbalizers that cause encoding issues
                    continue

            # Sum probabilities for all No verbalizers
            no_prob_total = 0.0
            for verbalizer in no_verbalizers:
                try:
                    # Get token IDs for this verbalizer (handle multi-token cases)
                    token_ids = self.processor.tokenizer.encode(
                        verbalizer,
                        add_special_tokens=False
                    )
                    # Sum probabilities for all tokens of this verbalizer
                    for token_id in token_ids:
                        if token_id < len(all_probs):
                            no_prob_total += all_probs[token_id].item()
                except Exception as e:
                    # Skip verbalizers that cause encoding issues
                    continue

            # Calculate P(statement is true) using verbalizer probabilities
            total_verbalizer_prob = yes_prob_total + no_prob_total

            if total_verbalizer_prob > 0:
                prob_statement_true = yes_prob_total / total_verbalizer_prob
            else:
                # No verbalizer tokens found - return neutral probability for failed extraction
                print(f"Warning: No verbalizer tokens found in response: '{response}'")
                prob_statement_true = 0.5  # Neutral probability for failed verbalizer extraction

            return float(prob_statement_true)

        except Exception as e:
            print(f"Warning: Failed to extract yes/no probability with verbalizers: {e}")
            # Return neutral probability for extraction failure
            return 0.5

    def extract_yes_no_probability_with_proper_softmax(
        self,
        logits_sequence: List[torch.Tensor],
        response: str
    ) -> float:
        """
        Extract P(statement is true) using verbalizer summing + 2-token softmax.

        This method delegates to the unified get_verifier_probability() function
        which sums logits for all Yes/No variants and applies proper softmax.

        Args:
            logits_sequence: List of logit tensors from generation
            response: The actual response text (for validation)

        Returns:
            float: P(statement is true) between 0.0 and 1.0
        """
        return get_verifier_probability(
            logits_sequence,
            response,
            self.processor.tokenizer
        )

    def validate_yes_no_response(self, response: str) -> bool:
        """
        Validate that response matches expected Yes/No format.

        Args:
            response: The model's response text

        Returns:
            bool: True if response is a valid Yes/No answer
        """
        clean_response = response.strip().lower()
        valid_responses = ["yes", "no"]
        return clean_response in valid_responses

    def run_inference(self, image: Union[Image.Image, str], prompt: str) -> str:
        """
        Simple inference method that returns only the response text.
        
        Args:
            image: PIL Image object or path to image file
            prompt: Text prompt for the model
            
        Returns:
            str: Model response
        """
        response, _ = self.run_inference_with_logits(image, prompt)
        return response
    
    def format_bbox_prompt(self, bbox: List[float], label: str) -> str:
        """
        Format bounding box coordinates for Qwen's native format.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            label: Object label
            
        Returns:
            str: Formatted bounding box string for Qwen
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        return f"<box>({x1},{y1}),({x2},{y2})</box>{label}"
    
    def get_model_name(self) -> str:
        """Get the name of the Qwen model."""
        return self.model_name
    
    def is_available(self) -> bool:
        """Check if Qwen model is loaded and ready."""
        return (self._model_loaded and 
                hasattr(self, 'model') and 
                hasattr(self, 'processor'))
    
    def get_memory_info(self) -> dict:
        """
        Get GPU memory usage information.
        
        Returns:
            dict: Memory usage statistics
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3)
            
            return {
                "allocated_gb": round(memory_allocated, 2),
                "cached_gb": round(memory_cached, 2), 
                "free_gb": round(memory_free, 2),
                "device": self.device
            }
        except Exception as e:
            return {"error": f"Failed to get memory info: {e}"}
    
    def cleanup(self):
        """Clean up GPU memory."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._model_loaded = False
            print("✓ Qwen VL model cleaned up successfully")
            
        except Exception as e:
            print(f"Warning: Failed to cleanup Qwen VL model: {e}")


# Utility functions for bounding box handling
def convert_florence_to_qwen_bbox(florence_bbox: List[float], image_size: Tuple[int, int] = None) -> str:
    """
    Convert Florence-2 bounding box format to Qwen format.
    
    Args:
        florence_bbox: [x1, y1, x2, y2] coordinates from Florence-2
        image_size: Optional (width, height) for coordinate validation
        
    Returns:
        str: Qwen format bounding box string
    """
    x1, y1, x2, y2 = [int(coord) for coord in florence_bbox]
    
    # Validate coordinates if image size provided
    if image_size:
        width, height = image_size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
    
    return f"<box>({x1},{y1}),({x2},{y2})</box>"


def create_dual_bbox_prompt(obj1_bbox: List[float], obj1_label: str,
                           obj2_bbox: List[float], obj2_label: str) -> str:
    """
    Create prompt with two labeled bounding boxes.
    
    Args:
        obj1_bbox: First object bounding box
        obj1_label: First object label
        obj2_bbox: Second object bounding box  
        obj2_label: Second object label
        
    Returns:
        str: Formatted prompt with both bounding boxes
    """
    box1 = convert_florence_to_qwen_bbox(obj1_bbox)
    box2 = convert_florence_to_qwen_bbox(obj2_bbox)
    
    return f"Object 1: {box1}{obj1_label}\nObject 2: {box2}{obj2_label}"