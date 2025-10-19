"""
Utility functions for common image operations across the pipeline.
Consolidates repeated patterns for cleaner, more maintainable code.
"""

from PIL import Image
from typing import Dict
import os


def load_rgb_image(image_path: str) -> Image.Image:
    """
    Load an image and ensure it's in RGB format.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image in RGB mode

    Raises:
        FileNotFoundError: If image file doesn't exist
        IOError: If image cannot be opened
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def load_multiple_images(image_paths: Dict[str, str]) -> Dict[str, Image.Image]:
    """
    Load multiple images from a dictionary of paths.

    Args:
        image_paths: Dictionary mapping image_id to image path

    Returns:
        Dictionary mapping image_id to loaded PIL Image in RGB mode

    Raises:
        FileNotFoundError: If any image file doesn't exist
        IOError: If any image cannot be opened
    """
    loaded_images = {}

    for image_id, image_path in image_paths.items():
        loaded_images[image_id] = load_rgb_image(image_path)

    return loaded_images
