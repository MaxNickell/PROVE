"""
Image processing utilities for PROVE pipeline.
Provides union crops, bounding box overlays, and batch processing for LLaVA verification.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import os

from src.core.types import ObjectDetection


class ImageProcessingError(RuntimeError):
    """Custom exception for image processing failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


def union_crop(image: Image.Image, bbox_list: List[List[float]], 
               padding: int = 10) -> Image.Image:
    """
    Create union crop of multiple bounding boxes.
    
    Args:
        image: PIL Image object
        bbox_list: List of bounding boxes in [x1, y1, x2, y2] format
        padding: Extra padding around union in pixels
        
    Returns:
        Image.Image: Cropped image containing all bounding boxes
        
    Raises:
        ImageProcessingError: If cropping fails
    """
    try:
        if not bbox_list:
            raise ImageProcessingError("No bounding boxes provided for union crop")
        
        # Calculate union of all bounding boxes
        min_x1 = min(bbox[0] for bbox in bbox_list)
        min_y1 = min(bbox[1] for bbox in bbox_list)
        max_x2 = max(bbox[2] for bbox in bbox_list)
        max_y2 = max(bbox[3] for bbox in bbox_list)
        
        # Add padding while staying within image bounds
        width, height = image.size
        
        union_x1 = max(0, min_x1 - padding)
        union_y1 = max(0, min_y1 - padding)
        union_x2 = min(width, max_x2 + padding)
        union_y2 = min(height, max_y2 + padding)
        
        # Ensure valid crop coordinates
        if union_x1 >= union_x2 or union_y1 >= union_y2:
            raise ImageProcessingError("Invalid union crop coordinates")
        
        # Crop and return
        return image.crop((union_x1, union_y1, union_x2, union_y2))
        
    except Exception as e:
        raise ImageProcessingError(f"Union crop failed: {e}")


def draw_bounding_boxes(image: Image.Image, subject_bbox: List[float], 
                       object_bbox: List[float], line_width: int = 3) -> Image.Image:
    """
    Draw red rectangle around subject, yellow rectangle around object.
    
    Args:
        image: PIL Image object
        subject_bbox: Subject bounding box [x1, y1, x2, y2]
        object_bbox: Object bounding box [x1, y1, x2, y2] 
        line_width: Width of bounding box lines
        
    Returns:
        Image.Image: Image with colored bounding boxes drawn
        
    Raises:
        ImageProcessingError: If drawing fails
    """
    try:
        # Create a copy to avoid modifying original
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Draw subject bounding box in red
        draw.rectangle(subject_bbox, outline="red", width=line_width)
        
        # Draw object bounding box in yellow
        draw.rectangle(object_bbox, outline="yellow", width=line_width)
        
        # Add labels if there's space
        try:
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Add "SUBJECT" label above subject bbox
            subject_label_pos = (subject_bbox[0], max(0, subject_bbox[1] - 20))
            draw.text(subject_label_pos, "SUBJECT", fill="red", font=font)
            
            # Add "OBJECT" label above object bbox  
            object_label_pos = (object_bbox[0], max(0, object_bbox[1] - 20))
            draw.text(object_label_pos, "OBJECT", fill="yellow", font=font)
            
        except Exception:
            # If labeling fails, continue without labels
            pass
        
        return annotated_image
        
    except Exception as e:
        raise ImageProcessingError(f"Bounding box drawing failed: {e}")


def batch_crop_objects(image: Image.Image, objects: List[ObjectDetection], 
                      padding: int = 10) -> List[Image.Image]:
    """
    Efficiently crop multiple objects from same image.
    
    Args:
        image: PIL Image object
        objects: List of ObjectDetection instances
        padding: Padding around each crop in pixels
        
    Returns:
        List[Image.Image]: List of cropped images for each object
        
    Raises:
        ImageProcessingError: If batch cropping fails
    """
    try:
        crops = []
        width, height = image.size
        
        for obj in objects:
            bbox = obj.bbox
            
            # Add padding while staying within bounds
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)  
            x2 = min(width, bbox[2] + padding)
            y2 = min(height, bbox[3] + padding)
            
            # Ensure valid crop
            if x1 >= x2 or y1 >= y2:
                # Create a small fallback crop
                crops.append(Image.new('RGB', (50, 50), color='white'))
                continue
            
            # Crop object
            cropped = image.crop((x1, y1, x2, y2))
            crops.append(cropped)
        
        return crops
        
    except Exception as e:
        raise ImageProcessingError(f"Batch cropping failed: {e}")


def blackout_outside_union(image: Image.Image, bbox_list: List[List[float]], 
                          blackout_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Black out content outside the union of bounding boxes.
    
    Args:
        image: PIL Image object
        bbox_list: List of bounding boxes to preserve
        blackout_color: RGB color for blackout areas
        
    Returns:
        Image.Image: Image with areas outside union blacked out
        
    Raises:
        ImageProcessingError: If blackout processing fails
    """
    try:
        if not bbox_list:
            raise ImageProcessingError("No bounding boxes provided for blackout")
        
        # Calculate union bounds
        min_x1 = min(bbox[0] for bbox in bbox_list)
        min_y1 = min(bbox[1] for bbox in bbox_list) 
        max_x2 = max(bbox[2] for bbox in bbox_list)
        max_y2 = max(bbox[3] for bbox in bbox_list)
        
        # Create mask for union area
        width, height = image.size
        mask = Image.new('L', (width, height), 0)  # Black mask
        mask_draw = ImageDraw.Draw(mask)
        
        # White rectangle for union area (areas to preserve)
        union_coords = (max(0, min_x1), max(0, min_y1), 
                       min(width, max_x2), min(height, max_y2))
        mask_draw.rectangle(union_coords, fill=255)
        
        # Create blackout image
        blackout_img = Image.new('RGB', (width, height), blackout_color)
        
        # Composite: use original where mask is white, blackout where mask is black
        result = Image.composite(image, blackout_img, mask)
        
        return result
        
    except Exception as e:
        raise ImageProcessingError(f"Blackout processing failed: {e}")


def create_relationship_crop(image: Image.Image, subject_obj: ObjectDetection, 
                           object_obj: ObjectDetection, padding: int = 20,
                           use_blackout: bool = True) -> Image.Image:
    """
    Create a relationship crop for LLaVA verification.
    Combines union cropping, bounding box annotation, and optional blackout.
    
    Args:
        image: PIL Image object
        subject_obj: Subject ObjectDetection instance
        object_obj: Object ObjectDetection instance
        padding: Padding around union crop
        use_blackout: Whether to black out areas outside objects
        
    Returns:
        Image.Image: Processed relationship crop ready for LLaVA
        
    Raises:
        ImageProcessingError: If relationship crop creation fails
    """
    try:
        bbox_list = [subject_obj.bbox, object_obj.bbox]
        
        # Create union crop
        union_cropped = union_crop(image, bbox_list, padding)
        
        # Adjust bounding box coordinates for the crop
        crop_bbox = get_union_bounds(bbox_list, padding, image.size)
        crop_x1, crop_y1 = crop_bbox[0], crop_bbox[1]
        
        # Adjust coordinates relative to crop
        subject_bbox_adj = [
            subject_obj.bbox[0] - crop_x1,
            subject_obj.bbox[1] - crop_y1, 
            subject_obj.bbox[2] - crop_x1,
            subject_obj.bbox[3] - crop_y1
        ]
        
        object_bbox_adj = [
            object_obj.bbox[0] - crop_x1,
            object_obj.bbox[1] - crop_y1,
            object_obj.bbox[2] - crop_x1, 
            object_obj.bbox[3] - crop_y1
        ]
        
        # Apply blackout if requested (before drawing boxes)
        if use_blackout:
            bbox_list_adj = [subject_bbox_adj, object_bbox_adj]
            union_cropped = blackout_outside_union(union_cropped, bbox_list_adj)
        
        # Draw bounding boxes
        annotated = draw_bounding_boxes(union_cropped, subject_bbox_adj, object_bbox_adj)
        
        return annotated
        
    except Exception as e:
        raise ImageProcessingError(f"Relationship crop creation failed: {e}")


def get_union_bounds(bbox_list: List[List[float]], padding: int, 
                    image_size: Tuple[int, int]) -> List[float]:
    """
    Calculate union bounds with padding and image size constraints.
    
    Args:
        bbox_list: List of bounding boxes
        padding: Padding to add
        image_size: (width, height) of image
        
    Returns:
        List[float]: Union bounds [x1, y1, x2, y2]
    """
    if not bbox_list:
        return [0, 0, image_size[0], image_size[1]]
    
    min_x1 = min(bbox[0] for bbox in bbox_list)
    min_y1 = min(bbox[1] for bbox in bbox_list)
    max_x2 = max(bbox[2] for bbox in bbox_list) 
    max_y2 = max(bbox[3] for bbox in bbox_list)
    
    width, height = image_size
    
    union_x1 = max(0, min_x1 - padding)
    union_y1 = max(0, min_y1 - padding)
    union_x2 = min(width, max_x2 + padding)
    union_y2 = min(height, max_y2 + padding)
    
    return [union_x1, union_y1, union_x2, union_y2]


def save_debug_artifacts(crops: List[Image.Image], relationships: List[str], 
                        output_dir: str) -> List[str]:
    """
    Save cropped images and relationships for debugging.
    
    Args:
        crops: List of cropped images
        relationships: List of relationship descriptions
        output_dir: Directory to save artifacts
        
    Returns:
        List[str]: Paths to saved files
        
    Raises:
        ImageProcessingError: If saving fails
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_paths = []
        
        for i, (crop, relationship) in enumerate(zip(crops, relationships)):
            # Clean relationship text for filename
            safe_name = "".join(c for c in relationship if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')[:50]  # Limit length
            
            filename = f"relationship_crop_{i}_{safe_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            crop.save(filepath)
            saved_paths.append(filepath)
        
        return saved_paths
        
    except Exception as e:
        raise ImageProcessingError(f"Debug artifact saving failed: {e}")


# Utility functions for validation and testing
def validate_bbox(bbox: List[float], image_size: Tuple[int, int]) -> bool:
    """
    Validate that bounding box coordinates are valid.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_size: (width, height) of image
        
    Returns:
        bool: True if bbox is valid
    """
    try:
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        width, height = image_size
        
        # Check coordinates are in valid range
        if not (0 <= x1 < x2 <= width):
            return False
        if not (0 <= y1 < y2 <= height):
            return False
        
        return True
        
    except Exception:
        return False


def get_processing_summary(image: Image.Image, objects: List[ObjectDetection], 
                          operations: List[str]) -> dict:
    """
    Get summary of image processing operations.
    
    Args:
        image: Original image
        objects: List of objects processed
        operations: List of operation names performed
        
    Returns:
        dict: Processing summary
    """
    return {
        "original_size": image.size,
        "objects_processed": len(objects),
        "operations": operations,
        "total_bbox_area": sum(
            (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1]) 
            for obj in objects
        ),
        "image_area": image.size[0] * image.size[1]
    }


# Example usage and testing
if __name__ == "__main__":
    # Test image processing utilities
    print("Testing image processing utilities...")
    
    # Test bbox validation
    test_bbox = [10.0, 20.0, 100.0, 150.0]
    test_size = (400, 300)
    is_valid = validate_bbox(test_bbox, test_size)
    print(f"✓ Bbox validation: {is_valid}")
    
    # Test union bounds calculation
    test_bboxes = [
        [10.0, 20.0, 100.0, 150.0],
        [50.0, 80.0, 200.0, 220.0]
    ]
    union_bounds = get_union_bounds(test_bboxes, 10, test_size)
    print(f"✓ Union bounds: {union_bounds}")
    
    # Test with sample objects
    obj1 = ObjectDetection(0, "person", [10.0, 20.0, 100.0, 150.0], 0.9)
    obj2 = ObjectDetection(1, "car", [50.0, 80.0, 200.0, 220.0], 0.8)
    
    summary = get_processing_summary(
        Image.new('RGB', test_size, 'white'), 
        [obj1, obj2], 
        ["union_crop", "draw_boxes"]
    )
    print(f"✓ Processing summary: {summary}")
    
    print("✓ Image processing utilities ready!")