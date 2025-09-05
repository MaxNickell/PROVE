# Florence-2 Object Detection with Attribute Extraction

This implementation uses Microsoft's Florence-2 model for object detection with confidence scores and detailed attribute extraction through LLM-based normalization.

## Features

1. **Object Detection with Confidence Scores**: Detects objects in images and provides confidence scores for each detection
2. **Object Cropping**: Automatically crops detected objects from images
3. **Detailed Descriptions**: Uses Florence-2 to generate detailed descriptions of each cropped object
4. **Normalized Attributes**: Uses an LLM to extract and normalize attributes into categories like color, material, texture, shape, etc.

## Key Components

### Florence2 Class (`src/vision/florence2.py`)
- `detect()`: Detect objects with bounding boxes and confidence scores
- `crop_object()`: Crop a detected object from the image
- `describe_region()`: Get a detailed description of an image region
- `detect_and_describe()`: Integrated method that detects and describes all objects

### AttributeExtractor Class (`src/pipeline/attribute_extractor.py`)
- `extract_attributes()`: Extract normalized attributes for all detected objects
- Uses LLM to parse descriptions into structured attribute categories

## Usage Example

```python
from PIL import Image
from src.vision.florence2 import Florence2
from src.pipeline.attribute_extractor import AttributeExtractor

# Initialize
florence = Florence2()
attr_extractor = AttributeExtractor()

# Load image
image = Image.open("your_image.jpg")

# Detect objects with confidence scores
detections = florence.detect(image, return_scores=True)

# Get descriptions for each object
for det in detections:
    cropped = florence.crop_object(image, det['bbox'])
    description = florence.describe_region(cropped)
    det['description'] = description

# Extract normalized attributes
attributes = attr_extractor.extract_attributes("your_image.jpg", detections)

# Results include:
# - Bounding boxes with confidence scores
# - Detailed descriptions
# - Normalized attributes (color, material, shape, etc.)
```

## Attribute Categories

The system extracts attributes into these categories:
- **color**: Specific colors (red, blue, multicolored)
- **material**: What it's made of (metal, wood, plastic)
- **texture**: Surface qualities (smooth, rough, glossy)
- **shape**: Geometric descriptions (round, square, curved)
- **size**: Relative size (large, small, tall)
- **state**: Current state (open, closed, moving)
- **pattern**: Visual patterns (striped, dotted, solid)
- **style**: Design style (modern, vintage, casual)
- **condition**: Physical condition (new, worn, clean)
- **function**: Purpose or action (carrying, supporting)

## Integration with Orchestrator

The Orchestrator now automatically:
1. Detects objects with confidence scores
2. Extracts normalized attributes for each object
3. Stores attributes in `self.attributes[image_id]`
4. Adds attributes directly to each object dict for convenience

## Testing

Run the test scripts to verify functionality:
```bash
python test_florence_attributes.py
python example_florence_usage.py
```

## Notes

- Confidence scores are computed based on model outputs and object size
- The LLM-based normalization ensures consistent attribute formatting
- Cropped objects can be saved for debugging/visualization
- The system gracefully handles errors with fallback attributes
