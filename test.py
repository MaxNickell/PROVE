from vision.blip_image_text_to_text import BlipImageTextToText
from vision.grounding_dino_zero_shot_object_detection import GroundingDinoZeroShotObjectDetection
from vision.sam_mask import SamMask
from vision.vilt_vqa import ViltVqa
from vision.yolo_object_detection import YoloObjectDetection
from dotenv import load_dotenv
from PIL import Image
import requests


def generate_graph(image_url):
    # YOLO -> can show exits object
    # Verify that the object from YOLO is correct
    
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    # Step 0: Initialize models
    object_detector = YoloObjectDetection()
    zero_shot_object_detector = GroundingDinoZeroShotObjectDetection()
    vqa = ViltVqa()
    captioner = BlipImageTextToText()

    # Step 1: Object detection first pass
    print('----RUNNING YOLO----')
    results = object_detector.detect_and_save(image, "yolo_result.jpg")

    # Step 2: Noun Captioning
    print('----RUNNING BLIP----')
    prompt = (
        "Look at the image and list every distinct object you can see.\n"
        "FORMAT RULES – follow all four EXACTLY:\n"
        "1. Output only common-noun words in lowercase (no adjectives, verbs, numbers, or proper nouns).\n"
        "2. Separate nouns with a single comma and one space.\n"
        "3. List each noun only once (no duplicates).\n"
        "4. Do not add any other words, punctuation, or explanations.\n"
        "Example (not related to the image): cat, glove, lamp, car\n"
        "List the nouns now:"
    )
    nouns = captioner.caption_and_parse_nouns(image, prompt)
    print(f'Extracted nouns: {nouns}')

    # Step 3: Object detection second pass
    print('----RUNNING GROUNDING DINO----')
    result2 = zero_shot_object_detector.detect_and_save(image, nouns, "grounding_dino_result.jpg")



    # Step 4: Get attributes
    attributes = {
        "color": "What is the color of the object?",
        "size": "What is the size of the object?",
        "material": "What is the material of the object?",
        "orientation": "What is the orientation of the object?",
        "texture": "What is the texture of the object?",
    }


if __name__ == "__main__":
    load_dotenv()

    image_url = "http://www.louwphotography.com/Africa/Tanzania/Highlights/i-zVB3Zfj/0/L/IMG_5487-L.jpg"
    generate_graph(image_url)


