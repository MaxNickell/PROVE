from vision.blip_image_text_to_text import BlipImageTextToText
from vision.grounding_dino_zero_shot_object_detection import GroundingDinoZeroShotObjectDetection
from vision.sam_mask import SamMask
from vision.vilt_vqa import ViltVqa
from vision.yolo_object_detection import YoloObjectDetection
from dotenv import load_dotenv
from PIL import Image
import requests


def generate_graph(image_url):
    img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    # Step 0: Initialize models
    object_detector = YoloObjectDetection()
    zero_shot_object_detector = GroundingDinoZeroShotObjectDetection()
    vqa = ViltVqa()

    # Step 1: Object detection first pass
    result = object_detector.detect_and_save(image_url, "yolo_result.jpg") 

    # Step 2: Object detection second pass



    # Step 3: Get attributes
    attributes = {
        "color": "What is the color of the object?",
        "size": "What is the size of the object?",
        "material": "What is the material of the object?",
        "orientation": "What is the orientation of the object?",
        "texture": "What is the texture of the object?",
    }

    boxes = result.boxes
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f'---Box {i}---\nClass: {class_name}\nConfidence: {confidence}\nCoords: {x1, y1, x2, y2}')

        for key, value in zip(attributes.keys(), attributes.values()):
            crop = img.crop((x1, y1, x2, y2))

            answer = vqa.query_image(image_url, value)
            print(f'Attribute {key}: {answer}')

        



if __name__ == "__main__":
    load_dotenv()

    image_url = "http://2.bp.blogspot.com/_X1IWXuEbgXI/TNek4LTTeII/AAAAAAAACx8/ZAPpNypF-RA/s640/hyena_eating_zebra_vultures.jpg"
    generate_graph(image_url)



