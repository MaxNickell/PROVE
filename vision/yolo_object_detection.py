from ultralytics import YOLO
from PIL import Image
import requests


class YoloObjectDetection():
    def __init__(self, model_name = "yolo11m.pt"):
        self.model = YOLO(model_name)

    def detect(self, image_url):
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        results = self.model(image) 

        return results[0]

    def detect_and_save(self, image_url, save_path):
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        results = self.model(image) 
        results[0].save(save_path)

        return results[0]
