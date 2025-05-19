import sys
from vision.captioner import BLIPCaptioner
from vision.object_detector import GroundingDINODetector
from vision.attribute_extractor import ViLTAttributeExtractor
from vision.masker import SAMMasker
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests

if __name__ == "__main__":
    load_dotenv()
    
    captioner = BLIPCaptioner()
    detector = GroundingDINODetector()
    attribute_extractor = ViLTAttributeExtractor()
    masker = SAMMasker()

    all_masks = masker.mask_all("https://s-media-cache-ak0.pinimg.com/236x/6b/d8/29/6bd829145660996d805a0bd080233a40--wine-reviews-costco.jpg")

    image = Image.open(requests.get("https://s-media-cache-ak0.pinimg.com/236x/6b/d8/29/6bd829145660996d805a0bd080233a40--wine-reviews-costco.jpg", stream=True).raw).convert("RGB")

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(image)
    for mask in all_masks:
        masker.show_mask(mask, ax=ax, random_color=True)
    ax.axis("off")
    plt.savefig("masked_output.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    print("Wrote overlay to masked_output.png")