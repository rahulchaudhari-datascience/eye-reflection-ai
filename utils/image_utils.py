import cv2
import numpy as np
from PIL import Image


def load_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    image = np.array(image)

    return image


def resize_image(image, width=800):

    h, w = image.shape[:2]

    ratio = width / w

    new_h = int(h * ratio)

    return cv2.resize(image, (width, new_h))


def convert_bgr_to_rgb(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)