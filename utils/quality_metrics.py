import cv2
import numpy as np


def calculate_blur_score(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_contrast_score(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return gray.std()