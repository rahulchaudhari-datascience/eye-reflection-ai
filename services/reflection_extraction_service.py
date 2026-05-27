import cv2
import numpy as np


class ReflectionExtractionService:

    def extract(self, eye_image, pupil_circle):

        if pupil_circle is None:
            return eye_image

        x, y, r = pupil_circle

        crop = eye_image[
            max(0, y-r):y+r,
            max(0, x-r):x+r
        ]

        enhanced = cv2.detailEnhance(crop, sigma_s=10, sigma_r=0.15)

        return enhanced