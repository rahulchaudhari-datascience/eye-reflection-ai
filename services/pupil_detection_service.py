import cv2
import numpy as np


class PupilDetectionService:

    def detect(self, eye_image):

        if eye_image is None or eye_image.size == 0:
            return None

        gray = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)

        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=5,
            maxRadius=40
        )

        if circles is not None:

            circles = np.uint16(np.around(circles))

            x, y, r = circles[0][0]
            return int(x), int(y), int(r)

        return None