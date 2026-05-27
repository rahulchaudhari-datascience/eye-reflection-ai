from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


class EyeLandmarkService:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.3,
        )

        # Fallback when FaceMesh fails (small face, occlusions, etc.).
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    def _resize_for_detection(self, image: np.ndarray, max_width: int = 1024) -> np.ndarray:
        h, w = image.shape[:2]
        if w <= max_width:
            return image
        scale = max_width / float(w)
        new_size = (max_width, max(1, int(h * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _safe_crop(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return image[0:0, 0:0]
        return image[y1:y2, x1:x2]

    def _fallback_haar_eyes(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eyes = self._eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )
        if eyes is None or len(eyes) < 2:
            return None

        # pick two largest detections
        eyes = sorted(eyes, key=lambda r: int(r[2] * r[3]), reverse=True)[:2]
        # left/right order by x
        eyes = sorted(eyes, key=lambda r: r[0])

        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[0], eyes[1]

        pad1 = int(0.15 * max(w1, h1))
        pad2 = int(0.15 * max(w2, h2))

        left_eye = self._safe_crop(image, x1 - pad1, y1 - pad1, x1 + w1 + pad1, y1 + h1 + pad1)
        right_eye = self._safe_crop(image, x2 - pad2, y2 - pad2, x2 + w2 + pad2, y2 + h2 + pad2)

        if left_eye.size == 0 or right_eye.size == 0:
            return None

        return {"left_eye": left_eye, "right_eye": right_eye}

    def extract_eyes(self, image):
        if image is None:
            return None

        # mediapipe works better with moderately sized RGB uint8 images.
        if not isinstance(image, np.ndarray):
            return None

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim != 3 or image.shape[2] != 3:
            return None

        image = self._resize_for_detection(image)

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return self._fallback_haar_eyes(image)

        h, w, _ = image.shape
        left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
        right_eye_indices = [362, 263, 387, 386, 385, 384, 398]

        face_landmarks = results.multi_face_landmarks[0]

        left_points = []
        right_points = []

        for idx in left_eye_indices:
            landmark = face_landmarks.landmark[idx]
            left_points.append((int(landmark.x * w), int(landmark.y * h)))

        for idx in right_eye_indices:
            landmark = face_landmarks.landmark[idx]
            right_points.append((int(landmark.x * w), int(landmark.y * h)))

        left_x = [p[0] for p in left_points]
        left_y = [p[1] for p in left_points]

        right_x = [p[0] for p in right_points]
        right_y = [p[1] for p in right_points]

        # Add padding so we don't crop too tightly.
        lx1, lx2 = min(left_x), max(left_x)
        ly1, ly2 = min(left_y), max(left_y)
        rx1, rx2 = min(right_x), max(right_x)
        ry1, ry2 = min(right_y), max(right_y)

        lpad = int(0.25 * max(1, lx2 - lx1))
        rpad = int(0.25 * max(1, rx2 - rx1))

        left_eye = self._safe_crop(image, lx1 - lpad, ly1 - lpad, lx2 + lpad, ly2 + lpad)
        right_eye = self._safe_crop(image, rx1 - rpad, ry1 - rpad, rx2 + rpad, ry2 + rpad)

        if left_eye.size == 0 or right_eye.size == 0:
            return self._fallback_haar_eyes(image)

        return {
            "left_eye": left_eye,
            "right_eye": right_eye
        }