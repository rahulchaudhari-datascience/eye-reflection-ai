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

        # Fallback when FaceMesh fails (small face, occlusions, glasses glare, etc.).
        # We try a couple variants because performance differs per image.
        self._eye_cascades = [
            cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
            ),
            cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            ),
        ]

        # Prefer richer landmark sets when available (more robust than a few hardcoded points).
        self._left_eye_indices, self._right_eye_indices = self._resolve_eye_landmark_indices()

    def _resolve_eye_landmark_indices(self):
        # MediaPipe provides connection sets; we convert them into index sets.
        try:
            fm = self.mp_face_mesh
            left = set()
            right = set()

            for a, b in getattr(fm, "FACEMESH_LEFT_EYE"):
                left.add(int(a))
                left.add(int(b))
            for a, b in getattr(fm, "FACEMESH_RIGHT_EYE"):
                right.add(int(a))
                right.add(int(b))

            # Iris landmarks exist when refine_landmarks=True.
            if hasattr(fm, "FACEMESH_LEFT_IRIS"):
                for a, b in getattr(fm, "FACEMESH_LEFT_IRIS"):
                    left.add(int(a))
                    left.add(int(b))
            if hasattr(fm, "FACEMESH_RIGHT_IRIS"):
                for a, b in getattr(fm, "FACEMESH_RIGHT_IRIS"):
                    right.add(int(a))
                    right.add(int(b))

            if left and right:
                return sorted(left), sorted(right)
        except Exception:
            pass

        # Fallback: small, stable subset.
        return (
            [33, 133, 160, 159, 158, 157, 173],
            [362, 263, 387, 386, 385, 384, 398],
        )

    def _resize_for_detection(
        self,
        image: np.ndarray,
        *,
        max_width: int = 1024,
        min_width: int = 320,
    ) -> tuple[np.ndarray, float, float]:
        """Resize for detection and return (resized, scale_x, scale_y).

        scale_x/scale_y map coordinates from resized -> original.
        """

        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
            return image, 1.0, 1.0

        if w > max_width:
            scale = max_width / float(w)
            new_size = (max_width, max(1, int(h * scale)))
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            scale_x = w / float(resized.shape[1])
            scale_y = h / float(resized.shape[0])
            return resized, scale_x, scale_y

        if w < min_width:
            scale = min_width / float(w)
            new_size = (min_width, max(1, int(h * scale)))
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            scale_x = w / float(resized.shape[1])
            scale_y = h / float(resized.shape[0])
            return resized, scale_x, scale_y

        return image, 1.0, 1.0

    def _safe_crop(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return image[0:0, 0:0]
        return image[y1:y2, x1:x2]

    def _fallback_haar_eyes(self, image: np.ndarray, *, scale_x: float = 1.0, scale_y: float = 1.0):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Improve contrast; helps with glasses glare and low-light images.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        def detect(gray_roi: np.ndarray):
            for cascade in self._eye_cascades:
                eyes_ = cascade.detectMultiScale(
                    gray_roi,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(18, 18),
                )
                if eyes_ is not None and len(eyes_) > 0:
                    return eyes_
            return None

        eyes = detect(gray)

        # If global scan fails (common for very wide eye-only crops), scan halves separately.
        h, w = gray.shape[:2]
        if eyes is None or len(eyes) < 2:
            left_half = gray[:, : w // 2]
            right_half = gray[:, w // 2 :]

            left_det = detect(left_half)
            right_det = detect(right_half)

            if left_det is None or len(left_det) == 0 or right_det is None or len(right_det) == 0:
                return None

            # Choose the largest detection per half.
            (lx, ly, lw, lh) = max(left_det, key=lambda r: int(r[2] * r[3]))
            (rx, ry, rw, rh) = max(right_det, key=lambda r: int(r[2] * r[3]))
            (x1, y1, w1, h1) = (int(lx), int(ly), int(lw), int(lh))
            (x2, y2, w2, h2) = (int(rx + w // 2), int(ry), int(rw), int(rh))
        else:
            # pick two largest detections
            eyes = sorted(eyes, key=lambda r: int(r[2] * r[3]), reverse=True)[:2]
            # left/right order by x
            eyes = sorted(eyes, key=lambda r: r[0])
            (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[0], eyes[1]

        pad1 = int(0.15 * max(w1, h1))
        pad2 = int(0.15 * max(w2, h2))

        # Map detection coords back to the original scale.
        ox1 = int(round((x1 - pad1) * scale_x))
        oy1 = int(round((y1 - pad1) * scale_y))
        ox2 = int(round((x1 + w1 + pad1) * scale_x))
        oy2 = int(round((y1 + h1 + pad1) * scale_y))

        px1 = int(round((x2 - pad2) * scale_x))
        py1 = int(round((y2 - pad2) * scale_y))
        px2 = int(round((x2 + w2 + pad2) * scale_x))
        py2 = int(round((y2 + h2 + pad2) * scale_y))

        # NOTE: `image` passed here should be the *original* image when scale != 1.
        left_eye = self._safe_crop(self._orig_image, ox1, oy1, ox2, oy2)
        right_eye = self._safe_crop(self._orig_image, px1, py1, px2, py2)

        if left_eye.size == 0 or right_eye.size == 0:
            return None

        return {"left_eye": left_eye, "right_eye": right_eye}

    def _fallback_split_eyes(self, image: np.ndarray):
        """Last-resort fallback for extreme crops.

        When both FaceMesh and Haar cascades fail (common for low-res, partial-face,
        strong glare, or highly cropped images), we approximate eye regions by
        splitting the image into left/right halves.
        """

        h, w = image.shape[:2]
        if h < 20 or w < 40:
            return None

        # Focus on the central vertical band where eyes typically are.
        y1 = int(0.10 * h)
        y2 = int(0.90 * h)
        pad_x = int(0.05 * w)
        mid = w // 2

        left_eye = self._safe_crop(image, 0 + pad_x, y1, mid - pad_x, y2)
        right_eye = self._safe_crop(image, mid + pad_x, y1, w - pad_x, y2)

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

        # Keep original for high-quality crops; detect on a resized copy.
        self._orig_image = image
        detect_img, scale_x, scale_y = self._resize_for_detection(image)

        results = self.face_mesh.process(detect_img)

        if not results.multi_face_landmarks:
            haar = self._fallback_haar_eyes(detect_img, scale_x=scale_x, scale_y=scale_y)
            if haar is not None:
                return haar
            return self._fallback_split_eyes(self._orig_image)

        h, w, _ = detect_img.shape
        left_eye_indices = self._left_eye_indices
        right_eye_indices = self._right_eye_indices

        face_landmarks = results.multi_face_landmarks[0]

        left_points = []
        right_points = []

        for idx in left_eye_indices:
            landmark = face_landmarks.landmark[idx]
            # Map from detect_img -> original image coords
            left_points.append(
                (
                    int(round((landmark.x * w) * scale_x)),
                    int(round((landmark.y * h) * scale_y)),
                )
            )

        for idx in right_eye_indices:
            landmark = face_landmarks.landmark[idx]
            right_points.append(
                (
                    int(round((landmark.x * w) * scale_x)),
                    int(round((landmark.y * h) * scale_y)),
                )
            )

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

        left_eye = self._safe_crop(self._orig_image, lx1 - lpad, ly1 - lpad, lx2 + lpad, ly2 + lpad)
        right_eye = self._safe_crop(self._orig_image, rx1 - rpad, ry1 - rpad, rx2 + rpad, ry2 + rpad)

        if left_eye.size == 0 or right_eye.size == 0:
            haar = self._fallback_haar_eyes(detect_img, scale_x=scale_x, scale_y=scale_y)
            if haar is not None:
                return haar
            return self._fallback_split_eyes(self._orig_image)

        return {
            "left_eye": left_eye,
            "right_eye": right_eye
        }