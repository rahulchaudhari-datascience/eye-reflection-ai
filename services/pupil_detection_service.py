from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


PupilCircle = Tuple[int, int, int]


class PupilDetectionService:

    def detect(self, eye_image: np.ndarray) -> Optional[PupilCircle]:
        if eye_image is None or not isinstance(eye_image, np.ndarray) or eye_image.size == 0:
            return None

        img = eye_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = gray.shape[:2]
        min_r = max(4, int(min(h, w) * 0.08))
        max_r = max(min_r + 1, int(min(h, w) * 0.45))

        # 1) HoughCircles (fast, good when edges are clean)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, int(min(h, w) * 0.25)),
            param1=60,
            param2=22,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None and len(circles) > 0:
            circles = np.around(circles[0]).astype(np.int32)

            def circle_score(x: int, y: int, r: int) -> float:
                # Prefer dark interiors near the center of the eye crop.
                if r <= 0:
                    return -1e9
                if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
                    edge_penalty = 1.0
                else:
                    edge_penalty = 0.0
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                mean_inside = float(cv2.mean(gray, mask=mask)[0])
                cx, cy = w / 2.0, h / 2.0
                dist = float(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) / max(1.0, min(h, w))
                # Lower mean_inside (darker) is better; lower dist is better.
                return -mean_inside - 35.0 * dist - 40.0 * edge_penalty

            best = None
            best_s = -1e18
            # Evaluate up to first 10 candidates for speed.
            for (x, y, r) in circles[:10]:
                x_i, y_i, r_i = int(x), int(y), int(r)
                s = circle_score(x_i, y_i, r_i)
                if s > best_s:
                    best_s = s
                    best = (x_i, y_i, r_i)

            if best is not None:
                return self._refine_circle(gray, best)

        # 2) Adaptive-threshold + contour circle fit (robust to glare/blur)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)

        # Pupil tends to be darker; invert so pupil becomes white.
        inv = cv2.bitwise_not(eq)
        bw = cv2.adaptiveThreshold(
            inv,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=2,
        )
        bw = cv2.medianBlur(bw, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        def score(cnt):
            area = cv2.contourArea(cnt)
            if area <= 0:
                return -1.0
            (x, y), r = cv2.minEnclosingCircle(cnt)
            circ_area = np.pi * (r ** 2)
            if circ_area <= 0:
                return -1.0
            circularity = float(area) / float(circ_area)
            # Prefer medium-sized, fairly circular shapes near center.
            cx, cy = w / 2.0, h / 2.0
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            dist_penalty = dist / max(1.0, min(h, w))
            # Prefer darker interior too.
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(max(1.0, r)), 255, -1)
            mean_inside = float(cv2.mean(gray, mask=mask)[0])
            darkness = (255.0 - mean_inside) / 255.0
            return (0.55 * darkness + 0.45 * circularity) * area * (1.0 - 0.6 * dist_penalty)

        best = max(contours, key=score)
        (x, y), r = cv2.minEnclosingCircle(best)
        r = float(r)
        if r < min_r or r > max_r:
            return None

        return self._refine_circle(gray, (int(round(x)), int(round(y)), int(round(r))))

    def _refine_circle(self, gray: np.ndarray, circle: PupilCircle) -> PupilCircle:
        """Local refinement: search small offsets to maximize darkness inside the circle."""

        h, w = gray.shape[:2]
        x0, y0, r0 = [int(v) for v in circle]
        r0 = max(3, r0)

        def mean_inside(x: int, y: int, r: int) -> float:
            if r <= 0:
                return 1e9
            if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
                return 1e9
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            return float(cv2.mean(gray, mask=mask)[0])

        # Small search window proportional to radius.
        step = max(1, int(r0 * 0.12))
        offsets = [-2 * step, -step, 0, step, 2 * step]

        best = (x0, y0, r0)
        best_val = mean_inside(x0, y0, r0)

        # Try slight radius adjustments too.
        for dr in (-step, 0, step):
            rr = max(3, r0 + dr)
            for dx in offsets:
                for dy in offsets:
                    x = x0 + dx
                    y = y0 + dy
                    val = mean_inside(x, y, rr)
                    if val < best_val:
                        best_val = val
                        best = (x, y, rr)

        return best