from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


PupilCircle = Tuple[int, int, int]


class ReflectionExtractionService:

    def extract(self, eye_image: np.ndarray, pupil_circle: Optional[PupilCircle]) -> np.ndarray:
        if eye_image is None or not isinstance(eye_image, np.ndarray) or eye_image.size == 0:
            raise ValueError("Invalid eye image")

        img = eye_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Scale up tiny eye crops to stabilize specularity detection.
        img, scaled_circle = self._maybe_upscale_for_reflection(img, pupil_circle)

        # 1) If we have a pupil circle, ALWAYS focus extraction on the pupil region.
        # This matches enterprise pipelines (crop pupil/iris first, then isolate reflection).
        if scaled_circle is None and pupil_circle is not None:
            scaled_circle = pupil_circle
        if scaled_circle is None:
            scaled_circle = self._estimate_pupil_circle(img)

        if scaled_circle is not None:
            pupil_crop = self._extract_pupil_crop(img, scaled_circle)
            if pupil_crop is not None and pupil_crop.size > 0:
                # Prefer specular highlight extraction inside the pupil crop.
                try:
                    reflection = self._extract_specular(pupil_crop, None)
                    if reflection is not None and reflection.size > 0:
                        return reflection
                except Exception:
                    pass

                return pupil_crop

            iris_crop = self._extract_iris_crop(img, scaled_circle)
            if iris_crop is not None and iris_crop.size > 0:
                try:
                    reflection = self._extract_specular(iris_crop, None)
                    if reflection is not None and reflection.size > 0:
                        return reflection
                except Exception:
                    pass
                return iris_crop

        # 2) Otherwise, try specular highlight extraction on the full eye crop.
        try:
            reflection = self._extract_specular(img, None)
            if reflection is not None and reflection.size > 0:
                return reflection
        except Exception:
            pass

        # Last resort: return the whole eye crop.
        return img

    def _maybe_upscale_for_reflection(
        self, eye_image: np.ndarray, pupil_circle: Optional[PupilCircle]
    ) -> tuple[np.ndarray, Optional[PupilCircle]]:
        h, w = eye_image.shape[:2]
        if min(h, w) >= 120:
            return eye_image, pupil_circle

        scale = 2.0
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        up = cv2.resize(eye_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        if pupil_circle is None:
            return up, None

        x, y, r = pupil_circle
        scaled = (int(round(x * scale)), int(round(y * scale)), int(round(r * scale)))
        return up, scaled

    def _estimate_pupil_circle(self, eye_image: np.ndarray) -> Optional[PupilCircle]:
        """Estimate pupil center from the darkest region when Hough fails."""

        gray = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Focus on central region to avoid eyebrow/skin.
        x1, x2 = int(0.15 * w), int(0.85 * w)
        y1, y2 = int(0.20 * h), int(0.85 * h)
        roi = blur[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        thresh_val = np.percentile(roi, 20)
        _, bw = cv2.threshold(roi, int(thresh_val), 255, cv2.THRESH_BINARY_INV)
        bw = cv2.medianBlur(bw, 5)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(best)
        r = max(3, int(round(r)))
        cx = int(round(cx + x1))
        cy = int(round(cy + y1))

        return (cx, cy, r)

    def _extract_pupil_crop(self, eye_image: np.ndarray, pupil_circle: PupilCircle) -> Optional[np.ndarray]:
        """Return a tight crop around the pupil/iris region.

        The reference output shows a small ROI around the pupil first, not the entire eye.
        """

        h, w = eye_image.shape[:2]
        x, y, r = [int(v) for v in pupil_circle]
        if r <= 0:
            return None

        # Use a slightly larger radius than the pupil to include corneal reflection context.
        rr = int(max(12, min(0.45 * min(h, w), 2.8 * r)))

        x1, y1 = max(0, x - rr), max(0, y - rr)
        x2, y2 = min(w, x + rr), min(h, y + rr)
        crop = eye_image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Circular mask to match the "pupil/iris patch" visual.
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.circle(mask, (cw // 2, ch // 2), int(0.98 * min(ch, cw) / 2), 255, -1)
        out = crop.copy()
        out[mask == 0] = 0
        return out

    def _extract_iris_crop(self, eye_image: np.ndarray, pupil_circle: PupilCircle) -> Optional[np.ndarray]:
        h, w = eye_image.shape[:2]
        x, y, r = [int(v) for v in pupil_circle]
        if r <= 0:
            return None

        # Iris radius is typically larger than pupil. Use a conservative multiplier.
        iris_r = int(max(10, min(0.48 * min(h, w), 2.2 * r)))

        x1, y1 = max(0, x - iris_r), max(0, y - iris_r)
        x2, y2 = min(w, x + iris_r), min(h, y + iris_r)
        crop = eye_image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Apply a circular mask so we keep only the globe region.
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cx, cy = cw // 2, ch // 2
        rr = int(0.98 * min(cx, cy))
        cv2.circle(mask, (cx, cy), rr, 255, -1)

        out = crop.copy()
        out[mask == 0] = 0

        # Mild detail enhancement.
        return cv2.detailEnhance(out, sigma_s=10, sigma_r=0.15)

    def _extract_specular(self, eye_image: np.ndarray, pupil_circle: Optional[PupilCircle]) -> Optional[np.ndarray]:
        h, w = eye_image.shape[:2]
        if h < 10 or w < 10:
            return None

        hsv = cv2.cvtColor(eye_image, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]

        gray = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)

        # Mask A: raw brightness threshold in V channel.
        thr_v = int(np.percentile(v, 97.0))
        thr_v = max(160, min(245, thr_v))
        mask_v = (v >= thr_v).astype(np.uint8) * 255

        # Mask B: morphological top-hat to isolate small bright blobs (very robust on blur).
        # Kernel size scales with image; we want to highlight small specular regions.
        k = int(max(9, min(h, w) * 0.12))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        thr_th = int(np.percentile(tophat, 99.0))
        thr_th = max(8, thr_th)
        mask_th = (tophat >= thr_th).astype(np.uint8) * 255

        # Combine both signals.
        mask = cv2.bitwise_or(mask_v, mask_th)

        # Restrict to near pupil/iris region if available.
        if pupil_circle is not None:
            x, y, r = pupil_circle
            rr = int(max(6, 1.8 * r))
            circ = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(circ, (int(x), int(y)), rr, 255, -1)
            mask = cv2.bitwise_and(mask, circ)
        else:
            # Otherwise focus on central region (eye whites can be too bright).
            x1, x2 = int(0.10 * w), int(0.90 * w)
            y1, y2 = int(0.10 * h), int(0.90 * h)
            roi = np.zeros((h, w), dtype=np.uint8)
            roi[y1:y2, x1:x2] = 255
            mask = cv2.bitwise_and(mask, roi)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=1)
        mask = cv2.dilate(mask, kernel2, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Choose the best candidate by area and proximity to center/pupil.
        if pupil_circle is not None:
            cx, cy = pupil_circle[0], pupil_circle[1]
        else:
            cx, cy = w / 2.0, h / 2.0

        img_area = float(h * w)

        def score(cnt):
            area = float(cv2.contourArea(cnt))
            if area <= 0:
                return -1.0
            # Reflection highlight should be a small blob.
            if area < 6.0 or area > 0.08 * img_area:
                return -1.0
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                return -1.0
            x = float(M["m10"] / M["m00"])
            y = float(M["m01"] / M["m00"])
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

            # Prefer brighter regions in V channel.
            x_i, y_i = int(round(x)), int(round(y))
            x_i = max(0, min(w - 1, x_i))
            y_i = max(0, min(h - 1, y_i))
            bright = float(v[y_i, x_i]) / 255.0

            return (1.2 * bright + 0.3) * (area ** 0.5) / (1.0 + 0.015 * dist)

        scored = [(score(c), c) for c in contours]
        scored.sort(key=lambda t: t[0], reverse=True)
        if not scored or scored[0][0] <= 0:
            return None

        best = scored[0][1]
        x, y, bw, bh = cv2.boundingRect(best)

        # Compute centroid for a stable highlight-centered crop.
        M = cv2.moments(best)
        if M["m00"] != 0:
            bx = int(round(M["m10"] / M["m00"]))
            by = int(round(M["m01"] / M["m00"]))
        else:
            bx = int(x + bw / 2)
            by = int(y + bh / 2)

        # Tight reflection crop size:
        # - proportional to highlight size
        # - clamped so we don't return the whole eye
        blob_scale = int(round(3.6 * max(bw, bh)))
        min_side = 48
        max_side = int(round(0.75 * min(h, w)))
        side = int(max(min_side, min(max_side, blob_scale)))
        half = side // 2

        x1, y1 = max(0, bx - half), max(0, by - half)
        x2, y2 = min(w, bx + half), min(h, by + half)
        crop = eye_image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Mild detail enhancement + contrast to emphasize micro-structure.
        crop = cv2.detailEnhance(crop, sigma_s=10, sigma_r=0.15)
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        crop = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

        # Circular mask to emphasize pupil reflection aesthetics.
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.circle(mask, (cw // 2, ch // 2), int(0.48 * min(ch, cw)), 255, -1)
        out = crop.copy()
        out[mask == 0] = 0
        return out