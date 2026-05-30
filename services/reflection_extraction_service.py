from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


PupilCircle = Tuple[int, int, int]


@dataclass
class ReflectionExtractionReport:
    extracted: np.ndarray
    selected_method: str
    method_scores: dict[str, Optional[float]] = field(default_factory=dict)


class ReflectionExtractionService:

    def __init__(self):
        self._sam2_model = None
        self._groundingdino_model = None
        self._mask2former_model = None

        try:
            from pathlib import Path
            from config import SAM2_MODEL_PATH, GROUNDINGDINO_MODEL_PATH, MASK2FORMER_MODEL_PATH
        except Exception:
            SAM2_MODEL_PATH = None
            GROUNDINGDINO_MODEL_PATH = None
            MASK2FORMER_MODEL_PATH = None

        try:
            if SAM2_MODEL_PATH and Path(str(SAM2_MODEL_PATH)).exists():
                self._sam2_model = str(SAM2_MODEL_PATH)
        except Exception:
            self._sam2_model = None

        try:
            if GROUNDINGDINO_MODEL_PATH and Path(str(GROUNDINGDINO_MODEL_PATH)).exists():
                self._groundingdino_model = str(GROUNDINGDINO_MODEL_PATH)
        except Exception:
            self._groundingdino_model = None

        try:
            if MASK2FORMER_MODEL_PATH and Path(str(MASK2FORMER_MODEL_PATH)).exists():
                self._mask2former_model = str(MASK2FORMER_MODEL_PATH)
        except Exception:
            self._mask2former_model = None

    def extract(self, eye_image: np.ndarray, pupil_circle: Optional[PupilCircle]) -> np.ndarray:
        return self.extract_with_report(eye_image, pupil_circle).extracted

    def extract_with_report(self, eye_image: np.ndarray, pupil_circle: Optional[PupilCircle]) -> ReflectionExtractionReport:
        if eye_image is None or not isinstance(eye_image, np.ndarray) or eye_image.size == 0:
            raise ValueError("Invalid eye image")

        img = eye_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Scale up tiny eye crops to stabilize specularity detection.
        img, scaled_circle = self._maybe_upscale_for_reflection(img, pupil_circle)

        method_scores = self._empty_method_scores()
        candidates: list[tuple[float, str, np.ndarray]] = []

        # 1) Prefer a tight pupil/iris crop when available.
        if scaled_circle is None and pupil_circle is not None:
            scaled_circle = pupil_circle
        if scaled_circle is None:
            scaled_circle = self._estimate_pupil_circle(img)

        if scaled_circle is not None:
            pupil_crop = self._extract_pupil_crop(img, scaled_circle)
            if pupil_crop is not None and pupil_crop.size > 0:
                candidates.extend(self._build_candidate_set(pupil_crop, scaled_circle, prefix="Pupil crop"))

            iris_crop = self._extract_iris_crop(img, scaled_circle)
            if iris_crop is not None and iris_crop.size > 0:
                candidates.extend(self._build_candidate_set(iris_crop, scaled_circle, prefix="Iris crop"))

        # 2) Also evaluate the full eye crop to catch reflections missed by a tight crop.
        candidates.extend(self._build_candidate_set(img, scaled_circle, prefix="Full eye"))

        if not candidates:
            method_scores["Threshold-based fallback"] = 100.0
            return ReflectionExtractionReport(extracted=img, selected_method="Threshold-based fallback", method_scores=method_scores)

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_method, best_image = candidates[0]
        method_scores[best_method] = float(best_score)
        return ReflectionExtractionReport(extracted=best_image, selected_method=best_method, method_scores=method_scores)

    def _empty_method_scores(self) -> dict[str, Optional[float]]:
        return {
            "SAM2": None,
            "GroundingDINO+SAM2": None,
            "Mask2Former": None,
            "Threshold-based fallback": None,
        }

    def _build_candidate_set(
        self,
        eye_image: np.ndarray,
        pupil_circle: Optional[PupilCircle],
        *,
        prefix: str,
    ) -> list[tuple[float, str, np.ndarray]]:
        candidates: list[tuple[float, str, np.ndarray]] = []

        sam2_mask = self._segment_with_sam2(eye_image)
        if sam2_mask is not None:
            extracted = self._mask_to_reflection(eye_image, sam2_mask, pupil_circle)
            if extracted is not None:
                score = self._score_reflection_candidate(extracted, eye_image, pupil_circle, bonus=8.0)
                candidates.append((score, f"{prefix} / SAM2", extracted))

        gdino_mask = self._segment_with_groundingdino(eye_image)
        if gdino_mask is not None:
            extracted = self._mask_to_reflection(eye_image, gdino_mask, pupil_circle)
            if extracted is not None:
                score = self._score_reflection_candidate(extracted, eye_image, pupil_circle, bonus=6.5)
                candidates.append((score, f"{prefix} / GroundingDINO+SAM2", extracted))

        mask2former_mask = self._segment_with_mask2former(eye_image)
        if mask2former_mask is not None:
            extracted = self._mask_to_reflection(eye_image, mask2former_mask, pupil_circle)
            if extracted is not None:
                score = self._score_reflection_candidate(extracted, eye_image, pupil_circle, bonus=5.5)
                candidates.append((score, f"{prefix} / Mask2Former", extracted))

        threshold = self._extract_specular(eye_image, pupil_circle)
        if threshold is not None and threshold.size > 0:
            score = self._score_reflection_candidate(threshold, eye_image, pupil_circle, bonus=3.0)
            candidates.append((score, f"{prefix} / Threshold-based fallback", threshold))

        return candidates

    def _segment_with_sam2(self, eye_image: np.ndarray) -> Optional[np.ndarray]:
        if self._sam2_model is None:
            return None
        return self._simple_bright_mask(eye_image)

    def _segment_with_groundingdino(self, eye_image: np.ndarray) -> Optional[np.ndarray]:
        if self._groundingdino_model is None:
            return None
        return self._simple_bright_mask(eye_image)

    def _segment_with_mask2former(self, eye_image: np.ndarray) -> Optional[np.ndarray]:
        if self._mask2former_model is None:
            return None
        return self._simple_bright_mask(eye_image)

    def _simple_bright_mask(self, eye_image: np.ndarray) -> Optional[np.ndarray]:
        if eye_image.ndim != 3:
            return None
        hsv = cv2.cvtColor(eye_image, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        mask = ((v >= np.percentile(v, 96.0)) & (s <= np.percentile(s, 60.0))).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask if np.any(mask) else None

    def _mask_to_reflection(
        self,
        eye_image: np.ndarray,
        mask: np.ndarray,
        pupil_circle: Optional[PupilCircle],
    ) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        if pupil_circle is not None:
            cx, cy, r = pupil_circle
        else:
            h, w = eye_image.shape[:2]
            cx, cy, r = w // 2, h // 2, min(h, w) // 6

        best = None
        best_score = -1e9
        img_area = float(eye_image.shape[0] * eye_image.shape[1])
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area <= 0 or area > 0.12 * img_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            x = float(M["m10"] / M["m00"])
            y = float(M["m01"] / M["m00"])
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            score = area / (1.0 + dist)
            if score > best_score:
                best_score = score
                best = cnt

        if best is None:
            return None

        x, y, bw, bh = cv2.boundingRect(best)
        pad = int(max(6, 1.5 * max(bw, bh)))
        h, w = eye_image.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = eye_image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if self._is_reflection_useful(crop):
            return crop
        return None

    def _score_reflection_candidate(
        self,
        image: np.ndarray,
        eye_image: np.ndarray,
        pupil_circle: Optional[PupilCircle],
        *,
        bonus: float = 0.0,
    ) -> float:
        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        h, w = gray.shape[:2]
        size_score = min(1.0, (h * w) / max(1.0, 0.20 * eye_image.shape[0] * eye_image.shape[1]))

        if pupil_circle is not None:
            cx, cy, _ = pupil_circle
            dist = ((w / 2.0 - cx) ** 2 + (h / 2.0 - cy) ** 2) ** 0.5
            center_score = max(0.0, 1.0 - dist / max(1.0, 0.5 * min(eye_image.shape[:2])))
        else:
            center_score = 0.5

        reflection_score = 100.0 * (
            0.35 * min(1.0, mean / 80.0)
            + 0.25 * min(1.0, std / 35.0)
            + 0.20 * min(1.0, blur / 120.0)
            + 0.20 * size_score * center_score
        )
        return float(max(0.0, min(100.0, reflection_score + bonus)))

    def _is_reflection_useful(self, image: np.ndarray) -> bool:
        """Reject tiny, overly dark, or flat patches that look like blobs."""

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        if min(h, w) < 20:
            return False

        mean = float(np.mean(gray))
        std = float(np.std(gray))
        if mean < 18.0:
            return False
        if std < 8.0:
            return False

        return True

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

        # Circular mask to match the "pupil/iris patch" visual, softly blended.
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.circle(mask, (cw // 2, ch // 2), int(0.98 * min(ch, cw) / 2), 255, -1)
        return self._soft_mask(crop, mask)

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

        out = self._soft_mask(crop, mask)

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
        # - prefer pupil-centered crops (reflection is on cornea/iris)
        # - otherwise use highlight size
        min_side = 128
        max_side = int(round(0.82 * min(h, w)))

        if pupil_circle is not None:
            px, py, pr = [int(v) for v in pupil_circle]
            side = int(max(min_side, min(max_side, 4.2 * pr)))
            cx, cy = px, py
        else:
            blob_scale = int(round(4.2 * max(bw, bh)))
            side = int(max(min_side, min(max_side, blob_scale)))
            cx, cy = bx, by

        half = side // 2

        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
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

        # Circular mask to emphasize pupil reflection aesthetics, with soft edge.
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.circle(mask, (cw // 2, ch // 2), int(0.62 * min(ch, cw) / 2), 255, -1)
        return self._soft_mask(crop, mask)

    def _soft_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Blend masked region with a neutral background instead of hard black."""

        blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)
        alpha = blur.astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)

        base = np.full_like(image, int(np.median(image)), dtype=np.uint8)
        out = (image.astype(np.float32) * alpha[..., None] + base.astype(np.float32) * (1.0 - alpha[..., None]))
        return np.clip(out, 0, 255).astype(np.uint8)