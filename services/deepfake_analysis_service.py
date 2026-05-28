from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


class DeepfakeAnalysisService:

    def validate(self, left_reflection, right_reflection) -> Dict[str, Any]:
        try:
            score = float(self._combined_similarity(left_reflection, right_reflection))
        except Exception:
            score = float(self._legacy_similarity(left_reflection, right_reflection))

        # Higher score => more consistent reflections (more likely genuine).
        status = "Likely Real" if score >= 0.35 else "Possible Deepfake"
        return {"status": status, "score": score}

    def _legacy_similarity(self, img1, img2) -> float:
        a = cv2.resize(img1, (128, 128))
        b = cv2.resize(img2, (128, 128))
        diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
        return float(1.0 / (1.0 + diff))

    def _combined_similarity(self, img1, img2) -> float:
        a = self._to_gray(img1)
        b = self._to_gray(img2)
        a = cv2.resize(a, (160, 160))
        b = cv2.resize(b, (160, 160))

        # Normalize illumination a bit (helps with slight exposure differences).
        a = cv2.equalizeHist(a)
        b = cv2.equalizeHist(b)

        # 1) SSIM (structure)
        ssim = self._ssim(a, b)

        # 2) Histogram correlation (overall tone distribution)
        hist_a = cv2.calcHist([a], [0], None, [64], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [64], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        hist_corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
        hist_corr = max(-1.0, min(1.0, hist_corr))

        # 3) Gradient orientation similarity (light consistency proxy)
        grad = self._gradient_orientation_similarity(a, b)

        # Combine into a [0,1] score.
        # Map hist_corr from [-1,1] -> [0,1]
        hist_01 = (hist_corr + 1.0) / 2.0
        ssim_01 = max(0.0, min(1.0, float(ssim)))
        grad_01 = max(0.0, min(1.0, float(grad)))

        return float(0.50 * ssim_01 + 0.25 * grad_01 + 0.25 * hist_01)

    def _to_gray(self, img):
        arr = np.asarray(img)
        if arr.ndim == 2:
            g = arr
        else:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        return g

    def _ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        try:
            from skimage.metrics import structural_similarity as ssim  # type: ignore

            return float(ssim(a, b))
        except Exception:
            # Quick fallback: normalized MSE -> similarity
            mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
            return float(1.0 / (1.0 + mse / (255.0 ** 2)))

    def _gradient_orientation_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a32 = a.astype(np.float32) / 255.0
        b32 = b.astype(np.float32) / 255.0
        ax = cv2.Sobel(a32, cv2.CV_32F, 1, 0, ksize=3)
        ay = cv2.Sobel(a32, cv2.CV_32F, 0, 1, ksize=3)
        bx = cv2.Sobel(b32, cv2.CV_32F, 1, 0, ksize=3)
        by = cv2.Sobel(b32, cv2.CV_32F, 0, 1, ksize=3)

        a_mag = np.sqrt(ax * ax + ay * ay) + 1e-6
        b_mag = np.sqrt(bx * bx + by * by) + 1e-6

        # Cosine similarity of gradient vectors, weighted by magnitude.
        dot = (ax * bx + ay * by)
        cos = dot / (a_mag * b_mag)
        weight = (a_mag + b_mag) / 2.0
        return float(np.sum(cos * weight) / (np.sum(weight) + 1e-6) * 0.5 + 0.5)