from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from utils.quality_metrics import calculate_blur_score, calculate_contrast_score


@dataclass
class FusionReport:
    fused: np.ndarray
    selected_method: str
    method_scores: dict[str, Optional[float]] = field(default_factory=dict)
    frame_count: int = 0
    reference_index: int = 0


class MultiFrameFusionService:
    def __init__(self, *, top_k_frames: int = 12):
        self._top_k_frames = max(1, int(top_k_frames))

    def fuse(self, frames: Sequence[np.ndarray]) -> FusionReport:
        normalized = self._normalize_frames(frames)
        if not normalized:
            raise ValueError("No frames provided for fusion")

        if len(normalized) == 1:
            frame = normalized[0]
            return FusionReport(
                fused=frame,
                selected_method="Single frame",
                method_scores={
                    "Optical Flow": None,
                    "RAFT": None,
                    "RIFE": None,
                    "Frame Alignment": 100.0,
                    "Multi-frame Super Resolution": None,
                },
                frame_count=1,
                reference_index=0,
            )

        sampled = self._sample_frames(normalized)
        reference_index = self._select_reference_index(sampled)
        reference = sampled[reference_index]

        aligned_stack = self._align_stack(sampled, reference_index, mode="ecc")
        aligned_flow_stack = self._align_stack(sampled, reference_index, mode="flow")

        candidates: list[tuple[str, np.ndarray]] = []
        candidates.append(("Frame Alignment", self._median_fuse(aligned_stack)))
        candidates.append(("Optical Flow", self._median_fuse(aligned_flow_stack)))

        if min(reference.shape[:2]) < 128 or len(sampled) >= 3:
            candidates.append(("Multi-frame Super Resolution", self._multi_frame_sr(aligned_stack)))

        scores = self._score_candidates(reference, aligned_stack, candidates)
        scores["RAFT"] = None
        scores["RIFE"] = None

        best_method = max(
            (name for name in scores.keys() if scores[name] is not None),
            key=lambda name: float(scores[name] or 0.0),
        )
        best_image = next(image for name, image in candidates if name == best_method)

        return FusionReport(
            fused=best_image,
            selected_method=best_method,
            method_scores=scores,
            frame_count=len(sampled),
            reference_index=reference_index,
        )

    def _normalize_frames(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        normalized: list[np.ndarray] = []
        for frame in frames:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                continue
            img = frame
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.ndim != 3 or img.shape[2] not in (3, 4):
                continue
            if img.shape[2] == 4:
                img = img[:, :, :3]
            normalized.append(img)
        return normalized

    def _sample_frames(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        if len(frames) <= self._top_k_frames:
            return list(frames)

        indices = np.linspace(0, len(frames) - 1, self._top_k_frames).round().astype(int)
        return [frames[int(index)] for index in indices]

    def _select_reference_index(self, frames: Sequence[np.ndarray]) -> int:
        best_index = 0
        best_score = -1e9
        for index, frame in enumerate(frames):
            blur = calculate_blur_score(frame)
            contrast = calculate_contrast_score(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray))
            score = 0.55 * blur + 0.35 * contrast + 0.10 * brightness
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _align_stack(self, frames: Sequence[np.ndarray], reference_index: int, *, mode: str) -> list[np.ndarray]:
        reference = frames[reference_index]
        aligned: list[np.ndarray] = []
        for frame in frames:
            if mode == "flow":
                aligned_frame = self._align_with_sparse_flow(frame, reference)
            else:
                aligned_frame = self._align_with_ecc(frame, reference)
            aligned.append(aligned_frame)
        return aligned

    def _align_with_ecc(self, frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                1e-5,
            )
            cv2.findTransformECC(reference_gray, frame_gray, warp, cv2.MOTION_AFFINE, criteria)
            h, w = reference.shape[:2]
            return cv2.warpAffine(
                frame,
                warp,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
        except Exception:
            return cv2.resize(frame, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    def _align_with_sparse_flow(self, frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
            points = cv2.goodFeaturesToTrack(reference_gray, maxCorners=120, qualityLevel=0.01, minDistance=7)
            if points is None:
                return self._align_with_ecc(frame, reference)

            tracked, status, _ = cv2.calcOpticalFlowPyrLK(reference_gray, frame_gray, points, None)
            if tracked is None or status is None:
                return self._align_with_ecc(frame, reference)

            reference_points = points[status.flatten() == 1]
            tracked_points = tracked[status.flatten() == 1]
            if len(reference_points) < 4 or len(tracked_points) < 4:
                return self._align_with_ecc(frame, reference)

            matrix, _ = cv2.estimateAffinePartial2D(tracked_points, reference_points)
            if matrix is None:
                return self._align_with_ecc(frame, reference)

            h, w = reference.shape[:2]
            return cv2.warpAffine(
                frame,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        except Exception:
            return self._align_with_ecc(frame, reference)

    def _median_fuse(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        stack = np.stack([frame.astype(np.float32) for frame in frames], axis=0)
        fused = np.median(stack, axis=0)
        return np.clip(fused, 0, 255).astype(np.uint8)

    def _multi_frame_sr(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        reference = frames[0]
        h, w = reference.shape[:2]
        scale = 4 if min(h, w) < 96 else 2
        target_size = (max(1, w * scale), max(1, h * scale))
        upscaled = [cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4) for frame in frames]
        fused = np.mean(np.stack([frame.astype(np.float32) for frame in upscaled], axis=0), axis=0)
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        return cv2.detailEnhance(fused, sigma_s=10, sigma_r=0.15)

    def _score_candidates(
        self,
        reference: np.ndarray,
        aligned_stack: Sequence[np.ndarray],
        candidates: Sequence[tuple[str, np.ndarray]],
    ) -> dict[str, Optional[float]]:
        ref_median = self._median_fuse(aligned_stack)
        ref_h, ref_w = reference.shape[:2]

        raw_scores = []
        for name, image in candidates:
            comparable = image
            if comparable.shape[:2] != (ref_h, ref_w):
                comparable = cv2.resize(comparable, (ref_w, ref_h), interpolation=cv2.INTER_LANCZOS4)

            blur = calculate_blur_score(comparable)
            contrast = calculate_contrast_score(comparable)
            residual = float(np.mean(np.abs(comparable.astype(np.float32) - ref_median.astype(np.float32))))
            raw_scores.append((name, blur, contrast, residual, image))

        blur_values = [item[1] for item in raw_scores]
        contrast_values = [item[2] for item in raw_scores]
        residual_values = [item[3] for item in raw_scores]

        def normalize(values: list[float]) -> list[float]:
            if not values:
                return []
            lo = min(values)
            hi = max(values)
            if abs(hi - lo) < 1e-6:
                return [0.5 for _ in values]
            return [(value - lo) / (hi - lo) for value in values]

        blur_norm = normalize(blur_values)
        contrast_norm = normalize(contrast_values)
        residual_norm = normalize(residual_values)
        tiny_factor = max(0.0, min(1.0, (128.0 - float(min(ref_h, ref_w))) / 128.0))

        scores: dict[str, Optional[float]] = {
            "Optical Flow": None,
            "RAFT": None,
            "RIFE": None,
            "Frame Alignment": None,
            "Multi-frame Super Resolution": None,
        }

        for index, (name, _, _, _, image) in enumerate(raw_scores):
            score = 100.0 * (
                0.42 * blur_norm[index]
                + 0.33 * contrast_norm[index]
                + 0.25 * (1.0 - residual_norm[index])
            )
            if name == "Multi-frame Super Resolution":
                score += 12.0 * tiny_factor + 4.0
            elif name == "Optical Flow":
                score += 4.0
            elif name == "Frame Alignment":
                score += 2.5
            scores[name] = float(max(0.0, min(100.0, score)))

        return scores