from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency handling
    mp = None


Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


@dataclass
class CornealReconstruction:
    limbus: Optional[Ellipse]
    panorama: Optional[np.ndarray]
    foveated: Optional[np.ndarray]
    limbus_source: Optional[str] = None
    limbus_scores: dict[str, Optional[float]] = field(default_factory=dict)


class CornealImagingService:
    """Approximate corneal imaging reconstruction based on Nishino CVPR04.

    This implementation prefers MediaPipe iris landmarks for limbus estimation
    and falls back to classical ellipse fitting on edge contours when landmark
    detection fails.
    """

    def __init__(self):
        self._face_mesh = None
        self._mp_face_mesh = None
        self._yolo_eye_model = None

        if mp is not None:
            try:
                self._mp_face_mesh = mp.solutions.face_mesh
                self._face_mesh = self._mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=1,
                    min_detection_confidence=0.3,
                )
            except Exception:
                self._face_mesh = None
                self._mp_face_mesh = None

        try:
            from pathlib import Path
            from ultralytics import YOLO  # type: ignore

            try:
                from config import YOLO_EYE_MODEL_PATH  # type: ignore

                model_path = YOLO_EYE_MODEL_PATH
            except Exception:
                model_path = None

            if model_path and Path(str(model_path)).exists():
                self._yolo_eye_model = YOLO(str(model_path))
        except Exception:
            self._yolo_eye_model = None

    def _prepare_image(self, eye_image: np.ndarray) -> tuple[np.ndarray, float, float]:
        img = eye_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return img, 1.0, 1.0

        max_dim = 640
        min_dim = 256
        scale_x = 1.0
        scale_y = 1.0

        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = w / float(new_w)
            scale_y = h / float(new_h)
            return resized, scale_x, scale_y

        if min(h, w) < min_dim:
            scale = min_dim / float(min(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            scale_x = w / float(new_w)
            scale_y = h / float(new_h)
            return resized, scale_x, scale_y

        return img, scale_x, scale_y

    def _score_limbus_ellipse(self, ellipse: Ellipse, eye_image: np.ndarray, *, confidence: float = 0.0) -> float:
        h, w = eye_image.shape[:2]
        (cx, cy), (ma, mi), _ = ellipse
        if h <= 0 or w <= 0 or ma <= 0 or mi <= 0:
            return 0.0

        min_side = float(min(h, w))
        center_dist = float(((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5)
        center_score = max(0.0, 1.0 - center_dist / max(1.0, 0.5 * min_side))

        major = max(ma, mi)
        minor = min(ma, mi)
        size_ratio = major / max(1.0, min_side)
        size_score = 1.0 - min(1.0, abs(size_ratio - 0.55) / 0.55)
        shape_score = min(1.0, minor / max(1.0, major))

        score = 100.0 * (0.45 * center_score + 0.35 * size_score + 0.20 * shape_score)
        score += 15.0 * max(0.0, min(1.0, confidence))
        return float(max(0.0, min(100.0, score)))

    def _empty_limbus_scores(self) -> dict[str, Optional[float]]:
        return {
            "MediaPipe Iris": None,
            "SAM2 segmentation": None,
            "YOLOv11 eye detector": None,
            "Ellipse fitting": None,
        }

    def _ellipse_from_points(self, points: np.ndarray, scale_x: float, scale_y: float) -> Optional[Ellipse]:
        if points.shape[0] < 5:
            return None

        try:
            ellipse = cv2.fitEllipse(points.reshape(-1, 1, 2).astype(np.float32))
        except Exception:
            return None

        (cx, cy), (ma, mi), angle = ellipse
        if ma <= 0 or mi <= 0:
            return None

        limbus_scale = 1.18
        return (
            (float(cx * scale_x), float(cy * scale_y)),
            (float(ma * scale_x * limbus_scale), float(mi * scale_y * limbus_scale)),
            float(angle),
        )

    def _detect_limbus_with_mediapipe(self, eye_image: np.ndarray) -> Optional[Ellipse]:
        if self._face_mesh is None or self._mp_face_mesh is None:
            return None

        img, scale_x, scale_y = self._prepare_image(eye_image)
        if img.size == 0:
            return None

        results = self._face_mesh.process(img)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = img.shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0
        candidates = []

        for side in ("LEFT", "RIGHT"):
            connections = getattr(self._mp_face_mesh, f"FACEMESH_{side}_IRIS", None)
            if not connections:
                continue

            iris_indices = sorted({int(idx) for pair in connections for idx in pair})
            points = []
            for idx in iris_indices:
                if idx >= len(face_landmarks.landmark):
                    continue
                landmark = face_landmarks.landmark[idx]
                points.append((landmark.x * w, landmark.y * h))

            ellipse = self._ellipse_from_points(np.asarray(points, dtype=np.float32), scale_x, scale_y)
            if ellipse is None:
                continue

            (ex, ey), (ma, mi), angle = ellipse
            dist = float(((ex - center_x * scale_x) ** 2 + (ey - center_y * scale_y) ** 2) ** 0.5)
            size = float(max(ma, mi))
            score = size - 0.7 * dist
            candidates.append((score, ellipse))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _detect_limbus_with_yolo(self, eye_image: np.ndarray) -> Optional[Ellipse]:
        if self._yolo_eye_model is None:
            return None

        img = eye_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        try:
            results = self._yolo_eye_model.predict(img, verbose=False)
        except Exception:
            return None

        if not results:
            return None

        best_mask = None
        best_box = None
        best_score = -1.0
        h, w = img.shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        for result in results:
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is None:
                continue

            box_count = len(boxes)
            for idx in range(box_count):
                try:
                    box = boxes[idx]
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                    conf = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
                except Exception:
                    continue

                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                area = float((x2 - x1) * (y2 - y1))
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                center_dist = float(((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5)
                score = conf + 0.00001 * area - 0.002 * center_dist

                mask = None
                if masks is not None and getattr(masks, "data", None) is not None:
                    try:
                        mask_data = masks.data[idx].cpu().numpy()
                        if mask_data.ndim == 2:
                            mask = cv2.resize(mask_data.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                            mask = (mask > 0.5).astype(np.uint8) * 255
                    except Exception:
                        mask = None

                if score > best_score:
                    best_score = score
                    best_mask = mask
                    best_box = (x1, y1, x2, y2)

        if best_box is None:
            return None

        if best_mask is not None:
            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = None
            best_area = -1.0
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area > best_area and len(cnt) >= 5:
                    best_area = area
                    best_contour = cnt
            if best_contour is not None:
                try:
                    ellipse = cv2.fitEllipse(best_contour)
                    return self._expand_limbus_ellipse(ellipse, 1.0, 1.0)
                except Exception:
                    pass

        x1, y1, x2, y2 = best_box
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        roi_limbus = self._detect_limbus_from_contours(roi)
        if roi_limbus is None:
            return None

        (cx, cy), (ma, mi), angle = roi_limbus
        return self._expand_limbus_ellipse(
            ((cx + x1, cy + y1), (ma, mi), angle),
            1.0,
            1.0,
        )

    def _expand_limbus_ellipse(self, ellipse: Ellipse, scale_x: float, scale_y: float) -> Ellipse:
        (cx, cy), (ma, mi), angle = ellipse
        limbus_scale = 1.18
        return (
            (float(cx * scale_x), float(cy * scale_y)),
            (float(ma * scale_x * limbus_scale), float(mi * scale_y * limbus_scale)),
            float(angle),
        )

    def _detect_limbus_from_contours(self, eye_image: np.ndarray) -> Optional[Ellipse]:
        img = eye_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = gray.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        best = None
        best_score = -1e9

        for cnt in contours:
            if len(cnt) < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
            except Exception:
                continue

            (ex, ey), (ma, mi), angle = ellipse
            if ma <= 0 or mi <= 0:
                continue

            dist = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            size = max(ma, mi)
            if size < 0.35 * min(h, w):
                continue

            score = size - 0.8 * dist
            if score > best_score:
                best_score = score
                best = ellipse

        return best

    def _detect_limbus_report(self, eye_image: np.ndarray) -> tuple[Optional[Ellipse], Optional[str], dict[str, Optional[float]]]:
        scores = self._empty_limbus_scores()
        if eye_image is None or not isinstance(eye_image, np.ndarray) or eye_image.size == 0:
            return None, None, scores

        candidates: list[tuple[float, str, Ellipse]] = []

        yolo_ellipse = self._detect_limbus_with_yolo(eye_image)
        if yolo_ellipse is not None:
            yolo_score = self._score_limbus_ellipse(yolo_ellipse, eye_image, confidence=0.75)
            scores["YOLOv11 eye detector"] = yolo_score
            candidates.append((yolo_score, "YOLOv11 eye detector", yolo_ellipse))

        mediapipe_ellipse = self._detect_limbus_with_mediapipe(eye_image)
        if mediapipe_ellipse is not None:
            mediapipe_score = self._score_limbus_ellipse(mediapipe_ellipse, eye_image, confidence=0.90)
            scores["MediaPipe Iris"] = mediapipe_score
            candidates.append((mediapipe_score, "MediaPipe Iris", mediapipe_ellipse))

        # SAM2 is not available in this workspace; keep the slot so the UI can show it explicitly.
        scores["SAM2 segmentation"] = None

        contour_ellipse = self._detect_limbus_from_contours(eye_image)
        if contour_ellipse is not None:
            contour_score = self._score_limbus_ellipse(contour_ellipse, eye_image, confidence=0.55)
            scores["Ellipse fitting"] = contour_score
            candidates.append((contour_score, "Ellipse fitting", contour_ellipse))

        if not candidates:
            return None, None, scores

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_source, best_ellipse = candidates[0]
        scores[best_source] = float(best_score)
        return best_ellipse, best_source, scores

    def detect_limbus(self, eye_image: np.ndarray) -> Optional[Ellipse]:
        limbus, _, _ = self._detect_limbus_report(eye_image)
        return limbus

    def detect_limbus_report(self, eye_image: np.ndarray) -> tuple[Optional[Ellipse], Optional[str], dict[str, Optional[float]]]:
        return self._detect_limbus_report(eye_image)

    def reconstruct(self, eye_image: np.ndarray) -> CornealReconstruction:
        limbus, limbus_source, limbus_scores = self._detect_limbus_report(eye_image)
        if limbus is None:
            return CornealReconstruction(
                limbus=None,
                panorama=None,
                foveated=None,
                limbus_source=None,
                limbus_scores=limbus_scores,
            )

        panorama = self._compute_panorama(eye_image, limbus)
        gaze_dir = self._estimate_gaze_direction(limbus)
        foveated = None
        if panorama is not None and gaze_dir is not None:
            foveated = self._compute_foveated(panorama, gaze_dir, fov_deg=45.0)

        return CornealReconstruction(
            limbus=limbus,
            panorama=panorama,
            foveated=foveated,
            limbus_source=limbus_source,
            limbus_scores=limbus_scores,
        )

    def _estimate_gaze_direction(self, limbus: Ellipse) -> Optional[np.ndarray]:
        (_, _), (ma, mi), angle = limbus
        if ma <= 0 or mi <= 0:
            return None

        rmax = max(ma, mi) / 2.0
        rmin = min(ma, mi) / 2.0
        ratio = max(1e-6, min(1.0, rmin / rmax))
        tau = float(np.arccos(ratio))
        phi = np.deg2rad(angle)

        # Approximate gaze direction in camera coordinates.
        gaze = np.array([
            np.sin(tau) * np.cos(phi),
            np.sin(tau) * np.sin(phi),
            np.cos(tau),
        ])
        norm = np.linalg.norm(gaze)
        if norm == 0:
            return None
        return gaze / norm

    def _compute_panorama(self, eye_image: np.ndarray, limbus: Ellipse) -> Optional[np.ndarray]:
        img = eye_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        (ex, ey), (ma, mi), _ = limbus
        h, w = img.shape[:2]

        # Create normalized coordinates inside ellipse.
        yy, xx = np.mgrid[0:h, 0:w]
        x = (xx - ex) / (ma / 2.0)
        y = (yy - ey) / (mi / 2.0)
        mask = (x * x + y * y) <= 1.0
        if not np.any(mask):
            return None

        x = x[mask]
        y = y[mask]

        # Spherical cornea approximation (unit sphere).
        z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
        p = np.stack([x, y, z], axis=1)
        n = p

        # Camera position along +Z.
        cam = np.array([0.0, 0.0, 2.2])
        v = cam - p
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        # Reflect viewing direction around surface normal.
        dot = np.sum(v * n, axis=1, keepdims=True)
        r = v - 2.0 * dot * n

        # Convert reflection directions to spherical coordinates.
        rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
        lon = np.arctan2(rx, rz)
        lat = np.arcsin(np.clip(ry, -1.0, 1.0))

        pano_w, pano_h = 512, 256
        u = ((lon + np.pi) / (2.0 * np.pi) * pano_w).astype(np.int32)
        v = ((np.pi / 2.0 - lat) / np.pi * pano_h).astype(np.int32)
        u = np.clip(u, 0, pano_w - 1)
        v = np.clip(v, 0, pano_h - 1)

        pano = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight = np.zeros((pano_h, pano_w), dtype=np.float32)

        colors = img[mask].astype(np.float32)
        for idx in range(u.size):
            pano[v[idx], u[idx]] += colors[idx]
            weight[v[idx], u[idx]] += 1.0

        weight = np.maximum(weight, 1.0)
        pano = (pano / weight[..., None]).astype(np.uint8)
        return pano

    def _compute_foveated(self, panorama: np.ndarray, gaze_dir: np.ndarray, fov_deg: float) -> np.ndarray:
        pano_h, pano_w = panorama.shape[:2]
        out_w, out_h = 256, 256

        fov = np.deg2rad(fov_deg)
        fx = 0.5 * out_w / np.tan(fov / 2.0)
        fy = 0.5 * out_h / np.tan(fov / 2.0)

        # Build rotation matrix from gaze direction.
        z = gaze_dir / np.linalg.norm(gaze_dir)
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)

        yy, xx = np.mgrid[0:out_h, 0:out_w]
        nx = (xx - out_w / 2.0) / fx
        ny = (yy - out_h / 2.0) / fy
        dirs = np.stack([nx, ny, np.ones_like(nx)], axis=-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        dirs = dirs.reshape(-1, 3) @ R.T
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        lon = np.arctan2(dx, dz)
        lat = np.arcsin(np.clip(dy, -1.0, 1.0))

        u = ((lon + np.pi) / (2.0 * np.pi) * pano_w).astype(np.int32)
        v = ((np.pi / 2.0 - lat) / np.pi * pano_h).astype(np.int32)
        u = np.clip(u, 0, pano_w - 1)
        v = np.clip(v, 0, pano_h - 1)

        out = panorama[v, u].reshape(out_h, out_w, 3)
        return out
