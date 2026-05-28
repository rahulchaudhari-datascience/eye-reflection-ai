from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


@dataclass
class CornealReconstruction:
    limbus: Optional[Ellipse]
    panorama: Optional[np.ndarray]
    foveated: Optional[np.ndarray]


class CornealImagingService:
    """Approximate corneal imaging reconstruction based on Nishino CVPR04.

    This implementation detects the limbus as an ellipse and uses a spherical
    mirror approximation to map the corneal reflection to a panorama.
    """

    def detect_limbus(self, eye_image: np.ndarray) -> Optional[Ellipse]:
        if eye_image is None or not isinstance(eye_image, np.ndarray) or eye_image.size == 0:
            return None

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

            # Limbus should be close to center and not too small.
            dist = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            size = max(ma, mi)
            if size < 0.35 * min(h, w):
                continue

            score = size - 0.8 * dist
            if score > best_score:
                best_score = score
                best = ellipse

        return best

    def reconstruct(self, eye_image: np.ndarray) -> CornealReconstruction:
        limbus = self.detect_limbus(eye_image)
        if limbus is None:
            return CornealReconstruction(limbus=None, panorama=None, foveated=None)

        panorama = self._compute_panorama(eye_image, limbus)
        gaze_dir = self._estimate_gaze_direction(limbus)
        foveated = None
        if panorama is not None and gaze_dir is not None:
            foveated = self._compute_foveated(panorama, gaze_dir, fov_deg=45.0)

        return CornealReconstruction(limbus=limbus, panorama=panorama, foveated=foveated)

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
