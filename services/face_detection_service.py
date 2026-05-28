from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np


FaceBox = Tuple[int, int, int, int]  # x, y, w, h


class FaceDetectionService:

    def __init__(self):
        self._backend: str = "haar"
        self._model: Optional[Any] = None
        self._init_error: Optional[BaseException] = None

        # Prefer RetinaFace when available (best accuracy among our deps).
        try:
            from facexlib.detection import RetinaFace  # type: ignore

            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

            self._model = RetinaFace(model_path=None, device=device)
            self._backend = "retinaface"
            return
        except BaseException as exc:
            self._init_error = exc
            self._model = None

        # Secondary: Ultralytics, but only if a local face model is configured.
        try:
            from ultralytics import YOLO  # type: ignore

            model_path = None
            try:
                from config import YOLO_FACE_MODEL_PATH  # type: ignore

                model_path = YOLO_FACE_MODEL_PATH
            except Exception:
                model_path = None

            if model_path and Path(str(model_path)).exists():
                self._model = YOLO(str(model_path))
                self._backend = "ultralytics"
                return

            self._model = None
        except BaseException:
            self._model = None

        # Final fallback: OpenCV Haar.
        self._model = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._backend = "haar"

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, image: np.ndarray) -> Sequence[FaceBox]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if self._backend == "retinaface" and self._model is not None:
            # facexlib RetinaFace expects BGR images.
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                # Returns dict with 'bbox' (N,4) in xyxy and 'score'.
                out = self._model.detect_faces(bgr)
            except AttributeError:
                out = self._model(bgr)

            boxes: List[FaceBox] = []
            bboxes = None
            if isinstance(out, dict):
                bboxes = out.get("bbox")
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                bboxes = out[0]

            if bboxes is None:
                return []
            bboxes = np.asarray(bboxes)
            if bboxes.ndim == 2 and bboxes.shape[1] >= 4:
                for row in bboxes:
                    x1, y1, x2, y2 = [int(v) for v in row[:4]]
                    boxes.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1)))
            return boxes

        if self._backend == "ultralytics" and self._model is not None:
            # This is a best-effort fallback; if model isn't face-specific it may be noisy.
            try:
                results = self._model.predict(image, verbose=False)
                boxes: List[FaceBox] = []
                for r in results:
                    if getattr(r, "boxes", None) is None:
                        continue
                    for b in r.boxes:
                        xyxy = b.xyxy[0].tolist()
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        boxes.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1)))
                return boxes
            except Exception:
                return []

        # Haar fallback
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self._model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )
        return faces if faces is not None else []