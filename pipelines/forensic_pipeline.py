from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from services.deepfake_analysis_service import DeepfakeAnalysisService
from services.eye_landmark_service import EyeLandmarkService
from services.face_detection_service import FaceDetectionService
from services.multimodal_reasoning_service import MultimodalReasoningService
from services.pupil_detection_service import PupilDetectionService
from services.reflection_enhancement_service import ReflectionEnhancementService
from services.reflection_extraction_service import ReflectionExtractionService


PupilCircle = Tuple[int, int, int]


@dataclass
class ForensicPipelineResult:
    faces: Any
    left_eye: np.ndarray
    right_eye: np.ndarray
    left_pupil: Optional[PupilCircle]
    right_pupil: Optional[PupilCircle]
    left_reflection: np.ndarray
    right_reflection: np.ndarray
    enhanced_left_reflection: np.ndarray
    enhanced_right_reflection: np.ndarray
    reasoning_left: str
    reasoning_right: str
    deepfake: Dict[str, Any]
    scene_left: Optional[Any] = None
    scene_right: Optional[Any] = None


class ForensicVisionPipeline:

    def __init__(self, *, enable_scene_reconstruction: bool = False):
        self.face_detector = FaceDetectionService()
        self.eye_service = EyeLandmarkService()
        self.pupil_service = PupilDetectionService()
        self.reflection_service = ReflectionExtractionService()
        self.enhancement_service = ReflectionEnhancementService()
        self.reasoning_service = MultimodalReasoningService()
        self.deepfake_service = DeepfakeAnalysisService()

        self._scene_enabled = enable_scene_reconstruction
        self._scene_service = None

    def _get_scene_service(self):
        if not self._scene_enabled:
            return None
        if self._scene_service is None:
            from services.scene_reconstruction_service import SceneReconstructionService

            self._scene_service = SceneReconstructionService()
        return self._scene_service

    def process(self, image: np.ndarray) -> ForensicPipelineResult:
        faces = self.face_detector.detect(image)

        # Eye detection is more reliable when we focus on the face ROI.
        eye_input = image
        try:
            if faces is not None and len(faces) > 0:
                # Pick largest face box.
                x, y, w, h = max(faces, key=lambda r: int(r[2] * r[3]))
                pad = int(0.25 * max(w, h))
                ih, iw = image.shape[:2]
                x1 = max(0, int(x - pad))
                y1 = max(0, int(y - pad))
                x2 = min(iw, int(x + w + pad))
                y2 = min(ih, int(y + h + pad))
                if x2 > x1 and y2 > y1:
                    eye_input = image[y1:y2, x1:x2]
        except Exception:
            eye_input = image

        eyes = self.eye_service.extract_eyes(eye_input)
        if eyes is None and eye_input is not image:
            eyes = self.eye_service.extract_eyes(image)
        if eyes is None:
            raise ValueError("No eyes detected")

        left_eye = eyes["left_eye"]
        right_eye = eyes["right_eye"]

        left_pupil = self.pupil_service.detect(left_eye)
        right_pupil = self.pupil_service.detect(right_eye)

        left_reflection = self.reflection_service.extract(left_eye, left_pupil)
        right_reflection = self.reflection_service.extract(right_eye, right_pupil)

        enhanced_left = self.enhancement_service.enhance(left_reflection)
        enhanced_right = self.enhancement_service.enhance(right_reflection)

        # Reasoning should describe the reflection features; analyze the tight reflection ROI.
        try:
            reasoning_left = self.reasoning_service.analyze(left_reflection)
        except Exception as exc:
            reasoning_left = f"[reasoning failed] {type(exc).__name__}: {exc}"

        try:
            reasoning_right = self.reasoning_service.analyze(right_reflection)
        except Exception as exc:
            reasoning_right = f"[reasoning failed] {type(exc).__name__}: {exc}"

        deepfake = self.deepfake_service.validate(enhanced_left, enhanced_right)

        scene_left = None
        scene_right = None
        scene_service = self._get_scene_service()
        if scene_service is not None:
            try:
                scene_left = scene_service.reconstruct(reasoning_left, enhanced_left)
            except Exception:
                scene_left = None

            try:
                scene_right = scene_service.reconstruct(reasoning_right, enhanced_right)
            except Exception:
                scene_right = None

        return ForensicPipelineResult(
            faces=faces,
            left_eye=left_eye,
            right_eye=right_eye,
            left_pupil=left_pupil,
            right_pupil=right_pupil,
            left_reflection=left_reflection,
            right_reflection=right_reflection,
            enhanced_left_reflection=enhanced_left,
            enhanced_right_reflection=enhanced_right,
            reasoning_left=reasoning_left,
            reasoning_right=reasoning_right,
            deepfake=deepfake,
            scene_left=scene_left,
            scene_right=scene_right,
        )
