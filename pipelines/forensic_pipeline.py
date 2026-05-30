from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from services.deepfake_analysis_service import DeepfakeAnalysisService
from services.eye_landmark_service import EyeLandmarkService
from services.face_detection_service import FaceDetectionService
from services.multi_frame_fusion_service import MultiFrameFusionService
from services.multimodal_reasoning_service import MultimodalReasoningService
from services.pupil_detection_service import PupilDetectionService
from services.reflection_enhancement_service import ReflectionEnhancementService
from services.reflection_extraction_service import ReflectionExtractionService
from services.corneal_imaging_service import CornealImagingService


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
    left_pre_sr: Optional[np.ndarray]
    right_pre_sr: Optional[np.ndarray]
    left_sr: Optional[np.ndarray]
    right_sr: Optional[np.ndarray]
    enhanced_left_reflection: np.ndarray
    enhanced_right_reflection: np.ndarray
    reasoning_left: str
    reasoning_right: str
    deepfake: Dict[str, Any]
    fusion_image: Optional[np.ndarray] = None
    fusion_method: Optional[str] = None
    fusion_scores: Optional[Dict[str, Optional[float]]] = None
    fusion_frame_count: int = 0
    left_panorama: Optional[np.ndarray] = None
    right_panorama: Optional[np.ndarray] = None
    left_foveated: Optional[np.ndarray] = None
    right_foveated: Optional[np.ndarray] = None
    left_reflection_method: Optional[str] = None
    right_reflection_method: Optional[str] = None
    left_reflection_scores: Optional[Dict[str, Optional[float]]] = None
    right_reflection_scores: Optional[Dict[str, Optional[float]]] = None
    left_enhancement_method: Optional[str] = None
    right_enhancement_method: Optional[str] = None
    left_enhancement_scores: Optional[Dict[str, Optional[float]]] = None
    right_enhancement_scores: Optional[Dict[str, Optional[float]]] = None
    left_limbus_source: Optional[str] = None
    right_limbus_source: Optional[str] = None
    left_limbus_scores: Optional[Dict[str, Optional[float]]] = None
    right_limbus_scores: Optional[Dict[str, Optional[float]]] = None
    scene_left: Optional[Any] = None
    scene_right: Optional[Any] = None


class ForensicVisionPipeline:

    def __init__(self, *, enable_scene_reconstruction: bool = False):
        self.face_detector = FaceDetectionService()
        self.eye_service = EyeLandmarkService()
        self.pupil_service = PupilDetectionService()
        self.reflection_service = ReflectionExtractionService()
        self.enhancement_service = ReflectionEnhancementService()
        self.fusion_service = MultiFrameFusionService()
        self.corneal_service = CornealImagingService()
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

    def process(self, image: Any) -> ForensicPipelineResult:
        fusion_image = None
        fusion_method = None
        fusion_scores = None
        fusion_frame_count = 0

        working_image: np.ndarray
        if isinstance(image, np.ndarray) and image.ndim == 4:
            frames = [image[idx] for idx in range(image.shape[0])]
            fusion_report = self.fusion_service.fuse(frames)
            working_image = fusion_report.fused
            fusion_image = fusion_report.fused
            fusion_method = fusion_report.selected_method
            fusion_scores = fusion_report.method_scores
            fusion_frame_count = fusion_report.frame_count
        elif isinstance(image, (list, tuple)):
            fusion_report = self.fusion_service.fuse(image)
            working_image = fusion_report.fused
            fusion_image = fusion_report.fused
            fusion_method = fusion_report.selected_method
            fusion_scores = fusion_report.method_scores
            fusion_frame_count = fusion_report.frame_count
        elif isinstance(image, np.ndarray):
            working_image = image
        else:
            raise ValueError("Unsupported input type for forensic pipeline")

        faces = self.face_detector.detect(working_image)

        # Eye detection is more reliable when we focus on the face ROI.
        eye_input = working_image
        try:
            if faces is not None and len(faces) > 0:
                # Pick largest face box.
                x, y, w, h = max(faces, key=lambda r: int(r[2] * r[3]))
                pad = int(0.25 * max(w, h))
                ih, iw = working_image.shape[:2]
                x1 = max(0, int(x - pad))
                y1 = max(0, int(y - pad))
                x2 = min(iw, int(x + w + pad))
                y2 = min(ih, int(y + h + pad))
                if x2 > x1 and y2 > y1:
                    eye_input = working_image[y1:y2, x1:x2]
        except Exception:
            eye_input = working_image

        eyes = self.eye_service.extract_eyes(eye_input)
        if eyes is None and eye_input is not image:
            eyes = self.eye_service.extract_eyes(image)
        if eyes is None:
            raise ValueError("No eyes detected")

        left_eye = eyes["left_eye"]
        right_eye = eyes["right_eye"]

        left_pupil = self.pupil_service.detect(left_eye)
        right_pupil = self.pupil_service.detect(right_eye)

        left_reflection_report = self.reflection_service.extract_with_report(left_eye, left_pupil)
        right_reflection_report = self.reflection_service.extract_with_report(right_eye, right_pupil)

        left_reflection = left_reflection_report.extracted
        right_reflection = right_reflection_report.extracted

        left_enhancement = self.enhancement_service.enhance_with_report(left_reflection)
        right_enhancement = self.enhancement_service.enhance_with_report(right_reflection)

        left_pre_sr, left_sr, enhanced_left = (
            left_enhancement.pre_sr,
            left_enhancement.sr,
            left_enhancement.final,
        )
        right_pre_sr, right_sr, enhanced_right = (
            right_enhancement.pre_sr,
            right_enhancement.sr,
            right_enhancement.final,
        )

        corneal_left = self.corneal_service.reconstruct(left_eye)
        corneal_right = self.corneal_service.reconstruct(right_eye)

        scene_left = None
        scene_right = None
        scene_service = self._get_scene_service()
        if scene_service is not None:
            try:
                left_geometry_context = (
                    f"left_eye limbus={corneal_left.limbus}, "
                    f"panorama={'available' if corneal_left.panorama is not None else 'unavailable'}, "
                    f"foveated={'available' if corneal_left.foveated is not None else 'unavailable'}"
                )
                scene_left = scene_service.reconstruct(
                    left_geometry_context,
                    corneal_left.panorama if corneal_left.panorama is not None else (corneal_left.foveated if corneal_left.foveated is not None else enhanced_left),
                )
            except Exception:
                scene_left = None

            try:
                right_geometry_context = (
                    f"right_eye limbus={corneal_right.limbus}, "
                    f"panorama={'available' if corneal_right.panorama is not None else 'unavailable'}, "
                    f"foveated={'available' if corneal_right.foveated is not None else 'unavailable'}"
                )
                scene_right = scene_service.reconstruct(
                    right_geometry_context,
                    corneal_right.panorama if corneal_right.panorama is not None else (corneal_right.foveated if corneal_right.foveated is not None else enhanced_right),
                )
            except Exception:
                scene_right = None

        # Reasoning should describe the reconstructed scene, then the panorama, then the reflection ROI.
        try:
            reasoning_left = self.reasoning_service.analyze(scene_left if scene_left is not None else (corneal_left.foveated if corneal_left.foveated is not None else (corneal_left.panorama if corneal_left.panorama is not None else left_reflection)))
        except Exception as exc:
            reasoning_left = f"[reasoning failed] {type(exc).__name__}: {exc}"

        try:
            reasoning_right = self.reasoning_service.analyze(scene_right if scene_right is not None else (corneal_right.foveated if corneal_right.foveated is not None else (corneal_right.panorama if corneal_right.panorama is not None else right_reflection)))
        except Exception as exc:
            reasoning_right = f"[reasoning failed] {type(exc).__name__}: {exc}"

        deepfake = self.deepfake_service.validate(enhanced_left, enhanced_right)

        return ForensicPipelineResult(
            faces=faces,
            left_eye=left_eye,
            right_eye=right_eye,
            left_pupil=left_pupil,
            right_pupil=right_pupil,
            left_reflection=left_reflection,
            right_reflection=right_reflection,
            left_reflection_method=left_reflection_report.selected_method,
            right_reflection_method=right_reflection_report.selected_method,
            left_reflection_scores=left_reflection_report.method_scores,
            right_reflection_scores=right_reflection_report.method_scores,
            left_pre_sr=left_pre_sr,
            right_pre_sr=right_pre_sr,
            left_sr=left_sr,
            right_sr=right_sr,
            enhanced_left_reflection=enhanced_left,
            enhanced_right_reflection=enhanced_right,
            reasoning_left=reasoning_left,
            reasoning_right=reasoning_right,
            deepfake=deepfake,
            left_panorama=corneal_left.panorama,
            right_panorama=corneal_right.panorama,
            left_foveated=corneal_left.foveated,
            right_foveated=corneal_right.foveated,
            fusion_image=fusion_image,
            fusion_method=fusion_method,
            fusion_scores=fusion_scores,
            fusion_frame_count=fusion_frame_count,
            left_enhancement_method=left_enhancement.selected_method,
            right_enhancement_method=right_enhancement.selected_method,
            left_enhancement_scores=left_enhancement.method_scores,
            right_enhancement_scores=right_enhancement.method_scores,
            left_limbus_source=corneal_left.limbus_source,
            right_limbus_source=corneal_right.limbus_source,
            left_limbus_scores=corneal_left.limbus_scores,
            right_limbus_scores=corneal_right.limbus_scores,
            scene_left=scene_left,
            scene_right=scene_right,
        )
