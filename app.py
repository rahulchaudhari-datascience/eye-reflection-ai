# Project Structure
# text
# image-restoration-app/
# ├── app.py                 # Main streamlined application
# ├── restoration_pipeline.py # Optimized pipeline with best methods
# ├── utils.py               # Utility functions
# ├── requirements.txt       # Dependencies
# └── models/                # Model cache directory


# # Delete the old venv folder
# rmdir /s venv  # On Windows
# # OR
# rm -rf venv    # On Mac/Linux

# # Create new virtual environment with Python 3.10 (recommended)
# python -m venv venv

# # Activate it
# venv\Scripts\activate  # Windows
# # OR
# source venv/bin/activate  # Mac/Linux

# # Install requirements
# pip install -r requirements.txt

# # Run the app
# streamlit run app.py

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
import numpy as np
from typing import Optional

from pipelines.forensic_pipeline import ForensicVisionPipeline
from config import FUSION_TOP_K_FRAMES


st.set_page_config(
    page_title="Eye Reflection Forensics",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline(enable_scene_reconstruction: bool) -> ForensicVisionPipeline:
    return ForensicVisionPipeline(enable_scene_reconstruction=enable_scene_reconstruction)


def display_pipeline_summary(
    original_image: Image.Image,
    raw_eye: np.ndarray,
    extracted_reflection: np.ndarray,
    pre_sr: Optional[np.ndarray],
    sr: Optional[np.ndarray],
    enhanced_reflection: np.ndarray,
    panorama: Optional[np.ndarray],
    foveated: Optional[np.ndarray],
    scene: Optional[Image.Image],
    reasoning: str,
    enhancement_method: Optional[str],
    enhancement_scores: Optional[dict[str, Optional[float]]],
    reflection_method: Optional[str],
    reflection_scores: Optional[dict[str, Optional[float]]],
    limbus_source: Optional[str],
    limbus_scores: Optional[dict[str, Optional[float]]],
    eye_name: str,
) -> None:
    """Display a comprehensive pipeline summary with all processing steps."""
    
    # Pipeline Steps
    st.subheader(f"Processing Pipeline - {eye_name} Eye")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Input Image")
        st.image(original_image, use_container_width=True, caption="Original photo")
    
    with col2:
        st.markdown("### 2️⃣ Crop & Detect")
        st.image(raw_eye, use_container_width=True, caption="Extracted eye region")
    
    with col3:
        st.markdown("### 3️⃣ Extract Reflection")
        st.image(extracted_reflection, use_container_width=True, caption="Raw reflection")

    if reflection_method or reflection_scores:
        st.markdown("**Reflection extraction scores**")
        if reflection_method:
            st.caption(f"Selected: {reflection_method}")
        rows = []
        for name in [
            "SAM2",
            "GroundingDINO+SAM2",
            "Mask2Former",
            "Threshold-based fallback",
        ]:
            value = None if reflection_scores is None else reflection_scores.get(name)
            rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
        st.table(rows)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 4️⃣ Enhance Image")
        if pre_sr is not None:
            st.image(pre_sr, use_container_width=True, caption="Pre-SR enhancement")
        else:
            st.text("Denoise + Contrast\nenhancement applied")
    
    with col2:
        st.markdown("### 5️⃣ Super Resolution")
        if sr is not None:
            st.image(sr, use_container_width=True, caption="Super-resolution output")
        else:
            st.text("ESRGAN upscaling\n(4x magnification)")
    
    with col3:
        st.markdown("### 6️⃣ Final Result")
        st.image(enhanced_reflection, use_container_width=True, caption="Enhanced reflection")

    if enhancement_method or enhancement_scores:
        st.markdown("**Enhancement scores**")
        if enhancement_method:
            st.caption(f"Selected: {enhancement_method}")
        rows = []
        for name in [
            "RealESRGAN",
            "SwinIR",
            "HAT",
            "DAT",
            "BSRGAN",
            "DiffBIR",
            "SeeSR",
        ]:
            value = None if enhancement_scores is None else enhancement_scores.get(name)
            rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
        st.table(rows)

    if panorama is not None or foveated is not None:
        st.divider()
        st.subheader("Corneal Imaging Outputs")
        st.info(
            "Single-eye reconstruction follows the original paper assumptions: "
            "accurate limbus detection, corneal geometry estimation, and reflection ray tracing."
        )
        if limbus_source or limbus_scores:
            st.markdown("**Limbus detection scores**")
            if limbus_source:
                st.caption(f"Selected: {limbus_source}")
            rows = []
            for name in [
                "MediaPipe Iris",
                "SAM2 segmentation",
                "YOLOv11 eye detector",
                "Ellipse fitting",
            ]:
                value = None if limbus_scores is None else limbus_scores.get(name)
                rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
            st.table(rows)
        pano_col, fovea_col = st.columns(2)
        with pano_col:
            st.markdown("**Spherical panorama (cropped)**")
            if panorama is not None:
                st.image(panorama, use_container_width=True)
            else:
                st.info("Panorama unavailable")
        with fovea_col:
            st.markdown("**Foveated retinal image (45° FOV)**")
            if foveated is not None:
                st.image(foveated, use_container_width=True)
            else:
                st.info("Foveated image unavailable")
    
    st.divider()
    
    # Techniques Used
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Techniques Used")
        techniques = [
            "✓ OpenCV (Eye Detection & Face detection)",
            "✓ Multi-frame Fusion (Optical Flow / Frame Alignment / Multi-frame SR)",
            "✓ MediaPipe Iris + Ellipse Fitting",
            "✓ ESRGAN (Super Resolution)",
            "✓ Bilateral Filter (Denoising)",
            "✓ CLAHE (Contrast Enhancement)",
            "✓ Multimodal LLM (Scene Analysis)",
        ]
        for tech in techniques:
            st.text(tech)
    
    with col2:
        st.markdown("### 📋 Result Analysis")
        st.markdown(f"**Scene Description:**")
        st.info(reasoning)
        
        if scene is not None:
            st.markdown("**Reconstructed Scene:**")
            st.image(scene, use_container_width=True, caption="AI-generated scene reconstruction")


def extract_video_frames(uploaded_file, *, max_frames: int) -> list[np.ndarray]:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[np.ndarray] = []

        if frame_count > 0:
            indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count)).round().astype(int)
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    continue
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return frames
    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def main() -> None:
    st.title("Eye Reflection Forensics")

    with st.sidebar:
        st.header("Options")
        enable_scene = st.checkbox(
            "Enable scene reconstruction (downloads Stable Diffusion model)",
            value=False,
        )

    uploaded = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "mov", "avi", "mkv", "webm"],
    )

    if uploaded is None:
        st.info("Upload an image or video to begin.")
        return

    is_video = uploaded.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
    input_frames = None

    if is_video:
        input_frames = extract_video_frames(uploaded, max_frames=FUSION_TOP_K_FRAMES)
        if not input_frames:
            st.error("Could not extract frames from the uploaded video.")
            return
        preview_image = Image.fromarray(input_frames[0])
        process_input = input_frames
        st.video(uploaded.getvalue())
    else:
        preview_image = Image.open(uploaded).convert("RGB")
        process_input = np.array(preview_image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(preview_image, use_container_width=True)

    run = st.button("Run Forensic Pipeline", type="primary")

    if not run:
        return

    pipeline = get_pipeline(enable_scene)

    try:
        with st.spinner("Processing..."):
            result = pipeline.process(process_input)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return

    display_image = Image.fromarray(result.fusion_image) if result.fusion_image is not None else preview_image

    with col2:
        st.subheader("Deepfake check")
        st.write(result.deepfake)

    if result.fusion_method or result.fusion_scores:
        st.divider()
        st.subheader("Frame Fusion")
        if result.fusion_method:
            st.caption(f"Selected: {result.fusion_method} | Frames used: {result.fusion_frame_count}")
        fusion_rows = []
        for name in ["Optical Flow", "RAFT", "RIFE", "Frame Alignment", "Multi-frame Super Resolution"]:
            value = None if result.fusion_scores is None else result.fusion_scores.get(name)
            fusion_rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
        st.table(fusion_rows)
        if result.fusion_image is not None:
            st.image(result.fusion_image, use_container_width=True, caption="Fused frame used for reflection extraction")

    st.divider()

    eyes_col1, eyes_col2 = st.columns(2)
    with eyes_col1:
        st.subheader("Left eye")
        st.image(result.left_eye, use_container_width=True)
        st.caption(f"Pupil circle: {result.left_pupil}")

        st.markdown("**Extracted reflection**")
        st.image(result.left_reflection, use_container_width=True)

        if result.left_pre_sr is not None:
            st.markdown("**Enhanced (pre-SR)**")
            st.image(result.left_pre_sr, use_container_width=True)

        if result.left_sr is not None:
            st.markdown("**Super-resolution**")
            st.image(result.left_sr, use_container_width=True)

        st.markdown("**Enhanced reflection**")
        st.image(result.enhanced_left_reflection, use_container_width=True)

        if result.left_panorama is not None:
            st.markdown("**Spherical panorama (cropped)**")
            st.image(result.left_panorama, use_container_width=True)

        if result.left_foveated is not None:
            st.markdown("**Foveated retinal image (45° FOV)**")
            st.image(result.left_foveated, use_container_width=True)

        if result.left_enhancement_method or result.left_enhancement_scores:
            st.markdown("**Enhancement scores**")
            if result.left_enhancement_method:
                st.caption(f"Selected: {result.left_enhancement_method}")
            left_enh_rows = []
            for name in [
                "RealESRGAN",
                "SwinIR",
                "HAT",
                "DAT",
                "BSRGAN",
                "DiffBIR",
                "SeeSR",
            ]:
                value = None if result.left_enhancement_scores is None else result.left_enhancement_scores.get(name)
                left_enh_rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
            st.table(left_enh_rows)

        if result.left_reflection_method or result.left_reflection_scores:
            st.markdown("**Reflection extraction scores**")
            if result.left_reflection_method:
                st.caption(f"Selected: {result.left_reflection_method}")
            left_ref_rows = []
            for name in [
                "SAM2",
                "GroundingDINO+SAM2",
                "Mask2Former",
                "Threshold-based fallback",
            ]:
                value = None if result.left_reflection_scores is None else result.left_reflection_scores.get(name)
                left_ref_rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
            st.table(left_ref_rows)

        if result.left_limbus_source or result.left_limbus_scores:
            st.markdown("**Limbus detection scores**")
            if result.left_limbus_source:
                st.caption(f"Selected: {result.left_limbus_source}")
            left_rows = []
            for name in [
                "MediaPipe Iris",
                "SAM2 segmentation",
                "YOLOv11 eye detector",
                "Ellipse fitting",
            ]:
                value = None if result.left_limbus_scores is None else result.left_limbus_scores.get(name)
                left_rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
            st.table(left_rows)

        st.markdown("**Scene reasoning**")
        st.write(result.reasoning_left)

        if result.scene_left is not None:
            st.markdown("**Scene reconstruction**")
            st.image(result.scene_left, use_container_width=True)

    with eyes_col2:
        st.subheader("Right eye")
        st.image(result.right_eye, use_container_width=True)
        st.caption(f"Pupil circle: {result.right_pupil}")

        st.markdown("**Extracted reflection**")
        st.image(result.right_reflection, use_container_width=True)

        if result.right_pre_sr is not None:
            st.markdown("**Enhanced (pre-SR)**")
            st.image(result.right_pre_sr, use_container_width=True)

        if result.right_sr is not None:
            st.markdown("**Super-resolution**")
            st.image(result.right_sr, use_container_width=True)

        st.markdown("**Enhanced reflection**")
        st.image(result.enhanced_right_reflection, use_container_width=True)

        if result.right_panorama is not None:
            st.markdown("**Spherical panorama (cropped)**")
            st.image(result.right_panorama, use_container_width=True)

        if result.right_foveated is not None:
            st.markdown("**Foveated retinal image (45° FOV)**")
            st.image(result.right_foveated, use_container_width=True)

        if result.right_enhancement_method or result.right_enhancement_scores:
            st.markdown("**Enhancement scores**")
            if result.right_enhancement_method:
                st.caption(f"Selected: {result.right_enhancement_method}")
            right_enh_rows = []
            for name in [
                "RealESRGAN",
                "SwinIR",
                "HAT",
                "DAT",
                "BSRGAN",
                "DiffBIR",
                "SeeSR",
            ]:
                value = None if result.right_enhancement_scores is None else result.right_enhancement_scores.get(name)
                right_enh_rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
            st.table(right_enh_rows)

        if result.right_reflection_method or result.right_reflection_scores:
            st.markdown("**Reflection extraction scores**")
            if result.right_reflection_method:
                st.caption(f"Selected: {result.right_reflection_method}")
            right_ref_rows = []
            for name in [
                "SAM2",
                "GroundingDINO+SAM2",
                "Mask2Former",
                "Threshold-based fallback",
            ]:
                value = None if result.right_reflection_scores is None else result.right_reflection_scores.get(name)
                right_ref_rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
            st.table(right_ref_rows)

        if result.right_limbus_source or result.right_limbus_scores:
            st.markdown("**Limbus detection scores**")
            if result.right_limbus_source:
                st.caption(f"Selected: {result.right_limbus_source}")
            right_rows = []
            for name in [
                "MediaPipe Iris",
                "SAM2 segmentation",
                "YOLOv11 eye detector",
                "Ellipse fitting",
            ]:
                value = None if result.right_limbus_scores is None else result.right_limbus_scores.get(name)
                right_rows.append({"Method": name, "Score": "unavailable" if value is None else f"{value:.1f}"})
            st.table(right_rows)

        st.markdown("**Scene reasoning**")
        st.write(result.reasoning_right)

        if result.scene_right is not None:
            st.markdown("**Scene reconstruction**")
            st.image(result.scene_right, use_container_width=True)

    st.divider()
    
    # Pipeline Summary Section
    st.header("📊 Pipeline Summary")
    
    tab1, tab2 = st.tabs(["Left Eye Pipeline", "Right Eye Pipeline"])
    
    with tab1:
        display_pipeline_summary(
            original_image=display_image,
            raw_eye=result.left_eye,
            extracted_reflection=result.left_reflection,
            pre_sr=result.left_pre_sr,
            sr=result.left_sr,
            enhanced_reflection=result.enhanced_left_reflection,
            panorama=result.left_panorama,
            foveated=result.left_foveated,
            scene=result.scene_left,
            reasoning=result.reasoning_left,
            enhancement_method=result.left_enhancement_method,
            enhancement_scores=result.left_enhancement_scores,
            reflection_method=result.left_reflection_method,
            reflection_scores=result.left_reflection_scores,
            limbus_source=result.left_limbus_source,
            limbus_scores=result.left_limbus_scores,
            eye_name="Left"
        )
    
    with tab2:
        display_pipeline_summary(
            original_image=display_image,
            raw_eye=result.right_eye,
            extracted_reflection=result.right_reflection,
            pre_sr=result.right_pre_sr,
            sr=result.right_sr,
            enhanced_reflection=result.enhanced_right_reflection,
            panorama=result.right_panorama,
            foveated=result.right_foveated,
            scene=result.scene_right,
            reasoning=result.reasoning_right,
            enhancement_method=result.right_enhancement_method,
            enhancement_scores=result.right_enhancement_scores,
            reflection_method=result.right_reflection_method,
            reflection_scores=result.right_reflection_scores,
            limbus_source=result.right_limbus_source,
            limbus_scores=result.right_limbus_scores,
            eye_name="Right"
        )


if __name__ == "__main__":
    main()
