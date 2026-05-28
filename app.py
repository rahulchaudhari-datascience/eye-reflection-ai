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

import streamlit as st
from PIL import Image
import numpy as np
from typing import Optional

from pipelines.forensic_pipeline import ForensicVisionPipeline


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
    enhanced_reflection: np.ndarray,
    scene: Optional[Image.Image],
    reasoning: str,
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 4️⃣ Enhance Image")
        st.text("Denoise + Contrast\nenhancement applied")
    
    with col2:
        st.markdown("### 5️⃣ Super Resolution")
        st.text("ESRGAN upscaling\n(4x magnification)")
    
    with col3:
        st.markdown("### 6️⃣ Final Result")
        st.image(enhanced_reflection, use_container_width=True, caption="Enhanced reflection")
    
    st.divider()
    
    # Techniques Used
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Techniques Used")
        techniques = [
            "✓ OpenCV (Eye Detection & Face detection)",
            "✓ MediaPipe (Facial Landmarks)",
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


def main() -> None:
    st.title("Eye Reflection Forensics")

    with st.sidebar:
        st.header("Options")
        enable_scene = st.checkbox(
            "Enable scene reconstruction (downloads Stable Diffusion model)",
            value=False,
        )

    uploaded = st.file_uploader(
        "Upload an image (jpg/png)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(pil)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil, use_container_width=True)

    run = st.button("Run Forensic Pipeline", type="primary")

    if not run:
        return

    pipeline = get_pipeline(enable_scene)

    try:
        with st.spinner("Processing..."):
            result = pipeline.process(image_np)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return

    with col2:
        st.subheader("Deepfake check")
        st.write(result.deepfake)

    st.divider()

    eyes_col1, eyes_col2 = st.columns(2)
    with eyes_col1:
        st.subheader("Left eye")
        st.image(result.left_eye, use_container_width=True)
        st.caption(f"Pupil circle: {result.left_pupil}")

        st.markdown("**Extracted reflection**")
        st.image(result.left_reflection, use_container_width=True)

        st.markdown("**Enhanced reflection**")
        st.image(result.enhanced_left_reflection, use_container_width=True)

        st.markdown("**Reasoning (caption)**")
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

        st.markdown("**Enhanced reflection**")
        st.image(result.enhanced_right_reflection, use_container_width=True)

        st.markdown("**Reasoning (caption)**")
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
            original_image=pil,
            raw_eye=result.left_eye,
            extracted_reflection=result.left_reflection,
            enhanced_reflection=result.enhanced_left_reflection,
            scene=result.scene_left,
            reasoning=result.reasoning_left,
            eye_name="Left"
        )
    
    with tab2:
        display_pipeline_summary(
            original_image=pil,
            raw_eye=result.right_eye,
            extracted_reflection=result.right_reflection,
            enhanced_reflection=result.enhanced_right_reflection,
            scene=result.scene_right,
            reasoning=result.reasoning_right,
            eye_name="Right"
        )


if __name__ == "__main__":
    main()
