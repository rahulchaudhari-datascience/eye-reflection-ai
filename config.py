import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REAL_ESRGAN_MODEL_PATH = "models/realesrgan/RealESRGAN_x4plus.pth"

# Optional: set to a local Ultralytics face model path if you have one.
# Example: "models/yolo/yolov11_face.pt"
YOLO_FACE_MODEL_PATH = None

# Optional: set to a local Ultralytics eye/iris model path if you have one.
# Example: "models/yolo/yolov11_eye.pt"
YOLO_EYE_MODEL_PATH = None

# Optional advanced reflection-segmentation model paths.
# Example: "models/sam2/sam2_hiera_large.pt"
SAM2_MODEL_PATH = None
GROUNDINGDINO_MODEL_PATH = None
MASK2FORMER_MODEL_PATH = None

# Multi-frame fusion uses up to this many frames when a video is uploaded.
FUSION_TOP_K_FRAMES = 12

# Reflection enhancement defaults for higher-resolution outputs.
REFLECTION_SR_SCALE = 4
REFLECTION_TARGET_MIN_SIDE = 1536
REFLECTION_MAX_SIDE = 8192
REFLECTION_MAX_PASSES = 6
REFLECTION_DENOISE_H = 9
REFLECTION_CLAHE_CLIP = 3.0

# Optional external tool hooks. Provide command templates with {input} and {output}.
# Example: r"C:\\Tools\\topaz\\topaz.exe --input {input} --output {output}"
EXTERNAL_SR_CMD = None
EXTERNAL_DEBLUR_CMD = None
EXTERNAL_DENOISE_CMD = None

OUTPUT_DIR = "outputs"