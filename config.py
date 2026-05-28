import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REAL_ESRGAN_MODEL_PATH = "models/realesrgan/RealESRGAN_x4plus.pth"

# Optional: set to a local Ultralytics face model path if you have one.
# Example: "models/yolo/yolov11_face.pt"
YOLO_FACE_MODEL_PATH = None

OUTPUT_DIR = "outputs"