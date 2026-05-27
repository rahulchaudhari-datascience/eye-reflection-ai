import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REAL_ESRGAN_MODEL_PATH = "models/realesrgan/RealESRGAN_x4plus.pth"

OUTPUT_DIR = "outputs"