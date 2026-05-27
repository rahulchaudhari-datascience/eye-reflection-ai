from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


class ReflectionEnhancementService:
    """Upscales reflection crops.

    Tries to use RealESRGAN if available, but falls back to OpenCV resize so the
    Streamlit app can run even when the RealESRGAN Python API varies by package
    version.
    """

    def __init__(self, *, scale: int = 4):
        self.scale = int(scale)

        self._backend: str = "cv2"
        self._model: Optional[Any] = None
        self._init_error: Optional[BaseException] = None

        try:
            from config import DEVICE, REAL_ESRGAN_MODEL_PATH
            import torch  # type: ignore

            device = torch.device(DEVICE)

            # API variant 1 (some packages expose RealESRGAN class).
            try:
                from realesrgan import RealESRGAN  # type: ignore
                from PIL import Image

                model = RealESRGAN(device, scale=self.scale)
                model.load_weights(REAL_ESRGAN_MODEL_PATH)

                self._backend = "RealESRGAN"
                self._model = (model, Image)
                return
            except BaseException:
                pass

            # API variant 2 (official xinntao/realesrgan pip package).
            try:
                from realesrgan import RealESRGANer  # type: ignore
                from realesrgan.archs.rrdbnet_arch import RRDBNet  # type: ignore

                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=self.scale,
                )

                half = str(DEVICE).lower().startswith("cuda")
                upsampler = RealESRGANer(
                    scale=self.scale,
                    model_path=REAL_ESRGAN_MODEL_PATH,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=half,
                    device=device,
                )

                self._backend = "RealESRGANer"
                self._model = upsampler
                return
            except BaseException:
                pass

        except BaseException as exc:
            self._init_error = exc
            self._backend = "cv2"
            self._model = None

    @property
    def backend(self) -> str:
        return self._backend

    def upscale(self, image: np.ndarray) -> np.ndarray:
        # Normalize to HxWx3 uint8
        if image is None:
            raise ValueError("No image provided")

        img = image
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(f"Unexpected image shape: {img.shape}")
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if self._backend == "RealESRGAN" and self._model is not None:
            model, Image = self._model  # type: ignore[misc]
            pil_image = Image.fromarray(img)
            sr_image = model.predict(pil_image)
            return np.array(sr_image)

        if self._backend == "RealESRGANer" and self._model is not None:
            import cv2

            # RealESRGANer commonly assumes OpenCV BGR images.
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output_bgr, _ = self._model.enhance(bgr, outscale=self.scale)
            return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        # Safe fallback: basic upscale via OpenCV.
        import cv2

        h, w = img.shape[:2]
        new_size: Tuple[int, int] = (int(w * self.scale), int(h * self.scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)