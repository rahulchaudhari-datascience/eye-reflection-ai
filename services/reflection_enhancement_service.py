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

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enterprise-ish enhancement chain.

        Denoise -> CLAHE -> slight sharpening -> Super-resolution -> final sharpening.
        """

        pre = self._preprocess(image)
        pre = self._deblur(pre)

        # Multi-stage SR for tiny crops (closer to enterprise pipelines).
        sr = pre
        max_side = 1024
        passes = 0
        while passes < 2:
            h, w = sr.shape[:2]
            if min(h, w) >= 220:
                break
            if max(h, w) >= max_side:
                break
            sr = self.upscale(sr)
            # Small post-pass sharpen helps SR outputs.
            sr = self._sharpen(sr, amount=0.25)
            passes += 1

        # Final sharpen.
        return self._sharpen(sr, amount=0.65)

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

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # Denoising: avoid over-smoothing tiny crops (it destroys the reflection signal).
        h, w = img.shape[:2]
        if min(h, w) < 120:
            den = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)
        else:
            den = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

        # Contrast enhancement (CLAHE on luminance).
        lab = cv2.cvtColor(den, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

        # Mild pre-sharpen to help SR latch onto edges.
        return self._sharpen(out, amount=0.35)

    def _deblur(self, image: np.ndarray) -> np.ndarray:
        """Classical deblurring on luminance.

        Uses scikit-image Richardson–Lucy (preferred) with a small Gaussian PSF,
        falling back to Wiener if RL isn't available.
        """

        try:
            import cv2
            import numpy as np

            img = image
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            # Work on LAB luminance for stability.
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            l_f = l.astype(np.float32) / 255.0

            # PSF: small Gaussian blur kernel.
            k = 9
            sigma = 1.2
            ax = np.arange(-(k // 2), k // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)
            psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
            psf /= float(np.sum(psf))

            deconv = None
            try:
                from skimage.restoration import richardson_lucy  # type: ignore

                # Fewer iterations for speed; prevents ringing.
                deconv = richardson_lucy(l_f, psf, num_iter=10, clip=False)
            except Exception:
                try:
                    from skimage.restoration import wiener  # type: ignore

                    deconv = wiener(l_f, psf, balance=0.08, clip=False)
                except Exception:
                    deconv = None

            if deconv is None:
                return img

            l2 = np.clip(deconv, 0.0, 1.0)
            l2 = (l2 * 255.0).astype(np.uint8)
            lab2 = cv2.merge([l2, a, b])
            out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

            # Very light sharpen to counteract deconvolution softness.
            return self._sharpen(out, amount=0.20)
        except Exception:
            return image

    def _sharpen(self, image: np.ndarray, *, amount: float = 0.5) -> np.ndarray:
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(img, 1.0 + float(amount), blur, -float(amount), 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)