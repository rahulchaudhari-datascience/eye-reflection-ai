from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


class ReflectionEnhancementService:
    """Upscales reflection crops.

    Tries to use RealESRGAN if available, but falls back to OpenCV resize so the
    Streamlit app can run even when the RealESRGAN Python API varies by package
    version.
    """

    def __init__(
        self,
        *,
        scale: Optional[int] = None,
        target_min_side: Optional[int] = None,
        max_side: Optional[int] = None,
        max_passes: Optional[int] = None,
        denoise_h: Optional[int] = None,
        clahe_clip: Optional[float] = None,
    ):
        try:
            from config import (
                REFLECTION_CLAHE_CLIP,
                REFLECTION_DENOISE_H,
                REFLECTION_MAX_PASSES,
                REFLECTION_MAX_SIDE,
                REFLECTION_SR_SCALE,
                REFLECTION_TARGET_MIN_SIDE,
                EXTERNAL_DEBLUR_CMD,
                EXTERNAL_DENOISE_CMD,
                EXTERNAL_SR_CMD,
            )
        except Exception:
            REFLECTION_SR_SCALE = 4
            REFLECTION_TARGET_MIN_SIDE = 320
            REFLECTION_MAX_SIDE = 2048
            REFLECTION_MAX_PASSES = 3
            REFLECTION_DENOISE_H = 6
            REFLECTION_CLAHE_CLIP = 2.4
            EXTERNAL_SR_CMD = None
            EXTERNAL_DEBLUR_CMD = None
            EXTERNAL_DENOISE_CMD = None

        self.scale = int(scale if scale is not None else REFLECTION_SR_SCALE)
        self._target_min_side = int(
            target_min_side if target_min_side is not None else REFLECTION_TARGET_MIN_SIDE
        )
        self._max_side = int(max_side if max_side is not None else REFLECTION_MAX_SIDE)
        self._max_passes = int(max_passes if max_passes is not None else REFLECTION_MAX_PASSES)
        self._denoise_h = int(denoise_h if denoise_h is not None else REFLECTION_DENOISE_H)
        self._clahe_clip = float(clahe_clip if clahe_clip is not None else REFLECTION_CLAHE_CLIP)
        self._external_sr_cmd = EXTERNAL_SR_CMD
        self._external_deblur_cmd = EXTERNAL_DEBLUR_CMD
        self._external_denoise_cmd = EXTERNAL_DENOISE_CMD

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
                    tile=256,
                    tile_pad=16,
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
        _, _, final = self.enhance_with_stages(image)
        return final

    def enhance_with_stages(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pre_sr, sr, final) for UI display and debugging."""

        # Step 1: Detect specularity regions
        spec_mask = self._detect_specularity(image)

        # Step 2: Wavelet-based denoising (preserves structure better)
        denoised = self._wavelet_denoise(image)

        # Step 3: Highlight-aware enhancement
        enhanced = self._enhance_highlights(denoised, spec_mask)

        # Step 4: Morphological refinement
        refined = self._morphological_refine(enhanced, spec_mask)

        # Step 5: Subpixel pre-upscale to stabilize micro-details
        pre_sr = self._subpixel_enhance(refined)

        # Step 6: Multi-stage SR for tiny crops
        sr = pre_sr
        if self._external_sr_cmd:
            external_sr = self._run_external_tool(sr, self._external_sr_cmd, "external_sr")
            if external_sr is not None:
                sr = external_sr
        else:
            passes = 0
            while passes < max(1, self._max_passes):
                h, w = sr.shape[:2]
                if min(h, w) >= self._target_min_side:
                    break
                if max(h, w) >= self._max_side:
                    break
                sr = self.upscale(sr)
                passes += 1

        # Step 7: Deblur + post-SR denoise + contrast recovery
        post = sr
        if self._external_deblur_cmd:
            external_deblur = self._run_external_tool(post, self._external_deblur_cmd, "external_deblur")
            if external_deblur is not None:
                post = external_deblur
            else:
                post = self._deblur_sr(post)
        else:
            post = self._deblur_sr(post)

        if self._external_denoise_cmd:
            external_denoise = self._run_external_tool(post, self._external_denoise_cmd, "external_denoise")
            if external_denoise is not None:
                post = external_denoise
            else:
                post = self._denoise_sr(post)
        else:
            post = self._denoise_sr(post)

        post = self._recover_contrast(post)

        # Step 8: Moderate, quality-aware sharpening on SR result
        spec_mask_sr = self._detect_specularity(post)
        final = self._adaptive_sharpen(post, spec_mask_sr)

        return pre_sr, sr, final

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

        # Safe fallback: advanced upscale via OpenCV with edge preservation.
        import cv2

        h, w = img.shape[:2]
        new_size: Tuple[int, int] = (int(w * self.scale), int(h * self.scale))
        # Use Lanczos4 for better quality than INTER_CUBIC
        return cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Legacy preprocessing - no longer heavily used."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        return img

    def _detect_specularity(self, image: np.ndarray) -> np.ndarray:
        """Detect specular highlights based on saturation and brightness.
        
        Based on Nishino et al. - specularity is characterized by high brightness
        and low saturation.
        """
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Specularity: high V (brightness) and low S (saturation)
        brightness_mask = v > 200
        saturation_mask = s < 50
        spec_mask = (brightness_mask & saturation_mask).astype(np.uint8) * 255

        # Dilate to get slightly larger highlight regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        spec_mask = cv2.dilate(spec_mask, kernel, iterations=1)

        return spec_mask

    def _wavelet_denoise(self, image: np.ndarray) -> np.ndarray:
        """Wavelet-based denoising using BayesShrink.
        
        More sophisticated than bilateral filtering - preserves edges better.
        """
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        try:
            import pywt
            
            # Denoise each channel independently using wavelet
            denoised_channels = []
            for c in range(3):
                channel = img[:, :, c].astype(np.float32) / 255.0
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec2(channel, 'db1', level=2)
                
                # Apply soft thresholding (Bayesian shrink)
                coeffs_thresh = list(coeffs)
                
                # Estimate noise sigma from detail coefficients
                sigma = np.median(np.abs(coeffs[1][0])) / 0.6745
                
                # Apply thresholding to detail coefficients
                for i in range(1, len(coeffs_thresh)):
                    for j in range(3):  # cA, cH, cV, cD for each level
                        if isinstance(coeffs_thresh[i], tuple):
                            c_sub = coeffs_thresh[i][j]
                        else:
                            c_sub = coeffs_thresh[i]
                        
                        # Soft threshold
                        threshold = sigma * np.sqrt(2 * np.log(channel.size))
                        coeffs_thresh[i] = tuple(np.sign(c_sub) * np.maximum(np.abs(c_sub) - threshold, 0) 
                                                if isinstance(coeffs_thresh[i], tuple) else 
                                                np.sign(c_sub) * np.maximum(np.abs(c_sub) - threshold, 0))
                
                # Reconstruct
                denoised = pywt.waverec2(coeffs_thresh, 'db1')
                denoised = np.clip(denoised, 0, 1) * 255
                denoised_channels.append(denoised.astype(np.uint8))
            
            return np.stack(denoised_channels, axis=-1)
        
        except Exception:
            # Fallback to bilateral filtering if wavelet not available
            return cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)

    def _enhance_highlights(self, image: np.ndarray, spec_mask: np.ndarray) -> np.ndarray:
        """Enhance highlight regions while preserving surrounding areas.
        
        Uses selective CLAHE only on highlight regions.
        """
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE selectively based on specularity mask
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Blend: full CLAHE in highlight regions, moderate in others
        spec_mask_norm = spec_mask.astype(np.float32) / 255.0
        l_blended = (l_enhanced * spec_mask_norm + l * (1 - spec_mask_norm)).astype(np.uint8)

        lab_out = cv2.merge([l_blended, a, b])
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    def _morphological_refine(self, image: np.ndarray, spec_mask: np.ndarray) -> np.ndarray:
        """Morphological operations to refine specularity regions."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Close operation: fills small holes while preserving overall structure
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        spec_mask_refined = cv2.morphologyEx(spec_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Apply selective sharpening in highlight regions
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Slight sharpening via unsharp mask
        l_float = l.astype(np.float32)
        l_blur = cv2.GaussianBlur(l, (0, 0), sigmaX=1.0).astype(np.float32)
        l_sharp = l_float + 0.3 * (l_float - l_blur)
        
        # Apply selectively in highlight regions
        spec_norm = spec_mask_refined.astype(np.float32) / 255.0
        l_final = (l_sharp * spec_norm + l_float * (1 - spec_norm)).astype(np.uint8)

        lab_out = cv2.merge([l_final, a, b])
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    def _adaptive_sharpen(self, image: np.ndarray, spec_mask: np.ndarray) -> np.ndarray:
        """Adaptive sharpening based on content - moderate strength."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Detect edges using Sobel
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        edges_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = (edges / (edges.max() + 1e-6)).astype(np.uint8)

        # Apply moderate sharpening (amount = 0.45 for subtle enhancement)
        sharp = self._sharpen(img, amount=0.45)

        # Blend sharpening with a soft specularity mask to avoid amplifying noise.
        if spec_mask is None:
            return sharp

        mask = cv2.GaussianBlur(spec_mask, (0, 0), sigmaX=2.0)
        alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
        blended = sharp.astype(np.float32) * alpha[..., None] + img.astype(np.float32) * (1.0 - alpha[..., None])
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _subpixel_enhance(self, image: np.ndarray) -> np.ndarray:
        """Gentle pre-upscale to stabilize tiny reflections before SR."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        if min(h, w) >= 96:
            return img

        scale = 1.5
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _deblur_sr(self, image: np.ndarray) -> np.ndarray:
        """Mild Wiener deconvolution for blur recovery without heavy models."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        if min(h, w) < 32:
            return img

        ksize = 9
        sigma = 1.2
        psf_1d = cv2.getGaussianKernel(ksize, sigma)
        psf = (psf_1d @ psf_1d.T).astype(np.float32)
        psf /= max(1e-6, float(psf.sum()))

        def wiener(channel: np.ndarray, k: float = 0.01) -> np.ndarray:
            channel_f = channel.astype(np.float32)
            H = np.fft.fft2(psf, s=channel_f.shape)
            G = np.fft.fft2(channel_f)
            H_conj = np.conj(H)
            denom = (np.abs(H) ** 2) + k
            F = (H_conj / denom) * G
            out = np.fft.ifft2(F)
            out = np.real(out)
            return np.clip(out, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            return wiener(img)

        channels = [wiener(img[:, :, c]) for c in range(3)]
        return np.stack(channels, axis=-1)

    def _denoise_sr(self, image: np.ndarray) -> np.ndarray:
        """Mild post-SR denoising to reduce GAN texture noise."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        h = max(1, int(self._denoise_h))
        # Fast non-local means is a good tradeoff for small crops.
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    def _run_external_tool(self, image: np.ndarray, cmd_template: str, label: str) -> Optional[np.ndarray]:
        """Run an external CLI tool if configured; returns None on failure."""
        import os
        import shlex
        import subprocess
        import tempfile
        from PIL import Image

        if not cmd_template:
            return None

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_in:
                input_path = f_in.name
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out:
                output_path = f_out.name

            Image.fromarray(img).save(input_path)

            cmd = cmd_template.format(input=input_path, output=output_path)
            args = shlex.split(cmd, posix=False)
            subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not os.path.exists(output_path):
                return None

            out = Image.open(output_path).convert("RGB")
            return np.array(out)
        except Exception:
            return None
        finally:
            if input_path and os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except Exception:
                    pass
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

    def _recover_contrast(self, image: np.ndarray) -> np.ndarray:
        """Recover local contrast after SR and denoising."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self._clahe_clip, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

    def _deblur(self, image: np.ndarray) -> np.ndarray:
        """Legacy deblurring - no longer used in new pipeline."""
        return image

    def _sharpen(self, image: np.ndarray, *, amount: float = 0.5) -> np.ndarray:
        """Moderate sharpening using Gaussian blur subtraction."""
        import cv2

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(img, 1.0 + float(amount), blur, -float(amount), 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _edge_enhance(self, image: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        """Legacy edge enhancement - no longer used."""
        return image

    def _unsharp_mask(self, image: np.ndarray, *, radius: float = 1.0, amount: float = 1.0, threshold: float = 0) -> np.ndarray:
        """Legacy unsharp masking - no longer used."""
        return image