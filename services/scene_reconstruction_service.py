from __future__ import annotations

from typing import Any, Optional


class SceneReconstructionService:
    """Generate a scene estimate from eye-reflection reasoning.

    The original single-eye reconstruction setup assumes accurate limbus
    detection, corneal geometry estimation, and reflection ray tracing.
    This service uses the reflection reasoning text and optional reference
    image as a practical downstream approximation of that reconstruction step.
    """

    def __init__(
        self,
        *,
        model: str = "runwayml/stable-diffusion-v1-5",
    ):
        self._model = model
        self._pipe: Optional[Any] = None
        self._img2img: Optional[Any] = None
        self._init_error: Optional[BaseException] = None

        try:
            import torch  # type: ignore
            from diffusers import StableDiffusionPipeline  # type: ignore
            from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore

            self._pipe = StableDiffusionPipeline.from_pretrained(
                self._model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")

            try:
                self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self._model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                if torch.cuda.is_available():
                    self._img2img = self._img2img.to("cuda")
            except BaseException:
                self._img2img = None
        except BaseException as exc:
            self._init_error = exc
            self._pipe = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def reconstruct(self, geometry_context: str, reference_image: Optional[Any] = None):
        if self._pipe is None:
            raise RuntimeError(
                "Scene reconstruction is unavailable. "
                f"Init error: {type(self._init_error).__name__}: {self._init_error}"
            )

        prompt = (
            "Single-eye image reconstruction from corneal geometry and panorama, "
            "following the pipeline reflection -> geometry -> panorama -> scene, "
            "natural lighting, sharp details, wide-angle view: "
            f"{geometry_context}"
        )

        if reference_image is not None and self._img2img is not None:
            from PIL import Image
            import numpy as np

            if isinstance(reference_image, np.ndarray):
                ref = Image.fromarray(reference_image)
            else:
                ref = reference_image

            ref = ref.convert("RGB")
            ref = ref.resize((512, 512), Image.LANCZOS)

            image = self._img2img(
                prompt=prompt,
                image=ref,
                strength=0.65,
                guidance_scale=7.5,
            ).images[0]
        else:
            image = self._pipe(prompt, guidance_scale=7.5).images[0]
        return image