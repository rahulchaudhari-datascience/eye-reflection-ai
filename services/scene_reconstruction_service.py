from __future__ import annotations

from typing import Any, Optional


class SceneReconstructionService:
    def __init__(
        self,
        *,
        model: str = "runwayml/stable-diffusion-v1-5",
    ):
        self._model = model
        self._pipe: Optional[Any] = None
        self._init_error: Optional[BaseException] = None

        try:
            import torch  # type: ignore
            from diffusers import StableDiffusionPipeline  # type: ignore

            self._pipe = StableDiffusionPipeline.from_pretrained(
                self._model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")
        except BaseException as exc:
            self._init_error = exc
            self._pipe = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def reconstruct(self, reasoning_text: str):
        if self._pipe is None:
            raise RuntimeError(
                "Scene reconstruction is unavailable. "
                f"Init error: {type(self._init_error).__name__}: {self._init_error}"
            )

        prompt = (
            "Highly realistic scene reconstruction from eye reflection: "
            f"{reasoning_text}"
        )
        image = self._pipe(prompt).images[0]
        return image