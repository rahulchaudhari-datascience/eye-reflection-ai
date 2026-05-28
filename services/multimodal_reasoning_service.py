from __future__ import annotations

from typing import Any, Optional

import numpy as np


class MultimodalReasoningService:
    """Image captioning/reasoning.

    HuggingFace `pipeline()` requires an ML backend (PyTorch/TensorFlow/Flax).
    If those aren't installed correctly, importing/initializing at module load
    time will crash Streamlit before the UI renders.
    """

    def __init__(
        self,
        *,
        model: str = "Salesforce/blip-image-captioning-base",
    ):
        self._model = model
        self._task: Optional[str] = None
        self._caption_pipeline: Optional[Any] = None
        self._init_error: Optional[BaseException] = None

        try:
            from transformers import pipeline  # type: ignore
            import torch  # type: ignore

            device = 0 if torch.cuda.is_available() else -1
            last_exc: Optional[BaseException] = None
            for task in ("image-to-text", "image-text-to-text"):
                try:
                    self._caption_pipeline = pipeline(
                        task,
                        model=self._model,
                        device=device,
                    )
                    self._task = task
                    last_exc = None
                    break
                except (KeyError, ValueError) as exc:
                    # Different Transformers versions expose different task names.
                    last_exc = exc
                    self._caption_pipeline = None
                    self._task = None

            if self._caption_pipeline is None and last_exc is not None:
                raise last_exc
        except BaseException as exc:
            self._init_error = exc
            self._caption_pipeline = None

    @property
    def available(self) -> bool:
        return self._caption_pipeline is not None

    def analyze(self, image) -> str:
        if self._caption_pipeline is None:
            msg = "[multimodal reasoning disabled]"
            if self._init_error is not None:
                return f"{msg} {type(self._init_error).__name__}: {self._init_error}"
            return msg

        pil = self._to_pil(image)

        # Prefer deterministic, short outputs for responsiveness.
        gen_kwargs = {"max_new_tokens": 90, "do_sample": False}

        prompt_text = self._load_prompt()

        # Task-specific calling conventions vary across transformers versions.
        result = None
        if self._task == "image-text-to-text":
            # Some pipelines require both an image + a text prompt.
            try:
                result = self._caption_pipeline({"image": pil, "text": prompt_text}, **gen_kwargs)
            except TypeError:
                # Some versions don't accept generation kwargs at pipeline level.
                result = self._caption_pipeline({"image": pil, "text": prompt_text})
        else:
            try:
                result = self._caption_pipeline(pil, **gen_kwargs)
            except TypeError:
                result = self._caption_pipeline(pil)

        if not result:
            return ""
        first = result[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"])
        return str(first)

    def _load_prompt(self) -> str:
        default = "Describe the image."
        try:
            from pathlib import Path

            path = Path(__file__).resolve().parent.parent / "prompts" / "reflection_reasoning_prompt.txt"
            if not path.exists():
                return default
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            return text or default
        except Exception:
            return default

    def _to_pil(self, image):
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim != 3:
                raise TypeError(f"Unsupported numpy image shape: {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")

        raise TypeError(
            "Incorrect format used for image. Provide a PIL image or a numpy array."
        )