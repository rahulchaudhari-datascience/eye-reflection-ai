from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from services.eye_landmark_service import EyeLandmarkService


def iter_images(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def load_rgb(path: Path) -> np.ndarray:
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return np.array(pil)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("image forensicTestsamples"),
        help="Folder containing sample images",
    )
    parser.add_argument(
        "--debug-out",
        type=Path,
        default=None,
        help="Optional folder to save detected left/right eye crops",
    )
    args = parser.parse_args()

    samples: Path = args.samples
    if not samples.exists():
        raise SystemExit(f"Samples folder not found: {samples}")

    debug_out: Path | None = args.debug_out
    if debug_out is not None:
        debug_out.mkdir(parents=True, exist_ok=True)

    svc = EyeLandmarkService()

    total = 0
    ok = 0
    failed: list[Path] = []

    for img_path in iter_images(samples):
        total += 1
        image = load_rgb(img_path)
        eyes = svc.extract_eyes(image)
        if eyes is None or "left_eye" not in eyes or "right_eye" not in eyes:
            failed.append(img_path)
            print(f"FAIL: {img_path}")
            continue

        left = eyes["left_eye"]
        right = eyes["right_eye"]
        if left is None or right is None or left.size == 0 or right.size == 0:
            failed.append(img_path)
            print(f"FAIL(empty crop): {img_path}")
            continue

        ok += 1
        print(f"OK:   {img_path}  left={left.shape} right={right.shape}")

        if debug_out is not None:
            stem = img_path.stem.replace(" ", "_")
            Image.fromarray(left).save(debug_out / f"{stem}__left.png")
            Image.fromarray(right).save(debug_out / f"{stem}__right.png")

    print("\nSummary")
    print(f"- total: {total}")
    print(f"- ok:    {ok}")
    print(f"- fail:  {len(failed)}")
    if failed:
        print("\nFailed files:")
        for p in failed:
            print(f"- {p}")


if __name__ == "__main__":
    main()
