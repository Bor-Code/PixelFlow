from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np

from pixelflow.core.io import save_image
from pixelflow.core.operation import ImageArray


def build_sample_document_image() -> ImageArray:
    canvas = np.full((240, 360, 3), 235, dtype=np.uint8)
    rng = np.random.default_rng(42)
    noise = rng.integers(-15, 15, size=canvas.shape, dtype=np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(noisy, "INVOICE 2026", (30, 60), font, 1.0, (25, 25, 25), 2)
    cv2.putText(noisy, "Total: 128.00 USD", (30, 120), font, 0.8, (40, 40, 40), 2)
    cv2.rectangle(noisy, (25, 150), (330, 210), (50, 50, 50), 2)
    return cast(ImageArray, noisy)


def build_sample_photo_image() -> ImageArray:
    h, w = 360, 480
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    r = np.clip((x_grid * 255) // w, 0, 255).astype(np.uint8)
    g = np.clip((y_grid * 255) // h, 0, 255).astype(np.uint8)
    b = np.full((h, w), 160, dtype=np.uint8)
    photo = np.dstack([r, g, b])
    cv2.circle(photo, (240, 180), 80, (255, 220, 80), -1)
    return cast(ImageArray, photo)


def generate_example_assets(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path = save_image(build_sample_document_image(), output_dir / "sample_document.png")
    photo_path = save_image(build_sample_photo_image(), output_dir / "sample_photo.png")
    return doc_path, photo_path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "assets"
    generate_example_assets(target)
