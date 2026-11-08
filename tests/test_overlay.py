from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixelflow.core.io import save_image
from pixelflow.core.pipeline import Pipeline
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.color import BrightnessContrastOperation, GrayscaleOperation
from pixelflow.operations.filters import GaussianBlurOperation
from pixelflow.operations.geometry import FlipOperation, ResizeOperation
from pixelflow.operations.overlay import WatermarkOperation


def test_watermark_registered_and_validation() -> None:
    assert "watermark" in default_registry.list_names()
    with pytest.raises(InvalidParametersError):
        WatermarkOperation({})


def test_text_watermark_positions() -> None:
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    for pos in ("top-left", "top-right", "bottom-left", "bottom-right", "center", "custom"):
        out = WatermarkOperation(
            {"text": "PF", "position": pos, "opacity": 0.8, "x": 5, "y": 5}
        ).apply(img)
        assert out.shape == img.shape and np.any(out > 0)


def test_image_watermark_and_grayscale(tmp_path: Path) -> None:
    wm = np.full((16, 16, 4), 255, dtype=np.uint8)
    save_image(wm, tmp_path / "wm.png")
    base_gray = np.zeros((64, 64), dtype=np.uint16)
    out = WatermarkOperation(
        {
            "watermark_path": "wm.png",
            "base_dir": str(tmp_path),
            "position": "center",
            "opacity": 0.5,
        }
    ).apply(base_gray)
    assert out.shape == (64, 64) and out.dtype == np.uint16 and np.any(out > 0)
    out_of_bounds = WatermarkOperation(
        {"text": "X", "position": "custom", "x": 500, "y": 500}
    ).apply(np.zeros((32, 32, 3), dtype=np.uint8))
    assert np.all(out_of_bounds == 0)


def test_golden_image_pipeline_with_tolerance() -> None:
    src = np.zeros((32, 32, 3), dtype=np.uint8)
    src[8:24, 8:24] = [200, 100, 50]
    pipeline = Pipeline(
        [
            ResizeOperation({"width": 16, "height": 16, "interpolation": "nearest"}),
            FlipOperation({"mode": "horizontal"}),
            BrightnessContrastOperation({"brightness": 0.1, "contrast": 1.0}),
            GaussianBlurOperation({"kernel_size": 3, "sigma_x": 0.5}),
            GrayscaleOperation(),
        ]
    )
    actual = pipeline.execute(src)
    expected_center = 148.0
    assert actual.shape == (16, 16)
    assert np.isclose(float(actual[8, 8]), expected_center, atol=3.0)
    assert np.isclose(float(actual[0, 0]), 26.0, atol=2.0)
