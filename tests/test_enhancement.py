from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.enhancement import HistogramEqualizationOperation, SharpenOperation


def test_enhancement_registered() -> None:
    names = default_registry.list_names()
    assert "histogram_equalization" in names and "sharpen" in names


def test_histogram_equalization_global_and_clahe() -> None:
    gray = np.arange(64, dtype=np.uint8).reshape((8, 8))
    eq_gray = HistogramEqualizationOperation({"use_clahe": False}).apply(gray)
    assert eq_gray.shape == (8, 8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    clahe_rgb = HistogramEqualizationOperation(
        {"use_clahe": True, "clip_limit": 2.5, "tile_grid_size": 4}
    ).apply(rgb)
    assert clahe_rgb.shape == (8, 8, 3)


def test_sharpen_unsharp_mask_and_threshold() -> None:
    img = np.full((12, 12, 4), 100, dtype=np.uint16)
    img[4:8, 4:8, :3] = 40000
    sharp = SharpenOperation({"kernel_size": 3, "sigma": 1.0, "amount": 1.5, "threshold": 5}).apply(
        img
    )
    assert sharp.shape == (12, 12, 4) and sharp.dtype == np.uint16
    with pytest.raises(InvalidParametersError):
        SharpenOperation({"kernel_size": 4})


@settings(max_examples=12, deadline=None)
@given(use_clahe=st.booleans(), amount=st.floats(min_value=0.5, max_value=3.0))
def test_enhancement_preserves_shape_and_dtype(use_clahe: bool, amount: float) -> None:
    arr: ImageArray = np.full((12, 12, 3), 120, dtype=np.uint8)
    h = HistogramEqualizationOperation({"use_clahe": use_clahe}).apply(arr)
    s = SharpenOperation({"amount": amount}).apply(arr)
    assert h.shape == arr.shape and h.dtype == arr.dtype
    assert s.shape == arr.shape and s.dtype == arr.dtype
