from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.color import (
    BrightnessContrastOperation,
    GammaOperation,
    GrayscaleOperation,
)


def test_color_operations_registered() -> None:
    names = default_registry.list_names()
    for expected in ("grayscale", "brightness_contrast", "gamma"):
        assert expected in names


def test_grayscale_variants() -> None:
    rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
    gray_2d = GrayscaleOperation().apply(rgb)
    assert gray_2d.shape == (8, 8)
    rgba = np.full((8, 8, 4), 120, dtype=np.uint8)
    gray_keep = GrayscaleOperation({"keep_channels": True}).apply(rgba)
    assert gray_keep.shape == (8, 8, 4)


def test_brightness_contrast_variants() -> None:
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    brighter = BrightnessContrastOperation({"brightness": 0.2, "contrast": 1.2}).apply(img)
    assert np.all(brighter > 100)
    u16_rgba = np.full((6, 6, 4), 30000, dtype=np.uint16)
    out_u16 = BrightnessContrastOperation({"brightness": -0.1, "contrast": 0.8}).apply(u16_rgba)
    assert out_u16.dtype == np.uint16 and out_u16.shape == (6, 6, 4)
    with pytest.raises(InvalidParametersError):
        BrightnessContrastOperation({"brightness": 2.5})


def test_gamma_variants() -> None:
    img = np.full((6, 6, 3), 128, dtype=np.uint8)
    bright_gamma = GammaOperation({"gamma": 2.0}).apply(img)
    assert np.all(bright_gamma > 128)
    u16 = np.full((6, 6), 32768, dtype=np.uint16)
    out_u16 = GammaOperation({"gamma": 0.5}).apply(u16)
    assert out_u16.dtype == np.uint16 and np.all(out_u16 < 32768)
    with pytest.raises(InvalidParametersError):
        GammaOperation({"gamma": 0.0})


@settings(max_examples=15, deadline=None)
@given(
    brightness=st.floats(min_value=-1.0, max_value=1.0),
    contrast=st.floats(min_value=0.1, max_value=3.0),
    gamma=st.floats(min_value=0.2, max_value=4.0),
)
def test_color_ops_preserve_shape_and_dtype(
    brightness: float,
    contrast: float,
    gamma: float,
) -> None:
    arr: ImageArray = np.full((8, 10, 3), 120, dtype=np.uint8)
    bc = BrightnessContrastOperation({"brightness": brightness, "contrast": contrast}).apply(arr)
    gm = GammaOperation({"gamma": gamma}).apply(arr)
    assert bc.shape == arr.shape and bc.dtype == arr.dtype
    assert gm.shape == arr.shape and gm.dtype == arr.dtype
