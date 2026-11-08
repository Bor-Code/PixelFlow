from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.geometry import (
    CropOperation,
    FlipOperation,
    ResizeOperation,
    RotateOperation,
)


def test_geometry_registered() -> None:
    names = default_registry.list_names()
    for expected in ("resize", "crop", "rotate", "flip"):
        assert expected in names


def test_resize_modes_and_aspect_ratio() -> None:
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    exact = ResizeOperation({"width": 10, "height": 10, "interpolation": "nearest"}).apply(img)
    assert exact.shape == (10, 10, 3)
    by_w = ResizeOperation({"width": 20, "keep_aspect_ratio": True}).apply(img)
    assert by_w.shape == (10, 20, 3)
    by_h = ResizeOperation({"height": 10, "keep_aspect_ratio": True}).apply(img)
    assert by_h.shape == (10, 20, 3)
    both_ar = ResizeOperation({"width": 20, "height": 30, "keep_aspect_ratio": True}).apply(img)
    assert both_ar.shape == (10, 20, 3)
    one_ch = np.zeros((10, 20, 1), dtype=np.uint16)
    res_1ch = ResizeOperation({"width": 5, "height": 5, "interpolation": "cubic"}).apply(one_ch)
    assert res_1ch.shape == (5, 5, 1) and res_1ch.dtype == np.uint16
    with pytest.raises(InvalidParametersError):
        ResizeOperation({})


def test_crop_valid_and_out_of_bounds() -> None:
    img = np.arange(100, dtype=np.uint8).reshape((10, 10))
    cropped = CropOperation({"x": 2, "y": 3, "width": 4, "height": 5}).apply(img)
    assert cropped.shape == (5, 4)
    assert np.array_equal(cropped, img[3:8, 2:6])
    with pytest.raises(InvalidParametersError):
        CropOperation({"x": 8, "y": 0, "width": 5, "height": 5}).apply(img)


def test_rotate_cardinal_and_arbitrary() -> None:
    img = np.zeros((10, 20, 4), dtype=np.uint8)
    assert RotateOperation({"angle": 0}).apply(img).shape == (10, 20, 4)
    assert RotateOperation({"angle": 90}).apply(img).shape == (20, 10, 4)
    assert RotateOperation({"angle": 180}).apply(img).shape == (10, 20, 4)
    assert RotateOperation({"angle": 270}).apply(img).shape == (20, 10, 4)
    no_exp = RotateOperation({"angle": 45, "expand": False}).apply(img)
    assert no_exp.shape == (10, 20, 4)
    exp_1ch = RotateOperation({"angle": 33, "expand": True}).apply(
        np.zeros((10, 12, 1), dtype=np.uint8)
    )
    assert exp_1ch.ndim == 3 and exp_1ch.shape[2] == 1


def test_flip_modes() -> None:
    img = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    h_flip = FlipOperation({"mode": "horizontal"}).apply(img)
    assert np.array_equal(h_flip, np.array([[2, 1], [4, 3]], dtype=np.uint16))
    v_flip = FlipOperation({"mode": "vertical"}).apply(img)
    assert np.array_equal(v_flip, np.array([[3, 4], [1, 2]], dtype=np.uint16))
    b_flip = FlipOperation({"mode": "both"}).apply(img[:, :, np.newaxis])
    assert b_flip.shape == (2, 2, 1)


@settings(max_examples=15, deadline=None)
@given(
    h=st.integers(min_value=4, max_value=32),
    w=st.integers(min_value=4, max_value=32),
    target_w=st.integers(min_value=2, max_value=24),
    target_h=st.integers(min_value=2, max_value=24),
)
def test_resize_property_dtype_and_shape(h: int, w: int, target_w: int, target_h: int) -> None:
    arr: ImageArray = np.zeros((h, w, 3), dtype=np.uint8)
    out = ResizeOperation({"width": target_w, "height": target_h}).apply(arr)
    assert out.shape == (target_h, target_w, 3)
    assert out.dtype == arr.dtype
