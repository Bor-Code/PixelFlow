from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.filters import (
    BilateralFilterOperation,
    GaussianBlurOperation,
    MedianBlurOperation,
)


def test_filter_operations_registered() -> None:
    names = default_registry.list_names()
    for expected in ("gaussian_blur", "median_blur", "bilateral_filter"):
        assert expected in names


def test_gaussian_blur_and_even_kernel_error() -> None:
    img = np.zeros((10, 10, 4), dtype=np.uint8)
    img[5, 5, :3] = 255
    out = GaussianBlurOperation({"kernel_size": 3, "sigma_x": 1.0}).apply(img)
    assert out.shape == (10, 10, 4) and out[5, 5, 0] < 255
    with pytest.raises(InvalidParametersError):
        GaussianBlurOperation({"kernel_size": 4})


def test_median_blur_variants() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[4, 4] = 255
    out = MedianBlurOperation({"kernel_size": 3}).apply(img)
    assert out[4, 4, 0] == 0
    k1 = MedianBlurOperation({"kernel_size": 1}).apply(img)
    assert np.array_equal(k1, img)
    u16 = np.full((12, 12), 10000, dtype=np.uint16)
    out_u16 = MedianBlurOperation({"kernel_size": 7}).apply(u16)
    assert out_u16.dtype == np.uint16
    with pytest.raises(InvalidParametersError):
        MedianBlurOperation({"kernel_size": 2})


def test_bilateral_filter_variants() -> None:
    img = np.full((10, 10, 3), 100, dtype=np.uint16)
    out = BilateralFilterOperation({"diameter": 5, "sigma_color": 40.0, "sigma_space": 40.0}).apply(
        img
    )
    assert out.shape == (10, 10, 3) and out.dtype == np.uint16


@settings(max_examples=12, deadline=None)
@given(k=st.sampled_from([1, 3, 5, 7]))
def test_filters_preserve_shape_and_dtype(k: int) -> None:
    arr: ImageArray = np.full((12, 12, 3), 80, dtype=np.uint8)
    g = GaussianBlurOperation({"kernel_size": k}).apply(arr)
    m = MedianBlurOperation({"kernel_size": k}).apply(arr)
    assert g.shape == arr.shape and g.dtype == arr.dtype
    assert m.shape == arr.shape and m.dtype == arr.dtype
