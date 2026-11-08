from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.edges import CannyOperation, SobelOperation


def test_edge_operations_registered() -> None:
    names = default_registry.list_names()
    assert "canny" in names and "sobel" in names


def test_canny_edges_and_validation() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, 8:] = 255
    edges = CannyOperation({"threshold1": 50.0, "threshold2": 150.0}).apply(img)
    assert edges.shape == (16, 16) and np.any(edges > 0)
    with pytest.raises(InvalidParametersError):
        CannyOperation({"aperture_size": 4})
    with pytest.raises(InvalidParametersError):
        CannyOperation({"threshold1": 200.0, "threshold2": 100.0})


def test_sobel_edges_and_validation() -> None:
    img = np.zeros((16, 16, 4), dtype=np.uint16)
    img[:, 8:] = 65535
    sobel = SobelOperation({"dx": 1, "dy": 0, "kernel_size": 3}).apply(img)
    assert sobel.shape == (16, 16) and sobel.dtype == np.uint16 and np.any(sobel > 0)
    with pytest.raises(InvalidParametersError):
        SobelOperation({"kernel_size": 4})
    with pytest.raises(InvalidParametersError):
        SobelOperation({"dx": 0, "dy": 0})


@settings(max_examples=12, deadline=None)
@given(t1=st.floats(min_value=10.0, max_value=100.0))
def test_edges_preserve_2d_and_dtype(t1: float) -> None:
    arr: ImageArray = np.zeros((12, 14, 3), dtype=np.uint8)
    arr[4:8, 4:8] = 200
    c = CannyOperation({"threshold1": t1, "threshold2": t1 + 50.0}).apply(arr)
    s = SobelOperation({"dx": 1, "dy": 1}).apply(arr)
    assert c.shape == (12, 14) and c.dtype == arr.dtype
    assert s.shape == (12, 14) and s.dtype == arr.dtype
