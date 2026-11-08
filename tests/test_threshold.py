from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.threshold import AdaptiveThresholdOperation, OtsuThresholdOperation


def test_threshold_operations_registered() -> None:
    names = default_registry.list_names()
    assert "adaptive_threshold" in names and "otsu_threshold" in names


def test_adaptive_threshold_variants() -> None:
    img = np.arange(256, dtype=np.uint8).reshape((16, 16))
    out = AdaptiveThresholdOperation({"method": "mean", "block_size": 5}).apply(img)
    assert out.shape == (16, 16)
    out_inv = AdaptiveThresholdOperation(
        {"method": "gaussian", "threshold_type": "binary_inv", "block_size": 5}
    ).apply(img)
    assert out_inv.shape == (16, 16)
    with pytest.raises(InvalidParametersError):
        AdaptiveThresholdOperation({"block_size": 4})


def test_otsu_threshold_variants() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint16)
    img[:, 8:] = 60000
    out = OtsuThresholdOperation({"threshold_type": "binary"}).apply(img)
    assert out.shape == (16, 16) and out.dtype == np.uint16
    assert out[0, 0] == 0 and out[0, 15] == 65535
    inv = OtsuThresholdOperation({"threshold_type": "binary_inv"}).apply(img)
    assert inv[0, 0] == 65535 and inv[0, 15] == 0


@settings(max_examples=12, deadline=None)
@given(block=st.sampled_from([3, 5, 7, 9]))
def test_threshold_ops_preserve_2d_and_dtype(block: int) -> None:
    arr: ImageArray = np.full((12, 12, 3), 120, dtype=np.uint8)
    ad = AdaptiveThresholdOperation({"block_size": block}).apply(arr)
    ot = OtsuThresholdOperation().apply(arr)
    assert ad.shape == (12, 12) and ad.dtype == arr.dtype
    assert ot.shape == (12, 12) and ot.dtype == arr.dtype
