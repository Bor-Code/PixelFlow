from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from pixelflow.core.registry import default_registry
from pixelflow.operations.curves import CurvesOperation, CurvesParams


def test_curves_registry() -> None:
    assert default_registry.get("curves") is CurvesOperation


def test_curves_all_channels() -> None:
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    img_copy = img.copy()
    params = CurvesParams(points=((0, 0), (100, 150), (255, 255)))
    op = CurvesOperation(params)
    out = op.apply(img)
    assert np.array_equal(img, img_copy)
    assert out.shape == (50, 50, 3)
    assert out[0, 0, 0] == 150
    assert out.dtype == np.uint8


def test_curves_single_channel() -> None:
    img = np.full((30, 30, 3), 100, dtype=np.uint8)
    params = CurvesParams(channel="r", points=((0, 0), (100, 200), (255, 255)))
    op = CurvesOperation(params)
    out = op.apply(img)
    assert out[0, 0, 0] == 200
    assert out[0, 0, 1] == 100
    assert out[0, 0, 2] == 100


def test_curves_grayscale_and_alpha() -> None:
    rgba = np.full((20, 20, 4), 120, dtype=np.uint8)
    rgba[:, :, 3] = 180
    params = CurvesParams(points=((0, 50), (255, 200)))
    op = CurvesOperation(params)
    out = op.apply(rgba)
    assert out.shape == (20, 20, 4)
    assert out[0, 0, 3] == 180

    gray = np.full((20, 20), 120, dtype=np.uint8)
    out_gray = op.apply(gray)
    assert out_gray.shape == (20, 20)


def test_curves_validation() -> None:
    with pytest.raises(ValidationError):
        CurvesParams(points=((0, 0),))
    with pytest.raises(ValidationError):
        CurvesParams(points=((0, -10), (255, 255)))
    with pytest.raises(ValidationError):
        CurvesParams(points=((0, 0), (300, 255)))
