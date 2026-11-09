from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.operations.perspective import (
    PerspectiveWarpOperation,
    PerspectiveWarpParams,
)
from pixelflow.operations.vibrance import (
    VibranceOperation,
    VibranceParams,
)


def _make_tilted_card() -> np.ndarray:
    img = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (140, 100), (40, 40, 40), -1)
    return img


def test_perspective_registry_and_warp() -> None:
    assert default_registry.get("perspective_warp") is PerspectiveWarpOperation
    src_img = _make_tilted_card()
    copy_img = src_img.copy()
    pts = ((20.0, 20.0), (140.0, 20.0), (140.0, 100.0), (20.0, 100.0))
    params = PerspectiveWarpParams(src_points=pts, output_width=100, output_height=80)
    op = PerspectiveWarpOperation(params)
    res = op.apply(src_img)
    assert np.array_equal(src_img, copy_img)
    assert res.shape == (80, 100, 3)
    assert res.dtype == src_img.dtype


def test_perspective_noop_and_rgba() -> None:
    src_rgba = cast(ImageArray, cv2.cvtColor(_make_tilted_card(), cv2.COLOR_BGR2BGRA))
    op_noop = PerspectiveWarpOperation()
    res_noop = op_noop.apply(src_rgba)
    assert np.array_equal(res_noop, src_rgba)

    pts = ((10.0, 10.0), (150.0, 15.0), (145.0, 95.0), (15.0, 90.0))
    op_auto = PerspectiveWarpOperation(PerspectiveWarpParams(src_points=pts))
    res = op_auto.apply(src_rgba)
    assert res.ndim == 3 and res.shape[2] == 4


def test_perspective_validation() -> None:
    with pytest.raises(ValidationError):
        PerspectiveWarpParams(output_width=5)
    with pytest.raises(ValidationError):
        PerspectiveWarpParams(output_height=10000)


def test_vibrance_registry_and_modes() -> None:
    assert default_registry.get("vibrance") is VibranceOperation
    for ch in (1, 3, 4):
        if ch == 1:
            img = np.full((40, 40), 100, dtype=np.uint8)
        elif ch == 3:
            img = np.full((40, 40, 3), (120, 80, 40), dtype=np.uint8)
        else:
            img = np.full((40, 40, 4), (120, 80, 40, 255), dtype=np.uint8)
        copy_img = img.copy()
        op = VibranceOperation(VibranceParams(vibrance=0.5, saturation=0.2))
        res = op.apply(img)
        assert np.array_equal(img, copy_img)
        assert res.shape == img.shape
        assert res.dtype == img.dtype


def test_vibrance_validation() -> None:
    with pytest.raises(ValidationError):
        VibranceParams(vibrance=2.0)
    with pytest.raises(ValidationError):
        VibranceParams(saturation=-2.0)


def test_perspective_auto_detect() -> None:
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (125, 125), 255, -1)
    op = PerspectiveWarpOperation(
        PerspectiveWarpParams(auto_detect=True, output_width=100, output_height=100)
    )
    res = op.apply(img)
    assert res.shape == (100, 100)

    empty = np.zeros((50, 50), dtype=np.uint8)
    op_fail = PerspectiveWarpOperation(PerspectiveWarpParams(auto_detect=True))
    assert np.array_equal(op_fail.apply(empty), empty)
