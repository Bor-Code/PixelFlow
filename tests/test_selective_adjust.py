from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.selective_adjust import (
    SelectiveAdjustOperation,
    SelectiveAdjustParams,
)


def test_selective_adjust_radial_brightness() -> None:
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    params = SelectiveAdjustParams(
        shape="radial",
        center=(0.5, 0.5),
        size=(0.2, 0.2),
        exposure=1.0,
        contrast=1.0,
    )
    op = SelectiveAdjustOperation(params)
    res = op.apply(img)
    assert res[25, 25, 0] > 150
    assert res[0, 0, 0] == 100


def test_selective_adjust_linear_gradient() -> None:
    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    params = SelectiveAdjustParams(
        shape="linear",
        center=(0.5, 0.5),
        angle=0.0,
        exposure=0.5,
    )
    op = SelectiveAdjustOperation(params)
    res = op.apply(img)
    assert res.shape == (40, 40, 3)
    assert res[5, 20, 0] >= res[35, 20, 0]


def test_selective_adjust_box_mask() -> None:
    img = np.full((30, 30, 3), 80, dtype=np.uint8)
    params = SelectiveAdjustParams(
        shape="box",
        center=(0.5, 0.5),
        size=(0.4, 0.4),
        feather=0.1,
        exposure=1.0,
    )
    op = SelectiveAdjustOperation(params)
    res = op.apply(img)
    assert res[15, 15, 0] > 120
    assert res[0, 0, 0] == 80


def test_selective_adjust_invert_and_warmth() -> None:
    img = np.full((20, 20, 3), 120, dtype=np.uint8)
    params = SelectiveAdjustParams(
        shape="radial",
        center=(0.5, 0.5),
        size=(0.3, 0.3),
        invert=True,
        warmth=0.8,
        saturation=1.5,
    )
    op = SelectiveAdjustOperation(params)
    res = op.apply(img)
    assert res[0, 0, 0] > res[0, 0, 2]


def test_selective_adjust_blur_radius() -> None:
    img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    params = SelectiveAdjustParams(
        shape="radial",
        center=(0.5, 0.5),
        size=(0.4, 0.4),
        blur_radius=5,
    )
    op = SelectiveAdjustOperation(params)
    res = op.apply(img)
    assert res.shape == (30, 30, 3)


def test_selective_adjust_alpha_preservation() -> None:
    rgba = np.full((20, 20, 4), 100, dtype=np.uint8)
    rgba[:, :, 3] = 220
    params = SelectiveAdjustParams(exposure=0.5)
    op = SelectiveAdjustOperation(params)
    res = op.apply(rgba)
    assert res.shape == (20, 20, 4)
    assert np.all(res[:, :, 3] == 220)


def test_selective_adjust_registered() -> None:
    op_cls = default_registry.get("selective_adjust")
    assert op_cls is SelectiveAdjustOperation
