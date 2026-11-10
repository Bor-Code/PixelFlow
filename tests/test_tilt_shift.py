from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.tilt_shift import TiltShiftOperation, TiltShiftParams


def test_tilt_shift_registered() -> None:
    op_cls = default_registry.get("tilt_shift")
    assert op_cls is TiltShiftOperation
    alias_cls = default_registry.get("miniature")
    assert alias_cls is TiltShiftOperation


def test_tilt_shift_horizontal() -> None:
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    img[45:55, 45:55] = 250
    op = TiltShiftOperation(
        params={
            "orientation": "horizontal",
            "focus_position": 0.5,
            "focus_width": 0.2,
            "blur_amount": 7,
            "saturation_boost": 1.0,
        }
    )
    res = op.apply(img)
    assert res.shape == (100, 100, 3)
    assert res.dtype == np.uint8


def test_tilt_shift_vertical() -> None:
    img = np.full((80, 80, 3), 100, dtype=np.uint8)
    params = TiltShiftParams(
        orientation="vertical",
        focus_position=0.3,
        focus_width=0.2,
        transition_width=0.1,
        blur_amount=5,
        saturation_boost=1.3,
    )
    op = TiltShiftOperation(params=params)
    res = op.apply(img)
    assert res.shape == (80, 80, 3)


def test_tilt_shift_rgba_preserves_alpha() -> None:
    img = np.full((50, 50, 4), 150, dtype=np.uint8)
    img[:, :, 3] = 210
    op = TiltShiftOperation(params={"blur_amount": 5})
    res = op.apply(img)
    assert res.shape == (50, 50, 4)
    assert np.all(res[:, :, 3] == 210)


def test_tilt_shift_grayscale() -> None:
    img = np.full((60, 60), 128, dtype=np.uint8)
    op = TiltShiftOperation(params={"orientation": "horizontal", "blur_amount": 5})
    res = op.apply(img)
    assert res.shape == (60, 60)
    assert res.dtype == np.uint8
