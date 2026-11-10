from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.glitch import GlitchOperation, GlitchParams


def test_glitch_registered() -> None:
    op_cls = default_registry.get("glitch")
    assert op_cls is GlitchOperation
    alias_cls = default_registry.get("rgb_split")
    assert alias_cls is GlitchOperation


def test_glitch_rgb_offset() -> None:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[15:25, 15:25, :] = 255
    params = GlitchParams(offset=5, scanlines=False)
    op = GlitchOperation(params=params)
    res = op.apply(img)

    assert res.shape == (40, 40, 3)
    assert res.dtype == np.uint8
    assert res[20, 26, 0] > 0


def test_glitch_scanlines_and_slices() -> None:
    img = np.full((60, 60, 3), 150, dtype=np.uint8)
    params = GlitchParams(
        offset=3,
        color_channel="red_green",
        scanlines=True,
        scanline_spacing=4,
        scanline_intensity=0.4,
        slice_glitch=True,
        slice_count=3,
        slice_displacement=10,
    )
    op = GlitchOperation(params=params)
    res = op.apply(img)

    assert res.shape == (60, 60, 3)
    assert res[0, 10, 0] < 150


def test_glitch_green_blue_channel() -> None:
    img = np.full((30, 30, 3), 120, dtype=np.uint8)
    op = GlitchOperation(params={"color_channel": "green_blue", "offset": 4})
    res = op.apply(img)
    assert res.shape == (30, 30, 3)


def test_glitch_rgba_preserves_alpha() -> None:
    img = np.full((30, 30, 4), 100, dtype=np.uint8)
    img[:, :, 3] = 180
    op = GlitchOperation(params={"offset": 2})
    res = op.apply(img)

    assert res.shape == (30, 30, 4)
    assert np.all(res[:, :, 3] == 180)


def test_glitch_grayscale() -> None:
    img = np.full((30, 30), 128, dtype=np.uint8)
    op = GlitchOperation(params={"offset": 2})
    res = op.apply(img)

    assert res.shape == (30, 30)
    assert res.dtype == np.uint8


def test_glitch_zero_offset() -> None:
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    op = GlitchOperation(params={"offset": 0, "scanlines": False, "slice_glitch": False})
    res = op.apply(img)
    assert np.array_equal(res, img)
