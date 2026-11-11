from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.blend import BlendOperation, BlendParams


def test_blend_multiply() -> None:
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    op = BlendOperation(BlendParams(mode="multiply", color=(255, 255, 255), opacity=1.0))
    res = op.apply(img)
    assert np.allclose(res, 200, atol=1)

    op_black = BlendOperation(BlendParams(mode="multiply", color=(0, 0, 0), opacity=1.0))
    res_black = op_black.apply(img)
    assert np.all(res_black == 0)


def test_blend_screen() -> None:
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    op = BlendOperation(BlendParams(mode="screen", color=(0, 0, 0), opacity=1.0))
    res = op.apply(img)
    assert np.allclose(res, 100, atol=1)

    op_white = BlendOperation(BlendParams(mode="screen", color=(255, 255, 255), opacity=1.0))
    res_white = op_white.apply(img)
    assert np.all(res_white == 255)


def test_blend_overlay() -> None:
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    op = BlendOperation(BlendParams(mode="overlay", color=(100, 100, 100), opacity=1.0))
    res = op.apply(img)
    assert res.shape == (10, 10, 3)

    img_bright = np.full((10, 10, 3), 200, dtype=np.uint8)
    res_bright = op.apply(img_bright)
    assert res_bright.shape == (10, 10, 3)


def test_blend_soft_light() -> None:
    img = np.full((10, 10, 3), 150, dtype=np.uint8)
    op_dark = BlendOperation(BlendParams(mode="soft_light", color=(50, 50, 50), opacity=1.0))
    res_dark = op_dark.apply(img)
    assert res_dark[0, 0, 0] < 150

    op_bright = BlendOperation(BlendParams(mode="soft_light", color=(220, 220, 220), opacity=1.0))
    res_bright = op_bright.apply(img)
    assert res_bright[0, 0, 0] > 150


def test_blend_darken_lighten_difference() -> None:
    img = np.full((10, 10, 3), 120, dtype=np.uint8)
    darken = BlendOperation(BlendParams(mode="darken", color=(80, 80, 80))).apply(img)
    assert np.all(darken == 80)

    lighten = BlendOperation(BlendParams(mode="lighten", color=(180, 180, 180))).apply(img)
    assert np.all(lighten == 180)

    diff = BlendOperation(BlendParams(mode="difference", color=(100, 100, 100))).apply(img)
    assert np.all(diff == 20)


def test_blend_opacity_and_alpha() -> None:
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    params_zero = BlendParams(mode="normal", color=(255, 255, 255), opacity=0.0)
    res_zero = BlendOperation(params_zero).apply(img)
    assert np.all(res_zero == 100)

    params_half = BlendParams(mode="normal", color=(200, 200, 200), opacity=0.5)
    res_half = BlendOperation(params_half).apply(img)
    assert np.allclose(res_half, 150, atol=2)

    rgba = np.full((10, 10, 4), 100, dtype=np.uint8)
    rgba[:, :, 3] = 180
    res_rgba = BlendOperation(BlendParams(mode="normal", color=(0, 0, 0), opacity=1.0)).apply(rgba)
    assert res_rgba.shape == (10, 10, 4)
    assert np.all(res_rgba[:, :, 3] == 180)


def test_blend_registry() -> None:
    op = default_registry.create("blend", {"mode": "multiply", "opacity": 0.8})
    assert isinstance(op, BlendOperation)
