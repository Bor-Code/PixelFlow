from __future__ import annotations

import numpy as np

from pixelflow.operations.color_splash import ColorSplashOperation, ColorSplashParams


def test_color_splash_red_preset() -> None:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:20, :, :] = [255, 0, 0]
    img[20:, :, :] = [0, 255, 0]

    op = ColorSplashOperation(params={"preset_color": "red", "feather": 0})
    res = op.apply(img)

    assert res.shape == (40, 40, 3)
    assert res.dtype == np.uint8
    assert res[5, 5, 0] > 200
    assert abs(int(res[25, 25, 0]) - int(res[25, 25, 1])) < 10


def test_color_splash_custom_hue() -> None:
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    img[:, :, :] = [0, 0, 255]

    params = ColorSplashParams(
        preset_color="custom",
        hue=120.0,
        hue_tolerance=20.0,
        feather=0,
    )
    op = ColorSplashOperation(params=params)
    res = op.apply(img)

    assert res.shape == (30, 30, 3)
    assert res[10, 10, 2] > 200


def test_color_splash_rgba_preserves_alpha() -> None:
    img = np.zeros((30, 30, 4), dtype=np.uint8)
    img[:, :, 0] = 255
    img[:, :, 3] = 175

    op = ColorSplashOperation(params={"preset_color": "red", "feather": 3})
    res = op.apply(img)

    assert res.shape == (30, 30, 4)
    assert np.all(res[:, :, 3] == 175)


def test_color_splash_grayscale() -> None:
    img = np.full((30, 30), 120, dtype=np.uint8)
    op = ColorSplashOperation(params={"preset_color": "blue"})
    res = op.apply(img)

    assert res.shape == (30, 30)
    assert np.array_equal(res, img)


def test_color_splash_desaturate_level() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, :, 1] = 200
    op = ColorSplashOperation(
        params={
            "preset_color": "blue",
            "desaturate_level": 0.5,
            "feather": 0,
        }
    )
    res = op.apply(img)
    assert res.shape == (20, 20, 3)
