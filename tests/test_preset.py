from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.preset import PresetOperation, PresetParams


def test_preset_registered() -> None:
    assert default_registry.get("preset") is PresetOperation
    assert default_registry.get("cinematic_preset") is PresetOperation


def test_preset_teal_orange() -> None:
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    params = PresetParams(preset="teal_orange", intensity=1.0)
    op = PresetOperation(params)
    out = op.apply(img)
    assert out.shape == (40, 40, 3)
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_preset_cyberpunk() -> None:
    img = np.full((30, 30, 3), 100, dtype=np.uint8)
    params = PresetParams(preset="cyberpunk", intensity=0.9)
    op = PresetOperation(params)
    out = op.apply(img)
    assert out.shape == (30, 30, 3)
    assert out[15, 15, 0] > out[15, 15, 1]


def test_preset_vintage_film() -> None:
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    params = PresetParams(preset="vintage_film", intensity=1.0)
    op = PresetOperation(params)
    out = op.apply(img)
    assert out.shape == (30, 30, 3)
    assert out[0, 0, 0] >= 20


def test_preset_warm_sunset_and_cool_breeze() -> None:
    img = np.full((20, 20, 3), 120, dtype=np.uint8)
    out_warm = PresetOperation(PresetParams(preset="warm_sunset")).apply(img)
    out_cool = PresetOperation(PresetParams(preset="cool_breeze")).apply(img)
    assert out_warm.shape == (20, 20, 3)
    assert out_cool.shape == (20, 20, 3)


def test_preset_moody_noir() -> None:
    img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    params = PresetParams(preset="moody_noir", intensity=1.0)
    op = PresetOperation(params)
    out = op.apply(img)
    assert out.shape == (40, 40, 3)
    assert np.all(out[:, :, 0] == out[:, :, 1])


def test_preset_rgba_and_zero_intensity() -> None:
    img = np.full((30, 30, 4), 140, dtype=np.uint8)
    img[:, :, 3] = 222
    out_zero = PresetOperation(PresetParams(intensity=0.0)).apply(img)
    assert np.array_equal(out_zero, img)
    out_rgba = PresetOperation(PresetParams(intensity=1.0)).apply(img)
    assert out_rgba.shape == (30, 30, 4)
    assert np.all(out_rgba[:, :, 3] == 222)


def test_preset_grayscale() -> None:
    gray = np.full((30, 30), 100, dtype=np.uint8)
    out = PresetOperation(PresetParams(preset="teal_orange")).apply(gray)
    assert out.shape == (30, 30)
    assert out.dtype == np.uint8
