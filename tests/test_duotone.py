from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.duotone import DuotoneOperation, DuotoneParams


def test_duotone_registered() -> None:
    op_cls = default_registry.get("duotone")
    assert op_cls is DuotoneOperation
    alias_cls = default_registry.get("gradient_map")
    assert alias_cls is DuotoneOperation


def test_duotone_spotify_preset() -> None:
    img = np.full((30, 30, 3), 128, dtype=np.uint8)
    op = DuotoneOperation(params={"preset": "spotify", "intensity": 1.0})
    res = op.apply(img)
    assert res.shape == (30, 30, 3)
    assert res.dtype == np.uint8


def test_duotone_custom_colors() -> None:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:20, :, :] = 50
    img[20:, :, :] = 200

    params = DuotoneParams(
        preset="custom",
        shadow_color="#102030",
        highlight_color="#f0a020",
        contrast=1.2,
    )
    op = DuotoneOperation(params=params)
    res = op.apply(img)
    assert res.shape == (40, 40, 3)
    assert res[5, 5, 0] < res[25, 25, 0]


def test_duotone_rgba_preserves_alpha() -> None:
    img = np.full((25, 25, 4), 140, dtype=np.uint8)
    img[:, :, 3] = 160
    op = DuotoneOperation(params={"preset": "cyberpunk"})
    res = op.apply(img)
    assert res.shape == (25, 25, 4)
    assert np.all(res[:, :, 3] == 160)


def test_duotone_grayscale() -> None:
    img = np.full((30, 30), 100, dtype=np.uint8)
    op = DuotoneOperation(params={"preset": "amber"})
    res = op.apply(img)
    assert res.shape == (30, 30)
    assert res.dtype == np.uint8


def test_duotone_intensity_zero() -> None:
    img = np.full((20, 20, 3), 90, dtype=np.uint8)
    op = DuotoneOperation(params={"intensity": 0.0})
    res = op.apply(img)
    assert np.array_equal(res, img)


def test_duotone_short_hex() -> None:
    params = DuotoneParams(preset="custom", shadow_color="#fff", highlight_color="invalid")
    op = DuotoneOperation(params=params)
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    res = op.apply(img)
    assert res.shape == (10, 10, 3)
