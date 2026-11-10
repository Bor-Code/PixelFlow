from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.vignette import VignetteOperation, VignetteParams


def test_vignette_registered() -> None:
    op_cls = default_registry.get("vignette")
    assert op_cls is VignetteOperation


def test_vignette_darken_edges() -> None:
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    params = VignetteParams(amount=0.8, radius=0.5, softness=0.3)
    op = VignetteOperation(params=params)
    res = op.apply(img)

    assert res.shape == (100, 100, 3)
    assert res.dtype == np.uint8
    assert res[50, 50, 0] > res[2, 2, 0]


def test_vignette_lighten_edges() -> None:
    img = np.full((100, 100, 3), 80, dtype=np.uint8)
    params = VignetteParams(amount=-0.7, radius=0.6, softness=0.4)
    op = VignetteOperation(params=params)
    res = op.apply(img)

    assert res.shape == (100, 100, 3)
    assert res[2, 2, 0] > res[50, 50, 0]


def test_vignette_rgba_preserves_alpha() -> None:
    img = np.full((50, 50, 4), 160, dtype=np.uint8)
    img[:, :, 3] = 195
    op = VignetteOperation(params={"amount": 0.5})
    res = op.apply(img)

    assert res.shape == (50, 50, 4)
    assert np.all(res[:, :, 3] == 195)


def test_vignette_grayscale() -> None:
    img = np.full((60, 60), 180, dtype=np.uint8)
    op = VignetteOperation(params={"amount": 0.5})
    res = op.apply(img)

    assert res.shape == (60, 60)
    assert res.dtype == np.uint8
    assert res[30, 30] > res[0, 0]


def test_vignette_zero_amount() -> None:
    img = np.full((30, 30, 3), 120, dtype=np.uint8)
    op = VignetteOperation(params={"amount": 0.0})
    res = op.apply(img)
    assert np.array_equal(res, img)
