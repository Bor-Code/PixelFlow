from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.auto_enhance import AutoEnhanceOperation, AutoEnhanceParams


def test_auto_enhance_registered() -> None:
    op_cls = default_registry.get("auto_enhance")
    assert op_cls is AutoEnhanceOperation


def test_auto_enhance_rgb_full() -> None:
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 120
    img[:, :, 2] = 160
    params = AutoEnhanceParams(strength=1.0)
    op = AutoEnhanceOperation(params)
    out = op.apply(img)
    assert out.shape == (60, 60, 3)
    assert out.dtype == np.uint8


def test_auto_enhance_zero_strength() -> None:
    img = np.full((40, 40, 3), 110, dtype=np.uint8)
    params = AutoEnhanceParams(strength=0.0)
    op = AutoEnhanceOperation(params)
    out = op.apply(img)
    assert np.array_equal(out, img)


def test_auto_enhance_grayscale() -> None:
    gray = np.linspace(50, 200, 2500, dtype=np.uint8).reshape((50, 50))
    params = AutoEnhanceParams(strength=1.0, clip_limit=3.0)
    op = AutoEnhanceOperation(params)
    out = op.apply(gray)
    assert out.shape == (50, 50)
    assert out.dtype == np.uint8


def test_auto_enhance_rgba_preserves_alpha() -> None:
    img = np.full((50, 50, 4), 140, dtype=np.uint8)
    img[:, :, 3] = 210
    params = AutoEnhanceParams(strength=1.2, auto_white_balance=True, unsharp=True)
    op = AutoEnhanceOperation(params)
    out = op.apply(img)
    assert out.shape == (50, 50, 4)
    assert np.all(out[:, :, 3] == 210)


def test_auto_enhance_individual_toggles() -> None:
    img = np.full((30, 30, 3), 90, dtype=np.uint8)
    params = AutoEnhanceParams(
        strength=0.8, auto_white_balance=False, auto_contrast=False, unsharp=True
    )
    op = AutoEnhanceOperation(params)
    out = op.apply(img)
    assert out.shape == (30, 30, 3)
