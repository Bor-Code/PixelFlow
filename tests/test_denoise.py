from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.denoise import DenoiseOperation, DenoiseParams


def test_denoise_registered() -> None:
    op_cls = default_registry.get("denoise")
    assert op_cls is DenoiseOperation
    alias_cls = default_registry.get("nlmeans_denoise")
    assert alias_cls is DenoiseOperation


def test_denoise_rgb() -> None:
    rng = np.random.default_rng(42)
    img = (rng.normal(128, 20, (30, 30, 3))).clip(0, 255).astype(np.uint8)

    params = DenoiseParams(
        h=10.0,
        h_color=10.0,
        template_window_size=5,
        search_window_size=15,
    )
    op = DenoiseOperation(params=params)
    res = op.apply(img)

    assert res.shape == (30, 30, 3)
    assert res.dtype == np.uint8


def test_denoise_rgba_preserves_alpha() -> None:
    img = np.full((25, 25, 4), 100, dtype=np.uint8)
    img[:, :, 3] = 190

    op = DenoiseOperation(
        params={
            "h": 5.0,
            "template_window_size": 5,
            "search_window_size": 15,
        }
    )
    res = op.apply(img)

    assert res.shape == (25, 25, 4)
    assert np.all(res[:, :, 3] == 190)


def test_denoise_grayscale() -> None:
    img = np.full((25, 25), 140, dtype=np.uint8)
    op = DenoiseOperation(params={"h": 6.0, "template_window_size": 5, "search_window_size": 15})
    res = op.apply(img)

    assert res.shape == (25, 25)
    assert res.dtype == np.uint8
