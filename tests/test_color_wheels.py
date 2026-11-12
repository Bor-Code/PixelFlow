from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.color_wheels import ColorWheelsOperation, ColorWheelsParams


def test_color_wheels_identity() -> None:
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    op = ColorWheelsOperation(ColorWheelsParams())
    result = op.apply(img)
    assert np.allclose(result, 128, atol=2)


def test_color_wheels_lift_brightens_shadows() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    params = ColorWheelsParams(lift=(0.2, 0.1, 0.0))
    op = ColorWheelsOperation(params)
    result = op.apply(img)
    assert result[0, 0, 0] > 40
    assert result[0, 0, 1] > 20
    assert result[0, 0, 2] == 0


def test_color_wheels_gain_affects_highlights() -> None:
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    params = ColorWheelsParams(gain=(1.2, 0.8, 1.0))
    op = ColorWheelsOperation(params)
    result = op.apply(img)
    assert result[0, 0, 0] > 200
    assert result[0, 0, 1] < 200


def test_color_wheels_gamma_shifts_midtones() -> None:
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    params = ColorWheelsParams(gamma=(0.5, 2.0, 1.0))
    op = ColorWheelsOperation(params)
    result = op.apply(img)
    assert result[0, 0, 0] < 128
    assert result[0, 0, 1] > 128


def test_color_wheels_master_controls() -> None:
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    params = ColorWheelsParams(lift_master=0.1, gain_master=1.1, gamma_master=0.9)
    op = ColorWheelsOperation(params)
    result = op.apply(img)
    assert result[0, 0, 0] > 100


def test_color_wheels_alpha_preserved() -> None:
    rgba = np.full((10, 10, 4), 128, dtype=np.uint8)
    rgba[:, :, 3] = 200
    params = ColorWheelsParams(lift=(0.1, 0.1, 0.1))
    op = ColorWheelsOperation(params)
    result = op.apply(rgba)
    assert result.shape == (10, 10, 4)
    assert np.all(result[:, :, 3] == 200)


def test_color_wheels_registered_in_registry() -> None:
    op_cls = default_registry.get("color_wheels")
    assert op_cls is ColorWheelsOperation
