from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.markup import (
    MarkupOperation,
    MarkupParams,
)


def test_markup_badge_render() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    params = MarkupParams(kind="badge", x1=10, y1=10, text="REVISION", color=(255, 0, 0))
    op = MarkupOperation(params)
    res = op.apply(img)
    assert np.any(res > 0)


def test_markup_arrow_render() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    params = MarkupParams(kind="arrow", x1=10, y1=10, x2=80, y2=80, color=(0, 255, 0))
    op = MarkupOperation(params)
    res = op.apply(img)
    assert np.any(res[:, :, 1] == 255)


def test_markup_box_with_fill() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    params = MarkupParams(
        kind="box",
        x1=20,
        y1=20,
        x2=80,
        y2=80,
        color=(0, 0, 255),
        fill_opacity=0.5,
    )
    op = MarkupOperation(params)
    res = op.apply(img)
    assert res[50, 50, 2] > 50


def test_markup_alpha_preservation() -> None:
    rgba = np.full((50, 50, 4), 100, dtype=np.uint8)
    rgba[:, :, 3] = 200
    params = MarkupParams(kind="arrow", x1=5, y1=5, x2=40, y2=40)
    op = MarkupOperation(params)
    res = op.apply(rgba)
    assert res.shape == (50, 50, 4)
    assert np.all(res[:, :, 3] == 200)


def test_markup_registered() -> None:
    op_cls = default_registry.get("markup")
    assert op_cls is MarkupOperation
