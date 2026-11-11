from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.frame import FrameOperation, FrameParams


def test_frame_registered() -> None:
    op_cls = default_registry.get("frame")
    assert op_cls is FrameOperation


def test_frame_solid_rgb() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    op = FrameOperation(params=FrameParams(style="solid", thickness=10, color=(255, 0, 0)))
    res = op.apply(img)
    assert res.shape == (120, 120, 3)
    assert tuple(res[0, 0]) == (255, 0, 0)


def test_frame_polaroid_rgb() -> None:
    img = np.ones((80, 80, 3), dtype=np.uint8) * 128
    op = FrameOperation(
        params=FrameParams(
            style="polaroid",
            thickness=20,
            bottom_extra=40,
            color=(250, 250, 250),
        )
    )
    res = op.apply(img)
    assert res.shape == (80 + 20 + 60, 80 + 20 + 20, 3)
    assert tuple(res[0, 0]) == (250, 250, 250)


def test_frame_passepartout() -> None:
    img = np.full((120, 120, 3), 100, dtype=np.uint8)
    op = FrameOperation(
        params=FrameParams(
            style="passepartout",
            thickness=30,
            inner_border=True,
            inner_color=(10, 10, 10),
        )
    )
    res = op.apply(img)
    assert res.shape == (180, 180, 3)


def test_frame_filmstrip() -> None:
    img = np.full((100, 150, 3), 50, dtype=np.uint8)
    op = FrameOperation(params=FrameParams(style="filmstrip", thickness=25))
    res = op.apply(img)
    assert res.shape == (150, 150, 3)


def test_frame_drop_shadow() -> None:
    img = np.full((90, 90, 3), 200, dtype=np.uint8)
    op = FrameOperation(params=FrameParams(style="drop_shadow", thickness=20))
    res = op.apply(img)
    assert res.shape == (130, 130, 3)


def test_frame_rgba() -> None:
    img = np.zeros((60, 60, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    op = FrameOperation(
        params=FrameParams(style="polaroid", thickness=15, bottom_extra=30)
    )
    res = op.apply(img)
    assert res.shape == (60 + 15 + 45, 60 + 30, 4)
    assert np.all(res[:, :, 3] == 255)
