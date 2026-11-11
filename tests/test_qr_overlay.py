from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.qr_overlay import QROverlayOperation, QROverlayParams


def test_qr_overlay_registered() -> None:
    op_cls = default_registry.get("qr_overlay")
    assert op_cls is QROverlayOperation
    alias_cls = default_registry.get("qr_code")
    assert alias_cls is QROverlayOperation


def test_qr_overlay_rgb_default() -> None:
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    params = QROverlayParams(data="https://example.com", size=100, margin=10)
    op = QROverlayOperation(params=params)
    res = op.apply(img)

    assert res.shape == (300, 300, 3)
    assert res.dtype == np.uint8
    assert not np.array_equal(res[190:290, 190:290], img[190:290, 190:290])


def test_qr_overlay_custom_position() -> None:
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    params = QROverlayParams(
        data="CustomPos",
        position="custom",
        x=20,
        y=30,
        size=60,
        fg_color="#0000ff",
        bg_color="#ffff00",
    )
    op = QROverlayOperation(params=params)
    res = op.apply(img)
    assert res.shape == (200, 200, 3)


def test_qr_overlay_all_positions() -> None:
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    positions = ["top-left", "top-right", "bottom-left", "center"]
    for pos in positions:
        op = QROverlayOperation(params={"data": "test", "position": pos, "size": 50, "margin": 5})
        res = op.apply(img)
        assert res.shape == (200, 200, 3)


def test_qr_overlay_rgba_preserves_alpha() -> None:
    img = np.full((150, 150, 4), 180, dtype=np.uint8)
    img[:, :, 3] = 205
    op = QROverlayOperation(params={"size": 60})
    res = op.apply(img)

    assert res.shape == (150, 150, 4)
    assert np.all(res[:, :, 3] == 205)


def test_qr_overlay_grayscale() -> None:
    img = np.full((150, 150), 128, dtype=np.uint8)
    op = QROverlayOperation(params={"size": 50})
    res = op.apply(img)
    assert res.shape == (150, 150)
    assert res.dtype == np.uint8


def test_qr_overlay_zero_opacity() -> None:
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    op = QROverlayOperation(params={"opacity": 0.0})
    res = op.apply(img)
    assert np.array_equal(res, img)


def test_qr_overlay_out_of_bounds() -> None:
    img = np.full((50, 50, 3), 120, dtype=np.uint8)
    op = QROverlayOperation(params={"position": "custom", "x": 60, "y": 60, "size": 50})
    res = op.apply(img)
    assert np.array_equal(res, img)
