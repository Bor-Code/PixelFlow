from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.inpaint import InpaintOperation, InpaintParams


def test_inpaint_registered() -> None:
    op_cls = default_registry.get("inpaint")
    assert op_cls is InpaintOperation


def test_inpaint_rgb_telea() -> None:
    img = np.full((80, 80, 3), 120, dtype=np.uint8)
    img[30:50, 30:50] = 0
    params = InpaintParams(x=30, y=30, width=20, height=20, method="telea", inpaint_radius=3)
    op = InpaintOperation(params)
    out = op.apply(img)
    assert out.shape == (80, 80, 3)
    assert np.mean(out[35:45, 35:45]) > 50


def test_inpaint_rgb_ns() -> None:
    img = np.full((80, 80, 3), 180, dtype=np.uint8)
    img[20:40, 20:40] = 0
    params = InpaintParams(
        x=20, y=20, width=20, height=20, method="ns", inpaint_radius=4, dilation=0
    )
    op = InpaintOperation(params)
    out = op.apply(img)
    assert out.shape == (80, 80, 3)
    assert np.mean(out[25:35, 25:35]) > 80


def test_inpaint_rgba_preserves_alpha() -> None:
    img = np.full((60, 60, 4), 200, dtype=np.uint8)
    img[:, :, 3] = 128
    img[20:30, 20:30, :3] = 10
    params = InpaintParams(x=20, y=20, width=10, height=10, inpaint_radius=2)
    op = InpaintOperation(params)
    out = op.apply(img)
    assert out.shape == (60, 60, 4)
    assert np.all(out[:, :, 3] == 128)
    assert np.mean(out[22:28, 22:28, :3]) > 100


def test_inpaint_grayscale() -> None:
    gray = np.full((50, 50), 150, dtype=np.uint8)
    gray[15:25, 15:25] = 0
    params = InpaintParams(x=15, y=15, width=10, height=10, inpaint_radius=2, dilation=1)
    op = InpaintOperation(params)
    out = op.apply(gray)
    assert out.shape == (50, 50)
    assert np.mean(out[17:23, 17:23]) > 70


def test_inpaint_out_of_bounds() -> None:
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    params = InpaintParams(x=100, y=100, width=50, height=50)
    op = InpaintOperation(params)
    out = op.apply(img)
    assert np.array_equal(out, img)


def test_inpaint_zero_area() -> None:
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    params = InpaintParams(x=0, y=0, width=1, height=1)
    op = InpaintOperation(params)
    out = op.apply(img)
    assert out.shape == img.shape
