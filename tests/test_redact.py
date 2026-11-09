from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from pixelflow.core.registry import default_registry
from pixelflow.operations.redact import (
    RedactOperation,
    RedactParams,
    StampOperation,
    StampParams,
)


def _make_canvas(channels: int = 3, h: int = 120, w: int = 160) -> np.ndarray:
    if channels == 1:
        img = np.full((h, w), 200, dtype=np.uint8)
        cv2.putText(img, "TOP SECRET", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 20, 2)
        return img
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.putText(img, "TOP SECRET", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    if channels == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


def test_redact_registry_and_modes() -> None:
    assert default_registry.get("redact") is RedactOperation
    for mode in ("solid", "blur", "pixelate"):
        for ch in (1, 3, 4):
            img = _make_canvas(channels=ch)
            copy_img = img.copy()
            params = RedactParams(x=10, y=20, width=80, height=40, mode=mode)
            op = RedactOperation(params)
            res = op.apply(img)
            assert np.array_equal(img, copy_img)
            assert res.shape == img.shape
            assert res.dtype == img.dtype
            assert not np.array_equal(res, img)


def test_redact_bounds_and_noop() -> None:
    img = _make_canvas(channels=3)
    copy_img = img.copy()
    op_outside = RedactOperation(RedactParams(x=500, y=500, width=50, height=50))
    res = op_outside.apply(img)
    assert np.array_equal(res, copy_img)

    op_negative = RedactOperation(RedactParams(x=-50, y=-50, width=20, height=20))
    res_neg = op_negative.apply(img)
    assert np.array_equal(res_neg, copy_img)

    op_partial = RedactOperation(RedactParams(x=-10, y=-10, width=50, height=50))
    res_part = op_partial.apply(img)
    assert not np.array_equal(res_part, copy_img)


def test_redact_params_validation() -> None:
    with pytest.raises(ValidationError):
        RedactParams(width=0)
    with pytest.raises(ValidationError):
        RedactParams(height=-5)
    with pytest.raises(ValidationError):
        RedactParams(blur_ksize=2)
    with pytest.raises(ValidationError):
        RedactParams(pixel_size=1)


def test_stamp_registry_and_text_overlay() -> None:
    assert default_registry.get("stamp") is StampOperation
    for ch in (1, 3, 4):
        img = _make_canvas(channels=ch)
        copy_img = img.copy()
        params = StampParams(
            text="APPROVED",
            x=15,
            y=15,
            scale=0.8,
            angle=-10.0,
            opacity=0.9,
            box_border=True,
        )
        op = StampOperation(params)
        res = op.apply(img)
        assert np.array_equal(img, copy_img)
        assert res.shape == img.shape
        assert res.dtype == img.dtype
        assert not np.array_equal(res, img)


def test_stamp_file_and_bounds(tmp_path: Path) -> None:
    stamp_file = tmp_path / "stamp.png"
    icon = np.zeros((40, 40, 4), dtype=np.uint8)
    icon[:, :] = (0, 0, 255, 255)
    cv2.imwrite(str(stamp_file), icon)

    img = _make_canvas(channels=3)
    op = StampOperation(StampParams(stamp_path=str(stamp_file), x=10, y=10, angle=0.0))
    res = op.apply(img)
    assert res.shape == img.shape

    op_outside = StampOperation(StampParams(stamp_path=str(stamp_file), x=500, y=500))
    res_outside = op_outside.apply(img)
    assert np.array_equal(res_outside, img)


def test_stamp_params_validation() -> None:
    with pytest.raises(ValidationError):
        StampParams()
    with pytest.raises(ValidationError):
        StampParams(text="OK", scale=0.0)
    with pytest.raises(ValidationError):
        StampParams(text="OK", angle=200.0)
    with pytest.raises(ValidationError):
        StampParams(text="OK", opacity=1.5)
