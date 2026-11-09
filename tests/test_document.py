from __future__ import annotations

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from pixelflow.core.registry import default_registry
from pixelflow.operations.document import (
    DeskewOperation,
    DeskewParams,
    ShadowRemovalOperation,
    ShadowRemovalParams,
)


def _make_shadowed_document(channels: int = 3) -> np.ndarray:
    h, w = 120, 160
    base = np.full((h, w), 220, dtype=np.uint8)
    cv2.putText(base, "TEST DOC", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 10, 2)
    shadow = np.linspace(50, 220, w, dtype=np.uint8).reshape(1, w)
    shadow = np.repeat(shadow, h, axis=0)
    shadowed = np.clip((base.astype(np.int16) * shadow.astype(np.int16)) // 255, 0, 255).astype(
        np.uint8
    )
    if channels == 1:
        return shadowed
    if channels == 3:
        return cv2.cvtColor(shadowed, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(shadowed, cv2.COLOR_GRAY2BGRA)


def _make_skewed_document(angle: float = 12.0) -> np.ndarray:
    h, w = 200, 200
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 70), (160, 130), (0, 0, 0), -1)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    skewed = cv2.warpAffine(canvas, matrix, (w, h), borderValue=(255, 255, 255))
    return skewed


def test_shadow_removal_registry_and_channels() -> None:
    op_cls = default_registry.get("shadow_removal")
    assert op_cls is ShadowRemovalOperation

    for ch in (1, 3, 4):
        img = _make_shadowed_document(channels=ch)
        copy_img = img.copy()
        op = ShadowRemovalOperation(ShadowRemovalParams(kernel_size=15, blur_size=11))
        res = op.apply(img)
        assert np.array_equal(img, copy_img)
        assert res.shape == img.shape
        assert res.dtype == img.dtype


def test_shadow_removal_params_validation() -> None:
    with pytest.raises(ValidationError):
        ShadowRemovalParams(kernel_size=20)
    with pytest.raises(ValidationError):
        ShadowRemovalParams(blur_size=14)
    with pytest.raises(ValidationError):
        ShadowRemovalParams(kernel_size=1)


def test_deskew_registry_and_correction() -> None:
    op_cls = default_registry.get("deskew")
    assert op_cls is DeskewOperation

    skewed = _make_skewed_document(angle=15.0)
    copy_img = skewed.copy()
    op = DeskewOperation(DeskewParams(max_angle=45.0, fill_value=255))
    corrected = op.apply(skewed)
    assert np.array_equal(skewed, copy_img)
    assert corrected.ndim == skewed.ndim
    assert corrected.dtype == skewed.dtype

    straight = np.full((100, 100, 3), 255, dtype=np.uint8)
    op_straight = DeskewOperation()
    res_straight = op_straight.apply(straight)
    assert np.array_equal(straight, res_straight)


def test_deskew_params_validation() -> None:
    with pytest.raises(ValidationError):
        DeskewParams(max_angle=0.0)
    with pytest.raises(ValidationError):
        DeskewParams(max_angle=95.0)
    with pytest.raises(ValidationError):
        DeskewParams(fill_value=300)
