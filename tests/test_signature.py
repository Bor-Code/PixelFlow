from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from pixelflow.core.registry import default_registry
from pixelflow.operations.signature import SignatureExtractOperation, SignatureExtractParams


def test_signature_registry() -> None:
    assert default_registry.get("signature_extract") is SignatureExtractOperation


def test_signature_extract_defaults() -> None:
    canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
    canvas[40:60, 40:60] = [20, 20, 30]
    canvas_copy = canvas.copy()
    op = SignatureExtractOperation(SignatureExtractParams())
    out = op.apply(canvas)
    assert np.array_equal(canvas, canvas_copy)
    assert out.shape == (100, 100, 4)
    assert out[10, 10, 3] == 0
    assert out[50, 50, 3] > 180


def test_signature_extract_grayscale_input() -> None:
    gray = np.full((80, 80), 250, dtype=np.uint8)
    gray[30:50, 30:50] = 10
    op = SignatureExtractOperation(SignatureExtractParams(smooth_radius=0))
    out = op.apply(gray)
    assert out.shape == (80, 80, 4)
    assert out[0, 0, 3] == 0
    assert out[40, 40, 3] == 255


def test_signature_extract_custom_ink_color() -> None:
    canvas = np.full((60, 60, 3), 240, dtype=np.uint8)
    canvas[20:40, 20:40] = 0
    params = SignatureExtractParams(ink_color=(0, 0, 200), threshold=220)
    op = SignatureExtractOperation(params)
    out = op.apply(canvas)
    assert out.shape == (60, 60, 4)
    assert tuple(out[30, 30, :3]) == (0, 0, 200)
    assert out[30, 30, 3] > 0


def test_signature_extract_without_high_contrast() -> None:
    canvas = np.full((40, 40, 3), 230, dtype=np.uint8)
    canvas[15:25, 15:25] = [50, 60, 70]
    op = SignatureExtractOperation(SignatureExtractParams(high_contrast=False))
    out = op.apply(canvas)
    assert out.shape == (40, 40, 4)


def test_signature_extract_invalid_params() -> None:
    with pytest.raises(ValidationError):
        SignatureExtractParams(threshold=300)
    with pytest.raises(ValidationError):
        SignatureExtractParams(smooth_radius=-1)
