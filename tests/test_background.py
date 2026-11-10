from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from pixelflow.core.registry import default_registry
from pixelflow.operations.background import BackgroundRemovalOperation, BackgroundRemovalParams


def test_background_removal_registry() -> None:
    assert default_registry.get("background_removal") is BackgroundRemovalOperation


def test_background_removal_grabcut() -> None:
    img = np.full((80, 80, 3), 240, dtype=np.uint8)
    img[20:60, 20:60] = [20, 180, 20]
    img_copy = img.copy()
    params = BackgroundRemovalParams(method="grabcut", iterations=1, margin=5)
    op = BackgroundRemovalOperation(params)
    out = op.apply(img)
    assert np.array_equal(img, img_copy)
    assert out.shape == (80, 80, 4)
    assert out.dtype == np.uint8


def test_background_removal_threshold() -> None:
    img = np.full((60, 60, 3), 250, dtype=np.uint8)
    img[20:40, 20:40] = [10, 10, 10]
    op = BackgroundRemovalOperation(BackgroundRemovalParams(method="threshold"))
    out = op.apply(img)
    assert out.shape == (60, 60, 4)
    assert out[30, 30, 3] == 255


def test_background_removal_grayscale_input() -> None:
    gray = np.full((50, 50), 240, dtype=np.uint8)
    gray[15:35, 15:35] = 20
    op = BackgroundRemovalOperation(BackgroundRemovalParams(method="threshold"))
    out = op.apply(gray)
    assert out.shape == (50, 50, 4)


def test_background_removal_validation() -> None:
    with pytest.raises(ValidationError):
        BackgroundRemovalParams(iterations=0)
    with pytest.raises(ValidationError):
        BackgroundRemovalParams(margin=200)
