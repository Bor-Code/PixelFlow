from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class BackgroundRemovalParams(OperationParams):
    method: str = Field(default="grabcut")
    iterations: int = Field(default=5, ge=1, le=10)
    margin: int = Field(default=10, ge=0, le=100)


def _segment_grabcut(rgb: np.ndarray, iterations: int, margin: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = min(margin, max(0, h // 4), max(0, w // 4))
    rect = (m, m, max(1, w - 2 * m), max(1, h - 2 * m))
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(rgb, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    is_fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    return np.where(is_fg, 255, 0).astype(np.uint8)


def _segment_threshold(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cast(np.ndarray, binary).astype(np.uint8)


def _compose_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = rgb
    out[:, :, 3] = alpha
    return out


@register_operation("background_removal")
class BackgroundRemovalOperation(ImageOperation[BackgroundRemovalParams]):
    params_model = BackgroundRemovalParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB) if u8.ndim == 2 else ensure_rgb(u8[:, :, :3])
        if self.params.method == "threshold":
            alpha = _segment_threshold(rgb)
        else:
            alpha = _segment_grabcut(rgb, self.params.iterations, self.params.margin)
        rgba = _compose_rgba(rgb, alpha)
        return cast(ImageArray, rgba)
