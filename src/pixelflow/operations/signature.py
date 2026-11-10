from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class SignatureExtractParams(OperationParams):
    threshold: int = Field(default=200, ge=0, le=255)
    smooth_radius: int = Field(default=1, ge=0, le=10)
    ink_color: tuple[int, int, int] | None = Field(default=None)
    high_contrast: bool = Field(default=True)


def _compute_alpha_mask(gray: np.ndarray, threshold: int, smooth_radius: int) -> np.ndarray:
    diff = np.maximum(0.0, float(threshold) - gray.astype(np.float32))
    scale = 255.0 / max(1.0, float(threshold))
    alpha_calc = np.clip(diff * scale * 1.5, 0.0, 255.0).astype(np.uint8)
    if smooth_radius > 0:
        ksize = smooth_radius * 2 + 1
        blurred = cv2.GaussianBlur(alpha_calc, (ksize, ksize), 0)
        return cast(np.ndarray, blurred).astype(np.uint8)
    return cast(np.ndarray, alpha_calc)


def _build_rgba(
    rgb: np.ndarray,
    alpha: np.ndarray,
    ink_color: tuple[int, int, int] | None,
    high_contrast: bool,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    if ink_color is not None:
        out[:, :, 0] = ink_color[0]
        out[:, :, 1] = ink_color[1]
        out[:, :, 2] = ink_color[2]
    else:
        color = rgb.copy()
        if high_contrast:
            color = np.clip((color.astype(np.float32) - 40.0) * 1.25, 0.0, 255.0).astype(np.uint8)
        out[:, :, :3] = color
    out[:, :, 3] = alpha
    return out


@register_operation("signature_extract")
class SignatureExtractOperation(ImageOperation[SignatureExtractParams]):
    params_model = SignatureExtractParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if u8.ndim == 2:
            gray = u8
            rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
        else:
            rgb = ensure_rgb(u8[:, :, :3])
            gray = cast(np.ndarray, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        alpha = _compute_alpha_mask(gray, self.params.threshold, self.params.smooth_radius)
        rgba = _build_rgba(rgb, alpha, self.params.ink_color, self.params.high_contrast)
        return cast(ImageArray, rgba)
