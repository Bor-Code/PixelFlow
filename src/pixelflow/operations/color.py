from __future__ import annotations

from typing import cast

import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import (
    ensure_grayscale,
    merge_alpha,
    restore_dtype,
    split_alpha,
    to_uint8,
)
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class GrayscaleParams(OperationParams):
    keep_channels: bool = False


@register_operation("grayscale")
class GrayscaleOperation(ImageOperation[GrayscaleParams]):
    params_model = GrayscaleParams

    def _execute(self, image: ImageArray) -> ImageArray:
        _, alpha = split_alpha(image)
        gray = ensure_grayscale(image)
        if self.params.keep_channels:
            stacked = cast(ImageArray, np.stack([gray, gray, gray], axis=-1))
            return merge_alpha(stacked, alpha)
        return gray


class BrightnessContrastParams(OperationParams):
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.0, le=5.0)


@register_operation("brightness_contrast")
class BrightnessContrastOperation(ImageOperation[BrightnessContrastParams]):
    params_model = BrightnessContrastParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        max_val = 65535.0 if base.dtype == np.uint16 else 255.0
        midpoint = max_val / 2.0
        arr_f = base.astype(np.float32)
        adjusted = (arr_f - midpoint) * self.params.contrast + midpoint
        adjusted = adjusted + (self.params.brightness * max_val)
        clipped = cast(ImageArray, np.clip(adjusted, 0.0, max_val).round().astype(base.dtype))
        return merge_alpha(clipped, alpha)


class GammaParams(OperationParams):
    gamma: float = Field(default=1.0, gt=0.0, le=10.0)


@register_operation("gamma")
class GammaOperation(ImageOperation[GammaParams]):
    params_model = GammaParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        orig_dtype = base.dtype
        u8 = to_uint8(base)
        inv_gamma = 1.0 / self.params.gamma
        indices = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip((indices**inv_gamma) * 255.0, 0, 255).round().astype(np.uint8)
        corrected_u8 = lut[u8]
        restored = restore_dtype(corrected_u8, orig_dtype)
        return merge_alpha(restored, alpha)
