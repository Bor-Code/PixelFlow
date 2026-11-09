from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class VibranceParams(OperationParams):
    vibrance: float = Field(default=0.2, ge=-1.0, le=1.0)
    saturation: float = Field(default=0.0, ge=-1.0, le=1.0)


def _adjust_vibrance(rgb: np.ndarray, vibrance: float, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    if abs(saturation) > 0.001:
        s = s * (1.0 + saturation)
    if abs(vibrance) > 0.001:
        mask = (1.0 - s) * vibrance
        s = s + mask * s
    hsv[:, :, 1] = np.clip(s * 255.0, 0.0, 255.0)
    res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return cast(np.ndarray, res)


@register_operation("vibrance")
class VibranceOperation(ImageOperation[VibranceParams]):
    params_model = VibranceParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        if base.ndim == 2:
            return image.copy()
        rgb_u8 = to_uint8(ensure_rgb(base))
        adjusted = _adjust_vibrance(rgb_u8, self.params.vibrance, self.params.saturation)
        merged = merge_alpha(cast(ImageArray, adjusted), alpha)
        return cast(ImageArray, merged.astype(image.dtype))
