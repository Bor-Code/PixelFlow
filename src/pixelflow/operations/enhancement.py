from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator

from pixelflow.core.image_utils import merge_alpha, restore_dtype, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class HistogramEqualizationParams(OperationParams):
    use_clahe: bool = False
    clip_limit: float = Field(default=2.0, gt=0.0, le=40.0)
    tile_grid_size: int = Field(default=8, ge=2, le=64)


@register_operation("histogram_equalization")
class HistogramEqualizationOperation(ImageOperation[HistogramEqualizationParams]):
    params_model = HistogramEqualizationParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        u8 = to_uint8(base)
        equalized_u8 = self._equalize_u8(u8)
        restored = restore_dtype(equalized_u8, base.dtype)
        return merge_alpha(restored, alpha)

    def _equalize_u8(self, u8: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if u8.ndim == 2:
            return self._equalize_single_channel(u8)
        ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
        channels = list(cv2.split(ycrcb))
        channels[0] = self._equalize_single_channel(cast(npt.NDArray[np.uint8], channels[0]))
        merged = cv2.merge(channels)
        return cast(npt.NDArray[np.uint8], cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB))

    def _equalize_single_channel(self, channel: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if not self.params.use_clahe:
            return cast(npt.NDArray[np.uint8], cv2.equalizeHist(channel))
        grid = self.params.tile_grid_size
        clahe = cv2.createCLAHE(clipLimit=self.params.clip_limit, tileGridSize=(grid, grid))
        return cast(npt.NDArray[np.uint8], clahe.apply(channel))


class SharpenParams(OperationParams):
    kernel_size: int = Field(default=5, ge=1, le=31)
    sigma: float = Field(default=1.0, gt=0.0, le=20.0)
    amount: float = Field(default=1.5, ge=0.0, le=10.0)
    threshold: int = Field(default=0, ge=0, le=255)

    @field_validator("kernel_size")
    @classmethod
    def check_odd_kernel(cls, value: int) -> int:
        if value < 1 or value % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return value


@register_operation("sharpen")
class SharpenOperation(ImageOperation[SharpenParams]):
    params_model = SharpenParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        k = self.params.kernel_size
        blurred = cv2.GaussianBlur(base, (k, k), sigmaX=self.params.sigma)
        max_val = 65535.0 if base.dtype == np.uint16 else 255.0
        base_f = base.astype(np.float32)
        blur_f = blurred.astype(np.float32)
        sharpened_f = base_f * (1.0 + self.params.amount) - blur_f * self.params.amount
        sharpened = cast(ImageArray, np.clip(sharpened_f, 0.0, max_val).round().astype(base.dtype))
        if self.params.threshold > 0:
            scale = 257 if base.dtype == np.uint16 else 1
            low_contrast = np.abs(base_f - blur_f) < (self.params.threshold * scale)
            np.copyto(sharpened, base, where=low_contrast)
        return merge_alpha(sharpened, alpha)
