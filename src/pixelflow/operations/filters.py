from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field, field_validator

from pixelflow.core.image_utils import merge_alpha, restore_dtype, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


def _validate_odd_kernel(value: int) -> int:
    if value < 1 or value % 2 == 0:
        raise ValueError(f"Kernel size must be a positive odd integer, got {value}")
    return value


class GaussianBlurParams(OperationParams):
    kernel_size: int = Field(default=3, ge=1, le=99)
    sigma_x: float = Field(default=0.0, ge=0.0, le=100.0)
    sigma_y: float = Field(default=0.0, ge=0.0, le=100.0)

    @field_validator("kernel_size")
    @classmethod
    def check_odd_kernel(cls, value: int) -> int:
        return _validate_odd_kernel(value)


@register_operation("gaussian_blur")
class GaussianBlurOperation(ImageOperation[GaussianBlurParams]):
    params_model = GaussianBlurParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        k = self.params.kernel_size
        blurred = cast(
            ImageArray,
            cv2.GaussianBlur(base, (k, k), sigmaX=self.params.sigma_x, sigmaY=self.params.sigma_y),
        )
        return merge_alpha(blurred, alpha)


class MedianBlurParams(OperationParams):
    kernel_size: int = Field(default=3, ge=1, le=99)

    @field_validator("kernel_size")
    @classmethod
    def check_odd_kernel(cls, value: int) -> int:
        return _validate_odd_kernel(value)


@register_operation("median_blur")
class MedianBlurOperation(ImageOperation[MedianBlurParams]):
    params_model = MedianBlurParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        k = self.params.kernel_size
        if k == 1:
            return merge_alpha(base.copy(), alpha)
        if base.dtype == np.uint16 and k > 5:
            u8 = to_uint8(base)
            blurred_u8 = cv2.medianBlur(u8, k)
            restored = restore_dtype(blurred_u8, np.dtype(np.uint16))
            return merge_alpha(restored, alpha)
        blurred = cast(ImageArray, cv2.medianBlur(base, k))
        return merge_alpha(blurred, alpha)


class BilateralFilterParams(OperationParams):
    diameter: int = Field(default=5, ge=1, le=25)
    sigma_color: float = Field(default=50.0, gt=0.0, le=255.0)
    sigma_space: float = Field(default=50.0, gt=0.0, le=255.0)


@register_operation("bilateral_filter")
class BilateralFilterOperation(ImageOperation[BilateralFilterParams]):
    params_model = BilateralFilterParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        orig_dtype = base.dtype
        u8 = to_uint8(base)
        filtered_u8 = cv2.bilateralFilter(
            u8,
            d=self.params.diameter,
            sigmaColor=self.params.sigma_color,
            sigmaSpace=self.params.sigma_space,
        )
        restored = restore_dtype(filtered_u8, orig_dtype)
        return merge_alpha(restored, alpha)
