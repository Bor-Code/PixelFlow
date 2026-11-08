from __future__ import annotations

import cv2
from pydantic import Field, field_validator, model_validator

from pixelflow.core.image_utils import ensure_grayscale, restore_dtype, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class CannyParams(OperationParams):
    threshold1: float = Field(default=100.0, ge=0.0, le=1000.0)
    threshold2: float = Field(default=200.0, ge=0.0, le=1000.0)
    aperture_size: int = Field(default=3)
    l2_gradient: bool = False

    @field_validator("aperture_size")
    @classmethod
    def check_aperture(cls, value: int) -> int:
        if value not in (3, 5, 7):
            raise ValueError("aperture_size must be 3, 5, or 7")
        return value

    @model_validator(mode="after")
    def validate_threshold_order(self) -> CannyParams:
        if self.threshold1 > self.threshold2:
            raise ValueError("threshold1 must be less than or equal to threshold2")
        return self


@register_operation("canny")
class CannyOperation(ImageOperation[CannyParams]):
    params_model = CannyParams

    def _execute(self, image: ImageArray) -> ImageArray:
        gray = ensure_grayscale(image)
        u8 = to_uint8(gray)
        edges_u8 = cv2.Canny(
            u8,
            threshold1=self.params.threshold1,
            threshold2=self.params.threshold2,
            apertureSize=self.params.aperture_size,
            L2gradient=self.params.l2_gradient,
        )
        return restore_dtype(edges_u8, image.dtype)


class SobelParams(OperationParams):
    dx: int = Field(default=1, ge=0, le=2)
    dy: int = Field(default=1, ge=0, le=2)
    kernel_size: int = Field(default=3, ge=1, le=7)

    @field_validator("kernel_size")
    @classmethod
    def check_odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return value

    @model_validator(mode="after")
    def validate_derivatives(self) -> SobelParams:
        if self.dx == 0 and self.dy == 0:
            raise ValueError("At least one of dx or dy must be greater than 0")
        return self


@register_operation("sobel")
class SobelOperation(ImageOperation[SobelParams]):
    params_model = SobelParams

    def _execute(self, image: ImageArray) -> ImageArray:
        gray = ensure_grayscale(image)
        u8 = to_uint8(gray)
        k = self.params.kernel_size
        grad = cv2.Sobel(u8, cv2.CV_64F, self.params.dx, self.params.dy, ksize=k)
        abs_grad = cv2.convertScaleAbs(grad)
        return restore_dtype(abs_grad, image.dtype)
