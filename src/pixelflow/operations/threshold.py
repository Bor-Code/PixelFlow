from __future__ import annotations

from typing import Literal

import cv2
from pydantic import Field, field_validator

from pixelflow.core.image_utils import ensure_grayscale, restore_dtype, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

AdaptiveMethodName = Literal["mean", "gaussian"]
ThresholdTypeName = Literal["binary", "binary_inv"]


class AdaptiveThresholdParams(OperationParams):
    max_value: int = Field(default=255, ge=1, le=255)
    method: AdaptiveMethodName = "gaussian"
    threshold_type: ThresholdTypeName = "binary"
    block_size: int = Field(default=11, ge=3, le=99)
    constant: float = Field(default=2.0, ge=-100.0, le=100.0)

    @field_validator("block_size")
    @classmethod
    def check_odd_block_size(cls, value: int) -> int:
        if value < 3 or value % 2 == 0:
            raise ValueError("block_size must be an odd integer >= 3")
        return value


@register_operation("adaptive_threshold")
class AdaptiveThresholdOperation(ImageOperation[AdaptiveThresholdParams]):
    params_model = AdaptiveThresholdParams

    def _execute(self, image: ImageArray) -> ImageArray:
        gray = ensure_grayscale(image)
        u8 = to_uint8(gray)
        method_flag = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.params.method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        type_flag = (
            cv2.THRESH_BINARY if self.params.threshold_type == "binary" else cv2.THRESH_BINARY_INV
        )
        thresh_u8 = cv2.adaptiveThreshold(
            u8,
            maxValue=float(self.params.max_value),
            adaptiveMethod=method_flag,
            thresholdType=type_flag,
            blockSize=self.params.block_size,
            C=self.params.constant,
        )
        return restore_dtype(thresh_u8, image.dtype)


class OtsuThresholdParams(OperationParams):
    max_value: int = Field(default=255, ge=1, le=255)
    threshold_type: ThresholdTypeName = "binary"


@register_operation("otsu_threshold")
class OtsuThresholdOperation(ImageOperation[OtsuThresholdParams]):
    params_model = OtsuThresholdParams

    def _execute(self, image: ImageArray) -> ImageArray:
        gray = ensure_grayscale(image)
        u8 = to_uint8(gray)
        base_flag = (
            cv2.THRESH_BINARY if self.params.threshold_type == "binary" else cv2.THRESH_BINARY_INV
        )
        _, thresh_u8 = cv2.threshold(
            u8,
            thresh=0.0,
            maxval=float(self.params.max_value),
            type=base_flag | cv2.THRESH_OTSU,
        )
        return restore_dtype(thresh_u8, image.dtype)
