from __future__ import annotations

from typing import Literal, cast

import cv2
from pydantic import Field, field_validator

from pixelflow.core.image_utils import merge_alpha, split_alpha
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

MorphAction = Literal["erode", "dilate", "open", "close"]
KernelShapeName = Literal["rect", "ellipse", "cross"]

MORPH_OP_MAP: dict[MorphAction, int] = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
}

KERNEL_SHAPE_MAP: dict[KernelShapeName, int] = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


class MorphologyParams(OperationParams):
    action: MorphAction = "dilate"
    kernel_size: int = Field(default=3, ge=1, le=99)
    kernel_shape: KernelShapeName = "rect"
    iterations: int = Field(default=1, ge=1, le=20)

    @field_validator("kernel_size")
    @classmethod
    def check_odd_kernel(cls, value: int) -> int:
        if value < 1 or value % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return value


@register_operation("morphology")
class MorphologyOperation(ImageOperation[MorphologyParams]):
    params_model = MorphologyParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        k = self.params.kernel_size
        shape_flag = KERNEL_SHAPE_MAP[self.params.kernel_shape]
        kernel = cv2.getStructuringElement(shape_flag, (k, k))
        op_flag = MORPH_OP_MAP[self.params.action]
        morphed = cast(
            ImageArray,
            cv2.morphologyEx(base, op_flag, kernel, iterations=self.params.iterations),
        )
        return merge_alpha(morphed, alpha)
