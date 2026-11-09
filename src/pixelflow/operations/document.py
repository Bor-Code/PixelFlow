from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field, field_validator

from pixelflow.core.image_utils import merge_alpha, split_alpha
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class ShadowRemovalParams(OperationParams):
    kernel_size: int = Field(default=25, ge=3, le=101)
    blur_size: int = Field(default=21, ge=3, le=101)

    @field_validator("kernel_size", "blur_size")
    @classmethod
    def validate_odd_sizes(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError(f"Value must be odd, got {value}")
        return value


class DeskewParams(OperationParams):
    max_angle: float = Field(default=45.0, ge=1.0, le=89.0)
    fill_value: int = Field(default=255, ge=0, le=255)
    expand: bool = Field(default=True)


def _remove_channel_shadow(channel: np.ndarray, kernel_size: int, blur_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    bg_dilated = cv2.dilate(channel, kernel)
    bg = cv2.medianBlur(bg_dilated, blur_size)
    diff = 255 - cv2.absdiff(channel, bg)
    norm = np.empty_like(diff)
    cv2.normalize(diff, norm, 0, 255, cv2.NORM_MINMAX)
    return cast(np.ndarray, norm)


@register_operation("shadow_removal")
class ShadowRemovalOperation(ImageOperation[ShadowRemovalParams]):
    params_model = ShadowRemovalParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        k = self.params.kernel_size
        b = self.params.blur_size
        if base.ndim == 2:
            cleaned = _remove_channel_shadow(base, k, b)
        else:
            channels = [_remove_channel_shadow(base[:, :, i], k, b) for i in range(base.shape[2])]
            cleaned = np.dstack(channels)
        merged = merge_alpha(cleaned, alpha)
        return cast(ImageArray, merged.astype(image.dtype))


def _detect_skew_angle(gray: np.ndarray, max_angle: float) -> float:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(thresh)
    if pts is None or len(pts) < 30:
        return 0.0
    rect = cv2.minAreaRect(pts)
    raw_angle = float(rect[2])
    if raw_angle < -45.0:
        angle = -(90.0 + raw_angle)
    elif raw_angle > 45.0:
        angle = 90.0 - raw_angle
    else:
        angle = -raw_angle
    return angle if abs(angle) <= max_angle else 0.0


def _rotate_bound(image: np.ndarray, angle: float, fill_value: int) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    matrix[0, 2] += (nw / 2.0) - center[0]
    matrix[1, 2] += (nh / 2.0) - center[1]
    border_val = (fill_value,) * (image.shape[2] if image.ndim == 3 else 1)
    return cast(
        np.ndarray,
        cv2.warpAffine(
            image, matrix, (nw, nh), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val
        ),
    )


@register_operation("deskew")
class DeskewOperation(ImageOperation[DeskewParams]):
    params_model = DeskewParams

    def _execute(self, image: ImageArray) -> ImageArray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.ndim == 3 and gray.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        angle = _detect_skew_angle(gray, self.params.max_angle)
        if abs(angle) < 0.2:
            return image.copy()
        rotated = _rotate_bound(image, angle, self.params.fill_value)
        return cast(ImageArray, rotated.astype(image.dtype))
