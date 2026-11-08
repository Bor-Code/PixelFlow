from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation
from pixelflow.exceptions import InvalidParametersError

InterpolationName = Literal["nearest", "linear", "cubic", "area", "lanczos"]
FlipMode = Literal["horizontal", "vertical", "both"]

INTERPOLATION_MAP: dict[InterpolationName, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class ResizeParams(OperationParams):
    width: int | None = Field(default=None, gt=0, le=16384)
    height: int | None = Field(default=None, gt=0, le=16384)
    keep_aspect_ratio: bool = False
    interpolation: InterpolationName = "linear"

    @model_validator(mode="after")
    def validate_dimensions(self) -> ResizeParams:
        if self.width is None and self.height is None:
            raise ValueError("At least one of width or height must be provided")
        return self


@register_operation("resize")
class ResizeOperation(ImageOperation[ResizeParams]):
    params_model = ResizeParams

    def _execute(self, image: ImageArray) -> ImageArray:
        src_h, src_w = image.shape[0], image.shape[1]
        target_w, target_h = self._compute_target_size(src_w, src_h)
        interp_flag = INTERPOLATION_MAP[self.params.interpolation]
        resized = cv2.resize(image, (target_w, target_h), interpolation=interp_flag)
        if image.ndim == 3 and image.shape[2] == 1 and resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return cast(ImageArray, resized)

    def _compute_target_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        req_w, req_h = self.params.width, self.params.height
        if not self.params.keep_aspect_ratio:
            return (req_w or src_w, req_h or src_h)
        if req_w is not None and req_h is not None:
            scale = min(req_w / src_w, req_h / src_h)
            return max(1, round(src_w * scale)), max(1, round(src_h * scale))
        if req_w is not None:
            scale = req_w / src_w
            return req_w, max(1, round(src_h * scale))
        assert req_h is not None
        scale = req_h / src_h
        return max(1, round(src_w * scale)), req_h


class CropParams(OperationParams):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)


@register_operation("crop")
class CropOperation(ImageOperation[CropParams]):
    params_model = CropParams

    def _execute(self, image: ImageArray) -> ImageArray:
        img_h, img_w = image.shape[0], image.shape[1]
        x2 = self.params.x + self.params.width
        y2 = self.params.y + self.params.height
        if self.params.x >= img_w or self.params.y >= img_h or x2 > img_w or y2 > img_h:
            raise InvalidParametersError(
                self.operation_name,
                f"Crop box ({self.params.x}, {self.params.y}, {self.params.width}, "
                f"{self.params.height}) exceeds image bounds ({img_w}, {img_h})",
            )
        return image[self.params.y : y2, self.params.x : x2].copy()


class RotateParams(OperationParams):
    angle: float = Field(ge=-360.0, le=360.0)
    expand: bool = True
    fill_value: int = Field(default=0, ge=0, le=65535)


@register_operation("rotate")
class RotateOperation(ImageOperation[RotateParams]):
    params_model = RotateParams

    def _execute(self, image: ImageArray) -> ImageArray:
        normalized = self.params.angle % 360.0
        if np.isclose(normalized, 0.0):
            return image.copy()
        if self.params.expand and np.isclose(normalized, 90.0):
            return cast(ImageArray, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
        if self.params.expand and np.isclose(normalized, 180.0):
            return cast(ImageArray, cv2.rotate(image, cv2.ROTATE_180))
        if self.params.expand and np.isclose(normalized, 270.0):
            return cast(ImageArray, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
        return self._rotate_arbitrary(image)

    def _rotate_arbitrary(self, image: ImageArray) -> ImageArray:
        matrix, out_w, out_h = self._build_rotation_matrix(image.shape[1], image.shape[0])
        channels = 1 if image.ndim == 2 else image.shape[2]
        border_val: tuple[float, ...] = tuple(
            float(self.params.fill_value) for _ in range(channels)
        )
        rotated = cv2.warpAffine(
            image,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        if image.ndim == 3 and image.shape[2] == 1 and rotated.ndim == 2:
            rotated = rotated[:, :, np.newaxis]
        return cast(ImageArray, rotated)

    def _build_rotation_matrix(self, w: int, h: int) -> tuple[npt.NDArray[np.float64], int, int]:
        center = (w / 2.0, h / 2.0)
        matrix = cast(
            npt.NDArray[np.float64],
            cv2.getRotationMatrix2D(center, self.params.angle, 1.0),
        )
        if not self.params.expand:
            return matrix, w, h
        cos_v = abs(float(matrix[0, 0]))
        sin_v = abs(float(matrix[0, 1]))
        out_w = max(1, round(h * sin_v + w * cos_v))
        out_h = max(1, round(h * cos_v + w * sin_v))
        matrix[0, 2] += (out_w / 2.0) - center[0]
        matrix[1, 2] += (out_h / 2.0) - center[1]
        return matrix, out_w, out_h


class FlipParams(OperationParams):
    mode: FlipMode = "horizontal"


@register_operation("flip")
class FlipOperation(ImageOperation[FlipParams]):
    params_model = FlipParams

    def _execute(self, image: ImageArray) -> ImageArray:
        code_map: dict[FlipMode, int] = {"horizontal": 1, "vertical": 0, "both": -1}
        flipped = cv2.flip(image, code_map[self.params.mode])
        if image.ndim == 3 and image.shape[2] == 1 and flipped.ndim == 2:
            flipped = flipped[:, :, np.newaxis]
        return cast(ImageArray, flipped)
