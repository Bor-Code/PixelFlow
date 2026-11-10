from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field, field_validator

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class CurvesParams(OperationParams):
    channel: Literal["all", "r", "g", "b"] = Field(default="all")
    points: tuple[tuple[int, int], ...] = Field(default=((0, 0), (255, 255)))

    @field_validator("points")
    @classmethod
    def validate_points(cls, v: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
        if len(v) < 2:
            raise ValueError("Curves requires at least 2 control points")
        for x, y in v:
            if not (0 <= x <= 255 and 0 <= y <= 255):
                raise ValueError(f"Point coordinates must be between 0 and 255, got ({x}, {y})")
        return v


def _build_lut(points: tuple[tuple[int, int], ...]) -> np.ndarray:
    sorted_pts = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in sorted_pts]
    ys = [p[1] for p in sorted_pts]
    lut = np.interp(np.arange(256), xs, ys)
    return np.clip(lut, 0.0, 255.0).astype(np.uint8)


def _apply_lut(base: np.ndarray, lut: np.ndarray, channel: str) -> np.ndarray:
    if base.ndim == 2 or channel == "all":
        return cast(np.ndarray, cv2.LUT(base, lut))
    res = ensure_rgb(base).copy()
    ch_idx = {"r": 0, "g": 1, "b": 2}[channel]
    res[:, :, ch_idx] = cv2.LUT(res[:, :, ch_idx], lut)
    return cast(np.ndarray, res)


@register_operation("curves")
class CurvesOperation(ImageOperation[CurvesParams]):
    params_model = CurvesParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        u8 = to_uint8(base)
        lut = _build_lut(self.params.points)
        transformed = _apply_lut(u8, lut, self.params.channel)
        merged = merge_alpha(cast(ImageArray, transformed), alpha)
        return cast(ImageArray, merged.astype(image.dtype))
