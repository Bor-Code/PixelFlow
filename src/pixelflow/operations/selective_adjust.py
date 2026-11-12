from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

MaskShape = Literal["radial", "linear", "box"]


class SelectiveAdjustParams(OperationParams):
    shape: MaskShape = "radial"
    center: tuple[float, float] = (0.5, 0.5)
    size: tuple[float, float] = (0.3, 0.3)
    feather: float = Field(default=0.2, ge=0.0, le=1.0)
    angle: float = 0.0
    invert: bool = False
    exposure: float = Field(default=0.0, ge=-2.0, le=2.0)
    contrast: float = Field(default=1.0, ge=0.0, le=3.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0)
    warmth: float = Field(default=0.0, ge=-1.0, le=1.0)
    blur_radius: int = Field(default=0, ge=0, le=50)


def _build_radial_mask(h: int, w: int, params: SelectiveAdjustParams) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    cx = params.center[0] * w
    cy = params.center[1] * h
    rx = max(1.0, params.size[0] * w)
    ry = max(1.0, params.size[1] * h)
    dist = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    feather = max(0.01, params.feather)
    val = (1.0 - dist) / feather
    return cast(np.ndarray, np.clip(val, 0.0, 1.0))


def _build_linear_mask(h: int, w: int, params: SelectiveAdjustParams) -> np.ndarray:
    ys, xs = np.mgrid[:h, :w]
    cx = params.center[0] * w
    cy = params.center[1] * h
    rad = np.radians(params.angle)
    proj = (xs - cx) * np.sin(rad) + (ys - cy) * np.cos(rad)
    span = max(1.0, params.size[1] * h)
    feather = max(0.01, params.feather) * span
    val = 0.5 - (proj / (span + feather))
    return cast(np.ndarray, np.clip(val, 0.0, 1.0))


def _build_box_mask(h: int, w: int, params: SelectiveAdjustParams) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    cx = params.center[0] * w
    cy = params.center[1] * h
    rx = max(1.0, params.size[0] * w * 0.5)
    ry = max(1.0, params.size[1] * h * 0.5)
    dx = np.maximum(0.0, np.abs(xs - cx) - rx)
    dy = np.maximum(0.0, np.abs(ys - cy) - ry)
    dist = np.sqrt(dx**2 + dy**2)
    feather = max(1.0, params.feather * max(rx, ry))
    return cast(np.ndarray, np.clip(1.0 - (dist / feather), 0.0, 1.0))


def _generate_mask(h: int, w: int, params: SelectiveAdjustParams) -> np.ndarray:
    if params.shape == "radial":
        mask = _build_radial_mask(h, w, params)
    elif params.shape == "linear":
        mask = _build_linear_mask(h, w, params)
    else:
        mask = _build_box_mask(h, w, params)
    if params.invert:
        mask = 1.0 - mask
    return mask[:, :, np.newaxis]


def _apply_tone_adjustments(
    rgb: np.ndarray,
    params: SelectiveAdjustParams,
) -> np.ndarray:
    norm = rgb.astype(np.float32) / 255.0
    if params.exposure != 0.0:
        norm = norm * (2.0**params.exposure)
    if params.contrast != 1.0:
        norm = (norm - 0.5) * params.contrast + 0.5
    if params.saturation != 1.0:
        gray = norm[:, :, 0] * 0.299 + norm[:, :, 1] * 0.587 + norm[:, :, 2] * 0.114
        norm = gray[:, :, np.newaxis] + (norm - gray[:, :, np.newaxis]) * params.saturation
    if params.warmth != 0.0:
        norm[:, :, 0] += params.warmth * 0.1
        norm[:, :, 2] -= params.warmth * 0.1
    clipped = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    if params.blur_radius > 0:
        k = params.blur_radius if params.blur_radius % 2 == 1 else params.blur_radius + 1
        blurred = cv2.GaussianBlur(clipped, (k, k), 0)
        return cast(np.ndarray, blurred)
    return cast(np.ndarray, clipped)


@register_operation("selective_adjust")
class SelectiveAdjustOperation(ImageOperation[SelectiveAdjustParams]):
    operation_name = "selective_adjust"
    params_model = SelectiveAdjustParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        base, alpha = split_alpha(u8)
        rgb = ensure_rgb(base)
        mask = _generate_mask(rgb.shape[0], rgb.shape[1], self.params)
        adjusted = _apply_tone_adjustments(rgb, self.params).astype(np.float32)
        blended = (rgb.astype(np.float32) * (1.0 - mask) + adjusted * mask)
        out = np.clip(blended, 0.0, 255.0).round().astype(np.uint8)
        if alpha is not None:
            return merge_alpha(out, alpha)
        return cast(ImageArray, out)
