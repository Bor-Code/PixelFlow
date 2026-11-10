from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class TiltShiftParams(OperationParams):
    orientation: Literal["horizontal", "vertical"] = "horizontal"
    focus_position: float = Field(default=0.5, ge=0.0, le=1.0)
    focus_width: float = Field(default=0.25, ge=0.05, le=0.8)
    transition_width: float = Field(default=0.2, ge=0.05, le=0.8)
    blur_amount: int = Field(default=15, ge=3, le=51)
    saturation_boost: float = Field(default=1.2, ge=1.0, le=2.0)


def _compute_1d_blur_weight(
    size: int, pos: float, focus_w: float, trans_w: float
) -> np.ndarray:
    coords = np.arange(size, dtype=np.float32)
    center = pos * (size - 1)
    half_focus = (focus_w * size) / 2.0
    trans = max(1.0, trans_w * size)
    dist = np.abs(coords - center)
    raw = (dist - half_focus) / trans
    clipped = np.clip(raw, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _build_blur_weight_mask(
    shape: tuple[int, int], params: TiltShiftParams
) -> np.ndarray:
    h, w = shape
    if params.orientation == "horizontal":
        w1d = _compute_1d_blur_weight(
            h, params.focus_position, params.focus_width, params.transition_width
        )
        return np.repeat(w1d[:, np.newaxis], w, axis=1)
    w1d = _compute_1d_blur_weight(
        w, params.focus_position, params.focus_width, params.transition_width
    )
    return np.repeat(w1d[np.newaxis, :], h, axis=0)


def _boost_saturation(rgb: np.ndarray, boost: float) -> np.ndarray:
    if boost <= 1.0:
        return rgb
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)
    return cast(np.ndarray, cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


def _apply_tilt_shift_rgb(rgb: np.ndarray, params: TiltShiftParams) -> np.ndarray:
    boosted = _boost_saturation(rgb, params.saturation_boost)
    ksize = params.blur_amount if params.blur_amount % 2 == 1 else params.blur_amount + 1
    blurred = cv2.GaussianBlur(boosted, (ksize, ksize), 0)
    weight = _build_blur_weight_mask(rgb.shape[:2], params)
    weight_3d = np.expand_dims(weight, axis=2)
    out = (1.0 - weight_3d) * boosted.astype(np.float32) + weight_3d * blurred.astype(np.float32)
    return cast(np.ndarray, np.clip(out, 0, 255).astype(np.uint8))


@register_operation("tilt_shift")
@register_operation("miniature")
class TiltShiftOperation(ImageOperation[TiltShiftParams]):
    params_model = TiltShiftParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if u8.ndim == 2:
            rgb = ensure_rgb(u8)
            res_rgb = _apply_tilt_shift_rgb(rgb, self.params)
            return cast(ImageArray, cv2.cvtColor(res_rgb, cv2.COLOR_RGB2GRAY))

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _apply_tilt_shift_rgb(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
