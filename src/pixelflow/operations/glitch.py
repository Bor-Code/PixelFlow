from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class GlitchParams(OperationParams):
    offset: int = Field(default=8, ge=0, le=50)
    color_channel: Literal["red_blue", "red_green", "green_blue"] = "red_blue"
    scanlines: bool = Field(default=True)
    scanline_spacing: int = Field(default=4, ge=2, le=20)
    scanline_intensity: float = Field(default=0.25, ge=0.0, le=0.9)
    slice_glitch: bool = Field(default=False)
    slice_count: int = Field(default=4, ge=1, le=20)
    slice_displacement: int = Field(default=15, ge=1, le=50)


def _apply_rgb_split(rgb: np.ndarray, offset: int, mode: str) -> np.ndarray:
    if offset <= 0:
        return rgb.copy()
    _h, w = rgb.shape[:2]
    out = rgb.copy()
    c1, c2 = 0, 2
    if mode == "red_green":
        c1, c2 = 0, 1
    elif mode == "green_blue":
        c1, c2 = 1, 2

    out[:, offset:, c1] = rgb[:, : w - offset, c1]
    out[:, :offset, c1] = rgb[:, 0:1, c1]
    out[:, : w - offset, c2] = rgb[:, offset:, c2]
    out[:, w - offset :, c2] = rgb[:, w - 1 : w, c2]
    return out


def _apply_scanlines(rgb: np.ndarray, spacing: int, intensity: float) -> np.ndarray:
    if intensity <= 0.0 or spacing < 2:
        return rgb
    h = rgb.shape[0]
    out = rgb.copy().astype(np.float32)
    factor = 1.0 - intensity
    lines = np.arange(0, h, spacing)
    out[lines, :, :] *= factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_slices(rgb: np.ndarray, count: int, disp: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = rgb.copy()
    step = max(1, h // (count + 1))
    for i in range(1, count + 1):
        y = min(h - 5, i * step)
        slice_h = min(h - y, max(3, step // 4))
        direction = 1 if i % 2 == 1 else -1
        shift = (disp * direction) % w
        out[y : y + slice_h, :] = np.roll(out[y : y + slice_h, :], shift, axis=1)
    return out


def _apply_glitch(rgb: np.ndarray, params: GlitchParams) -> np.ndarray:
    res = _apply_rgb_split(rgb, params.offset, params.color_channel)
    if params.slice_glitch:
        res = _apply_slices(res, params.slice_count, params.slice_displacement)
    if params.scanlines:
        res = _apply_scanlines(res, params.scanline_spacing, params.scanline_intensity)
    return res


@register_operation("glitch")
@register_operation("rgb_split")
class GlitchOperation(ImageOperation[GlitchParams]):
    params_model = GlitchParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if u8.ndim == 2:
            rgb = ensure_rgb(u8)
            res_rgb = _apply_glitch(rgb, self.params)
            return cast(ImageArray, cv2.cvtColor(res_rgb, cv2.COLOR_RGB2GRAY))

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _apply_glitch(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
