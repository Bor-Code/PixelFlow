from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class ColorSplashParams(OperationParams):
    preset_color: Literal["red", "green", "blue", "yellow", "orange", "purple", "custom"] = "red"
    hue: float = Field(default=0.0, ge=0.0, le=179.0)
    hue_tolerance: float = Field(default=15.0, ge=1.0, le=90.0)
    min_saturation: float = Field(default=50.0, ge=0.0, le=255.0)
    min_value: float = Field(default=30.0, ge=0.0, le=255.0)
    feather: int = Field(default=5, ge=0, le=51)
    desaturate_level: float = Field(default=0.0, ge=0.0, le=1.0)


PRESET_HUES: dict[str, float] = {
    "red": 0.0,
    "orange": 15.0,
    "yellow": 30.0,
    "green": 60.0,
    "blue": 120.0,
    "purple": 145.0,
}


def _get_target_hue(preset: str, hue: float) -> float:
    if preset == "custom":
        return hue
    return PRESET_HUES.get(preset, 0.0)


def _build_hue_mask(
    hsv: np.ndarray, target_hue: float, tol: float, min_s: float, min_v: float
) -> np.ndarray:
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    valid_sv = (s >= min_s) & (v >= min_v)
    diff = np.abs(h - target_hue)
    circular_diff = np.minimum(diff, 180.0 - diff)
    mask = (circular_diff <= tol) & valid_sv
    return mask.astype(np.float32)


def _feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    if feather <= 0:
        return mask
    ksize = feather if feather % 2 == 1 else feather + 1
    return cast(np.ndarray, cv2.GaussianBlur(mask, (ksize, ksize), 0)).astype(np.float32)


def _apply_splash(rgb: np.ndarray, params: ColorSplashParams) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    target_hue = _get_target_hue(params.preset_color, params.hue)
    raw_mask = _build_hue_mask(
        hsv,
        target_hue,
        params.hue_tolerance,
        params.min_saturation,
        params.min_value,
    )
    mask = _feather_mask(raw_mask, params.feather)
    mask_3d = np.expand_dims(mask, axis=2)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    rgb_f = rgb.astype(np.float32)
    bg = gray_3ch + params.desaturate_level * (rgb_f - gray_3ch)

    res = mask_3d * rgb_f + (1.0 - mask_3d) * bg
    return np.clip(res, 0, 255).astype(np.uint8)


@register_operation("color_splash")
@register_operation("selective_color")
class ColorSplashOperation(ImageOperation[ColorSplashParams]):
    params_model = ColorSplashParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if u8.ndim == 2:
            return image.copy()

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _apply_splash(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
