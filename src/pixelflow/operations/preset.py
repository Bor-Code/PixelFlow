from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

PresetName = Literal[
    "teal_orange",
    "cyberpunk",
    "vintage_film",
    "moody_noir",
    "warm_sunset",
    "cool_breeze",
]


class PresetParams(OperationParams):
    preset: PresetName = "teal_orange"
    intensity: float = Field(default=1.0, ge=0.0, le=2.0)


def _lut_teal_orange() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(256, dtype=np.float32)
    r = np.clip(x * 1.15 - 10, 0, 255).astype(np.uint8)
    g = np.clip(x * 1.0, 0, 255).astype(np.uint8)
    b = np.clip(x * 0.85 + 25, 0, 255).astype(np.uint8)
    return r, g, b


def _lut_cyberpunk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(256, dtype=np.float32)
    r = np.clip(x * 1.25 + 15, 0, 255).astype(np.uint8)
    g = np.clip(x * 0.75 - 10, 0, 255).astype(np.uint8)
    b = np.clip(x * 1.30 + 20, 0, 255).astype(np.uint8)
    return r, g, b


def _lut_vintage_film() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(256, dtype=np.float32)
    r = np.clip(25 + x * 0.82 + 10, 0, 255).astype(np.uint8)
    g = np.clip(22 + x * 0.82 + 5, 0, 255).astype(np.uint8)
    b = np.clip(20 + x * 0.78, 0, 255).astype(np.uint8)
    return r, g, b


def _lut_warm_sunset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(256, dtype=np.float32)
    r = np.clip(x * 1.22 + 15, 0, 255).astype(np.uint8)
    g = np.clip(x * 1.08 + 5, 0, 255).astype(np.uint8)
    b = np.clip(x * 0.75 - 15, 0, 255).astype(np.uint8)
    return r, g, b


def _lut_cool_breeze() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(256, dtype=np.float32)
    r = np.clip(x * 0.88 - 5, 0, 255).astype(np.uint8)
    g = np.clip(x * 1.02 + 5, 0, 255).astype(np.uint8)
    b = np.clip(x * 1.20 + 15, 0, 255).astype(np.uint8)
    return r, g, b


def _apply_noir(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x = np.arange(256, dtype=np.float32)
    lut = np.clip((x / 255.0) ** 1.35 * 255.0, 0, 255).astype(np.uint8)
    noir = cv2.LUT(gray, lut)
    return cast(np.ndarray, cv2.cvtColor(noir, cv2.COLOR_GRAY2RGB))


def _transform_preset(rgb: np.ndarray, preset: PresetName) -> np.ndarray:
    if preset == "moody_noir":
        return _apply_noir(rgb)
    lut_map = {
        "teal_orange": _lut_teal_orange,
        "cyberpunk": _lut_cyberpunk,
        "vintage_film": _lut_vintage_film,
        "warm_sunset": _lut_warm_sunset,
        "cool_breeze": _lut_cool_breeze,
    }
    r_lut, g_lut, b_lut = lut_map[preset]()
    out = np.empty_like(rgb)
    out[:, :, 0] = cv2.LUT(rgb[:, :, 0], r_lut)
    out[:, :, 1] = cv2.LUT(rgb[:, :, 1], g_lut)
    out[:, :, 2] = cv2.LUT(rgb[:, :, 2], b_lut)
    return out


@register_operation("preset")
@register_operation("cinematic_preset")
class PresetOperation(ImageOperation[PresetParams]):
    params_model = PresetParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if self.params.intensity <= 0.0:
            return image.copy()
        alpha: np.ndarray | None = None
        work: np.ndarray = u8
        if u8.ndim == 2:
            work = ensure_rgb(u8)
        elif u8.shape[2] == 4:
            work = u8[:, :, :3]
            alpha = u8[:, :, 3]
        transformed = _transform_preset(work, self.params.preset)
        factor = float(self.params.intensity)
        blended = (1.0 - factor) * work.astype(np.float32) + factor * transformed.astype(np.float32)
        res = np.clip(blended, 0, 255).astype(np.uint8)
        if u8.ndim == 2:
            return cast(ImageArray, cv2.cvtColor(res, cv2.COLOR_RGB2GRAY))
        if alpha is not None:
            return merge_alpha(res, alpha)
        return cast(ImageArray, res)
