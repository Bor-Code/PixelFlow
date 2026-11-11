from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class DuotoneParams(OperationParams):
    preset: Literal["spotify", "sunset", "cyberpunk", "amber", "ocean", "neon_mint", "custom"] = (
        "spotify"
    )
    shadow_color: str = "#0d1b2a"
    highlight_color: str = "#1db954"
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.5, le=2.0)


DUOTONE_PRESETS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "spotify": ((13, 27, 42), (29, 185, 84)),
    "sunset": ((45, 0, 75), (255, 120, 50)),
    "cyberpunk": ((10, 10, 45), (255, 0, 128)),
    "amber": ((20, 15, 10), (255, 180, 50)),
    "ocean": ((5, 25, 45), (64, 224, 208)),
    "neon_mint": ((15, 23, 42), (52, 211, 153)),
}


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    clean = hex_str.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        return (0, 0, 0)
    return (
        int(clean[0:2], 16),
        int(clean[2:4], 16),
        int(clean[4:6], 16),
    )


def _get_duotone_colors(params: DuotoneParams) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if params.preset in DUOTONE_PRESETS:
        return DUOTONE_PRESETS[params.preset]
    shadow = _hex_to_rgb(params.shadow_color)
    highlight = _hex_to_rgb(params.highlight_color)
    return shadow, highlight


def _build_duotone_luts(
    shadow: tuple[int, int, int], highlight: tuple[int, int, int], contrast: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = (np.arange(256, dtype=np.float32) / 255.0) ** float(contrast)
    r_lut = np.clip(shadow[0] + t * (highlight[0] - shadow[0]), 0, 255).astype(np.uint8)
    g_lut = np.clip(shadow[1] + t * (highlight[1] - shadow[1]), 0, 255).astype(np.uint8)
    b_lut = np.clip(shadow[2] + t * (highlight[2] - shadow[2]), 0, 255).astype(np.uint8)
    return r_lut, g_lut, b_lut


def _apply_duotone_rgb(rgb: np.ndarray, params: DuotoneParams) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    shadow, highlight = _get_duotone_colors(params)
    r_lut, g_lut, b_lut = _build_duotone_luts(shadow, highlight, params.contrast)
    duo = np.empty_like(rgb)
    duo[:, :, 0] = cv2.LUT(gray, r_lut)
    duo[:, :, 1] = cv2.LUT(gray, g_lut)
    duo[:, :, 2] = cv2.LUT(gray, b_lut)

    if params.intensity >= 1.0:
        return duo
    factor = float(params.intensity)
    blended = (1.0 - factor) * rgb.astype(np.float32) + factor * duo.astype(np.float32)
    return cast(np.ndarray, np.clip(blended, 0, 255).astype(np.uint8))


@register_operation("duotone")
@register_operation("gradient_map")
class DuotoneOperation(ImageOperation[DuotoneParams]):
    params_model = DuotoneParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if self.params.intensity <= 0.0:
            return image.copy()

        if u8.ndim == 2:
            rgb = ensure_rgb(u8)
            res_rgb = _apply_duotone_rgb(rgb, self.params)
            return cast(ImageArray, cv2.cvtColor(res_rgb, cv2.COLOR_RGB2GRAY))

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _apply_duotone_rgb(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
