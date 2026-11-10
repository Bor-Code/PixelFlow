from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class VignetteParams(OperationParams):
    amount: float = Field(default=0.6, ge=-1.0, le=1.0)
    radius: float = Field(default=0.8, ge=0.1, le=2.0)
    softness: float = Field(default=0.5, ge=0.05, le=1.0)
    roundness: float = Field(default=1.0, ge=0.2, le=2.0)
    center_x: float = Field(default=0.5, ge=0.0, le=1.0)
    center_y: float = Field(default=0.5, ge=0.0, le=1.0)


def _compute_vignette_mask(
    h: int, w: int, params: VignetteParams
) -> np.ndarray:
    y_coords = (np.arange(h, dtype=np.float32) - (params.center_y * (h - 1))) / (h / 2.0)
    x_coords = (np.arange(w, dtype=np.float32) - (params.center_x * (w - 1))) / (w / 2.0)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    dist = np.sqrt(grid_x**2 + (grid_y * params.roundness) ** 2)

    r_inner = max(0.0, params.radius * (1.0 - params.softness))
    r_outer = params.radius + params.softness
    span = max(1e-4, r_outer - r_inner)

    t = np.clip((dist - r_inner) / span, 0.0, 1.0)
    return cast(np.ndarray, t * t * (3.0 - 2.0 * t))


def _apply_vignette_rgb(rgb: np.ndarray, params: VignetteParams) -> np.ndarray:
    if abs(params.amount) < 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    mask = _compute_vignette_mask(h, w, params)
    mask_3d = np.expand_dims(mask, axis=2)
    rgb_f = rgb.astype(np.float32)

    if params.amount > 0:
        darken_factor = 1.0 - (params.amount * mask_3d)
        res = rgb_f * darken_factor
    else:
        lighten_factor = (-params.amount) * mask_3d
        res = rgb_f + (255.0 - rgb_f) * lighten_factor

    return np.clip(res, 0, 255).astype(np.uint8)


@register_operation("vignette")
class VignetteOperation(ImageOperation[VignetteParams]):
    params_model = VignetteParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if abs(self.params.amount) < 1e-4:
            return image.copy()

        if u8.ndim == 2:
            rgb = ensure_rgb(u8)
            res_rgb = _apply_vignette_rgb(rgb, self.params)
            return cast(ImageArray, cv2.cvtColor(res_rgb, cv2.COLOR_RGB2GRAY))

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _apply_vignette_rgb(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
