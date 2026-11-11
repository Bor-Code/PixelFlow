from __future__ import annotations

from typing import Literal, cast

import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

BlendMode = Literal[
    "normal",
    "multiply",
    "screen",
    "overlay",
    "soft_light",
    "darken",
    "lighten",
    "difference",
]


class BlendParams(OperationParams):
    mode: BlendMode = "normal"
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    color: tuple[int, int, int] = (255, 255, 255)


def _compute_blend(
    a: np.ndarray,
    b: np.ndarray,
    mode: BlendMode,
) -> np.ndarray:
    if mode == "multiply":
        return cast(np.ndarray, a * b)
    if mode == "screen":
        return cast(np.ndarray, 1.0 - (1.0 - a) * (1.0 - b))
    if mode == "overlay":
        mask = a < 0.5
        res = np.empty_like(a)
        res[mask] = 2.0 * a[mask] * b[mask]
        res[~mask] = 1.0 - 2.0 * (1.0 - a[~mask]) * (1.0 - b[~mask])
        return res
    if mode == "soft_light":
        mask = b < 0.5
        res = np.empty_like(a)
        res[mask] = 2.0 * a[mask] * b[mask] + (a[mask] ** 2) * (1.0 - 2.0 * b[mask])
        sq = np.sqrt(np.maximum(a[~mask], 0.0))
        res[~mask] = 2.0 * a[~mask] * (1.0 - b[~mask]) + sq * (2.0 * b[~mask] - 1.0)
        return res
    if mode == "darken":
        return cast(np.ndarray, np.minimum(a, b))
    if mode == "lighten":
        return cast(np.ndarray, np.maximum(a, b))
    if mode == "difference":
        return cast(np.ndarray, np.abs(a - b))
    return b


def _apply_blend_rgb(
    base_rgb: np.ndarray,
    params: BlendParams,
) -> np.ndarray:
    a = base_rgb.astype(np.float32) / 255.0
    color_arr = np.array(params.color, dtype=np.float32) / 255.0
    b = np.full_like(a, color_arr)
    blended = _compute_blend(a, b, params.mode)
    out = a * (1.0 - params.opacity) + blended * params.opacity
    return np.clip(out * 255.0, 0.0, 255.0).round().astype(np.uint8)


@register_operation("blend")
class BlendOperation(ImageOperation[BlendParams]):
    operation_name = "blend"
    params_model = BlendParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        base, alpha = split_alpha(u8)
        rgb = ensure_rgb(base)
        blended = _apply_blend_rgb(rgb, self.params)
        if alpha is not None:
            return merge_alpha(blended, alpha)
        return cast(ImageArray, blended)
