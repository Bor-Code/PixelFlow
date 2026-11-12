from __future__ import annotations

from typing import cast

import numpy as np

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class ColorWheelsParams(OperationParams):
    lift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    lift_master: float = 0.0
    gamma_master: float = 1.0
    gain_master: float = 1.0


def _compute_channel_grading(
    channel: np.ndarray,
    lift: float,
    gamma: float,
    gain: float,
) -> np.ndarray:
    scaled = np.clip(channel * gain + lift, 0.0, 1.0)
    safe_gamma = max(0.01, gamma)
    graded = np.power(scaled, 1.0 / safe_gamma)
    return np.clip(graded, 0.0, 1.0)


def _apply_color_wheels_rgb(
    rgb: np.ndarray,
    params: ColorWheelsParams,
) -> np.ndarray:
    norm = rgb.astype(np.float32) / 255.0
    out = np.empty_like(norm)
    for c in range(3):
        eff_lift = params.lift[c] + params.lift_master
        eff_gain = max(0.0, params.gain[c] * params.gain_master)
        eff_gamma = max(0.01, params.gamma[c] * params.gamma_master)
        out[:, :, c] = _compute_channel_grading(
            norm[:, :, c], eff_lift, eff_gamma, eff_gain
        )
    return np.clip(out * 255.0, 0.0, 255.0).round().astype(np.uint8)


@register_operation("color_wheels")
class ColorWheelsOperation(ImageOperation[ColorWheelsParams]):
    operation_name = "color_wheels"
    params_model = ColorWheelsParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        base, alpha = split_alpha(u8)
        rgb = ensure_rgb(base)
        graded = _apply_color_wheels_rgb(rgb, self.params)
        if alpha is not None:
            return merge_alpha(graded, alpha)
        return cast(ImageArray, graded)
