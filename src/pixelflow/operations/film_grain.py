from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class FilmGrainParams(OperationParams):
    intensity: float = Field(default=0.2, ge=0.0, le=1.0)
    grain_size: float = Field(default=1.0, ge=1.0, le=4.0)
    colored: bool = Field(default=False)
    seed: int | None = Field(default=42)


def _generate_noise_map(
    shape: tuple[int, int], channels: int, grain_size: float, seed: int | None
) -> np.ndarray:
    h, w = shape
    rng = np.random.default_rng(seed)
    if grain_size <= 1.05:
        noise_shape = (h, w, channels) if channels > 1 else (h, w)
        return cast(np.ndarray, rng.standard_normal(noise_shape, dtype=np.float32))

    sw = max(2, int(w / grain_size))
    sh = max(2, int(h / grain_size))
    s_shape = (sh, sw, channels) if channels > 1 else (sh, sw)
    s_noise = rng.standard_normal(s_shape, dtype=np.float32)
    resized = cv2.resize(s_noise, (w, h), interpolation=cv2.INTER_LINEAR)
    return cast(np.ndarray, resized)


def _apply_film_grain_rgb(rgb: np.ndarray, params: FilmGrainParams) -> np.ndarray:
    if params.intensity <= 0.0:
        return rgb
    h, w = rgb.shape[:2]
    channels = 3 if params.colored else 1
    noise = _generate_noise_map((h, w), channels, params.grain_size, params.seed)
    if channels == 1:
        noise = np.expand_dims(noise, axis=2)

    scale = float(params.intensity) * 127.0
    blended = rgb.astype(np.float32) + noise * scale
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_film_grain_gray(gray: np.ndarray, params: FilmGrainParams) -> np.ndarray:
    if params.intensity <= 0.0:
        return gray
    h, w = gray.shape[:2]
    noise = _generate_noise_map((h, w), 1, params.grain_size, params.seed)
    scale = float(params.intensity) * 127.0
    blended = gray.astype(np.float32) + noise * scale
    return np.clip(blended, 0, 255).astype(np.uint8)


@register_operation("film_grain")
@register_operation("grain")
class FilmGrainOperation(ImageOperation[FilmGrainParams]):
    params_model = FilmGrainParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if self.params.intensity <= 0.0:
            return image.copy()

        if u8.ndim == 2:
            return cast(ImageArray, _apply_film_grain_gray(u8, self.params))

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _apply_film_grain_rgb(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
