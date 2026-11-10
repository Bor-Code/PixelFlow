from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class DenoiseParams(OperationParams):
    h: float = Field(default=10.0, ge=1.0, le=50.0)
    h_color: float = Field(default=10.0, ge=1.0, le=50.0)
    template_window_size: int = Field(default=7, ge=3, le=21)
    search_window_size: int = Field(default=21, ge=7, le=35)


def _ensure_odd(val: int) -> int:
    return val if val % 2 == 1 else val + 1


def _denoise_colored(rgb: np.ndarray, params: DenoiseParams) -> np.ndarray:
    t_win = _ensure_odd(params.template_window_size)
    s_win = _ensure_odd(params.search_window_size)
    res = cv2.fastNlMeansDenoisingColored(
        rgb,
        None,
        h=params.h,
        hColor=params.h_color,
        templateWindowSize=t_win,
        searchWindowSize=s_win,
    )
    return cast(np.ndarray, res)


def _denoise_gray(gray: np.ndarray, params: DenoiseParams) -> np.ndarray:
    t_win = _ensure_odd(params.template_window_size)
    s_win = _ensure_odd(params.search_window_size)
    res = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=params.h,
        templateWindowSize=t_win,
        searchWindowSize=s_win,
    )
    return cast(np.ndarray, res)


@register_operation("denoise")
@register_operation("nlmeans_denoise")
class DenoiseOperation(ImageOperation[DenoiseParams]):
    params_model = DenoiseParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if u8.ndim == 2:
            return cast(ImageArray, _denoise_gray(u8, self.params))

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _denoise_colored(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
