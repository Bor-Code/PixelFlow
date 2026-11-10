from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class AutoEnhanceParams(OperationParams):
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    auto_white_balance: bool = Field(default=True)
    auto_contrast: bool = Field(default=True)
    unsharp: bool = Field(default=True)
    clip_limit: float = Field(default=2.0, ge=1.0, le=5.0)


def _apply_white_balance(rgb: np.ndarray, strength: float) -> np.ndarray:
    means = np.mean(rgb, axis=(0, 1))
    avg = float(np.mean(means))
    scales = [float(avg / (m + 1e-5)) for m in means]
    out = rgb.astype(np.float32)
    for c in range(3):
        blended = (1.0 - strength) * out[:, :, c] + strength * (out[:, :, c] * scales[c])
        out[:, :, c] = blended
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_clahe_lab(rgb: np.ndarray, clip_limit: float, strength: float) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(lab[:, :, 0])
    l_orig = lab[:, :, 0].astype(np.float32)
    l_enh = enhanced_l.astype(np.float32)
    lab[:, :, 0] = np.clip((1.0 - strength) * l_orig + strength * l_enh, 0, 255).astype(np.uint8)
    return cast(np.ndarray, cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def _apply_unsharp(rgb: np.ndarray, strength: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, (0, 0), 1.5)
    weight = float(strength * 0.4)
    res = cv2.addWeighted(rgb, 1.0 + weight, blurred, -weight, 0)
    return cast(np.ndarray, np.clip(res, 0, 255).astype(np.uint8))


@register_operation("auto_enhance")
class AutoEnhanceOperation(ImageOperation[AutoEnhanceParams]):
    params_model = AutoEnhanceParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if self.params.strength <= 0.0:
            return image.copy()
        if u8.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=self.params.clip_limit, tileGridSize=(8, 8))
            return cast(ImageArray, clahe.apply(u8))
        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]
        if self.params.auto_white_balance:
            cur = _apply_white_balance(cur, min(1.0, self.params.strength))
        if self.params.auto_contrast:
            cur = _apply_clahe_lab(cur, self.params.clip_limit, min(1.0, self.params.strength))
        if self.params.unsharp:
            cur = _apply_unsharp(cur, self.params.strength)
        if alpha is not None:
            return merge_alpha(cur, alpha)
        return cast(ImageArray, cur)
