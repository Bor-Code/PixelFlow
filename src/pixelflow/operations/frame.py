from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class FrameParams(OperationParams):
    style: Literal["solid", "polaroid", "passepartout", "filmstrip", "drop_shadow"] = (
        "solid"
    )
    thickness: int = Field(default=24, ge=1, le=500)
    color: tuple[int, int, int] = (255, 255, 255)
    bottom_extra: int = Field(default=60, ge=0, le=500)
    inner_border: bool = Field(default=False)
    inner_color: tuple[int, int, int] = (30, 30, 30)


def _draw_film_perforations(
    canvas: np.ndarray, top_pad: int, b_pad: int, w: int
) -> None:
    slot_w, slot_h = 10, max(4, top_pad // 2)
    step = 24
    for x in range(step // 2, w - slot_w, step):
        y1_top = (top_pad - slot_h) // 2
        cv2.rectangle(
            canvas, (x, y1_top), (x + slot_w, y1_top + slot_h), (220, 220, 220), -1
        )
        ch = canvas.shape[0]
        y1_bot = ch - b_pad + (b_pad - slot_h) // 2
        cv2.rectangle(
            canvas, (x, y1_bot), (x + slot_w, y1_bot + slot_h), (220, 220, 220), -1
        )


def _apply_filmstrip(rgb: np.ndarray, params: FrameParams) -> np.ndarray:
    pad = params.thickness
    bg = (18, 18, 20)
    out = cv2.copyMakeBorder(
        rgb, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=bg
    )
    _draw_film_perforations(out, pad, pad, out.shape[1])
    return out


def _apply_drop_shadow(rgb: np.ndarray, pad: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    cw, ch = w + pad * 2, h + pad * 2
    canvas = np.full((ch, cw, 3), 242, dtype=np.uint8)
    shadow = np.zeros((ch, cw), dtype=np.uint8)
    sx, sy = pad + 6, pad + 8
    shadow[sy : sy + h, sx : sx + w] = 160
    blurred_shadow = cv2.GaussianBlur(shadow, (21, 21), 0)
    shadow_3d = np.repeat(blurred_shadow[:, :, None], 3, axis=2).astype(np.float32)
    canvas_f = canvas.astype(np.float32) * (1.0 - (shadow_3d / 255.0 * 0.45))
    canvas = np.clip(canvas_f, 0, 255).astype(np.uint8)
    canvas[pad : pad + h, pad : pad + w] = rgb
    return canvas


def _render_frame(rgb: np.ndarray, p: FrameParams) -> np.ndarray:
    if p.style == "filmstrip":
        return _apply_filmstrip(rgb, p)
    if p.style == "drop_shadow":
        return _apply_drop_shadow(rgb, p.thickness)
    top = p.thickness
    bottom = p.thickness + (p.bottom_extra if p.style == "polaroid" else 0)
    left = right = p.thickness
    c_rgb = (p.color[0], p.color[1], p.color[2])
    bordered = cv2.copyMakeBorder(
        rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=c_rgb
    )
    if p.style == "passepartout" or p.inner_border:
        in_rgb = (p.inner_color[0], p.inner_color[1], p.inner_color[2])
        h_orig = rgb.shape[0]
        w_orig = rgb.shape[1]
        cv2.rectangle(
            bordered, (left - 2, top - 2), (left + w_orig + 1, top + h_orig + 1), in_rgb, 1
        )
    return bordered


@register_operation("frame")
class FrameOperation(ImageOperation[FrameParams]):
    params_model = FrameParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        base, alpha = split_alpha(u8)
        rgb = ensure_rgb(base)
        framed = _render_frame(rgb, self.params)
        if alpha is None:
            return framed
        pad_top = self.params.thickness
        extra = self.params.bottom_extra if self.params.style == "polaroid" else 0
        pad_bot = self.params.thickness + extra
        pad_lr = 0 if self.params.style == "filmstrip" else self.params.thickness
        alpha_framed = cv2.copyMakeBorder(
            alpha, pad_top, pad_bot, pad_lr, pad_lr, cv2.BORDER_CONSTANT, value=255
        )
        return merge_alpha(framed, cast(ImageArray, alpha_framed.astype(np.uint8)))
