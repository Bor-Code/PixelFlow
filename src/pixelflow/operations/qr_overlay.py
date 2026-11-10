from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class QROverlayParams(OperationParams):
    data: str = Field(default="https://pixelflow.dev", min_length=1, max_length=1000)
    position: Literal[
        "top-left", "top-right", "bottom-left", "bottom-right", "center", "custom"
    ] = "bottom-right"
    size: int = Field(default=120, ge=30, le=1000)
    margin: int = Field(default=20, ge=0, le=200)
    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)
    fg_color: str = "#000000"
    bg_color: str = "#ffffff"
    opacity: float = Field(default=0.9, ge=0.0, le=1.0)
    border_modules: int = Field(default=2, ge=0, le=10)


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


def _generate_qr_rgb(params: QROverlayParams) -> np.ndarray:
    encoder_factory_name = "QRCodeEncoder_create"
    create_fn = getattr(cv2, encoder_factory_name)
    encoder = create_fn()
    matrix = cast(np.ndarray, encoder.encode(params.data))
    b = params.border_modules
    padded = cv2.copyMakeBorder(matrix, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
    resized = cv2.resize(
        padded, (params.size, params.size), interpolation=cv2.INTER_NEAREST
    )
    fg = _hex_to_rgb(params.fg_color)
    bg = _hex_to_rgb(params.bg_color)
    qr_rgb = np.empty((params.size, params.size, 3), dtype=np.uint8)
    is_bg = resized == 255
    qr_rgb[is_bg] = bg
    qr_rgb[~is_bg] = fg
    return qr_rgb


def _get_coordinates(
    img_w: int, img_h: int, size: int, margin: int, pos: str, custom_x: int, custom_y: int
) -> tuple[int, int]:
    if pos == "top-left":
        return margin, margin
    if pos == "top-right":
        return max(0, img_w - size - margin), margin
    if pos == "bottom-left":
        return margin, max(0, img_h - size - margin)
    if pos == "center":
        return max(0, (img_w - size) // 2), max(0, (img_h - size) // 2)
    if pos == "custom":
        return custom_x, custom_y
    return max(0, img_w - size - margin), max(0, img_h - size - margin)


def _blend_qr_into_rgb(rgb: np.ndarray, params: QROverlayParams) -> np.ndarray:
    if params.opacity <= 0.0:
        return rgb
    qr_rgb = _generate_qr_rgb(params)
    h, w = rgb.shape[:2]
    x, y = _get_coordinates(
        w, h, params.size, params.margin, params.position, params.x, params.y
    )
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + params.size), min(h, y + params.size)
    if x1 >= x2 or y1 >= y2:
        return rgb
    qr_crop = qr_rgb[: (y2 - y1), : (x2 - x1)]
    bg_crop = rgb[y1:y2, x1:x2].astype(np.float32)
    op = float(params.opacity)
    blended = (1.0 - op) * bg_crop + op * qr_crop.astype(np.float32)
    out = rgb.copy()
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


@register_operation("qr_overlay")
@register_operation("qr_code")
class QROverlayOperation(ImageOperation[QROverlayParams]):
    params_model = QROverlayParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        if self.params.opacity <= 0.0:
            return image.copy()

        if u8.ndim == 2:
            rgb = ensure_rgb(u8)
            res_rgb = _blend_qr_into_rgb(rgb, self.params)
            return cast(ImageArray, cv2.cvtColor(res_rgb, cv2.COLOR_RGB2GRAY))

        alpha: np.ndarray | None = None
        cur: np.ndarray = u8
        if u8.shape[2] == 4:
            cur = u8[:, :, :3]
            alpha = u8[:, :, 3]

        out_rgb = _blend_qr_into_rgb(cur, self.params)
        if alpha is not None:
            return merge_alpha(out_rgb, alpha)
        return cast(ImageArray, out_rgb)
