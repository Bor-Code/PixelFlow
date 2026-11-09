from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field, model_validator

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.io import load_image
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

RedactMode = Literal["solid", "blur", "pixelate"]


class RedactParams(OperationParams):
    x: int = Field(default=0)
    y: int = Field(default=0)
    width: int = Field(default=100, ge=1)
    height: int = Field(default=40, ge=1)
    mode: RedactMode = "solid"
    fill_color: tuple[int, int, int] = (0, 0, 0)
    blur_ksize: int = Field(default=25, ge=3, le=101)
    pixel_size: int = Field(default=10, ge=2, le=100)


def _apply_redact_patch(patch: np.ndarray, params: RedactParams) -> np.ndarray:
    if params.mode == "solid":
        res = patch.copy()
        if patch.ndim == 2:
            res[:] = params.fill_color[0]
        elif patch.shape[2] == 4:
            res[:, :, :3] = params.fill_color
            res[:, :, 3] = 255
        else:
            res[:] = params.fill_color
        return res
    if params.mode == "blur":
        k = params.blur_ksize if params.blur_ksize % 2 == 1 else params.blur_ksize + 1
        return cast(np.ndarray, cv2.GaussianBlur(patch, (k, k), 0))
    h, w = patch.shape[:2]
    p = max(2, min(params.pixel_size, min(h, w)))
    sw, sh = max(1, w // p), max(1, h // p)
    small = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return cast(np.ndarray, cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))


@register_operation("redact")
class RedactOperation(ImageOperation[RedactParams]):
    params_model = RedactParams

    def _execute(self, image: ImageArray) -> ImageArray:
        h, w = image.shape[:2]
        x1 = max(0, min(w, self.params.x))
        y1 = max(0, min(h, self.params.y))
        x2 = max(0, min(w, self.params.x + self.params.width))
        y2 = max(0, min(h, self.params.y + self.params.height))
        if x1 >= x2 or y1 >= y2:
            return image.copy()
        out = image.copy()
        out[y1:y2, x1:x2] = _apply_redact_patch(out[y1:y2, x1:x2], self.params)
        return out


class StampParams(OperationParams):
    text: str | None = Field(default=None, max_length=100)
    stamp_path: str | None = None
    x: int = 0
    y: int = 0
    scale: float = Field(default=1.0, gt=0.0, le=5.0)
    angle: float = Field(default=-12.0, ge=-180.0, le=180.0)
    opacity: float = Field(default=0.85, ge=0.0, le=1.0)
    color: tuple[int, int, int] = (200, 30, 30)
    box_border: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_source(self) -> StampParams:
        if not self.text and not self.stamp_path:
            raise ValueError("Either 'text' or 'stamp_path' must be provided")
        return self


def _build_text_stamp(params: StampParams) -> np.ndarray:
    text = params.text or ""
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0 * params.scale
    thickness = max(1, int(2 * params.scale))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = int(16 * params.scale)
    bw, bh = tw + pad * 2, th + baseline + pad * 2
    stamp = np.zeros((bh, bw, 4), dtype=np.uint8)
    color_bgr = (params.color[2], params.color[1], params.color[0])
    color_bgra = (*color_bgr, 255)
    if params.box_border:
        b_thick = max(2, int(3 * params.scale))
        cv2.rectangle(stamp, (4, 4), (bw - 5, bh - 5), color_bgra, b_thick)
        cv2.rectangle(stamp, (8, 8), (bw - 9, bh - 9), color_bgra, max(1, b_thick // 2))
    tx = pad
    ty = pad + th
    cv2.putText(stamp, text, (tx, ty), font, font_scale, color_bgra, thickness, cv2.LINE_AA)
    return stamp


def _rotate_stamp(stamp: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return stamp
    h, w = stamp.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    mat[0, 2] += (nw / 2.0) - center[0]
    mat[1, 2] += (nh / 2.0) - center[1]
    return cast(
        np.ndarray,
        cv2.warpAffine(
            stamp, mat, (nw, nh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        ),
    )


def _blend_stamp(
    base_rgb: np.ndarray, stamp_rgba: np.ndarray, x: int, y: int, opacity: float
) -> np.ndarray:
    bh, bw = base_rgb.shape[:2]
    sh, sw = stamp_rgba.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + sw), min(bh, y + sh)
    if x1 >= x2 or y1 >= y2:
        return base_rgb
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    stamp_crop = stamp_rgba[sy1:sy2, sx1:sx2]
    alpha = (stamp_crop[:, :, 3].astype(np.float32) / 255.0) * opacity
    alpha = np.expand_dims(alpha, axis=-1)
    color = stamp_crop[:, :, :3].astype(np.float32)
    roi = base_rgb[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - alpha) + color * alpha
    base_rgb[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return base_rgb


@register_operation("stamp")
class StampOperation(ImageOperation[StampParams]):
    params_model = StampParams

    def _execute(self, image: ImageArray) -> ImageArray:
        if self.params.stamp_path:
            raw = load_image(self.params.stamp_path)
            raw_rgb = to_uint8(ensure_rgb(raw))
            if raw.ndim == 3 and raw.shape[2] == 4:
                stamp_rgba: np.ndarray = raw.astype(np.uint8)
            else:
                stamp_rgba = cast(np.ndarray, cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2RGBA))
        else:
            stamp_rgba = _build_text_stamp(self.params)
        rotated = _rotate_stamp(stamp_rgba, self.params.angle)
        base, alpha = split_alpha(image)
        rgb_u8 = to_uint8(ensure_rgb(base))
        blended = _blend_stamp(rgb_u8, rotated, self.params.x, self.params.y, self.params.opacity)
        out_base = cv2.cvtColor(blended, cv2.COLOR_RGB2GRAY) if base.ndim == 2 else blended
        merged = merge_alpha(cast(ImageArray, out_base), alpha)
        return cast(ImageArray, merged.astype(image.dtype))
