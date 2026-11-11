from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, restore_dtype, split_alpha, to_uint8
from pixelflow.core.io import load_image
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

WatermarkPosition = Literal[
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "center",
    "custom",
]


class WatermarkParams(OperationParams):
    text: str | None = Field(default=None, min_length=1, max_length=200)
    watermark_path: str | None = None
    position: WatermarkPosition = "bottom-right"
    opacity: float = Field(default=0.5, ge=0.0, le=1.0)
    scale: float = Field(default=0.25, gt=0.0, le=2.0)
    margin: int = Field(default=10, ge=0, le=500)
    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)
    color: tuple[int, int, int] = (255, 255, 255)
    base_dir: str | None = None
    font_style: Literal["sans", "serif", "mono", "bold"] = "sans"
    box_background: bool = Field(default=False)
    box_color: tuple[int, int, int] = (0, 0, 0)

    @model_validator(mode="after")
    def validate_source(self) -> WatermarkParams:
        if not self.text and not self.watermark_path:
            raise ValueError("Either 'text' or 'watermark_path' must be provided")
        return self


@register_operation("watermark")
class WatermarkOperation(ImageOperation[WatermarkParams]):
    params_model = WatermarkParams

    def _execute(self, image: ImageArray) -> ImageArray:
        _, orig_alpha = split_alpha(image)
        rgb_u8 = to_uint8(ensure_rgb(image))
        overlay_rgba = self._build_watermark_rgba(rgb_u8.shape[1], rgb_u8.shape[0])
        ox, oy = self._resolve_coordinates(
            rgb_u8.shape[1],
            rgb_u8.shape[0],
            overlay_rgba.shape[1],
            overlay_rgba.shape[0],
        )
        blended_u8 = self._blend_patch(rgb_u8, overlay_rgba, ox, oy)
        if image.ndim == 2:
            gray_u8 = cv2.cvtColor(blended_u8, cv2.COLOR_RGB2GRAY)
            return restore_dtype(gray_u8, image.dtype)
        restored = restore_dtype(blended_u8, image.dtype)
        return merge_alpha(restored, orig_alpha)

    def _build_watermark_rgba(self, target_w: int, target_h: int) -> npt.NDArray[np.uint8]:
        if self.params.watermark_path:
            return self._load_image_watermark(target_w, target_h)
        return self._render_text_watermark(target_w)

    def _load_image_watermark(self, target_w: int, target_h: int) -> npt.NDArray[np.uint8]:
        assert self.params.watermark_path is not None
        raw_path = Path(self.params.watermark_path)
        if not raw_path.is_absolute() and self.params.base_dir:
            raw_path = Path(self.params.base_dir) / raw_path
        wm_img = load_image(raw_path)
        wm_rgb = to_uint8(ensure_rgb(wm_img))
        _, wm_alpha = split_alpha(wm_img)
        alpha_u8 = (
            to_uint8(wm_alpha)
            if wm_alpha is not None
            else np.full((wm_img.shape[0], wm_img.shape[1]), 255, dtype=np.uint8)
        )
        rgba = np.dstack([wm_rgb, alpha_u8])
        new_w = max(1, min(target_w, round(target_w * self.params.scale)))
        aspect = rgba.shape[0] / max(1, rgba.shape[1])
        new_h = max(1, min(target_h, round(new_w * aspect)))
        return cast(
            npt.NDArray[np.uint8],
            cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA),
        )

    @staticmethod
    def _select_font(style: str) -> int:
        fonts = {
            "serif": cv2.FONT_HERSHEY_COMPLEX,
            "mono": cv2.FONT_HERSHEY_PLAIN,
            "bold": cv2.FONT_HERSHEY_DUPLEX,
        }
        return fonts.get(style, cv2.FONT_HERSHEY_SIMPLEX)

    def _render_text_watermark(self, target_w: int) -> npt.NDArray[np.uint8]:
        text = self.params.text or "Pixelflow"
        font = self._select_font(self.params.font_style)
        font_scale = max(0.4, (target_w * self.params.scale) / max(60.0, len(text) * 18.0))
        thickness = max(1, round(font_scale * 2.0))
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 6
        h_box, w_box = max(1, th + baseline + pad * 2), max(1, tw + pad * 2)
        canvas = np.zeros((h_box, w_box, 4), dtype=np.uint8)
        if self.params.box_background:
            canvas[:, :, 0] = self.params.box_color[0]
            canvas[:, :, 1] = self.params.box_color[1]
            canvas[:, :, 2] = self.params.box_color[2]
            canvas[:, :, 3] = 180
        color_rgba = (
            int(self.params.color[0]),
            int(self.params.color[1]),
            int(self.params.color[2]),
            255,
        )
        cv2.putText(
            canvas,
            text,
            (pad, pad + th),
            font,
            font_scale,
            color_rgba,
            thickness,
            cv2.LINE_AA,
        )
        return canvas

    def _resolve_coordinates(self, img_w: int, img_h: int, wm_w: int, wm_h: int) -> tuple[int, int]:
        m = self.params.margin
        pos = self.params.position
        if pos == "top-left":
            return m, m
        if pos == "top-right":
            return max(0, img_w - wm_w - m), m
        if pos == "bottom-left":
            return m, max(0, img_h - wm_h - m)
        if pos == "bottom-right":
            return max(0, img_w - wm_w - m), max(0, img_h - wm_h - m)
        if pos == "center":
            return max(0, (img_w - wm_w) // 2), max(0, (img_h - wm_h) // 2)
        return self.params.x, self.params.y

    def _blend_patch(
        self,
        base_rgb: npt.NDArray[np.uint8],
        wm_rgba: npt.NDArray[np.uint8],
        ox: int,
        oy: int,
    ) -> npt.NDArray[np.uint8]:
        out = base_rgb.copy()
        img_h, img_w = out.shape[0], out.shape[1]
        if ox >= img_w or oy >= img_h:
            return out
        x2 = min(img_w, ox + wm_rgba.shape[1])
        y2 = min(img_h, oy + wm_rgba.shape[0])
        patch_w, patch_h = x2 - ox, y2 - oy
        if patch_w <= 0 or patch_h <= 0:
            return out
        wm_crop = wm_rgba[:patch_h, :patch_w]
        alpha = (wm_crop[:, :, 3:4].astype(np.float32) / 255.0) * self.params.opacity
        roi = out[oy:y2, ox:x2].astype(np.float32)
        blended = roi * (1.0 - alpha) + wm_crop[:, :, :3].astype(np.float32) * alpha
        out[oy:y2, ox:x2] = np.clip(blended, 0, 255).round().astype(np.uint8)
        return out
