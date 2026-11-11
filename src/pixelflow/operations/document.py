from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field, field_validator

from pixelflow.core.image_utils import merge_alpha, split_alpha
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class ShadowRemovalParams(OperationParams):
    kernel_size: int = Field(default=25, ge=3, le=101)
    blur_size: int = Field(default=21, ge=3, le=101)

    @field_validator("kernel_size", "blur_size")
    @classmethod
    def validate_odd_sizes(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError(f"Value must be odd, got {value}")
        return value


class DeskewParams(OperationParams):
    max_angle: float = Field(default=45.0, ge=1.0, le=89.0)
    fill_value: int = Field(default=255, ge=0, le=255)
    expand: bool = Field(default=True)


def _remove_channel_shadow(channel: np.ndarray, kernel_size: int, blur_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    bg_dilated = cv2.dilate(channel, kernel)
    bg = cv2.medianBlur(bg_dilated, blur_size)
    diff = 255 - cv2.absdiff(channel, bg)
    norm = np.empty_like(diff)
    cv2.normalize(diff, norm, 0, 255, cv2.NORM_MINMAX)
    return cast(np.ndarray, norm)


@register_operation("shadow_removal")
class ShadowRemovalOperation(ImageOperation[ShadowRemovalParams]):
    params_model = ShadowRemovalParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        k = self.params.kernel_size
        b = self.params.blur_size
        if base.ndim == 2:
            cleaned = _remove_channel_shadow(base, k, b)
        else:
            channels = [_remove_channel_shadow(base[:, :, i], k, b) for i in range(base.shape[2])]
            cleaned = np.dstack(channels)
        merged = merge_alpha(cleaned, alpha)
        return cast(ImageArray, merged.astype(image.dtype))


def _detect_skew_angle(gray: np.ndarray, max_angle: float) -> float:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(thresh)
    if pts is None or len(pts) < 30:
        return 0.0
    rect = cv2.minAreaRect(pts)
    raw_angle = float(rect[2])
    if raw_angle < -45.0:
        angle = -(90.0 + raw_angle)
    elif raw_angle > 45.0:
        angle = 90.0 - raw_angle
    else:
        angle = -raw_angle
    return angle if abs(angle) <= max_angle else 0.0


def _rotate_bound(image: np.ndarray, angle: float, fill_value: int) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    matrix[0, 2] += (nw / 2.0) - center[0]
    matrix[1, 2] += (nh / 2.0) - center[1]
    border_val = (fill_value,) * (image.shape[2] if image.ndim == 3 else 1)
    return cast(
        np.ndarray,
        cv2.warpAffine(
            image, matrix, (nw, nh), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val
        ),
    )


@register_operation("deskew")
class DeskewOperation(ImageOperation[DeskewParams]):
    params_model = DeskewParams

    def _execute(self, image: ImageArray) -> ImageArray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.ndim == 3 and gray.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        angle = _detect_skew_angle(gray, self.params.max_angle)
        if abs(angle) < 0.2:
            return image.copy()
        rotated = _rotate_bound(image, angle, self.params.fill_value)
        return cast(ImageArray, rotated.astype(image.dtype))


def _order_corners(pts: np.ndarray) -> list[list[int]]:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=int)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered.tolist()


def detect_document_corners(image: np.ndarray) -> list[list[int]]:
    h, w = image.shape[:2]
    fallback = [[0, 0], [w, 0], [w, h], [0, h]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.ndim == 3 and gray.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = (h * w) * 0.05
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            return _order_corners(pts)
    return fallback


class MagicColorParams(OperationParams):
    contrast: float = Field(default=1.2, ge=0.5, le=3.0)
    brightness: int = Field(default=10, ge=-100, le=100)
    sharpness: float = Field(default=1.2, ge=0.0, le=5.0)
    saturation_boost: float = Field(default=1.25, ge=0.5, le=3.0)


def _enhance_magic_channel(ch: np.ndarray, contrast: float, brightness: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
    divided = cv2.divide(ch, bg, scale=255)
    adjusted = cv2.convertScaleAbs(divided, alpha=contrast, beta=brightness)
    return cast(np.ndarray, adjusted)


def _apply_magic_color(base: np.ndarray, params: MagicColorParams) -> np.ndarray:
    if base.ndim == 2:
        return _enhance_magic_channel(base, params.contrast, params.brightness)
    channels = [
        _enhance_magic_channel(base[:, :, i], params.contrast, params.brightness) for i in range(3)
    ]
    merged = np.dstack(channels)
    hsv = cv2.cvtColor(merged, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params.saturation_boost, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if params.sharpness > 0.0:
        blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, 1.0 + params.sharpness, blur, -params.sharpness, 0)
    return cast(np.ndarray, enhanced)


@register_operation("magic_color")
class MagicColorOperation(ImageOperation[MagicColorParams]):
    params_model = MagicColorParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        processed = _apply_magic_color(base, self.params)
        merged = merge_alpha(processed, alpha)
        return cast(ImageArray, merged.astype(image.dtype))


def compose_id_card(
    front: np.ndarray,
    back: np.ndarray,
    layout: str = "vertical",
    card_width: int = 800,
    card_height: int = 504,
    margin: int = 40,
) -> np.ndarray:
    f_bgr = (
        cv2.cvtColor(front, cv2.COLOR_BGRA2BGR)
        if front.ndim == 3 and front.shape[2] == 4
        else front
    )
    b_bgr = (
        cv2.cvtColor(back, cv2.COLOR_BGRA2BGR) if back.ndim == 3 and back.shape[2] == 4 else back
    )
    if f_bgr.ndim == 2:
        f_bgr = cv2.cvtColor(f_bgr, cv2.COLOR_GRAY2BGR)
    if b_bgr.ndim == 2:
        b_bgr = cv2.cvtColor(b_bgr, cv2.COLOR_GRAY2BGR)
    f_res = cv2.resize(f_bgr, (card_width, card_height), interpolation=cv2.INTER_AREA)
    b_res = cv2.resize(b_bgr, (card_width, card_height), interpolation=cv2.INTER_AREA)
    if layout == "horizontal":
        cw = (card_width * 2) + (margin * 3)
        ch = card_height + (margin * 2)
        canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
        canvas[margin : margin + card_height, margin : margin + card_width] = f_res
        x2 = (margin * 2) + card_width
        canvas[margin : margin + card_height, x2 : x2 + card_width] = b_res
        return canvas
    cw = card_width + (margin * 2)
    ch = (card_height * 2) + (margin * 3)
    canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
    canvas[margin : margin + card_height, margin : margin + card_width] = f_res
    y2 = (margin * 2) + card_height
    canvas[y2 : y2 + card_height, margin : margin + card_width] = b_res
    return canvas
