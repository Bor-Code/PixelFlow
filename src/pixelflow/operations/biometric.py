from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

BiometricPreset = Literal["icao_35x45", "us_visa_50x50", "standard_50x60"]
BiometricLayout = Literal["4_up", "6_up", "8_up", "single"]

PRESET_DIMENSIONS_MM: dict[BiometricPreset, tuple[float, float]] = {
    "icao_35x45": (35.0, 45.0),
    "us_visa_50x50": (50.0, 50.0),
    "standard_50x60": (50.0, 60.0),
}


class BiometricSheetParams(OperationParams):
    preset: BiometricPreset = "icao_35x45"
    layout: BiometricLayout = "4_up"
    dpi: int = Field(default=300, ge=72, le=600)
    sheet_width_mm: float = 150.0
    sheet_height_mm: float = 100.0
    draw_cut_lines: bool = True
    background_color: tuple[int, int, int] = (255, 255, 255)


def _crop_and_scale_photo(
    rgb: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = round(h * target_ratio)
        start_x = (w - new_w) // 2
        cropped = rgb[:, start_x : start_x + new_w]
    else:
        new_h = round(w / target_ratio)
        start_y = (h - new_h) // 2
        cropped = rgb[start_y : start_y + new_h, :]
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return cast(np.ndarray, resized)


def _get_grid_dims(layout: BiometricLayout) -> tuple[int, int]:
    if layout == "6_up":
        return 2, 3
    if layout == "8_up":
        return 2, 4
    return 2, 2


def _tile_photos_on_sheet(
    sheet: np.ndarray,
    photo: np.ndarray,
    layout: BiometricLayout,
    draw_cut_lines: bool,
) -> np.ndarray:
    rows, cols = _get_grid_dims(layout)
    sh_h, sh_w = sheet.shape[:2]
    ph_h, ph_w = photo.shape[:2]
    gap_x = max(10, (sh_w - cols * ph_w) // (cols + 1))
    gap_y = max(10, (sh_h - rows * ph_h) // (rows + 1))
    for r in range(rows):
        for c in range(cols):
            x = gap_x + c * (ph_w + gap_x)
            y = gap_y + r * (ph_h + gap_y)
            sheet[y : y + ph_h, x : x + ph_w] = photo
            if draw_cut_lines:
                cv2.rectangle(sheet, (x - 1, y - 1), (x + ph_w, y + ph_h), (210, 210, 210), 1)
    return sheet


@register_operation("biometric_sheet")
class BiometricSheetOperation(ImageOperation[BiometricSheetParams]):
    operation_name = "biometric_sheet"
    params_model = BiometricSheetParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        rgb = ensure_rgb(u8)
        mm_to_px = self.params.dpi / 25.4
        pw_mm, ph_mm = PRESET_DIMENSIONS_MM[self.params.preset]
        target_pw = max(10, round(pw_mm * mm_to_px))
        target_ph = max(10, round(ph_mm * mm_to_px))
        photo = _crop_and_scale_photo(rgb, target_pw, target_ph)
        if self.params.layout == "single":
            return cast(ImageArray, photo)
        rows, cols = _get_grid_dims(self.params.layout)
        min_w = cols * target_pw + (cols + 1) * 10
        min_h = rows * target_ph + (rows + 1) * 10
        sheet_w = max(min_w, round(self.params.sheet_width_mm * mm_to_px))
        sheet_h = max(min_h, round(self.params.sheet_height_mm * mm_to_px))
        sheet = np.full((sheet_h, sheet_w, 3), self.params.background_color, dtype=np.uint8)
        result = _tile_photos_on_sheet(
            sheet, photo, self.params.layout, self.params.draw_cut_lines
        )
        return cast(ImageArray, result)
