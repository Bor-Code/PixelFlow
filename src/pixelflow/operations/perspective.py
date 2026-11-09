from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import merge_alpha, split_alpha
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

Points4 = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


class PerspectiveWarpParams(OperationParams):
    src_points: Points4 | None = None
    output_width: int | None = Field(default=None, ge=10, le=8000)
    output_height: int | None = Field(default=None, ge=10, le=8000)
    auto_detect: bool = False


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _auto_detect_corners(gray: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = (gray.shape[0] * gray.shape[1]) * 0.05
    for c in sorted_cnts[:5]:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2).astype(np.float32))
    return None


def _calc_target_dims(pts: np.ndarray, ow: int | None, oh: int | None) -> tuple[int, int]:
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
    w_a = float(np.linalg.norm(br - bl))
    w_b = float(np.linalg.norm(tr - tl))
    h_a = float(np.linalg.norm(tr - br))
    h_b = float(np.linalg.norm(tl - bl))
    tw = ow if ow is not None else max(int(w_a), int(w_b))
    th = oh if oh is not None else max(int(h_a), int(h_b))
    return max(10, tw), max(10, th)


@register_operation("perspective_warp")
class PerspectiveWarpOperation(ImageOperation[PerspectiveWarpParams]):
    params_model = PerspectiveWarpParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
        if self.params.src_points is not None:
            src = np.array(self.params.src_points, dtype=np.float32)
        elif self.params.auto_detect:
            found = _auto_detect_corners(gray)
            if found is None:
                return image.copy()
            src = found
        else:
            return image.copy()
        tw, th = _calc_target_dims(src, self.params.output_width, self.params.output_height)
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped_base = cv2.warpPerspective(base, matrix, (tw, th))
        warped_alpha = (
            cast(ImageArray, cv2.warpPerspective(alpha, matrix, (tw, th)))
            if alpha is not None
            else None
        )
        merged = merge_alpha(cast(ImageArray, warped_base), warped_alpha)
        return cast(ImageArray, merged.astype(image.dtype))
