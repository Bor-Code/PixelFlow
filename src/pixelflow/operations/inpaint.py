from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

InpaintMethod = Literal["telea", "ns"]


class InpaintParams(OperationParams):
    x: int = Field(default=0)
    y: int = Field(default=0)
    width: int = Field(default=50, ge=1)
    height: int = Field(default=50, ge=1)
    inpaint_radius: int = Field(default=3, ge=1, le=50)
    method: InpaintMethod = "telea"
    dilation: int = Field(default=2, ge=0, le=25)


def _build_inpaint_mask(h: int, w: int, params: InpaintParams) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, min(w, params.x))
    y1 = max(0, min(h, params.y))
    x2 = max(0, min(w, params.x + params.width))
    y2 = max(0, min(h, params.y + params.height))
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = 255
    if params.dilation > 0 and np.any(mask):
        k = params.dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(mask, kernel)
        mask = cast(np.ndarray, dilated).astype(np.uint8)
    return mask


def _run_inpaint_core(
    img: np.ndarray, mask: np.ndarray, radius: int, method: InpaintMethod
) -> np.ndarray:
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    return cast(np.ndarray, cv2.inpaint(img, mask, float(radius), flag))


@register_operation("inpaint")
class InpaintOperation(ImageOperation[InpaintParams]):
    params_model = InpaintParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        h, w = u8.shape[:2]
        mask = _build_inpaint_mask(h, w, self.params)
        if not np.any(mask):
            return image.copy()
        if u8.ndim == 2:
            res = _run_inpaint_core(u8, mask, self.params.inpaint_radius, self.params.method)
            return cast(ImageArray, res)
        if u8.shape[2] == 4:
            rgb, a = split_alpha(u8)
            inp = _run_inpaint_core(rgb, mask, self.params.inpaint_radius, self.params.method)
            return merge_alpha(inp, a)
        res = _run_inpaint_core(u8, mask, self.params.inpaint_radius, self.params.method)
        return cast(ImageArray, res)
