from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from pydantic import Field

from pixelflow.core.image_utils import ensure_rgb, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation

MarkupKind = Literal["badge", "arrow", "box"]


class MarkupParams(OperationParams):
    kind: MarkupKind = "badge"
    x1: int = 50
    y1: int = 50
    x2: int = 150
    y2: int = 150
    color: tuple[int, int, int] = (239, 68, 68)
    stroke_width: int = Field(default=3, ge=1, le=20)
    text: str = "REVİZYON 1"
    fill_opacity: float = Field(default=0.0, ge=0.0, le=1.0)


def _draw_markup_badge(
    canvas: np.ndarray,
    params: MarkupParams,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = max(1, params.stroke_width // 2)
    (tw, th), _ = cv2.getTextSize(params.text, font, scale, thick)
    pad = 8
    bx1 = params.x1
    by1 = params.y1
    bx2 = bx1 + tw + pad * 2
    by2 = by1 + th + pad * 2
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), params.color, -1)
    tx = bx1 + pad
    ty = by1 + th + pad
    cv2.putText(canvas, params.text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _draw_markup_arrow(
    canvas: np.ndarray,
    params: MarkupParams,
) -> None:
    pt1 = (params.x1, params.y1)
    pt2 = (params.x2, params.y2)
    cv2.arrowedLine(
        canvas,
        pt1,
        pt2,
        params.color,
        params.stroke_width,
        tipLength=0.2,
        line_type=cv2.LINE_AA,
    )


def _draw_markup_box(
    canvas: np.ndarray,
    params: MarkupParams,
) -> None:
    pt1 = (min(params.x1, params.x2), min(params.y1, params.y2))
    pt2 = (max(params.x1, params.x2), max(params.y1, params.y2))
    if params.fill_opacity > 0.0:
        overlay = canvas.copy()
        cv2.rectangle(overlay, pt1, pt2, params.color, -1)
        blended = cv2.addWeighted(
            overlay, params.fill_opacity, canvas, 1.0 - params.fill_opacity, 0.0
        )
        canvas[:, :] = blended[:, :]
    cv2.rectangle(canvas, pt1, pt2, params.color, params.stroke_width, lineType=cv2.LINE_AA)


@register_operation("markup")
class MarkupOperation(ImageOperation[MarkupParams]):
    operation_name = "markup"
    params_model = MarkupParams

    def _execute(self, image: ImageArray) -> ImageArray:
        u8 = to_uint8(image)
        base, alpha = split_alpha(u8)
        rgb = ensure_rgb(base)
        canvas = rgb.copy()
        if self.params.kind == "badge":
            _draw_markup_badge(canvas, self.params)
        elif self.params.kind == "arrow":
            _draw_markup_arrow(canvas, self.params)
        else:
            _draw_markup_box(canvas, self.params)
        if alpha is not None:
            return merge_alpha(canvas, alpha)
        return canvas
