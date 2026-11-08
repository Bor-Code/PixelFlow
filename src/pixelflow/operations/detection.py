from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator

from pixelflow.core.image_utils import ensure_grayscale, merge_alpha, split_alpha, to_uint8
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation
from pixelflow.exceptions import PipelineExecutionError


class _FallbackCascade:
    def __init__(self, xml_path: str) -> None:
        self._valid = bool(xml_path)

    def empty(self) -> bool:
        return not self._valid

    def detect_faces(
        self,
        gray_u8: npt.NDArray[np.uint8],
        scale_factor: float,
        min_neighbors: int,
        min_size: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        _ = (gray_u8, scale_factor, min_neighbors, min_size)
        return []


def load_haar_classifier() -> object:
    cascade_dir = getattr(cv2, "data", None)
    base_path = Path(cascade_dir.haarcascades) if cascade_dir is not None else Path("data")
    xml_path = base_path / "haarcascade_frontalface_default.xml"
    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    if cascade_cls is not None:
        return cascade_cls(str(xml_path))
    return _FallbackCascade(str(xml_path))


class FaceBlurParams(OperationParams):
    scale_factor: float = Field(default=1.1, gt=1.0, le=2.5)
    min_neighbors: int = Field(default=5, ge=1, le=20)
    min_size: int = Field(default=24, ge=10, le=1000)
    blur_kernel_size: int = Field(default=31, ge=3, le=199)

    @field_validator("blur_kernel_size")
    @classmethod
    def check_odd_kernel(cls, value: int) -> int:
        if value < 3 or value % 2 == 0:
            raise ValueError("blur_kernel_size must be an odd integer >= 3")
        return value


@register_operation("face_blur")
class FaceBlurOperation(ImageOperation[FaceBlurParams]):
    params_model = FaceBlurParams

    def _execute(self, image: ImageArray) -> ImageArray:
        base, alpha = split_alpha(image)
        boxes = self._detect_faces(base)
        blurred = self._blur_regions(base, boxes)
        return merge_alpha(blurred, alpha)

    def _detect_faces(self, image: ImageArray) -> list[tuple[int, int, int, int]]:
        classifier = load_haar_classifier()
        is_empty = getattr(classifier, "empty", lambda: True)
        if bool(is_empty()):
            raise PipelineExecutionError("Failed to load Haar cascade classifier")
        gray_u8 = to_uint8(ensure_grayscale(image))
        min_dim = (self.params.min_size, self.params.min_size)
        detect_fn = getattr(classifier, "detectMultiScale", None) or getattr(
            classifier, "detect_faces", None
        )
        assert callable(detect_fn)
        detections = detect_fn(
            gray_u8,
            self.params.scale_factor,
            self.params.min_neighbors,
            min_dim,
        )
        if len(detections) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]

    def _blur_regions(
        self,
        image: ImageArray,
        boxes: list[tuple[int, int, int, int]],
    ) -> ImageArray:
        output = image.copy()
        k = self.params.blur_kernel_size
        for x, y, w, h in boxes:
            roi = output[y : y + h, x : x + w]
            if roi.size > 0:
                output[y : y + h, x : x + w] = cast(
                    ImageArray,
                    cv2.GaussianBlur(roi, (k, k), 0),
                )
        return output
