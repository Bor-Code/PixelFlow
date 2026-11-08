from __future__ import annotations

import numpy as np
import pytest

from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError, PipelineExecutionError
from pixelflow.operations import detection
from pixelflow.operations.detection import FaceBlurOperation


def test_face_blur_registered_and_validation() -> None:
    assert "face_blur" in default_registry.list_names()
    with pytest.raises(InvalidParametersError):
        FaceBlurOperation({"blur_kernel_size": 10})


def test_face_blur_no_faces_and_detected_region(monkeypatch: pytest.MonkeyPatch) -> None:
    img = np.full((60, 60, 3), 120, dtype=np.uint8)
    img[10:40, 10:40] = 250
    out_none = FaceBlurOperation({"min_size": 20}).apply(img)
    assert out_none.shape == (60, 60, 3)

    class FakeCascade:
        def empty(self) -> bool:
            return False

        def detect_faces(
            self, *_args: object, **_kwargs: object
        ) -> list[tuple[int, int, int, int]]:
            return [(8, 8, 30, 30)]

    monkeypatch.setattr(detection, "load_haar_classifier", FakeCascade)
    out_blurred = FaceBlurOperation({"blur_kernel_size": 11}).apply(img)
    assert not np.array_equal(out_blurred[8:38, 8:38], img[8:38, 8:38])


def test_face_blur_empty_classifier_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyCascade:
        def empty(self) -> bool:
            return True

    monkeypatch.setattr(detection, "load_haar_classifier", EmptyCascade)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(PipelineExecutionError):
        FaceBlurOperation().apply(img)
