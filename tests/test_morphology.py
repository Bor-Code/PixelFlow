from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pixelflow.core.operation import ImageArray
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import InvalidParametersError
from pixelflow.operations.morphology import MorphologyOperation


def test_morphology_registered() -> None:
    assert "morphology" in default_registry.list_names()


def test_morphology_actions_and_shapes() -> None:
    img = np.zeros((11, 11, 4), dtype=np.uint8)
    img[5, 5, :3] = 255
    dilated = MorphologyOperation(
        {"action": "dilate", "kernel_size": 3, "kernel_shape": "rect"}
    ).apply(img)
    assert dilated[4, 5, 0] == 255
    eroded = MorphologyOperation(
        {"action": "erode", "kernel_size": 3, "kernel_shape": "cross"}
    ).apply(dilated)
    assert eroded[4, 4, 0] == 0
    opened = MorphologyOperation(
        {"action": "open", "kernel_size": 3, "kernel_shape": "ellipse"}
    ).apply(img)
    assert opened[5, 5, 0] == 0
    closed = MorphologyOperation({"action": "close", "kernel_size": 3}).apply(dilated)
    assert closed.shape == (11, 11, 4)
    with pytest.raises(InvalidParametersError):
        MorphologyOperation({"kernel_size": 4})


@settings(max_examples=12, deadline=None)
@given(
    action=st.sampled_from(["erode", "dilate", "open", "close"]),
    k=st.sampled_from([1, 3, 5]),
)
def test_morphology_preserves_shape_and_dtype(action: str, k: int) -> None:
    arr: ImageArray = np.full((12, 12, 3), 100, dtype=np.uint16)
    out = MorphologyOperation({"action": action, "kernel_size": k}).apply(arr)
    assert out.shape == arr.shape and out.dtype == arr.dtype
