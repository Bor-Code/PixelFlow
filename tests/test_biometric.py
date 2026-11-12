from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.biometric import (
    BiometricSheetOperation,
    BiometricSheetParams,
)


def test_biometric_single_photo_crop() -> None:
    img = np.full((200, 100, 3), 180, dtype=np.uint8)
    params = BiometricSheetParams(preset="icao_35x45", layout="single", dpi=100)
    op = BiometricSheetOperation(params)
    res = op.apply(img)
    h, w = res.shape[:2]
    assert abs((w / h) - (35 / 45)) < 0.05


def test_biometric_4_up_sheet_dimensions() -> None:
    img = np.full((120, 120, 3), 150, dtype=np.uint8)
    params = BiometricSheetParams(preset="icao_35x45", layout="4_up", dpi=100)
    op = BiometricSheetOperation(params)
    res = op.apply(img)
    assert res.shape == (394, 591, 3)


def test_biometric_us_visa_preset() -> None:
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    params = BiometricSheetParams(preset="us_visa_50x50", layout="6_up", dpi=72)
    op = BiometricSheetOperation(params)
    res = op.apply(img)
    assert res.ndim == 3
    assert res.shape[2] == 3


def test_biometric_8_up_layout() -> None:
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    params = BiometricSheetParams(preset="standard_50x60", layout="8_up", dpi=72)
    op = BiometricSheetOperation(params)
    res = op.apply(img)
    assert res.shape[0] > 0
    assert res.shape[1] > 0


def test_biometric_registered() -> None:
    op_cls = default_registry.get("biometric_sheet")
    assert op_cls is BiometricSheetOperation
