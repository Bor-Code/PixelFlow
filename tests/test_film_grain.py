from __future__ import annotations

import numpy as np

from pixelflow.core.registry import default_registry
from pixelflow.operations.film_grain import FilmGrainOperation, FilmGrainParams


def test_film_grain_registered() -> None:
    op_cls = default_registry.get("film_grain")
    assert op_cls is FilmGrainOperation
    alias_cls = default_registry.get("grain")
    assert alias_cls is FilmGrainOperation


def test_film_grain_monochrome() -> None:
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    params = FilmGrainParams(intensity=0.3, grain_size=1.0, colored=False)
    op = FilmGrainOperation(params=params)
    res = op.apply(img)

    assert res.shape == (50, 50, 3)
    assert res.dtype == np.uint8
    assert not np.array_equal(res, img)


def test_film_grain_colored_and_coarse() -> None:
    img = np.full((60, 60, 3), 120, dtype=np.uint8)
    params = FilmGrainParams(intensity=0.25, grain_size=2.0, colored=True, seed=123)
    op = FilmGrainOperation(params=params)
    res = op.apply(img)

    assert res.shape == (60, 60, 3)
    assert not np.array_equal(res, img)


def test_film_grain_rgba_preserves_alpha() -> None:
    img = np.full((30, 30, 4), 100, dtype=np.uint8)
    img[:, :, 3] = 175
    op = FilmGrainOperation(params={"intensity": 0.2})
    res = op.apply(img)

    assert res.shape == (30, 30, 4)
    assert np.all(res[:, :, 3] == 175)


def test_film_grain_grayscale() -> None:
    img = np.full((40, 40), 130, dtype=np.uint8)
    op = FilmGrainOperation(params={"intensity": 0.2, "grain_size": 2.0})
    res = op.apply(img)

    assert res.shape == (40, 40)
    assert res.dtype == np.uint8
    assert not np.array_equal(res, img)


def test_film_grain_zero_intensity() -> None:
    img = np.full((20, 20, 3), 90, dtype=np.uint8)
    op = FilmGrainOperation(params={"intensity": 0.0})
    res = op.apply(img)
    assert np.array_equal(res, img)
