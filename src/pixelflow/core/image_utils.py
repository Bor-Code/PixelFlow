from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from pixelflow.core.operation import ImageArray


def to_uint8(image: ImageArray) -> npt.NDArray[np.uint8]:
    if image.dtype == np.uint8:
        return cast(npt.NDArray[np.uint8], image.copy())
    scaled = (image.astype(np.float32) / 257.0).round()
    return cast(npt.NDArray[np.uint8], np.clip(scaled, 0, 255).astype(np.uint8))


def restore_dtype(
    image_u8: npt.NDArray[np.uint8],
    original_dtype: np.dtype[np.uint8] | np.dtype[np.uint16],
) -> ImageArray:
    if original_dtype == np.uint16:
        return cast(ImageArray, image_u8.astype(np.uint16) * np.uint16(257))
    return cast(ImageArray, image_u8.copy())


def squeeze_single_channel(image: ImageArray) -> ImageArray:
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].copy()
    return image.copy()


def split_alpha(image: ImageArray) -> tuple[ImageArray, ImageArray | None]:
    squeezed = squeeze_single_channel(image)
    if squeezed.ndim == 3 and squeezed.shape[2] == 4:
        return squeezed[:, :, :3].copy(), squeezed[:, :, 3].copy()
    return squeezed, None


def merge_alpha(image: ImageArray, alpha: ImageArray | None) -> ImageArray:
    if alpha is None:
        return image
    target_h, target_w = image.shape[0], image.shape[1]
    resized_alpha = alpha
    if alpha.shape[0] != target_h or alpha.shape[1] != target_w:
        resized_alpha = cast(
            ImageArray,
            cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR),
        )
    rgb: ImageArray = np.stack([image, image, image], axis=-1) if image.ndim == 2 else image
    alpha_3d: ImageArray = (
        resized_alpha[:, :, np.newaxis] if resized_alpha.ndim == 2 else resized_alpha
    )
    if alpha_3d.dtype != rgb.dtype:
        alpha_3d = (
            to_uint8(alpha_3d)
            if rgb.dtype == np.uint8
            else restore_dtype(to_uint8(alpha_3d), np.dtype(np.uint16))
        )
    return cast(ImageArray, np.concatenate([rgb, alpha_3d], axis=2))


def ensure_grayscale(image: ImageArray) -> ImageArray:
    base, _ = split_alpha(image)
    if base.ndim == 2:
        return base.copy()
    return cast(ImageArray, cv2.cvtColor(base, cv2.COLOR_RGB2GRAY))


def ensure_rgb(image: ImageArray) -> ImageArray:
    base, _ = split_alpha(image)
    if base.ndim == 2:
        return cast(ImageArray, np.stack([base, base, base], axis=-1))
    return base.copy()
