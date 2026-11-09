from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from pixelflow.core.operation import ImageArray, validate_image_array
from pixelflow.exceptions import InvalidImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False

DEFAULT_MAX_PIXELS = 64_000_000
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
)
FORMAT_TO_MIME: dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "PDF": "application/pdf",
}
MIME_TO_FORMAT: dict[str, str] = {v: k for k, v in FORMAT_TO_MIME.items()}


def load_image(path: str | Path, max_pixels: int = DEFAULT_MAX_PIXELS) -> ImageArray:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise InvalidImageError(f"Image file not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported image extension: '{file_path.suffix}'")
    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        raise InvalidImageError(f"Failed to read image file: {file_path}") from exc
    return decode_image_bytes(raw_bytes, max_pixels=max_pixels)


def decode_image_bytes(data: bytes, max_pixels: int = DEFAULT_MAX_PIXELS) -> ImageArray:
    if not data:
        raise InvalidImageError("Image byte payload is empty")
    if max_pixels <= 0:
        raise InvalidImageError("max_pixels must be positive")
    pil_image = _open_and_verify_pil(data, max_pixels=max_pixels)
    transposed = _apply_exif_orientation(pil_image)
    array = _pil_to_numpy(transposed)
    validate_image_array(array)
    return array


def _open_and_verify_pil(data: bytes, max_pixels: int) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in FORMAT_TO_MIME:
                raise InvalidImageError(f"Unsupported image format: '{fmt}'")
            width, height = probe.size
            if width <= 0 or height <= 0:
                raise InvalidImageError("Image has zero width or height")
            if width * height > max_pixels:
                raise InvalidImageError(
                    f"Image pixel count {width * height} exceeds limit {max_pixels}"
                )
            probe.verify()
        reopened = Image.open(BytesIO(data))
        reopened.load()
        return reopened
    except InvalidImageError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise InvalidImageError(f"Corrupted or invalid image data: {exc}") from exc


def _apply_exif_orientation(pil_image: Image.Image) -> Image.Image:
    try:
        transposed = ImageOps.exif_transpose(pil_image)
        return transposed if transposed is not None else pil_image
    except Exception:
        return pil_image


def _pil_to_numpy(pil_image: Image.Image) -> ImageArray:
    mode = pil_image.mode
    if mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.array(pil_image)
        if arr.dtype == np.int32 or arr.dtype == np.uint32:
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
        return arr.astype(np.uint16)
    if mode == "L":
        return np.array(pil_image, dtype=np.uint8)
    if mode in ("RGBA", "LA"):
        return np.array(pil_image.convert("RGBA"), dtype=np.uint8)
    return np.array(pil_image.convert("RGB"), dtype=np.uint8)


def save_image(image: ImageArray, path: str | Path) -> Path:
    validate_image_array(image)
    out_path = Path(path)
    ext = out_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported output extension: '{ext}'")
    fmt = _extension_to_format(ext)
    encoded = encode_image_bytes(image, image_format=fmt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encoded)
    return out_path


def encode_image_bytes(image: ImageArray, image_format: str = "PNG") -> bytes:
    validate_image_array(image)
    fmt = image_format.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in FORMAT_TO_MIME:
        raise InvalidImageError(f"Unsupported output image format: '{image_format}'")
    pil_img = _numpy_to_pil(image, target_format=fmt)
    buffer = BytesIO()
    pil_img.save(buffer, format=fmt)
    return buffer.getvalue()


def encode_multipage_pdf(images: list[ImageArray]) -> bytes:
    if not images:
        raise InvalidImageError("No images provided for multipage PDF")
    for img in images:
        validate_image_array(img)
    pil_images = [_numpy_to_pil(img, target_format="PDF") for img in images]
    buffer = BytesIO()
    first = pil_images[0]
    rest = pil_images[1:] if len(pil_images) > 1 else []
    first.save(buffer, format="PDF", save_all=True, append_images=rest)
    return buffer.getvalue()


def _numpy_to_pil(image: ImageArray, target_format: str) -> Image.Image:
    if image.dtype == np.uint16:
        if image.ndim == 2 and target_format in ("PNG", "TIFF"):
            return Image.fromarray(image)
        scaled = np.clip((image.astype(np.float32) / 257.0).round(), 0, 255).astype(np.uint8)
        return _numpy_to_pil(scaled, target_format=target_format)
    if image.ndim == 2:
        return Image.fromarray(image)
    if image.shape[2] == 1:
        return Image.fromarray(image[:, :, 0])
    if image.shape[2] == 4:
        if target_format in ("JPEG", "BMP"):
            return Image.fromarray(image[:, :, :3])
        return Image.fromarray(image)
    return Image.fromarray(image)


def _extension_to_format(ext: str) -> str:
    mapping = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".pdf": "PDF",
    }
    return mapping[ext]
