from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image
from pydantic import Field

from pixelflow.core.image_utils import (
    ensure_grayscale,
    ensure_rgb,
    merge_alpha,
    restore_dtype,
    split_alpha,
    squeeze_single_channel,
    to_uint8,
)
from pixelflow.core.io import (
    decode_image_bytes,
    encode_image_bytes,
    load_image,
    save_image,
)
from pixelflow.core.logging import StructuredJsonFormatter, get_structured_logger, log_structured
from pixelflow.core.operation import (
    ImageArray,
    ImageOperation,
    OperationParams,
    validate_image_array,
)
from pixelflow.core.pipeline import Pipeline
from pixelflow.core.registry import OperationRegistry, register_operation
from pixelflow.exceptions import (
    InvalidImageError,
    InvalidParametersError,
    PipelineExecutionError,
    UnknownOperationError,
)


class DummyParams(OperationParams):
    delta: int = Field(default=10, ge=0, le=100)


class DummyAddOperation(ImageOperation[DummyParams]):
    operation_name = "dummy_add"
    params_model = DummyParams

    def _execute(self, image: ImageArray) -> ImageArray:
        return np.clip(image.astype(np.int32) + self.params.delta, 0, 255).astype(image.dtype)


class FailingOperation(ImageOperation[OperationParams]):
    operation_name = "failing_op"
    params_model = OperationParams

    def _execute(self, image: ImageArray) -> ImageArray:
        raise ValueError("boom")


def test_operation_immutability_and_execution() -> None:
    original = np.full((8, 8, 3), 20, dtype=np.uint8)
    snapshot = original.copy()
    op = DummyAddOperation({"delta": 15})
    result = op(original)
    assert np.array_equal(original, snapshot)
    assert np.all(result == 35)
    op_from_model = DummyAddOperation(DummyParams(delta=5))
    assert np.all(op_from_model.apply(original) == 25)


def test_operation_invalid_parameters() -> None:
    with pytest.raises(InvalidParametersError) as exc_info:
        DummyAddOperation({"delta": 999}, line=12)
    assert exc_info.value.line == 12
    with pytest.raises(InvalidParametersError):
        bad_params = cast(dict[str, object], "not-a-dict")
        DummyAddOperation(bad_params)


def test_validate_image_array_errors() -> None:
    with pytest.raises(InvalidImageError):
        validate_image_array("not_an_array")
    with pytest.raises(InvalidImageError):
        validate_image_array(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(InvalidImageError):
        validate_image_array(np.zeros((4,), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        validate_image_array(np.zeros((0, 4, 3), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        validate_image_array(np.zeros((4, 4, 2), dtype=np.uint8))


def test_registry_register_describe_and_errors() -> None:
    reg = OperationRegistry()
    with pytest.raises(ValueError):
        reg.register("   ")

    @reg.register("Custom_Add")
    class CustomOp(DummyAddOperation):
        pass

    assert reg.list_names() == ["custom_add"]
    instance = reg.create("custom_add", {"delta": 5})
    assert isinstance(instance, CustomOp)
    desc = reg.describe("custom_add")
    assert desc["name"] == "custom_add"
    assert len(reg.list_descriptions()) == 1
    with pytest.raises(UnknownOperationError):
        reg.get("missing_op")


def test_default_register_decorator() -> None:
    @register_operation("temp_registered_op")
    class TempOp(DummyAddOperation):
        pass

    assert TempOp.operation_name == "temp_registered_op"


def test_image_utils_conversions() -> None:
    img_u16 = np.full((6, 6, 3), 65535, dtype=np.uint16)
    u8 = to_uint8(img_u16)
    assert u8.dtype == np.uint8 and np.all(u8 == 255)
    assert to_uint8(u8).dtype == np.uint8
    restored_u16 = restore_dtype(u8, np.dtype(np.uint16))
    assert restored_u16.dtype == np.uint16 and np.all(restored_u16 == 65535)
    restored_u8 = restore_dtype(u8, np.dtype(np.uint8))
    assert restored_u8.dtype == np.uint8
    single_ch = np.ones((4, 4, 1), dtype=np.uint8)
    assert squeeze_single_channel(single_ch).shape == (4, 4)


def test_image_utils_alpha_and_channels() -> None:
    rgba = np.full((6, 8, 4), 128, dtype=np.uint8)
    rgb, alpha = split_alpha(rgba)
    assert rgb.shape == (6, 8, 3) and alpha is not None and alpha.shape == (6, 8)
    merged = merge_alpha(rgb, alpha)
    assert merged.shape == (6, 8, 4)
    small_gray = np.full((3, 4), 100, dtype=np.uint8)
    merged_resized = merge_alpha(small_gray, alpha)
    assert merged_resized.shape == (3, 4, 4)
    merged_u16 = merge_alpha(np.full((6, 8, 3), 1000, dtype=np.uint16), alpha)
    assert merged_u16.dtype == np.uint16
    assert merge_alpha(rgb, None).shape == (6, 8, 3)
    assert ensure_grayscale(rgb).shape == (6, 8)
    assert ensure_grayscale(small_gray).shape == (3, 4)
    assert ensure_rgb(small_gray).shape == (3, 4, 3)
    assert ensure_rgb(rgb).shape == (6, 8, 3)


def test_io_load_save_and_exif(tmp_path: Path) -> None:
    raw_img = np.zeros((10, 20, 3), dtype=np.uint8)
    raw_img[:, :10] = [255, 0, 0]
    pil_img = Image.fromarray(raw_img, mode="RGB")
    exif = pil_img.getexif()
    exif[0x0112] = 6
    jpg_path = tmp_path / "oriented.jpg"
    pil_img.save(jpg_path, format="JPEG", exif=exif.tobytes())
    loaded = load_image(jpg_path)
    assert loaded.shape == (20, 10, 3)
    out_png = save_image(loaded, tmp_path / "sub" / "out.png")
    assert out_png.exists()


def test_io_modes_and_errors(tmp_path: Path) -> None:
    u16_arr = np.full((8, 8), 30000, dtype=np.uint16)
    png_bytes = encode_image_bytes(u16_arr, "PNG")
    decoded_u16 = decode_image_bytes(png_bytes)
    assert decoded_u16.dtype == np.uint16
    jpg_from_u16 = encode_image_bytes(u16_arr, "JPG")
    assert decode_image_bytes(jpg_from_u16).dtype == np.uint8
    rgba = np.full((8, 8, 4), 200, dtype=np.uint8)
    assert decode_image_bytes(encode_image_bytes(rgba, "PNG")).shape == (8, 8, 4)
    assert decode_image_bytes(encode_image_bytes(rgba, "JPEG")).shape == (8, 8, 3)
    one_ch = np.full((8, 8, 1), 100, dtype=np.uint8)
    assert decode_image_bytes(encode_image_bytes(one_ch, "PNG")).shape == (8, 8)


def test_io_edge_cases(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "nonexistent.png")
    bad_ext = tmp_path / "file.txt"
    bad_ext.write_text("hello")
    with pytest.raises(InvalidImageError):
        load_image(bad_ext)
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"")
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"not-an-image")
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"123", max_pixels=0)
    valid_bytes = encode_image_bytes(np.zeros((10, 10, 3), dtype=np.uint8), "PNG")
    with pytest.raises(InvalidImageError):
        decode_image_bytes(valid_bytes, max_pixels=50)
    with pytest.raises(InvalidImageError):
        decode_image_bytes(valid_bytes[:25])
    with pytest.raises(InvalidImageError):
        save_image(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "bad.gif")
    with pytest.raises(InvalidImageError):
        encode_image_bytes(np.zeros((4, 4, 3), dtype=np.uint8), "GIF")
    gif_buf = BytesIO()
    Image.new("RGB", (4, 4)).save(gif_buf, format="GIF")
    with pytest.raises(InvalidImageError):
        decode_image_bytes(gif_buf.getvalue())


def test_io_pdf_support(tmp_path: Path) -> None:
    img = np.full((50, 60, 3), 120, dtype=np.uint8)
    pdf_bytes = encode_image_bytes(img, "PDF")
    assert pdf_bytes.startswith(b"%PDF")
    pdf_file = tmp_path / "document.pdf"
    saved = save_image(img, pdf_file)
    assert saved.exists()
    assert saved.read_bytes().startswith(b"%PDF")


def test_pipeline_execution_and_logging() -> None:
    reg = OperationRegistry()
    reg.register("dummy_add")(DummyAddOperation)
    reg.register("failing_op")(FailingOperation)
    with pytest.raises(PipelineExecutionError):
        Pipeline([])
    with pytest.raises(PipelineExecutionError):
        Pipeline.from_step_specs([], registry=reg)
    pipeline = Pipeline.from_step_specs(
        [("dummy_add", {"delta": 10}, 1), ("dummy_add", {"delta": 20}, 2)],
        name="test_pipe",
        registry=reg,
    )
    img = np.full((6, 6, 3), 5, dtype=np.uint8)
    out = pipeline.execute(img)
    assert np.all(out == 35)
    assert len(pipeline.last_metrics) == 2
    fail_pipe = Pipeline([FailingOperation()])
    with pytest.raises(PipelineExecutionError):
        fail_pipe.execute(img)


def test_structured_logger() -> None:
    logger = get_structured_logger("pixelflow.test")
    log_structured(logger, logging.INFO, "hello_event", key="val")
    formatter = StructuredJsonFormatter()
    record = logger.makeRecord(
        "pixelflow.test",
        logging.INFO,
        "fn",
        1,
        "msg",
        (),
        None,
        extra={"structured_data": {"a": 1}},
    )
    formatted = formatter.format(record)
    assert '"a": 1' in formatted
