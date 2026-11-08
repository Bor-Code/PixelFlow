from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixelflow.core.io import save_image
from pixelflow.exceptions import (
    InvalidParametersError,
    PipelineExecutionError,
    UnknownOperationError,
)
from pixelflow.pipeline.loader import load_pipeline_from_file, load_pipeline_from_string


def test_load_valid_yaml_and_execute(tmp_path: Path) -> None:
    wm_img = np.full((8, 8, 4), 255, dtype=np.uint8)
    save_image(wm_img, tmp_path / "logo.png")
    yaml_content = """
name: sample_pipeline
version: "1.0"
steps:
  - op: resize
    params:
      width: 32
      height: 32
  - op: watermark
    params:
      watermark_path: logo.png
      position: bottom-right
      opacity: 0.5
"""
    yaml_file = tmp_path / "pipeline.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    pipeline = load_pipeline_from_file(yaml_file)
    assert pipeline.name == "sample_pipeline"
    assert len(pipeline.steps) == 2
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = pipeline.execute(img)
    assert out.shape == (32, 32, 3)


def test_load_valid_json_string() -> None:
    json_content = '{"name": "json_pipe", "version": "1.0", "steps": [{"op": "grayscale"}]}'
    pipeline = load_pipeline_from_string(json_content)
    assert pipeline.name == "json_pipe"
    assert len(pipeline.steps) == 1


def test_loader_line_number_and_schema_errors(tmp_path: Path) -> None:
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_file(tmp_path / "missing.yaml")
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("")
    with pytest.raises(InvalidParametersError) as syntax_err:
        load_pipeline_from_string("name: test\nsteps: [\n  unclosed")
    assert syntax_err.value.line is not None
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("- item1\n- item2")
    with pytest.raises(InvalidParametersError) as extra_top:
        load_pipeline_from_string("name: p\nversion: 1\nunknown_field: 1\nsteps:\n  - op: flip")
    assert "line 3" in str(extra_top.value)
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("name: p\nsteps:\n  - op: flip")
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("name: '  '\nversion: 1\nsteps:\n  - op: flip")
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("name: p\nversion: true\nsteps:\n  - op: flip")


def test_loader_step_validation_and_duplicate_names() -> None:
    with pytest.raises(PipelineExecutionError):
        load_pipeline_from_string("name: p\nversion: 1\nsteps: []")
    with pytest.raises(InvalidParametersError) as dup_op:
        load_pipeline_from_string(
            "name: p\nversion: 1\nsteps:\n  - op: grayscale\n  - op: grayscale\n"
        )
    assert dup_op.value.line == 5
    with pytest.raises(InvalidParametersError) as dup_name:
        load_pipeline_from_string(
            "name: p\nversion: 1\nsteps:\n  - op: grayscale\n    name: s1\n  - op: flip\n    name: s1\n"
        )
    assert dup_name.value.line == 6
    with pytest.raises(UnknownOperationError) as unk_op:
        load_pipeline_from_string("name: p\nversion: 1\nsteps:\n  - op: non_existent_op\n")
    assert "line 4" in str(unk_op.value)
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("name: p\nversion: 1\nsteps:\n  - not_a_dict\n")
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("name: p\nversion: 1\nsteps:\n  - op: flip\n    bad_key: 1\n")
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("name: p\nversion: 1\nsteps:\n  - params: {}\n")
    with pytest.raises(InvalidParametersError):
        load_pipeline_from_string("name: p\nversion: 1\nsteps:\n  - op: flip\n    params: 123\n")

