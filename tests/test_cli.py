from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pixelflow.batch import execute_batch, process_single_file_worker
from pixelflow.cli import app
from pixelflow.core.io import save_image

runner = CliRunner()


def _create_sample_pipeline(tmp_path: Path) -> Path:
    pipe_file = tmp_path / "pipeline.yaml"
    content = (
        "name: cli_pipe\nversion: '1.0'\nsteps:\n"
        "  - op: resize\n    params:\n      width: 16\n      height: 16\n"
    )
    pipe_file.write_text(content, encoding="utf-8")
    return pipe_file


def test_cli_ops_list_and_describe() -> None:
    res_list = runner.invoke(app, ["ops", "list"])
    assert res_list.exit_code == 0
    assert "resize" in res_list.stdout
    res_desc = runner.invoke(app, ["ops", "describe", "resize"])
    assert res_desc.exit_code == 0
    assert '"name": "resize"' in res_desc.stdout
    res_bad = runner.invoke(app, ["ops", "describe", "non_existent"])
    assert res_bad.exit_code != 0


def test_cli_validate(tmp_path: Path) -> None:
    pipe = _create_sample_pipeline(tmp_path)
    res_ok = runner.invoke(app, ["validate", str(pipe), "--verbose"])
    assert res_ok.exit_code == 0
    assert "Valid pipeline" in res_ok.stdout
    res_fail = runner.invoke(app, ["validate", str(tmp_path / "missing.yaml")])
    assert res_fail.exit_code != 0


def test_cli_run_and_errors(tmp_path: Path) -> None:
    pipe = _create_sample_pipeline(tmp_path)
    in_img = save_image(np.zeros((32, 32, 3), dtype=np.uint8), tmp_path / "in.png")
    out_img = tmp_path / "out.png"
    res = runner.invoke(app, ["run", str(pipe), str(in_img), "--output", str(out_img), "--verbose"])
    assert res.exit_code == 0 and out_img.exists()
    res_err = runner.invoke(
        app, ["run", str(pipe), str(tmp_path / "no.png"), "--output", str(out_img)]
    )
    assert res_err.exit_code == 1


def test_cli_batch_and_worker(tmp_path: Path) -> None:
    pipe = _create_sample_pipeline(tmp_path)
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    out_dir = tmp_path / "outputs"
    save_image(np.zeros((24, 24, 3), dtype=np.uint8), in_dir / "a.png")
    save_image(np.zeros((24, 24, 3), dtype=np.uint8), in_dir / "b.png")
    summary_seq = execute_batch(pipe, in_dir, out_dir, workers=1, use_processes=False)
    assert summary_seq.total == 2 and summary_seq.succeeded == 2
    res_batch = runner.invoke(
        app, ["batch", str(pipe), str(in_dir), "--output-dir", str(out_dir), "--workers", "1"]
    )
    assert res_batch.exit_code == 0
    assert "succeeded=2" in res_batch.stdout
    fail_worker = process_single_file_worker(
        str(pipe), str(tmp_path / "missing.png"), str(out_dir / "x.png")
    )
    assert not fail_worker.succeeded
    res_no_dir = runner.invoke(
        app, ["batch", str(pipe), str(tmp_path / "nodir"), "--output-dir", str(out_dir)]
    )
    assert res_no_dir.exit_code == 1
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    res_empty = runner.invoke(
        app, ["batch", str(pipe), str(empty_dir), "--output-dir", str(out_dir)]
    )
    assert res_empty.exit_code == 1
