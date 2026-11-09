from __future__ import annotations

from pathlib import Path

from pixelflow.core.io import load_image
from pixelflow.pipeline.loader import load_pipeline_from_file


def test_example_pipelines_and_sample_generator(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    doc_pipe_path = repo_root / "examples" / "document-cleanup.yaml"
    thumb_pipe_path = repo_root / "examples" / "thumbnail-with-watermark.yaml"
    doc_pipe = load_pipeline_from_file(doc_pipe_path)
    thumb_pipe = load_pipeline_from_file(thumb_pipe_path)
    assert doc_pipe.name == "document-cleanup"
    assert thumb_pipe.name == "thumbnail-with-watermark"
    from examples.generate_samples import generate_example_assets

    doc_img_path, photo_img_path = generate_example_assets(tmp_path)
    cleaned = doc_pipe.execute(load_image(doc_img_path))
    assert cleaned.ndim == 2
    thumb = thumb_pipe.execute(load_image(photo_img_path))
    assert thumb.shape == (240, 320, 3)
