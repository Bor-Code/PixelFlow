from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from pixelflow.core.io import SUPPORTED_EXTENSIONS, load_image, save_image
from pixelflow.pipeline.loader import load_pipeline_from_file


@dataclass(frozen=True)
class BatchItemResult:
    input_path: str
    output_path: str
    succeeded: bool
    error: str | None = None


@dataclass(frozen=True)
class BatchSummary:
    total: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]


def process_single_file_worker(
    pipeline_path_str: str,
    input_path_str: str,
    output_path_str: str,
) -> BatchItemResult:
    try:
        pipeline = load_pipeline_from_file(pipeline_path_str)
        image = load_image(input_path_str)
        output = pipeline.execute(image)
        save_image(output, output_path_str)
        return BatchItemResult(
            input_path=input_path_str,
            output_path=output_path_str,
            succeeded=True,
        )
    except Exception as exc:
        return BatchItemResult(
            input_path=input_path_str,
            output_path=output_path_str,
            succeeded=False,
            error=str(exc),
        )


def collect_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def execute_batch(
    pipeline_path: Path,
    input_dir: Path,
    output_dir: Path,
    workers: int = 2,
    use_processes: bool = True,
) -> BatchSummary:
    load_pipeline_from_file(pipeline_path)
    files = collect_image_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        return BatchSummary(total=0, succeeded=0, failed=0, results=[])
    results = _run_batch_tasks(pipeline_path, files, output_dir, max(1, workers), use_processes)
    succeeded = sum(1 for item in results if item.succeeded)
    failed = len(results) - succeeded
    return BatchSummary(total=len(results), succeeded=succeeded, failed=failed, results=results)


def _run_batch_tasks(
    pipeline_path: Path,
    files: list[Path],
    output_dir: Path,
    workers: int,
    use_processes: bool,
) -> list[BatchItemResult]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    with progress:
        task_id = progress.add_task("Processing images", total=len(files))
        if not use_processes:
            return [
                _step_advance(progress, task_id, pipeline_path, f, output_dir / f.name)
                for f in files
            ]
        return _run_parallel_pool(progress, task_id, pipeline_path, files, output_dir, workers)


def _step_advance(
    progress: Progress,
    task_id: TaskID,
    pipeline_path: Path,
    in_file: Path,
    out_file: Path,
) -> BatchItemResult:
    res = process_single_file_worker(str(pipeline_path), str(in_file), str(out_file))
    progress.advance(task_id)
    return res


def _run_parallel_pool(
    progress: Progress,
    task_id: TaskID,
    pipeline_path: Path,
    files: list[Path],
    output_dir: Path,
    workers: int,
) -> list[BatchItemResult]:
    results: list[BatchItemResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_single_file_worker,
                str(pipeline_path),
                str(file_path),
                str(output_dir / file_path.name),
            )
            for file_path in files
        ]
        for future in as_completed(futures):
            results.append(future.result())
            progress.advance(task_id)
    return results
