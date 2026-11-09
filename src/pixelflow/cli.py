from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

import pixelflow.operations
from pixelflow.batch import execute_batch
from pixelflow.core.io import load_image, save_image
from pixelflow.core.logging import get_structured_logger
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import PixelflowError
from pixelflow.pipeline.loader import load_pipeline_from_file

app = typer.Typer(name="pixelflow", no_args_is_help=True, add_completion=False)
ops_app = typer.Typer(name="ops", no_args_is_help=True)
app.add_typer(ops_app, name="ops")

_ = pixelflow.operations


def _configure_verbosity(verbose: bool) -> logging.Logger:
    logger = get_structured_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


@app.command("run")
def run_command(
    pipeline_file: Annotated[Path, typer.Argument()],
    input_file: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option("--output", "-o")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    logger = _configure_verbosity(verbose)
    try:
        logger.debug("Loading pipeline from %s", pipeline_file)
        pipeline = load_pipeline_from_file(pipeline_file)
        image = load_image(input_file)
        processed = pipeline.execute(image)
        saved = save_image(processed, output)
        typer.echo(f"Processed '{input_file}' -> '{saved}'")
    except PixelflowError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Unexpected error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command("batch")
def batch_command(
    pipeline_file: Annotated[Path, typer.Argument()],
    input_dir: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    workers: Annotated[int, typer.Option("--workers", "-w", min=1)] = 2,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    logger = _configure_verbosity(verbose)
    if not input_dir.exists() or not input_dir.is_dir():
        typer.echo(f"Error: Input directory not found: {input_dir}", err=True)
        raise typer.Exit(code=1)
    try:
        logger.debug("Starting batch with %d workers", workers)
        summary = execute_batch(pipeline_file, input_dir, output_dir, workers=workers)
    except PixelflowError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    counts = f"total={summary.total} succeeded={summary.succeeded} failed={summary.failed}"
    typer.echo(f"Batch summary: {counts}")
    if summary.failed > 0 or summary.total == 0:
        raise typer.Exit(code=1)


@app.command("validate")
def validate_command(
    pipeline_file: Annotated[Path, typer.Argument()],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    logger = _configure_verbosity(verbose)
    try:
        logger.debug("Validating pipeline %s", pipeline_file)
        pipeline = load_pipeline_from_file(pipeline_file)
        step_count = len(pipeline.steps)
        typer.echo(
            f"Valid pipeline '{pipeline.name}' (v{pipeline.version}) with {step_count} steps"
        )
    except PixelflowError as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@ops_app.command("list")
def ops_list_command() -> None:
    for name in default_registry.list_names():
        typer.echo(name)


@ops_app.command("describe")
def ops_describe_command(
    name: Annotated[str, typer.Argument()],
) -> None:
    try:
        info = default_registry.describe(name)
        typer.echo(json.dumps(info, indent=2))
    except PixelflowError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
