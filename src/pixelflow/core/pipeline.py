from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pixelflow.core.logging import get_structured_logger, log_structured
from pixelflow.core.operation import (
    ImageArray,
    ImageOperation,
    OperationParams,
    validate_image_array,
)
from pixelflow.core.registry import OperationRegistry, default_registry
from pixelflow.exceptions import PipelineExecutionError, PixelflowError


@dataclass(frozen=True)
class StepExecutionMetrics:
    step_index: int
    operation_name: str
    duration_ms: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


class Pipeline:
    def __init__(
        self,
        steps: list[ImageOperation[OperationParams]],
        name: str = "pipeline",
        version: str = "1.0",
        logger: logging.Logger | None = None,
    ) -> None:
        if not steps:
            raise PipelineExecutionError("Pipeline must contain at least one step")
        self.name = name
        self.version = version
        self.steps = list(steps)
        self.logger = logger or get_structured_logger()
        self.last_metrics: list[StepExecutionMetrics] = []

    @classmethod
    def from_step_specs(
        cls,
        step_specs: list[tuple[str, dict[str, object] | None, int | None]],
        name: str = "pipeline",
        version: str = "1.0",
        registry: OperationRegistry = default_registry,
    ) -> Pipeline:
        if not step_specs:
            raise PipelineExecutionError("Pipeline must contain at least one step")
        operations: list[ImageOperation[OperationParams]] = [
            registry.create(op_name, params=params, line=line)
            for op_name, params, line in step_specs
        ]
        return cls(steps=operations, name=name, version=version)

    def execute(self, image: ImageArray) -> ImageArray:
        validate_image_array(image)
        current: ImageArray = image.copy()
        metrics: list[StepExecutionMetrics] = []
        for idx, operation in enumerate(self.steps):
            current, step_metric = self._run_single_step(idx, operation, current)
            metrics.append(step_metric)
        self.last_metrics = metrics
        return current

    def _run_single_step(
        self,
        index: int,
        operation: ImageOperation[OperationParams],
        image: ImageArray,
    ) -> tuple[ImageArray, StepExecutionMetrics]:
        in_shape = tuple(int(dim) for dim in image.shape)
        start = time.perf_counter()
        output = self._invoke_operation(index, operation, image)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 4)
        out_shape = tuple(int(dim) for dim in output.shape)
        metric = StepExecutionMetrics(
            step_index=index,
            operation_name=operation.operation_name,
            duration_ms=elapsed_ms,
            input_shape=in_shape,
            output_shape=out_shape,
        )
        self._log_step_metric(metric)
        return output, metric

    def _invoke_operation(
        self,
        index: int,
        operation: ImageOperation[OperationParams],
        image: ImageArray,
    ) -> ImageArray:
        try:
            return operation.apply(image)
        except PixelflowError as exc:
            raise PipelineExecutionError(
                f"Step {index} ('{operation.operation_name}') failed: {exc}",
                step_index=index,
                operation_name=operation.operation_name,
            ) from exc
        except Exception as exc:
            raise PipelineExecutionError(
                f"Unexpected error in step {index} ('{operation.operation_name}'): {exc}",
                step_index=index,
                operation_name=operation.operation_name,
            ) from exc

    def _log_step_metric(self, metric: StepExecutionMetrics) -> None:
        log_structured(
            self.logger,
            logging.INFO,
            "pipeline_step_completed",
            pipeline=self.name,
            step_index=metric.step_index,
            operation=metric.operation_name,
            duration_ms=metric.duration_ms,
            input_shape=metric.input_shape,
            output_shape=metric.output_shape,
        )
