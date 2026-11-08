from __future__ import annotations


class PixelflowError(Exception):
    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidImageError(PixelflowError):
    pass


class UnknownOperationError(PixelflowError):
    def __init__(self, operation_name: str, available: list[str] | None = None) -> None:
        details: dict[str, object] = {"operation": operation_name}
        if available is not None:
            details["available_operations"] = available
        super().__init__(f"Unknown operation: '{operation_name}'", details=details)
        self.operation_name = operation_name


class InvalidParametersError(PixelflowError):
    def __init__(
        self,
        operation_name: str,
        errors: list[dict[str, object]] | str,
        line: int | None = None,
    ) -> None:
        details: dict[str, object] = {"operation": operation_name, "errors": errors}
        if line is not None:
            details["line"] = line
        location = f" at line {line}" if line is not None else ""
        super().__init__(
            f"Invalid parameters for operation '{operation_name}'{location}: {errors}",
            details=details,
        )
        self.operation_name = operation_name
        self.line = line


class PipelineExecutionError(PixelflowError):
    def __init__(
        self,
        message: str,
        step_index: int | None = None,
        operation_name: str | None = None,
        line: int | None = None,
    ) -> None:
        details: dict[str, object] = {}
        if step_index is not None:
            details["step_index"] = step_index
        if operation_name is not None:
            details["operation"] = operation_name
        if line is not None:
            details["line"] = line
        super().__init__(message, details=details)
        self.step_index = step_index
        self.operation_name = operation_name
        self.line = line
