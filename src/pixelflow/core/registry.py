from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pixelflow.core.operation import ImageOperation, OperationParams
from pixelflow.exceptions import UnknownOperationError

OpType = TypeVar("OpType", bound=type[ImageOperation[OperationParams]])


class OperationRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, type[ImageOperation[OperationParams]]] = {}

    def register(self, name: str) -> Callable[[OpType], OpType]:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Operation name cannot be empty")

        def decorator(cls: OpType) -> OpType:
            cls.operation_name = normalized
            self._operations[normalized] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[ImageOperation[OperationParams]]:
        normalized = name.strip().lower()
        if normalized not in self._operations:
            raise UnknownOperationError(name, available=self.list_names())
        return self._operations[normalized]

    def create(
        self,
        name: str,
        params: dict[str, object] | OperationParams | None = None,
        line: int | None = None,
    ) -> ImageOperation[OperationParams]:
        op_cls = self.get(name)
        return op_cls(params=params, line=line)

    def list_names(self) -> list[str]:
        return sorted(self._operations.keys())

    def describe(self, name: str) -> dict[str, object]:
        op_cls = self.get(name)
        schema = op_cls.params_model.model_json_schema()
        return {
            "name": op_cls.operation_name,
            "class_name": op_cls.__name__,
            "params_schema": schema,
        }

    def list_descriptions(self) -> list[dict[str, object]]:
        return [self.describe(name) for name in self.list_names()]


default_registry = OperationRegistry()


def register_operation(name: str) -> Callable[[OpType], OpType]:
    return default_registry.register(name)
