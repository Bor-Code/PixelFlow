from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import yaml
from yaml.nodes import MappingNode, Node, SequenceNode

import pixelflow.operations
from pixelflow.core.operation import ImageOperation, OperationParams
from pixelflow.core.pipeline import Pipeline
from pixelflow.core.registry import OperationRegistry, default_registry
from pixelflow.exceptions import (
    InvalidParametersError,
    PipelineExecutionError,
    UnknownOperationError,
)

_ = pixelflow.operations

ALLOWED_TOP_KEYS: frozenset[str] = frozenset({"name", "version", "steps"})
ALLOWED_STEP_KEYS: frozenset[str] = frozenset({"op", "name", "params"})
PATH_PARAM_KEYS: frozenset[str] = frozenset({"watermark_path", "path", "image_path", "mask_path"})


class LineTrackingLoader(yaml.SafeLoader):
    pass


def _parse_yaml_with_lines(
    content: str,
) -> tuple[dict[str, object], dict[str, int], list[int]]:
    try:
        root_node = yaml.compose(content, Loader=LineTrackingLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = (mark.line + 1) if mark is not None else 1
        raise InvalidParametersError(
            "pipeline", f"YAML syntax error at line {line}: {exc}", line=line
        ) from exc
    if root_node is None:
        raise InvalidParametersError("pipeline", "Pipeline definition is empty", line=1)
    if not isinstance(root_node, MappingNode):
        line = root_node.start_mark.line + 1
        raise InvalidParametersError(
            "pipeline", f"Pipeline root must be a mapping at line {line}", line=line
        )
    raw_data = cast(dict[str, object], yaml.safe_load(content))
    key_lines, step_lines = _extract_node_lines(root_node)
    return raw_data, key_lines, step_lines


def _extract_node_lines(root_node: MappingNode) -> tuple[dict[str, int], list[int]]:
    key_lines: dict[str, int] = {}
    step_lines: list[int] = []
    for key_node, val_node in root_node.value:
        key_name = str(key_node.value)
        key_lines[key_name] = key_node.start_mark.line + 1
        if key_name == "steps" and isinstance(val_node, SequenceNode):
            for item_node in val_node.value:
                step_lines.append(_node_line(item_node))
    return key_lines, step_lines


def _node_line(node: Node) -> int:
    return int(node.start_mark.line) + 1


def load_pipeline_from_file(
    path: str | Path,
    registry: OperationRegistry = default_registry,
) -> Pipeline:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise InvalidParametersError("pipeline", f"Pipeline file not found: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    base_dir = file_path.parent.resolve()
    return load_pipeline_from_string(content, base_dir=base_dir, registry=registry)


def load_pipeline_from_string(
    content: str,
    base_dir: Path | None = None,
    registry: OperationRegistry = default_registry,
) -> Pipeline:
    if not content or not content.strip():
        raise InvalidParametersError("pipeline", "Pipeline definition is empty", line=1)
    stripped = content.lstrip()
    if stripped.startswith("{"):
        return _load_from_json_or_yaml(content, base_dir=base_dir, registry=registry)
    raw_data, key_lines, step_lines = _parse_yaml_with_lines(content)
    return _build_validated_pipeline(
        raw_data,
        key_lines=key_lines,
        step_lines=step_lines,
        base_dir=base_dir,
        registry=registry,
    )


def _load_from_json_or_yaml(
    content: str,
    base_dir: Path | None,
    registry: OperationRegistry,
) -> Pipeline:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raw_data, key_lines, step_lines = _parse_yaml_with_lines(content)
        return _build_validated_pipeline(
            raw_data,
            key_lines=key_lines,
            step_lines=step_lines,
            base_dir=base_dir,
            registry=registry,
        )
    if not isinstance(parsed, dict):
        raise InvalidParametersError("pipeline", "Pipeline JSON root must be an object", line=1)
    return _build_validated_pipeline(
        parsed,
        key_lines={},
        step_lines=[],
        base_dir=base_dir,
        registry=registry,
    )


def _build_validated_pipeline(
    raw_data: dict[str, object],
    key_lines: dict[str, int],
    step_lines: list[int],
    base_dir: Path | None,
    registry: OperationRegistry,
) -> Pipeline:
    _validate_top_level_schema(raw_data, key_lines)
    name = str(raw_data["name"]).strip()
    version = str(raw_data["version"]).strip()
    steps_raw = raw_data["steps"]
    if not isinstance(steps_raw, list) or len(steps_raw) == 0:
        line = key_lines.get("steps", 1)
        raise PipelineExecutionError(
            f"Pipeline 'steps' must be a non-empty list at line {line}", line=line
        )
    operations = _build_operations_list(steps_raw, step_lines, base_dir, registry)
    return Pipeline(steps=operations, name=name, version=version)


def _validate_top_level_schema(raw_data: dict[str, object], key_lines: dict[str, int]) -> None:
    extra_keys = set(raw_data.keys()) - ALLOWED_TOP_KEYS
    if extra_keys:
        bad_key = sorted(extra_keys)[0]
        line = key_lines.get(bad_key, 1)
        raise InvalidParametersError(
            "pipeline", f"Unknown top-level field '{bad_key}' at line {line}", line=line
        )
    for req in ("name", "version", "steps"):
        if req not in raw_data:
            raise InvalidParametersError(
                "pipeline", f"Missing required top-level field '{req}'", line=1
            )
    _validate_name_and_version(raw_data, key_lines)


def _validate_name_and_version(raw_data: dict[str, object], key_lines: dict[str, int]) -> None:
    if not isinstance(raw_data["name"], str) or not raw_data["name"].strip():
        line = key_lines.get("name", 1)
        raise InvalidParametersError(
            "pipeline", f"Field 'name' must be a non-empty string at line {line}", line=line
        )
    if not isinstance(raw_data["version"], (str, int, float)) or isinstance(
        raw_data["version"], bool
    ):
        line = key_lines.get("version", 1)
        raise InvalidParametersError(
            "pipeline", f"Field 'version' must be a string or number at line {line}", line=line
        )


def _build_operations_list(
    steps_raw: list[object],
    step_lines: list[int],
    base_dir: Path | None,
    registry: OperationRegistry,
) -> list[ImageOperation[OperationParams]]:
    seen_names: set[str] = set()
    operations: list[ImageOperation[OperationParams]] = []
    for idx, step_obj in enumerate(steps_raw):
        line = step_lines[idx] if idx < len(step_lines) else idx + 1
        op_name, step_label, params = _parse_single_step(step_obj, line, base_dir)
        if step_label in seen_names:
            raise InvalidParametersError(
                op_name,
                f"Duplicate step name '{step_label}' at line {line}",
                line=line,
            )
        seen_names.add(step_label)
        try:
            operation = registry.create(op_name, params=params, line=line)
        except UnknownOperationError as exc:
            raise UnknownOperationError(
                f"{op_name} (at line {line})", available=registry.list_names()
            ) from exc
        operations.append(operation)
    return operations


def _parse_single_step(
    step_obj: object,
    line: int,
    base_dir: Path | None,
) -> tuple[str, str, dict[str, object]]:
    if not isinstance(step_obj, dict):
        raise InvalidParametersError(
            "pipeline", f"Step at line {line} must be a mapping", line=line
        )
    extra = set(step_obj.keys()) - ALLOWED_STEP_KEYS
    if extra:
        bad = sorted(extra)[0]
        raise InvalidParametersError(
            "pipeline", f"Unknown step field '{bad}' at line {line}", line=line
        )
    op_name, step_label = _extract_op_and_label(step_obj, line)
    params = _extract_step_params(op_name, step_obj.get("params"), line)
    resolved_params = _resolve_relative_paths(op_name, params, base_dir)
    return op_name, step_label, resolved_params


def _extract_op_and_label(step_obj: dict[str, object], line: int) -> tuple[str, str]:
    raw_op = step_obj.get("op")
    if not isinstance(raw_op, str) or not raw_op.strip():
        raise InvalidParametersError(
            "pipeline", f"Step at line {line} is missing valid 'op' field", line=line
        )
    op_name = raw_op.strip().lower()
    raw_step_name = step_obj.get("name")
    step_label = (
        raw_step_name.strip().lower()
        if isinstance(raw_step_name, str) and raw_step_name.strip()
        else op_name
    )
    return op_name, step_label


def _extract_step_params(op_name: str, raw_params: object, line: int) -> dict[str, object]:
    if raw_params is None:
        return {}
    if isinstance(raw_params, dict):
        return dict(raw_params)
    raise InvalidParametersError(
        op_name, f"Field 'params' at line {line} must be a mapping", line=line
    )


def _resolve_relative_paths(
    op_name: str,
    params: dict[str, object],
    base_dir: Path | None,
) -> dict[str, object]:
    resolved = dict(params)
    target_base = base_dir if base_dir is not None else Path.cwd()
    for key in PATH_PARAM_KEYS:
        val = resolved.get(key)
        if isinstance(val, str) and val.strip():
            candidate = Path(val)
            if not candidate.is_absolute():
                resolved[key] = (target_base / candidate).resolve().as_posix()
    if op_name == "watermark" and "watermark_path" in resolved and "base_dir" not in resolved:
        resolved["base_dir"] = target_base.resolve().as_posix()
    return resolved
