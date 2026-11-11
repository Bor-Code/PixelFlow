from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorDetailModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str
    request_id: str
    details: dict[str, object]


class ValidateRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: str


class ValidateResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    name: str
    version: str
    step_count: int
    steps: list[str]


class OperationDescriptionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    class_name: str
    params_schema: dict[str, object]


class OperationsListResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operations: list[OperationDescriptionModel]


class HealthResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str


class DetectCornersResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corners: list[list[int]]
