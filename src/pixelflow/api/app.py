from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse

import pixelflow.operations
from pixelflow import __version__
from pixelflow.api.config import ApiSettings, get_api_settings
from pixelflow.api.schemas import (
    ErrorDetailModel,
    HealthResponseModel,
    OperationDescriptionModel,
    OperationsListResponseModel,
    ValidateRequestModel,
    ValidateResponseModel,
)
from pixelflow.core.io import (
    MIME_TO_FORMAT,
    decode_image_bytes,
    encode_image_bytes,
)
from pixelflow.core.logging import get_structured_logger, log_structured
from pixelflow.core.registry import default_registry
from pixelflow.exceptions import (
    InvalidImageError,
    InvalidParametersError,
    PipelineExecutionError,
    PixelflowError,
    UnknownOperationError,
)
from pixelflow.pipeline.loader import load_pipeline_from_string

_ = pixelflow.operations


def _request_id_from(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid else str(uuid.uuid4())


def _error_status_code(exc: PixelflowError) -> int:
    if isinstance(exc, UnknownOperationError):
        return 404
    if isinstance(exc, (InvalidImageError, InvalidParametersError, PipelineExecutionError)):
        return 422
    return 400


def _register_middleware(app: FastAPI, logger: logging.Logger) -> None:
    @app.middleware("http")
    async def request_id_and_logging_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 4)
        response.headers["X-Request-ID"] = req_id
        log_structured(
            logger,
            logging.INFO,
            "http_request_completed",
            request_id=req_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=elapsed_ms,
        )
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PixelflowError)
    async def handle_pixelflow_error(request: Request, exc: PixelflowError) -> JSONResponse:
        status_code = _error_status_code(exc)
        payload = ErrorDetailModel(
            error_code=exc.__class__.__name__,
            message=exc.message,
            request_id=_request_id_from(request),
            details=exc.details,
        )
        return JSONResponse(
            status_code=status_code,
            content=payload.model_dump(),
            headers={"X-Request-ID": payload.request_id},
        )


def _register_health_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponseModel)
    async def health_check() -> HealthResponseModel:
        return HealthResponseModel(status="ok", version=__version__)

    @app.get("/ready", response_model=HealthResponseModel)
    async def readiness_check() -> HealthResponseModel:
        op_count = len(default_registry.list_names())
        status = "ready" if op_count > 0 else "degraded"
        return HealthResponseModel(status=status, version=__version__)


def _register_v1_routes(app: FastAPI) -> None:
    @app.get("/v1/operations", response_model=OperationsListResponseModel)
    async def list_operations() -> OperationsListResponseModel:
        raw_items = default_registry.list_descriptions()
        items = [OperationDescriptionModel.model_validate(item) for item in raw_items]
        return OperationsListResponseModel(operations=items)

    @app.post("/v1/validate", response_model=ValidateResponseModel)
    async def validate_pipeline_endpoint(
        body: ValidateRequestModel,
    ) -> ValidateResponseModel:
        pipeline = load_pipeline_from_string(body.pipeline)
        return ValidateResponseModel(
            valid=True,
            name=pipeline.name,
            version=pipeline.version,
            step_count=len(pipeline.steps),
            steps=[step.operation_name for step in pipeline.steps],
        )

    @app.post("/v1/process")
    async def process_image_endpoint(
        image: Annotated[UploadFile, File(...)],
        pipeline: Annotated[str, Form(...)],
        settings: Annotated[ApiSettings, Depends(get_api_settings)],
    ) -> Response:
        return await _handle_process_request(image, pipeline, settings)


async def _handle_process_request(
    upload: UploadFile,
    pipeline_text: str,
    settings: ApiSettings,
) -> Response:
    content_type = (upload.content_type or "").lower().strip()
    if content_type not in settings.allowed_mime_types:
        raise InvalidImageError(
            f"Unsupported MIME type: '{content_type}'",
            details={"allowed_mime_types": list(settings.allowed_mime_types)},
        )
    raw_bytes = await upload.read(settings.max_upload_bytes + 1)
    if len(raw_bytes) > settings.max_upload_bytes:
        raise InvalidImageError(
            f"Uploaded file exceeds maximum size of {settings.max_upload_bytes} bytes"
        )
    pipeline_obj = load_pipeline_from_string(pipeline_text)
    decoded = decode_image_bytes(raw_bytes, max_pixels=settings.max_image_pixels)
    processed = pipeline_obj.execute(decoded)
    target_format = MIME_TO_FORMAT.get(content_type, "PNG")
    output_bytes = encode_image_bytes(processed, image_format=target_format)
    return Response(content=output_bytes, media_type=content_type)


def create_app(settings_override: ApiSettings | None = None) -> FastAPI:
    logger = get_structured_logger("pixelflow.api")
    app = FastAPI(title="Pixelflow API", version=__version__)
    if settings_override is not None:
        app.dependency_overrides[get_api_settings] = lambda: settings_override
    _register_middleware(app, logger)
    _register_exception_handlers(app)
    _register_health_routes(app)
    _register_v1_routes(app)
    return app
