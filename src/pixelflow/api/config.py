from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIXELFLOW_", extra="ignore")

    app_name: str = "Pixelflow API"
    app_version: str = "0.1.0"
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=64_000_000, gt=0)
    allowed_mime_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/bmp",
        "image/tiff",
    )


def get_api_settings() -> ApiSettings:
    return ApiSettings()
