from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pytest

from pixelflow.api.app import create_app
from pixelflow.api.config import ApiSettings
from pixelflow.core.io import decode_image_bytes, encode_image_bytes


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_ready_and_operations() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ui_resp = await client.get("/")
        assert ui_resp.status_code == 200
        assert "PixelFlow" in ui_resp.text
        h_resp = await client.get("/health", headers={"X-Request-ID": "req-123"})
        assert h_resp.status_code == 200
        assert h_resp.headers["X-Request-ID"] == "req-123"
        assert h_resp.json()["status"] == "ok"
        r_resp = await client.get("/ready")
        assert r_resp.status_code == 200
        assert r_resp.json()["status"] == "ready"
        ops_resp = await client.get("/v1/operations")
        assert ops_resp.status_code == 200
        names = [item["name"] for item in ops_resp.json()["operations"]]
        assert "resize" in names


@pytest.mark.anyio
async def test_validate_endpoint() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        valid_yaml = "name: p\nversion: '1.0'\nsteps:\n  - op: flip\n"
        resp = await client.post("/v1/validate", json={"pipeline": valid_yaml})
        assert resp.status_code == 200
        assert resp.json()["valid"] is True
        unk_resp = await client.post(
            "/v1/validate", json={"pipeline": "name: p\nversion: 1\nsteps:\n  - op: unknown_op\n"}
        )
        assert unk_resp.status_code == 404
        bad_resp = await client.post("/v1/validate", json={"pipeline": "invalid: yaml"})
        assert bad_resp.status_code == 422


@pytest.mark.anyio
async def test_process_endpoint_success() -> None:
    settings = ApiSettings(max_upload_bytes=2048, max_image_pixels=1024)
    app = create_app(settings_override=settings)
    transport = httpx.ASGITransport(app=app)
    img_bytes = encode_image_bytes(np.full((16, 16, 3), 100, dtype=np.uint8), "PNG")
    pipe_json = (
        '{"name":"p","version":"1","steps":[{"op":"resize","params":{"width":8,"height":8}}]}'
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/process",
            files={"image": ("in.png", img_bytes, "image/png")},
            data={"pipeline": pipe_json},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        out_arr = decode_image_bytes(resp.content)
        assert out_arr.shape == (8, 8, 3)

        pdf_resp = await client.post(
            "/v1/process",
            files={"image": ("in.png", img_bytes, "image/png")},
            data={"pipeline": pipe_json, "output_format": "PDF"},
        )
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        assert pdf_resp.content.startswith(b"%PDF")


@pytest.mark.anyio
async def test_process_endpoint_limits() -> None:
    settings = ApiSettings(max_upload_bytes=2048, max_image_pixels=1024)
    app = create_app(settings_override=settings)
    transport = httpx.ASGITransport(app=app)
    img_bytes = encode_image_bytes(np.full((16, 16, 3), 100, dtype=np.uint8), "PNG")
    pipe_json = (
        '{"name":"p","version":"1","steps":[{"op":"resize","params":{"width":8,"height":8}}]}'
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        bad_mime = await client.post(
            "/v1/process",
            files={"image": ("in.gif", img_bytes, "image/gif")},
            data={"pipeline": pipe_json},
        )
        assert bad_mime.status_code == 422
        too_big = await client.post(
            "/v1/process",
            files={"image": ("in.png", b"x" * 3000, "image/png")},
            data={"pipeline": pipe_json},
        )
        assert too_big.status_code == 422


@pytest.mark.anyio
async def test_concurrent_api_requests_isolated() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    img_a = encode_image_bytes(np.zeros((20, 20, 3), dtype=np.uint8), "PNG")
    img_b = encode_image_bytes(np.full((20, 20, 3), 200, dtype=np.uint8), "PNG")
    pipe_a = "name: a\nversion: 1\nsteps:\n  - op: resize\n    params: {width: 10, height: 10}\n"
    pipe_b = "name: b\nversion: 1\nsteps:\n  - op: resize\n    params: {width: 14, height: 14}\n"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res_a, res_b = await asyncio.gather(
            client.post(
                "/v1/process",
                files={"image": ("a.png", img_a, "image/png")},
                data={"pipeline": pipe_a},
            ),
            client.post(
                "/v1/process",
                files={"image": ("b.png", img_b, "image/png")},
                data={"pipeline": pipe_b},
            ),
        )
        assert decode_image_bytes(res_a.content).shape == (10, 10, 3)
        assert decode_image_bytes(res_b.content).shape == (14, 14, 3)


@pytest.mark.anyio
async def test_process_multipage_endpoint() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    img_bytes = encode_image_bytes(np.full((16, 16, 3), 100, dtype=np.uint8), "PNG")
    pipe_json = (
        '{"name":"p","version":"1","steps":[{"op":"resize","params":{"width":8,"height":8}}]}'
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/process-multipage",
            files=[
                ("images", ("p1.png", img_bytes, "image/png")),
                ("images", ("p2.png", img_bytes, "image/png")),
            ],
            data={"pipeline": pipe_json},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content.startswith(b"%PDF")


@pytest.mark.anyio
async def test_process_batch_endpoint() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    img_bytes = encode_image_bytes(np.full((16, 16, 3), 100, dtype=np.uint8), "PNG")
    pipe_json = (
        '{"name":"p","version":"1","steps":[{"op":"resize","params":{"width":8,"height":8}}]}'
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/process-batch",
            files=[
                ("images", ("b1.png", img_bytes, "image/png")),
                ("images", ("b2.png", img_bytes, "image/png")),
            ],
            data={"pipeline": pipe_json, "output_format": "PNG"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert len(resp.content) > 0
