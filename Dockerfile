FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 pixelflow && \
    useradd --uid 10001 --gid pixelflow --shell /bin/bash --create-home pixelflow

WORKDIR /app

COPY --from=builder --chown=pixelflow:pixelflow /opt/venv /opt/venv
COPY --chown=pixelflow:pixelflow src ./src

USER pixelflow

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "pixelflow.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
