.PHONY: install lint test run docker

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy --strict src tests

test:
	uv run pytest --cov=pixelflow --cov-report=term-missing

run:
	uv run uvicorn pixelflow.api.app:create_app --factory --host 0.0.0.0 --port 8000

docker:
	docker build -t pixelflow:latest .

