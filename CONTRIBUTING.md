# Contributing to Pixelflow

## Development Setup

1. Install `uv`:
   ```bash
   pip install uv
   ```
2. Install dependencies:
   ```bash
   uv sync --all-extras
   ```
3. Run quality checks before committing:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src tests
   uv run pytest
   ```

## Guidelines

- Follow Conventional Commits for all commit messages and pull request titles.
- Do not include comments or docstrings in code; use descriptive names and small functions.
- Keep every function under 30 lines.
- Ensure strict typing (`mypy --strict`) and at least 90% test coverage.

