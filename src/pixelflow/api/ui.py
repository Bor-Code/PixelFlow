from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_studio_html() -> str:
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")
