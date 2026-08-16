"""Path shim: lets the Streamlit app find `aesim` even when the package has not
been pip-installed (falls back to the repo's src/ layout)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_package() -> None:
    try:
        import aesim  # noqa: F401
    except ImportError:
        src = Path(__file__).resolve().parents[1] / "src"
        if src.exists():
            sys.path.insert(0, str(src))


ensure_package()
