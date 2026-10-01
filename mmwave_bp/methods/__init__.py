"""One module per paper; `get(name)` imports it on demand (torch / scikit-learn only when needed)."""
from __future__ import annotations

from importlib import import_module

NAMES = ("wavebp", "rfbp", "airbp", "mmbp", "hbpfi")
SITES = {"wavebp": "chest", "rfbp": "chest", "airbp": "wrist", "mmbp": "wrist", "hbpfi": "arm (several sites)"}


def get(name: str):
    if name not in NAMES:
        raise SystemExit(f"unknown method {name!r}: choose from {', '.join(NAMES)}")
    return import_module(f"mmwave_bp.methods.{name}")
