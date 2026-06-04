"""Tiny config loader.

Configs are YAML (or JSON, which is valid YAML). Returned as a plain dict; we
deliberately avoid a heavy config framework. Access nested keys with `get`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        # YAML not installed -- fall back to JSON. Our config files are written
        # in a JSON-compatible subset so this works either way.
        return json.loads(text)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Fetch a nested key like ``"model.base_width"`` with a default."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
