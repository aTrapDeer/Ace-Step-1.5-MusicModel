"""Environment helpers for project-wide .env loading."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOTENV_PATH = _PROJECT_ROOT / ".env"
_DOTENV_LOADED = False


def load_project_env() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
    _DOTENV_LOADED = True


def get_env(*keys: str, default: str = "") -> str:
    load_project_env()
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def set_default_env_file_value(key: str, value: str) -> bool:
    """Set key=value in .env only if key is missing; returns True when file changed."""
    key = (key or "").strip()
    if not key:
        return False

    lines = []
    if _DOTENV_PATH.exists():
        lines = _DOTENV_PATH.read_text(encoding="utf-8").splitlines()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _ = line.split("=", 1)
        if k.strip() == key:
            return False

    lines.append(f"{key}={value}")
    _DOTENV_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return True
