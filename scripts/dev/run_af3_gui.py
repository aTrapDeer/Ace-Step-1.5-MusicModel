#!/usr/bin/env python
"""Build and launch the AF3 + ChatGPT GUI stack (API + React UI)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env_config import load_project_env


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _build_frontend(skip_npm_install: bool, skip_build: bool) -> None:
    react_dir = PROJECT_ROOT / "react-ui"
    if not react_dir.exists():
        raise FileNotFoundError(f"React UI folder missing: {react_dir}")

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("`npm` was not found. Install Node.js (includes npm) first.")

    if not skip_npm_install and not (react_dir / "node_modules").exists():
        _run([npm, "install"], cwd=react_dir)

    if not skip_build:
        _run([npm, "run", "build"], cwd=react_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch AF3 GUI (FastAPI + built React frontend)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode")
    p.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    p.add_argument("--skip-npm-install", action="store_true", help="Skip npm install")
    p.add_argument("--skip-build", action="store_true", help="Skip frontend build")
    return p


def main() -> int:
    args = build_parser().parse_args()
    load_project_env()

    _build_frontend(skip_npm_install=bool(args.skip_npm_install), skip_build=bool(args.skip_build))

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(
        "services.pipeline_api:app",
        host=args.host,
        port=int(args.port),
        reload=bool(args.reload),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
