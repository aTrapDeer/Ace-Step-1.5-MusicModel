#!/usr/bin/env python
"""Convenience entrypoint for AF3 GUI stack."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEV_SCRIPTS = ROOT / "scripts" / "dev"
if str(DEV_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DEV_SCRIPTS))

from run_af3_gui import main

if __name__ == "__main__":
    raise SystemExit(main())
