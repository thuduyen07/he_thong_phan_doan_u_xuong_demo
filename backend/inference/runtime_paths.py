from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
APP_DIR = PROJECT_ROOT / "app"
RUNTIME_STATIC_DIR = PROJECT_ROOT / "runtime_static"
RESOURCES_DIR = PROJECT_ROOT / "resources"
RUNTIME_SRC_DIR = PROJECT_ROOT / "runtime_src"


def ensure_runtime_paths() -> None:
    for path in (PROJECT_ROOT, RUNTIME_SRC_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
