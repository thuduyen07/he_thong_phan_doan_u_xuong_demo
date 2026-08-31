"""Small, model-free checks for the live inference runtime."""

from __future__ import annotations

import importlib
import sys


LIVE_INFERENCE_MODULES = ("numpy", "PIL", "yaml", "torch", "transformers")


def validate_inference_environment() -> dict[str, object]:
    """Import exactly the packages needed by the packaged SegFormer adapter.

    This intentionally does not construct a model or read checkpoint artifacts.
    """

    missing: list[dict[str, str]] = []
    versions: dict[str, str] = {}
    for module_name in LIVE_INFERENCE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            missing.append({"module": module_name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        versions[module_name] = str(getattr(module, "__version__", "unknown"))

    return {
        "available": not missing,
        "missing_dependencies": missing,
        "python_version": sys.version.split()[0],
        "dependencies": versions,
    }
