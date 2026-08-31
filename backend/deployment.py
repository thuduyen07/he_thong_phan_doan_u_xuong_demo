"""Explicit deployment capabilities shared by Flask routes and the frontend."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentConfig:
    mode: str

    @property
    def live_enabled(self) -> bool:
        return self.mode == "live"

    def as_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "features": {
                "static_demo": True,
                "live_demo": self.live_enabled,
            },
        }


def load_deployment_config(value: str | None = None) -> DeploymentConfig:
    mode = (value if value is not None else os.getenv("DEMO_MODE", "static")).strip().lower()
    if mode not in {"static", "live"}:
        raise RuntimeError("Invalid DEMO_MODE. Expected one of: static, live.")
    return DeploymentConfig(mode=mode)


DEPLOYMENT = load_deployment_config()
