from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionArtifacts:
    mask: np.ndarray
    overlay: np.ndarray
    original: np.ndarray
    model_id: str
    checkpoint_path: Path
    note: str
    extra_metadata: dict[str, Any]
    auxiliary_maps: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class PredictorSpec:
    model_id: str
    predictor_type: str
    display_name: str
    config_path: Path
    checkpoint_path: Path
    note: str
    config_overrides: dict[str, Any] | None = None
    cache_token: str | None = None
    extra_metadata: dict[str, Any] | None = None
