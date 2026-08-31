from __future__ import annotations

import json
import importlib
import logging
import re
from pathlib import Path

import yaml

from backend.inference.contracts import PredictorSpec
from backend.inference.runtime_paths import PROJECT_ROOT, RESOURCES_DIR


PREDICTOR_CLASSES = {
    "torch_segmentation": "backend.inference.predictors.torch_segmentation:TorchSegmentationPredictor",
}
LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


STATIC_MODEL_SPECS: dict[str, PredictorSpec] = {}

ALLOWED_EXPERIMENT_RUNS: dict[str, dict[str, str]] = {}
MODEL_REGISTRY_PATH = RESOURCES_DIR / "metadata" / "models.yaml"


def load_yaml(path: str | Path) -> dict:
    """Read dashboard metadata without importing the PyTorch research runtime."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return payload


def _load_declared_models() -> list[dict[str, object]]:
    if not MODEL_REGISTRY_PATH.exists():
        return []
    payload = load_yaml(str(MODEL_REGISTRY_PATH)) or {}
    models = payload.get("models", []) if isinstance(payload, dict) else []
    return [dict(item) for item in models if isinstance(item, dict)]


def _resolve_registry_path(item: dict[str, object], field: str) -> tuple[Path | None, str | None]:
    configured = str(item.get(field, "")).strip()
    if not configured:
        return None, f"{field}_missing"

    candidate = Path(configured)
    project_root = PROJECT_ROOT.resolve()
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        return None, f"{field}_outside_project"
    return resolved, None


def _validated_declared_models() -> list[dict[str, object]]:
    """Validate registry metadata by file existence only; never load model weights."""
    entries: list[dict[str, object]] = []
    for item in _load_declared_models():
        model_id = str(item.get("id", "")).strip()
        if not model_id:
            LOGGER.warning("Ignoring model registry entry without an id.")
            continue

        config_path, config_error = _resolve_registry_path(item, "config")
        checkpoint_path, checkpoint_error = _resolve_registry_path(item, "checkpoint")
        calibration_path, calibration_error = _resolve_registry_path(item, "calibration_artifact")
        requires_calibration = bool(item.get("requires_calibration", False))
        reasons: list[str] = []
        if item.get("enabled") is not True:
            reasons.append("disabled")
        if config_error:
            reasons.append(config_error)
        elif not config_path.is_file():
            reasons.append("config_not_found")
        if checkpoint_error:
            reasons.append(checkpoint_error)
        elif not checkpoint_path.is_file():
            reasons.append("checkpoint_not_found")
        if requires_calibration:
            if calibration_error:
                reasons.append(calibration_error)
            elif not calibration_path.is_file():
                reasons.append("calibration_artifact_not_found")

        entries.append(
            {
                "item": item,
                "model_id": model_id,
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "calibration_path": calibration_path,
                "requires_calibration": requires_calibration,
                "reasons": reasons,
            }
        )
    return entries


def list_model_records() -> list[dict[str, object]]:
    """Expose model availability without loading weights or server-side paths."""
    entries = _validated_declared_models()
    records: list[dict[str, object]] = []
    for entry in entries:
        item = entry["item"]
        reasons = entry["reasons"]
        assert isinstance(item, dict)
        assert isinstance(reasons, list)
        records.append(
            {
                "model_id": entry["model_id"],
                "display_name": str(item.get("display_name", entry["model_id"])),
                "dataset": str(item.get("dataset", "generic")),
                "architecture": str(item.get("architecture", "")),
                "method": str(item.get("method", "")),
                "threshold": item.get("threshold"),
                "calibration_available": bool(entry["calibration_path"] and entry["calibration_path"].is_file()),
                "available": not reasons,
                "unavailable_reason": reasons[0] if reasons else None,
                "unavailable_reasons": reasons,
                "description": str(item.get("description", "")),
            }
        )
    LOGGER.info("Model registry: %d declared, %d available.", len(records), sum(record["available"] for record in records))
    return records


def _declared_model_specs() -> dict[str, PredictorSpec]:
    specs: dict[str, PredictorSpec] = {}
    for entry in _validated_declared_models():
        if entry["reasons"]:
            continue
        item = entry["item"]
        config_path = entry["config_path"]
        checkpoint_path = entry["checkpoint_path"]
        assert isinstance(item, dict)
        assert isinstance(config_path, Path)
        assert isinstance(checkpoint_path, Path)
        config_overrides = item.get("config_overrides")
        specs[str(entry["model_id"])] = PredictorSpec(
            model_id=str(entry["model_id"]),
            predictor_type=str(item.get("predictor_type", "torch_segmentation")),
            display_name=str(item.get("display_name", entry["model_id"])),
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            note=str(item.get("description", "")),
            config_overrides=dict(config_overrides) if isinstance(config_overrides, dict) else None,
            extra_metadata={
                "dataset": str(item.get("dataset", "generic")),
                "architecture": str(item.get("architecture", "")),
                "method": str(item.get("method", "")),
                "threshold": item.get("threshold"),
                "calibration_required": bool(entry["requires_calibration"]),
            },
        )
    return specs


def _collect_run_dirs(root: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for path in sorted(root.glob("*/*/*")):
        if path.is_dir() and path not in seen:
            seen.add(path)
            candidates.append((path, "nested"))

    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.is_dir() and path not in seen:
            seen.add(path)
            candidates.append((path, "flat"))

    candidates.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return candidates


def _discover_experiment_specs() -> dict[str, PredictorSpec]:
    specs: dict[str, PredictorSpec] = {}
    root = RESOURCES_DIR / "outputs" / "thesis_experiments_seg_seed_v2"
    if not root.exists():
        return specs

    for run_dir, layout in _collect_run_dirs(root):
        run_manifest_path = run_dir / "run_manifest.json"
        config_snapshot_path = run_dir / "config_snapshot.yaml"
        checkpoint_path = run_dir / "seg_best.pt"
        if not run_manifest_path.exists() or not config_snapshot_path.exists() or not checkpoint_path.exists():
            continue

        try:
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if run_manifest.get("status") != "completed":
            continue

        try:
            config_snapshot = load_yaml(str(config_snapshot_path))
        except Exception:
            config_snapshot = {}
        metrics_summary = {}
        metrics_summary_candidates = [
            run_dir / "metrics_summary.json",
            run_dir / "evaluation_test" / "metrics_summary.json",
            run_dir / "eval_test_binary" / "metrics_summary.json",
        ]
        for metrics_summary_path in metrics_summary_candidates:
            if not metrics_summary_path.exists():
                continue
            try:
                metrics_summary = json.loads(metrics_summary_path.read_text(encoding="utf-8"))
                break
            except Exception:
                metrics_summary = {}

        run_name = run_dir.name
        if layout == "nested":
            run_date = run_dir.parent.parent.name
            phase = run_dir.parent.name
            run_key = f"{run_date}/{phase}/{run_name}"
        else:
            experiment_cfg = config_snapshot.get("experiment") or {}
            run_date = str(experiment_cfg.get("date") or "") or str(run_manifest.get("run_date") or "")[:10].replace("-", "")
            phase = str((run_manifest.get("experiment_tags") or {}).get("benchmark") or "standalone")
            run_key = run_name
        allowed_entry = ALLOWED_EXPERIMENT_RUNS.get(run_key)
        if allowed_entry is None:
            continue
        model_id_parts = ["thesis"]
        if run_date:
            model_id_parts.append(_slugify(run_date))
        if phase:
            model_id_parts.append(_slugify(phase))
        model_id_parts.append(_slugify(run_name))
        model_id = "_".join(part for part in model_id_parts if part)
        display_name = allowed_entry["display_name"]
        cache_token = f"{checkpoint_path.stat().st_mtime_ns}:{checkpoint_path.stat().st_size}"
        primary_metric_name = metrics_summary.get("primary_metric") or run_manifest.get("best_metric_name")
        primary_metric_value = metrics_summary.get("primary_value")
        if primary_metric_value is None:
            primary_metric_value = run_manifest.get("best_metric_value")
        secondary_metrics = metrics_summary.get("secondary_metrics") or {}
        model_type = str(config_snapshot.get("model_type") or (run_manifest.get("cfg") or {}).get("model_type") or "segformer")
        note_parts = [allowed_entry["note"], f"Run artifact: {run_name}.", f"Phase: {phase}."]
        if primary_metric_name and primary_metric_value is not None:
            note_parts.append(f"{primary_metric_name}={primary_metric_value:.6f}.")
        if secondary_metrics.get("best_primary_dice") is not None:
            note_parts.append(f"best_primary_dice={secondary_metrics['best_primary_dice']:.6f}.")

        specs[model_id] = PredictorSpec(
            model_id=model_id,
            predictor_type="torch_segmentation",
            display_name=display_name,
            config_path=config_snapshot_path,
            checkpoint_path=checkpoint_path,
            note=" ".join(note_parts),
            cache_token=cache_token,
            extra_metadata={
                "source": "thesis_local_batch",
                "run_date": run_date,
                "phase": phase,
                "run_name": run_name,
                "best_metric_name": primary_metric_name,
                "best_metric_value": primary_metric_value,
                "best_primary_dice": secondary_metrics.get("best_primary_dice"),
                "model_type": model_type,
            },
        )
    return specs


def list_model_specs() -> list[PredictorSpec]:
    specs = {
        **STATIC_MODEL_SPECS,
        **_discover_experiment_specs(),
        **_declared_model_specs(),
    }
    return sorted(specs.values(), key=lambda item: item.display_name.lower())


def build_predictor(model_id: str):
    spec = next((item for item in list_model_specs() if item.model_id == model_id), None)
    if spec is None:
        raise KeyError(f"Unknown model_id: {model_id}")
    predictor_entry = PREDICTOR_CLASSES[spec.predictor_type]
    if isinstance(predictor_entry, str):
        module_name, class_name = predictor_entry.split(":", 1)
        module = importlib.import_module(module_name)
        predictor_class = getattr(module, class_name)
    else:
        predictor_class = predictor_entry
    return predictor_class(spec)
