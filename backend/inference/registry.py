from __future__ import annotations

import json
import importlib
import re
from pathlib import Path

from backend.inference.contracts import PredictorSpec
from backend.inference.runtime_paths import RESOURCES_DIR
from runtime_src.common.io_utils import load_yaml


PREDICTOR_CLASSES = {
    "torch_segmentation": "backend.inference.predictors.torch_segmentation:TorchSegmentationPredictor",
}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


STATIC_MODEL_SPECS: dict[str, PredictorSpec] = {}

ALLOWED_EXPERIMENT_RUNS: dict[str, dict[str, str]] = {}


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
