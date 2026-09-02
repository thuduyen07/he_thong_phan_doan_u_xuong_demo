"""Serve curated, precomputed demo results without importing ML runtime modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from backend.inference.runtime_paths import RESOURCES_DIR


SAMPLE_DIR = RESOURCES_DIR / "samples"
SAMPLE_MANIFEST_PATH = SAMPLE_DIR / "samples.yaml"


def _static_model_display_name(entry: dict) -> str:
    source = dict(entry.get("source_evaluation") or {})
    dataset = str(entry.get("dataset", "")).strip().upper()
    method = str(source.get("method", "")).strip()
    if method:
        return f"SegFormer-B0 | {dataset} | {method.replace('_', ' ')}"
    return f"SegFormer-B0 | {dataset}"


def _safe_relative_asset(relative_path: str) -> Path | None:
    candidate = (SAMPLE_DIR / relative_path).resolve()
    try:
        candidate.relative_to(SAMPLE_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _asset_url(relative_path: str | None) -> str | None:
    if not relative_path or _safe_relative_asset(relative_path) is None:
        return None
    return f"/sample-assets/{relative_path}"


@lru_cache(maxsize=1)
def load_static_samples() -> list[dict]:
    if not SAMPLE_MANIFEST_PATH.is_file():
        return []
    payload = yaml.safe_load(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    entries = payload.get("samples", []) if isinstance(payload, dict) else []
    return [dict(entry) for entry in entries if isinstance(entry, dict) and str(entry.get("id", "")).strip()]


def list_static_samples() -> list[dict]:
    records: list[dict] = []
    for entry in load_static_samples():
        original = str(entry.get("file", "")).strip()
        artifacts = dict(entry.get("artifacts") or {})
        records.append(
            {
                "id": entry["id"],
                "display_name": str(entry.get("display_name", entry["id"])),
                "dataset": str(entry.get("dataset", "")),
                "case_type": str(entry.get("case_type", "")),
                "available": all(
                    [
                        _asset_url(original) is not None,
                        _asset_url(str(artifacts.get("overlay", "")).strip()) is not None,
                        _asset_url(str(artifacts.get("mask", "")).strip()) is not None,
                        _asset_url(str(artifacts.get("probability_heatmap", "")).strip()) is not None,
                        _asset_url(str(artifacts.get("entropy_heatmap", "")).strip()) is not None,
                    ]
                ),
            }
        )
    return records


def get_static_result(sample_id: str) -> dict:
    entry = next((item for item in load_static_samples() if item["id"] == sample_id), None)
    if entry is None:
        raise FileNotFoundError("Không tìm thấy mẫu đã chọn.")

    original_file = str(entry.get("file", "")).strip()
    original_url = _asset_url(original_file)
    if original_url is None:
        raise FileNotFoundError("Không thể tải dữ liệu mẫu này.")
    artifact_dir = str(entry.get("static_artifact_dir", "")).strip().rstrip("/")
    artifacts = {
        "overlay": f"{artifact_dir}/overlay.png" if artifact_dir else "",
        "mask": f"{artifact_dir}/mask.png" if artifact_dir else "",
        **dict(entry.get("artifacts") or {}),
    }
    uncertainty = dict(entry.get("uncertainty") or {})
    metrics = dict(entry.get("metrics") or {})

    # Static visual maps must be explicitly declared heatmaps. Never fall back
    # to legacy generic PNG names that may not preserve the display semantics.
    probability_heatmap_url = _asset_url(str(artifacts.get("probability_heatmap", "")).strip())
    entropy_heatmap_url = _asset_url(str(artifacts.get("entropy_heatmap", "")).strip())
    global_entropy = dict(uncertainty.get("global") or {
        "available": False,
        "name": "H_G",
        "value": None,
        "reason": "scalar_not_recorded_with_legacy_precomputed_artifact",
    })
    boundary_entropy = dict(uncertainty.get("boundary") or {
        "available": False,
        "name": "H_B",
        "value": None,
        "reason": "boundary_disabled_in_model_config",
    })
    return {
        "source": "static",
        "sample": {
            "id": entry["id"],
            "display_name": str(entry.get("display_name", entry["id"])),
            "dataset": str(entry.get("dataset", "")),
            "case_type": str(entry.get("case_type", "")),
        },
        "model": {
            "display_name": _static_model_display_name(entry),
            "method": str((entry.get("source_evaluation") or {}).get("method", "")),
        },
        "artifacts": {
            "original": original_url,
            "overlay": _asset_url(str(artifacts.get("overlay", "")).strip()),
            "mask": _asset_url(str(artifacts.get("mask", "")).strip()),
            "prediction_mask": _asset_url(str(artifacts.get("prediction_mask", artifacts.get("mask", ""))).strip()),
            "reference_mask": _asset_url(str(artifacts.get("reference_mask", "")).strip()),
            "probability_heatmap": probability_heatmap_url,
            "entropy_heatmap": entropy_heatmap_url,
        },
        "uncertainty": {
            "available": entropy_heatmap_url is not None,
            "pixel_entropy": {
                "available": entropy_heatmap_url is not None,
                "heatmap_url": entropy_heatmap_url,
                "normalization": "normalized_binary_entropy",
                "range": [0.0, 1.0],
            },
            "tumor_probability": {"available": probability_heatmap_url is not None, "heatmap_url": probability_heatmap_url, "range": [0.0, 1.0]},
            "global": global_entropy,
            "boundary": boundary_entropy,
            "summary": dict(uncertainty.get("summary") or {}),
            "note": "Artifact hậu kiểm được chuẩn bị offline từ cùng pipeline phân đoạn đã đăng ký.",
        },
        "reference_metrics": {
            "available": bool(metrics),
            **metrics,
        },
        "provenance": {
            "dataset": str(entry.get("dataset", "")),
            "original_image_id": str(entry.get("original_image_id", "")),
            **dict(entry.get("source_evaluation") or {}),
        },
    }
