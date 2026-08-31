#!/usr/bin/env python3
"""Curate a small, reproducible set of deployable segmentation-demo samples."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUPPORTED_DATASETS = ("btxrd", "fracatlas", "bv_chinhhinh")
REQUIRED_COLUMNS = {
    "patient_id", "image_path", "dice", "iou", "hd95", "precision", "recall",
    "reference_positive_pixels", "pred_positive_pixels",
}
CATEGORY_ORDER = (
    "fn", "fp", "tn", "tp", "best_dice", "worst_dice", "best_iou", "worst_iou", "best_hd95", "worst_hd95",
)
DISPLAY_NAMES = {
    "fn": "FN", "fp": "FP", "tn": "TN", "tp": "TP",
    "best_dice": "Best_Dice", "worst_dice": "Worst_Dice",
    "best_iou": "Best_IoU", "worst_iou": "Worst_IoU",
    "best_hd95": "Best_HD95", "worst_hd95": "Worst_HD95",
}


def finite(value: str) -> float | None:
    try:
        value_as_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_as_float if math.isfinite(value_as_float) else None


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            row = dict(row)
            row["reference_positive_pixels"] = int(row["reference_positive_pixels"])
            row["pred_positive_pixels"] = int(row["pred_positive_pixels"])
            row["gt_foreground"] = row["reference_positive_pixels"] > 0
            row["pred_foreground"] = row["pred_positive_pixels"] > 0
            row["case_class"] = (
                "tp" if row["gt_foreground"] and row["pred_foreground"] else
                "fn" if row["gt_foreground"] else
                "fp" if row["pred_foreground"] else "tn"
            )
            for metric in ("dice", "iou", "hd95", "precision", "recall"):
                row[f"{metric}_value"] = finite(row[metric])
            rows.append(row)
    if not rows:
        raise ValueError("Metrics CSV contains no rows.")
    return rows


def ranked_selections(rows: list[dict]) -> dict[str, dict]:
    positive = [row for row in rows if row["gt_foreground"]]
    tp_rows = [row for row in rows if row["case_class"] == "tp"]
    median_dice = sorted(row["dice_value"] for row in tp_rows if row["dice_value"] is not None)
    median = median_dice[len(median_dice) // 2] if median_dice else 0.0

    def first(items: list[dict], key):
        return sorted(items, key=key)[0] if items else None

    return {
        "fn": first([r for r in rows if r["case_class"] == "fn"], lambda r: (-r["reference_positive_pixels"], r["patient_id"])),
        "fp": first([r for r in rows if r["case_class"] == "fp"], lambda r: (-r["pred_positive_pixels"], r["patient_id"])),
        "tn": first([r for r in rows if r["case_class"] == "tn"], lambda r: r["patient_id"]),
        "tp": first(tp_rows, lambda r: (abs((r["dice_value"] or 0.0) - median), r["patient_id"])),
        "best_dice": first([r for r in positive if r["dice_value"] is not None], lambda r: (-r["dice_value"], r["patient_id"])),
        "worst_dice": first([r for r in positive if r["dice_value"] is not None], lambda r: (r["dice_value"], r["patient_id"])),
        "best_iou": first([r for r in positive if r["iou_value"] is not None], lambda r: (-r["iou_value"], r["patient_id"])),
        "worst_iou": first([r for r in positive if r["iou_value"] is not None], lambda r: (r["iou_value"], r["patient_id"])),
        "best_hd95": first([r for r in positive if r["hd95_value"] is not None], lambda r: (r["hd95_value"], r["patient_id"])),
        "worst_hd95": first([r for r in positive if r["hd95_value"] is not None], lambda r: (-r["hd95_value"], r["patient_id"])),
    }


def resolve_source(raw_path: str, source_root: Path) -> Path:
    original = Path(raw_path)
    if original.is_file():
        return original
    try:
        relative = Path(*original.parts[original.parts.index("outputs") + 1 :])
    except ValueError as exc:
        raise FileNotFoundError(f"Cannot relocate source image: {original}") from exc
    mapped = source_root / relative
    if not mapped.is_file():
        raise FileNotFoundError(f"Source image does not exist: {mapped}")
    return mapped


def build_manifest(dataset: str, source_evaluation: dict, selections: dict[str, dict], source_root: Path, output_dir: Path, copy_assets: bool) -> tuple[list[dict], int]:
    entries: list[dict] = []
    unique_sources: dict[str, Path] = {}
    for category in CATEGORY_ORDER:
        row = selections[category]
        if row is None:
            continue
        original_id = str(row["patient_id"])
        source = resolve_source(row["image_path"], source_root)
        unique_sources.setdefault(original_id, source)
        suffix = source.suffix.lower() or ".png"
        relative_file = Path(dataset) / f"{dataset}_{original_id}{suffix}"
        entries.append({
            "id": f"{dataset}_{category}_{original_id}",
            "display_name": f"[sample] {dataset}_{DISPLAY_NAMES[category]}_{original_id}",
            "dataset": dataset,
            "case_type": category,
            "original_image_id": original_id,
            "file": relative_file.as_posix(),
            "source_evaluation": source_evaluation,
            "selection": {"case_class": row["case_class"].upper(), "metric": category.rsplit("_", 1)[1] if "_" in category else None, "direction": "max" if category.startswith("best_") else "min" if category.startswith("worst_") else None},
            "metrics": {metric: row[f"{metric}_value"] for metric in ("dice", "iou", "hd95", "precision", "recall")},
            "foreground_presence": {"ground_truth": row["gt_foreground"], "prediction": row["pred_foreground"]},
        })
    if len(unique_sources) > 10:
        raise RuntimeError(f"Refusing to curate {len(unique_sources)} unique images; maximum is 10.")
    if copy_assets:
        for original_id, source in unique_sources.items():
            destination = output_dir / entries[next(i for i, e in enumerate(entries) if e["original_image_id"] == original_id)]["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
    return entries, len(unique_sources)


def prepare_static_artifacts(manifest_path: Path, output_dir: Path, model_id: str) -> None:
    """Materialize only curated samples through the already-registered live adapter.

    This command is offline preparation. It is never invoked by Flask startup or
    Static Demo, and reuses one predictor instance through the live service cache.
    """
    from backend.inference.service import run_segmentation

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {"samples": []}
    entries = payload.get("samples", []) if isinstance(payload, dict) else []
    generated: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("original_image_id", "")).strip()
        if not source_id:
            continue
        result = generated.get(source_id)
        if result is None:
            result = run_segmentation(image_id=f"default:{entry['id']}", model_id=model_id)
            generated[source_id] = result
        artifact_dir = Path("artifacts") / str(entry.get("dataset", "generic")) / source_id
        destination = output_dir / artifact_dir
        destination.mkdir(parents=True, exist_ok=True)
        source_names = {
            "mask": result["mask_image_url"],
            "overlay": result["overlay_image_url"],
            "probability": result["uncertainty"]["tumor_probability"].get("heatmap_url"),
            "entropy": result["uncertainty"]["pixel_entropy"].get("heatmap_url"),
        }
        for name, url in source_names.items():
            if not url:
                continue
            source = PROJECT_ROOT / url.lstrip("/")
            if not source.is_file():
                raise FileNotFoundError(f"Prepared live artifact does not exist: {source}")
            shutil.copy2(source, destination / f"{name}.png")
        entry["static_artifact_dir"] = artifact_dir.as_posix()
        entry["uncertainty"] = {
            "global": result["uncertainty"].get("global", {}),
            "boundary": result["uncertainty"].get("boundary", {}),
            "summary": result["uncertainty"].get("summary", {}),
        }
        entry["risk_control"] = result["uncertainty"].get("risk_control", {})
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"prepared_static_samples={len(generated)} manifest={manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--source-root", type=Path, help="Research repository's `outputs` directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("resources/samples"))
    parser.add_argument("--manifest", type=Path, default=Path("resources/samples/samples.yaml"))
    parser.add_argument("--method", default="mean_teacher_entropy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--confirm-public-deidentified", action="store_true", help="Required before copying medical-image assets.")
    parser.add_argument("--prepare-static-artifacts", action="store_true", help="Run only curated samples through the registered live adapter and materialize deployable artifacts.")
    parser.add_argument("--static-model-id", help="Registered model id used only with --prepare-static-artifacts.")
    args = parser.parse_args()
    if args.prepare_static_artifacts:
        if not args.static_model_id:
            raise ValueError("--static-model-id is required with --prepare-static-artifacts.")
        prepare_static_artifacts(args.manifest, args.output_dir, args.static_model_id)
        return
    if not args.dataset or not args.metrics_csv or not args.source_root or not args.metrics_csv.is_file() or not args.source_root.is_dir():
        raise FileNotFoundError("Metrics CSV or source root does not exist.")

    source_evaluation = {"report": args.metrics_csv.name, "dataset": args.dataset, "method": args.method, "seed": args.seed, "split": args.split}
    entries, unique_count = build_manifest(args.dataset, source_evaluation, ranked_selections(load_rows(args.metrics_csv)), args.source_root, args.output_dir, args.confirm_public_deidentified)
    if args.confirm_public_deidentified:
        existing = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {"samples": []}
        remaining = [entry for entry in existing.get("samples", []) if entry.get("dataset") != args.dataset]
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(yaml.safe_dump({"samples": [*remaining, *entries]}, sort_keys=False), encoding="utf-8")
    print(f"logical_categories={len(entries)} unique_images={unique_count} copied={unique_count if args.confirm_public_deidentified else 0}")
    for entry in entries:
        print(f"{entry['case_type']}: {entry['original_image_id']} -> {entry['file']}")
    categories_by_image: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        categories_by_image[entry["original_image_id"]].append(entry["case_type"])
    for original_id, categories in sorted(categories_by_image.items()):
        if len(categories) > 1:
            print(f"shared_asset: {original_id} <- {', '.join(categories)}")


if __name__ == "__main__":
    main()
