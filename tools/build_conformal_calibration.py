#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.inference.registry import build_predictor, list_model_specs
from backend.inference.runtime_paths import RESOURCES_DIR
from runtime_src.common.uncertainty import conformal_quantile_threshold
from runtime_src.seg.models import extract_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a split-conformal calibration artifact for the standalone bone-tumor segmentation web demo."
        )
    )
    parser.add_argument("--model_id", required=True, help="Model id exposed by `/models` in the standalone demo.")
    parser.add_argument(
        "--master_thesis_root",
        required=True,
        help="Absolute path to the local Master-Thesis repository that still contains the source dataset and splits.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--alpha", type=float, default=0.1, help="Target error rate for conformal prediction.")
    parser.add_argument("--output_path", default="", help="Optional explicit output JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional image cap for quick smoke runs.")
    return parser.parse_args()


def get_model_spec(model_id: str):
    for spec in list_model_specs():
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"Unknown model_id: {model_id}")


def resolve_cfg_paths(cfg: dict, master_thesis_root: Path) -> tuple[Path, Path, Path]:
    bone_seg_root = master_thesis_root / "source_code" / "bone_seg_option1"
    data_cfg = dict(cfg.get("data") or {})
    raw_images_value = str(data_cfg["images_dir"])
    raw_masks_value = str(data_cfg["pseudo_masks_dir"])

    def resolve_known_path(path_value: str, *, kind: str) -> Path:
        raw = Path(str(path_value))
        if raw.is_absolute():
            return raw.resolve()

        candidates = [
            (bone_seg_root / raw).resolve(),
            (master_thesis_root / raw).resolve(),
        ]
        if kind == "images":
            candidates.extend(
                [
                    (master_thesis_root / "datasets" / "BV_ChinhHinh_Data" / "changed_name_images").resolve(),
                    (master_thesis_root / "datasets" / "BV_ChinhHinh_Data" / "labeled_data_for_train").resolve(),
                ]
            )
        elif kind == "masks":
            candidates.extend(
                [
                    (bone_seg_root / "outputs" / "seg_seed_v2_data" / "masks").resolve(),
                ]
            )
        elif kind == "split":
            candidates.extend(
                [
                    (bone_seg_root / "splits_seed_v2").resolve(),
                    (bone_seg_root / "splits_seed").resolve(),
                    (bone_seg_root / "splits").resolve(),
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    masks_dir = resolve_known_path(raw_masks_value, kind="masks")
    if "seg_seed_v2_data/masks" in raw_masks_value.replace("\\", "/"):
        images_dir = (master_thesis_root / "datasets" / "BV_ChinhHinh_Data" / "changed_name_images").resolve()
    else:
        images_dir = resolve_known_path(raw_images_value, kind="images")
    split_dir = resolve_known_path(str(data_cfg["split_dir"]), kind="split")
    return images_dir, masks_dir, split_dir


def read_patient_ids(split_dir: Path, split: str) -> list[str]:
    split_path = split_dir / f"{split}_patients.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_cases(images_dir: Path, masks_dir: Path, patient_ids: list[str]) -> list[tuple[Path, Path]]:
    cases: list[tuple[Path, Path]] = []
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp")
    for patient_id in patient_ids:
        mask_dir = masks_dir / patient_id
        if not mask_dir.is_dir():
            continue
        for mask_path in sorted(mask_dir.glob("*.png")):
            image_path = None
            for suffix in image_suffixes:
                candidate = images_dir / f"{mask_path.stem}{suffix}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is not None:
                cases.append((image_path, mask_path))
    return cases


def run_case(predictor, image_path: Path, mask_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    processed_image = image.resize((predictor.img_size, predictor.img_size))
    mask_image = Image.open(mask_path).convert("L").resize(
        (predictor.img_size, predictor.img_size),
        resample=Image.NEAREST,
    )

    image_np = np.array(processed_image, dtype=np.float32) / 255.0
    mask_np = np.array(mask_image, dtype=np.int64)
    if predictor.num_classes <= 2:
        mask_np = (mask_np > 0).astype(np.int64)
    else:
        mask_np = np.clip(mask_np, 0, predictor.num_classes - 1)

    x = torch.from_numpy(image_np).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(predictor.device)
    model = predictor._load_model()
    with torch.no_grad():
        logits = extract_logits(model, x, predictor.model_type)
        logits = torch.nn.functional.interpolate(
            logits,
            size=(predictor.img_size, predictor.img_size),
            mode="bilinear",
            align_corners=False,
        )
        logits_np = logits.squeeze(0).detach().cpu().numpy()
        if logits.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-logits_np[0]))
            probs = np.stack([1.0 - positive, positive], axis=0)
        else:
            shifted = logits_np - np.max(logits_np, axis=0, keepdims=True)
            exp_values = np.exp(shifted)
            probs = exp_values / np.clip(exp_values.sum(axis=0, keepdims=True), a_min=1e-12, a_max=None)

    flat_probs = np.moveaxis(probs, 0, -1).reshape(-1, probs.shape[0])
    flat_labels = mask_np.reshape(-1)
    true_probs = flat_probs[np.arange(flat_labels.size), flat_labels]
    return 1.0 - np.clip(true_probs, a_min=0.0, a_max=1.0)


def main() -> None:
    args = parse_args()
    master_thesis_root = Path(args.master_thesis_root).resolve()
    spec = get_model_spec(args.model_id)
    predictor = build_predictor(args.model_id)

    images_dir, masks_dir, split_dir = resolve_cfg_paths(predictor.cfg, master_thesis_root)
    patient_ids = read_patient_ids(split_dir, args.split)
    cases = collect_cases(images_dir, masks_dir, patient_ids)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("No calibration cases were found for the requested split.")

    all_scores: list[np.ndarray] = []
    for image_path, mask_path in cases:
        case_scores = run_case(predictor, image_path, mask_path)
        if case_scores.size:
            all_scores.append(case_scores.astype(np.float64, copy=False))

    if not all_scores:
        raise RuntimeError("Calibration scores are empty after processing the split.")

    calibration_scores = np.concatenate(all_scores, axis=0)
    lambda_threshold = conformal_quantile_threshold(calibration_scores, alpha=float(args.alpha))
    empirical_coverage = float(np.mean(calibration_scores <= lambda_threshold))

    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else (RESOURCES_DIR / "conformal" / f"{args.model_id}.json").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": "split_conformal_prediction_sets",
        "score_name": "one_minus_true_class_probability",
        "coverage_scope": "pixelwise",
        "alpha": float(args.alpha),
        "target_coverage": float(1.0 - float(args.alpha)),
        "lambda_threshold": float(lambda_threshold),
        "probability_floor": float(1.0 - float(lambda_threshold)),
        "calibration_size": int(calibration_scores.size),
        "case_count": int(len(cases)),
        "source_split": args.split,
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "split_dir": str(split_dir),
        "model_id": args.model_id,
        "checkpoint_path": str(spec.checkpoint_path),
        "config_path": str(spec.config_path),
        "generated_date": "2026-08-13",
        "empirical_coverage": empirical_coverage,
        "notes": (
            "This artifact calibrates pixelwise true-class nonconformity scores on the local split. "
            "The resulting coverage interpretation should stay scoped to the exchangeability assumption "
            "between this calibration split and the live demo inputs."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved conformal calibration artifact to: {output_path}")


if __name__ == "__main__":
    main()
