from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.inference.runtime_paths import APP_DIR, RESOURCES_DIR
from backend.inference.service import DEFAULT_MODEL_ID, available_models
from runtime_src.common.io_utils import load_yaml


OUTPUTS_ROOT = RESOURCES_DIR / "outputs" / "thesis_experiments_seg_seed_v2"
CONFORMAL_ROOT = RESOURCES_DIR / "conformal"
DEFAULT_IMAGE_DIR = APP_DIR / "assets" / "cases" / "original"

PIPELINE_MODULES = [
    {
        "id": "supervised",
        "title": "Supervised warm-up",
        "summary": "Huấn luyện có nhãn để tạo student ổn định trước khi bật dữ liệu chưa gán nhãn.",
        "details": [
            "Loss supervised gồm BCE và Dice trên ảnh có mask tham chiếu.",
            "Checkpoint supervised tốt nhất được dùng làm khởi tạo cho các pha semi-supervised.",
            "Dataset loader, preprocessing, model factory và checkpointing đều được tái sử dụng từ pipeline gốc.",
        ],
    },
    {
        "id": "mean_teacher",
        "title": "EMA Teacher + pseudo-label",
        "summary": "Teacher EMA sinh probability map và pseudo-mask trên ảnh unlabeled để student học bán giám sát.",
        "details": [
            "Teacher dự đoán trên weak view, student học trên strong view.",
            "Pseudo-label loss được mở dần theo lambda_u ramp-up.",
            "Training mode mean_teacher giữ tương thích ngược với supervised loop hiện có.",
        ],
    },
    {
        "id": "uncertainty",
        "title": "Predictive entropy + boundary entropy",
        "summary": "Softmax hậu kiểm tạo entropy map theo pixel và mean boundary entropy để đo độ khó cấp ảnh.",
        "details": [
            "Predictive entropy làm uncertainty weighting cho pseudo-label.",
            "Mean boundary entropy đóng vai trò difficulty signal cho adaptive CRC.",
            "Boundary-aware weighting tăng tập trung quanh biên tumor dự đoán.",
        ],
    },
    {
        "id": "adaptive_crc",
        "title": "Adaptive CRC + reliability weighting",
        "summary": "Conformal risk control điều biến lambda(x) và reliability map theo mức độ khó của từng ảnh.",
        "details": [
            "Adaptive CRC chia difficulty bins và hiệu chỉnh ngưỡng riêng theo độ khó.",
            "Reliability map tách reliable, ambiguous và rejected để điều tiết loss unsupervised.",
            "Demo live dựng prediction set hậu kiểm nếu checkpoint có conformal artifact tương ứng.",
        ],
    },
]

TRAINING_SCHEDULE = [
    {
        "phase": "Warm-up",
        "epoch_range": "1-5",
        "purpose": "Chỉ train supervised để ổn định student ban đầu.",
        "signals": ["Labeled masks", "BCE + Dice"],
    },
    {
        "phase": "Mean teacher",
        "epoch_range": "6-15",
        "purpose": "Mở unlabeled data, EMA teacher, pseudo-label và entropy weighting.",
        "signals": ["EMA teacher", "Predictive entropy", "Mean boundary entropy", "lambda_u ramp-up"],
    },
    {
        "phase": "Adaptive CRC",
        "epoch_range": "16-30",
        "purpose": "Bật adaptive CRC và reliability-aware soft weighting cho pseudo-label.",
        "signals": ["Adaptive CRC", "Difficulty bins", "Reliability map", "Boundary-aware weights"],
    },
]

KEY_FORMULAS = [
    {"label": "Objective", "value": "L = L_sup + lambda_u(t) * L_unsup"},
    {"label": "Supervised", "value": "L_sup = lambda_BCE * L_BCE + lambda_Dice * L_Dice"},
    {"label": "Soft weight", "value": "w_ij = C_ij(lambda(x)) * (1 - H_tilde_ij) * W_B,ij"},
]


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_yaml(path: Path) -> dict[str, Any]:
    try:
        return load_yaml(str(path)) or {}
    except Exception:
        return {}


def _iter_run_dirs(root: Path) -> list[tuple[Path, str, str]]:
    run_dirs: list[tuple[Path, str, str]] = []
    if not root.exists():
        return run_dirs

    for path in sorted(root.glob("*/*/*")):
        if path.is_dir():
            run_dirs.append((path, path.parent.parent.name, path.parent.name))

    for path in sorted(root.iterdir()):
        if path.is_dir() and not any(path == existing[0] for existing in run_dirs):
            run_dirs.append((path, "", "standalone"))
    return run_dirs


def _run_key(run_dir: Path, run_date: str, phase: str) -> str:
    return f"{run_date}/{phase}/{run_dir.name}" if run_date else run_dir.name


def _conformal_profile_name(run_date: str, phase: str, run_name: str) -> str:
    normalized_run_name = run_name.replace("-", "_")
    return f"thesis_{run_date}_{phase}_{normalized_run_name}.json"


def _discover_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    for run_dir, run_date, phase in _iter_run_dirs(OUTPUTS_ROOT):
        run_key = _run_key(run_dir, run_date, phase)
        run_manifest_path = run_dir / "run_manifest.json"
        config_snapshot_path = run_dir / "config_snapshot.yaml"
        checkpoint_path = run_dir / "seg_best.pt"
        if not run_manifest_path.exists() or not config_snapshot_path.exists() or not checkpoint_path.exists():
            continue

        config_snapshot = _safe_read_yaml(run_dir / "config_snapshot.yaml")
        train_summary = _safe_read_json(run_dir / "metrics_summary.json")
        run_manifest = _safe_read_json(run_manifest_path)

        evaluation_paths = [
            run_dir / "eval_test_binary" / "metrics_summary.json",
            run_dir / "evaluation_test" / "metrics_summary.json",
        ]
        evaluation_summary: dict[str, Any] = {}
        evaluation_scope = ""
        for candidate in evaluation_paths:
            if candidate.exists():
                evaluation_summary = _safe_read_json(candidate)
                evaluation_scope = candidate.parent.name
                break

        conformal_summary = _safe_read_json(CONFORMAL_ROOT / _conformal_profile_name(run_date, phase, run_dir.name))
        train_metrics = dict(train_summary.get("secondary_metrics") or {})
        eval_metrics = dict(evaluation_summary.get("secondary_metrics") or {})
        tumor_metrics = dict((eval_metrics.get("per_class_summary") or {}).get("tumor") or {})
        positive_only = dict(tumor_metrics.get("positive_reference_only") or {})

        resolved_phase = phase or str((run_manifest.get("experiment_tags") or {}).get("benchmark") or "standalone")
        display_name = str((run_manifest.get("method_name") or run_dir.name)).strip() or run_dir.name
        runs.append(
            {
                "run_key": run_key,
                "run_name": run_dir.name,
                "display_name": display_name,
                "note": "Artifact này hiện chỉ còn được giữ để kiểm kê nội bộ.",
                "phase": resolved_phase,
                "run_date": run_date,
                "training_mode": train_metrics.get("training_mode") or config_snapshot.get("training", {}).get("mode") or "supervised",
                "model_type": train_metrics.get("model_type") or config_snapshot.get("model_type") or config_snapshot.get("model", {}).get("type") or "segformer",
                "primary_metric": train_summary.get("primary_metric"),
                "primary_value": train_summary.get("primary_value"),
                "best_epoch": train_summary.get("best_epoch"),
                "test_scope": evaluation_scope,
                "ce_weights": train_metrics.get("ce_weights") or config_snapshot.get("ce_weights") or [],
                "img_size": train_metrics.get("img_size") or config_snapshot.get("img_size"),
                "train_batch_size": train_metrics.get("train_batch_size") or config_snapshot.get("batch_size"),
                "parameter_count": train_metrics.get("parameter_count"),
                "best_primary_dice": train_metrics.get("best_primary_dice"),
                "test_dice": positive_only.get("dice_mean") if positive_only else tumor_metrics.get("dice_mean"),
                "test_iou": positive_only.get("iou_mean") if positive_only else tumor_metrics.get("iou_mean"),
                "test_precision": positive_only.get("precision_mean") if positive_only else tumor_metrics.get("precision_mean"),
                "test_recall": positive_only.get("recall_mean") if positive_only else tumor_metrics.get("recall_mean"),
                "num_cases": eval_metrics.get("num_cases"),
                "num_positive_reference_cases": tumor_metrics.get("num_positive_reference_cases"),
                "conformal": {
                    "available": bool(conformal_summary),
                    "target_coverage": conformal_summary.get("target_coverage"),
                    "probability_floor": conformal_summary.get("probability_floor"),
                    "lambda_threshold": conformal_summary.get("lambda_threshold"),
                    "empirical_coverage": conformal_summary.get("empirical_coverage"),
                    "calibration_size": conformal_summary.get("calibration_size"),
                },
            }
        )

    runs.sort(key=lambda item: (item["run_date"], item["phase"], item["display_name"]))
    return runs


def _count_default_images() -> int:
    if not DEFAULT_IMAGE_DIR.exists():
        return 0
    return sum(1 for path in DEFAULT_IMAGE_DIR.iterdir() if path.is_file())


def _hero_metrics(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    live_models = available_models()
    conformal_ready = sum(1 for item in runs if item.get("conformal", {}).get("available"))
    best_dice = max((item.get("test_dice") or 0.0) for item in runs) if runs else 0.0
    return [
        {
            "label": "Checkpoint live",
            "value": str(len(live_models)),
            "note": "Số checkpoint đang mở cho inference trực tiếp trên web demo.",
        },
        {
            "label": "Run thực nghiệm",
            "value": str(len(runs)),
            "note": "Các run được đọc trực tiếp từ artifact đã đóng gói trong resources/outputs.",
        },
        {
            "label": "Run có conformal",
            "value": str(conformal_ready),
            "note": "Số run hiện có calibration profile để dựng prediction set hậu kiểm.",
        },
        {
            "label": "Dice tốt nhất",
            "value": f"{best_dice:.4f}",
            "note": "Dice cao nhất trong các run hiện được giữ lại để trình diễn trên web.",
        },
    ]


@lru_cache(maxsize=1)
def get_dashboard_payload() -> dict[str, Any]:
    runs = _discover_runs()

    return {
        "thesis_title": "Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc transformer",
        "hero_metrics": _hero_metrics(runs),
        "overview": {
            "summary": [
                "Web demo này chỉ giữ các phần bám trực tiếp vào pipeline supervised segmentation -> mean teacher -> uncertainty -> adaptive CRC.",
                "Các checkpoint và artifact thực nghiệm của phương pháp cũ đã được gỡ khỏi repo demo này.",
                "Live lab chỉ hoạt động khi repo được bổ sung checkpoint mới phù hợp với pipeline hiện hành.",
            ],
            "sample_image_count": _count_default_images(),
            "default_model_id": DEFAULT_MODEL_ID,
        },
        "pipeline": {
            "modules": PIPELINE_MODULES,
            "schedule": TRAINING_SCHEDULE,
            "formulas": KEY_FORMULAS,
        },
        "experiments": {
            "runs": runs,
        },
    }
