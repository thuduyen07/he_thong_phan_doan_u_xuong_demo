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
def _legacy_dashboard_payload() -> dict[str, Any]:
    runs = _discover_runs()

    return {
        "thesis_title": "Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc transformer",
        "hero_metrics": _hero_metrics(runs),
        "overview": {
            "summary": [
                "Web demo này chỉ giữ các phần bám trực tiếp vào pipeline supervised segmentation -> mean teacher -> uncertainty -> adaptive CRC.",
                # "Các checkpoint và artifact thực nghiệm của phương pháp cũ đã được gỡ khỏi repo demo này.",
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


# These values are the verified thesis tables exported from the current research
# repository. They deliberately do not depend on checkpoint discovery at startup.
_BTXRD_BACKBONES = [
    {"model": "U-Net", "dice": 0.444, "iou": 0.344, "hd95": 127.869, "precision": 0.479, "recall": 0.557},
    {"model": "Swin-UNet", "dice": 0.508, "iou": 0.418, "hd95": 76.908, "precision": 0.556, "recall": 0.544},
    {"model": "EfficientViT", "dice": 0.541, "iou": 0.450, "hd95": 54.525, "precision": 0.635, "recall": 0.542},
    {"model": "YOLOv8-Seg", "dice": 0.494, "iou": 0.420, "hd95": 107.454, "precision": 0.499, "recall": 0.521},
    {"model": "SegFormer-B0", "dice": 0.587, "iou": 0.485, "hd95": 74.833, "precision": 0.602, "recall": 0.658},
]
_BTXRD_ABLATION = [
    {"method": "Supervised", "dice": 0.587, "iou": 0.485, "hd95": 74.833},
    {"method": "Mean Teacher", "dice": 0.592, "iou": 0.488, "hd95": 76.705},
    {"method": "Mean Teacher + entropy", "dice": 0.593, "iou": 0.490, "hd95": 64.621},
    {"method": "Global CRC", "dice": 0.570, "iou": 0.477, "hd95": 67.772},
    {"method": "Adaptive CRC", "dice": 0.562, "iou": 0.466, "hd95": 61.290},
    {"method": "Boundary-Adaptive CRC", "dice": 0.582, "iou": 0.481, "hd95": 68.455},
]
_FRACATLAS_BACKBONES = [
    {"model": "U-Net", "dice": 0.311, "iou": 0.230, "hd95": 105.657, "precision": 0.357, "recall": 0.316},
    {"model": "Swin-UNet", "dice": 0.401, "iou": 0.300, "hd95": 63.286, "precision": 0.442, "recall": 0.450},
    {"model": "EfficientViT", "dice": 0.326, "iou": 0.242, "hd95": 91.140, "precision": 0.375, "recall": 0.345},
    {"model": "YOLOv8-Seg", "dice": 0.281, "iou": 0.215, "hd95": 69.767, "precision": 0.282, "recall": 0.301},
    {"model": "SegFormer-B0", "dice": 0.457, "iou": 0.350, "hd95": 50.201, "precision": 0.476, "recall": 0.510},
]


def get_dashboard_payload() -> dict[str, Any]:
    live_models = available_models()
    return {
        "thesis_title": "Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc Transformer",
        "disclaimer": "Kết quả chỉ phục vụ minh họa nghiên cứu và hậu kiểm mô hình; không phải kết luận chẩn đoán lâm sàng.",
        "overview": {
            "summary": [
                "Bài toán là phân đoạn nhị phân vùng tổn thương trên ảnh X-quang, không phải chẩn đoán hoặc phân loại ác tính.",
                "SegFormer-B0 là kiến trúc chính; Mean Teacher, entropy và các CRC variants là các nhánh thực nghiệm có trade-off riêng.",
                "Với đầu ra một channel, foreground probability dùng sigmoid và predictive entropy nhị phân được chuẩn hóa về [0, 1].",
            ],
            "datasets": [
                {"name": "BTXRD", "description": "3.746 ảnh X-quang u xương có nhãn (1.867 dương tính, 1.879 âm tính).", "split": "seed 42, chia theo mẫu; train 2.247, val/dev/final calibration mỗi tập 375, test 374."},
                {"name": "FracAtlas", "description": "Benchmark phân đoạn gãy xương, không phải bộ dữ liệu u xương.", "split": "seed 42, chia theo mẫu; train 2.414, val 403, dev calibration 403, final calibration 402, test 402."},
            ],
            "default_model_id": live_models[0]["model_id"] if live_models else None,
        },
        "pipeline": {
            "modules": [
                {"title": "Supervised SegFormer-B0", "summary": "BCEWithLogits + Dice trên mask tham chiếu; FracAtlas có recipe positive weight 5.6 riêng.", "details": ["Một logit foreground", "sigmoid ở binary path", "checkpoint selection là run-specific"]},
                {"title": "EMA Mean Teacher", "summary": "Teacher EMA sinh pseudo-mask từ weak view; student học strong view.", "details": ["Hard threshold 0.5", "teacher không nhận gradient", "EMA sau optimizer step"]},
                {"title": "Predictive entropy", "summary": "H_G là mean entropy toàn ảnh; H_B là mean entropy trên vùng biên pseudo-mask dự đoán.", "details": ["H(p)/log(2)", "không phải epistemic uncertainty", "boundary không dùng ground truth"]},
                {"title": "CRC variants", "summary": "Global/Adaptive/Boundary-Adaptive CRC là variants kiểm soát rủi ro FNR cho pseudo-supervision.", "details": ["target FNR 0.010 trong config hiện hành", "reliable/ambiguous/rejected", "không tự động tối ưu Dice"]},
            ],
            "formulas": [
                {"label": "Binary probability", "value": "p = sigmoid(z)"},
                {"label": "Predictive entropy", "value": "H(p) = [-p log p - (1-p) log(1-p)] / log(2)"},
                {"label": "Pseudo-label weight", "value": "w = w_conformal × clamp(1 - H, entropy_floor, 1) × w_boundary (when enabled)"},
            ],
        },
        "experiments": {
            "btxrd_backbones": _BTXRD_BACKBONES,
            "btxrd_ablation": _BTXRD_ABLATION,
            "fracatlas_backbones": _FRACATLAS_BACKBONES,
            "notes": [
                "BTXRD ablation dùng seed 42: Mean Teacher + entropy có Dice/IoU cao nhất; Adaptive CRC có HD95 thấp nhất.",
                "CRC variants biểu diễn trade-off, không phải chuỗi cải thiện cộng dồn.",
                "Không hiển thị FracAtlas ablation vì chưa có bảng final đã kiểm chứng cùng protocol.",
            ],
        },
        "models": {"available": len(live_models)},
    }
