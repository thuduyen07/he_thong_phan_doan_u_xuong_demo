from __future__ import annotations

from typing import Any

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

_DATASET_SPLITS = [
    {
        "id": "btxrd",
        "name": "BTXRD",
        "description": "Ảnh X-quang dùng cho thí nghiệm phân đoạn u xương.",
        "title": "Phân bố dữ liệu BTXRD trong các tập thực nghiệm với seed 42.",
        "rows": [
            {"split": "Huấn luyện", "images": 2247, "positive": 1120, "negative": 1127, "ratio": "60%"},
            {"split": "Xác thực", "images": 375, "positive": 187, "negative": 188, "ratio": "10%"},
            {"split": "Hiệu chỉnh phát triển", "images": 375, "positive": 187, "negative": 188, "ratio": "10%"},
            {"split": "Hiệu chỉnh cuối", "images": 375, "positive": 187, "negative": 188, "ratio": "10%"},
            {"split": "Kiểm thử", "images": 374, "positive": 186, "negative": 188, "ratio": "10%"},
            {"split": "Tổng", "images": 3746, "positive": 1867, "negative": 1879, "ratio": "100%", "is_total": True},
        ],
    },
    {
        "id": "fracatlas",
        "name": "FracAtlas",
        "description": "Ảnh X-quang cơ xương có nhãn gãy xương, dùng làm benchmark bổ sung.",
        "title": "Phân bố dữ liệu FracAtlas trong các tập thực nghiệm với seed 42.",
        "rows": [
            {"split": "Huấn luyện", "images": 2414, "positive": 431, "negative": 1983, "ratio": "60%"},
            {"split": "Xác thực", "images": 403, "positive": 72, "negative": 331, "ratio": "10%"},
            {"split": "Hiệu chỉnh phát triển", "images": 403, "positive": 72, "negative": 331, "ratio": "10%"},
            {"split": "Hiệu chỉnh cuối", "images": 402, "positive": 72, "negative": 330, "ratio": "10%"},
            {"split": "Kiểm thử", "images": 402, "positive": 72, "negative": 330, "ratio": "10%"},
            {"split": "Tổng", "images": 4024, "positive": 719, "negative": 3305, "ratio": "100%", "is_total": True},
        ],
    },
]


def _dataset_overview_rows() -> list[dict[str, object]]:
    """Compatibility summary derived from the explicit split-table source."""
    records = []
    for dataset in _DATASET_SPLITS:
        total = dataset["rows"][-1]
        records.append(
            {
                "name": dataset["name"],
                "description": dataset["description"],
                "total": total["images"],
                "positive": total["positive"],
                "negative": total["negative"],
                "train": dataset["rows"][0]["images"],
                "validation": dataset["rows"][1]["images"],
                "development_calibration": dataset["rows"][2]["images"],
                "final_calibration": dataset["rows"][3]["images"],
                "test": dataset["rows"][4]["images"],
            }
        )
    return records

_FRACATLAS_ABLATION_PLACEHOLDER = [
    {"method": "Supervised", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Mean Teacher", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "+ Entropy weighting", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Global CRC", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Adaptive CRC", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Boundary-Adaptive CRC", "dice": "—", "iou": "—", "hd95": "—"},
]

_EXPERIMENT_SECTIONS = [
    {
        "id": "btxrd_backbones",
        "title": "BTXRD: backbone comparison",
        "row_label": "Model",
        "row_key": "model",
        "metrics": [
            {"key": "dice", "label": "Dice ↑", "direction": "max"},
            {"key": "iou", "label": "IoU ↑", "direction": "max"},
            {"key": "precision", "label": "Precision ↑", "direction": "max"},
            {"key": "recall", "label": "Recall ↑", "direction": "max"},
            {"key": "hd95", "label": "HD95 ↓", "direction": "min"},
        ],
        "rows": _BTXRD_BACKBONES,
        "primary_value": "SegFormer-B0",
        "primary_badge": "Phương pháp đề xuất",
    },
    {
        "id": "btxrd_ablation",
        "title": "BTXRD: SegFormer-B0 ablation",
        "row_label": "Phương pháp",
        "row_key": "method",
        "metrics": [
            {"key": "dice", "label": "Dice ↑", "direction": "max"},
            {"key": "iou", "label": "IoU ↑", "direction": "max"},
            {"key": "hd95", "label": "HD95 ↓", "direction": "min"},
        ],
        "rows": _BTXRD_ABLATION,
        "primary_value": "Boundary-Adaptive CRC",
        "primary_badge": "Phương pháp đề xuất",
    },
    {
        "id": "fracatlas_backbones",
        "title": "FracAtlas: backbone comparison",
        "row_label": "Model",
        "row_key": "model",
        "metrics": [
            {"key": "dice", "label": "Dice ↑", "direction": "max"},
            {"key": "iou", "label": "IoU ↑", "direction": "max"},
            {"key": "precision", "label": "Precision ↑", "direction": "max"},
            {"key": "recall", "label": "Recall ↑", "direction": "max"},
            {"key": "hd95", "label": "HD95 ↓", "direction": "min"},
        ],
        "rows": _FRACATLAS_BACKBONES,
        "primary_value": "SegFormer-B0",
        "primary_badge": "Phương pháp đề xuất",
    },
    {
        "id": "fracatlas_ablation",
        "title": "FracAtlas: SegFormer-B0 ablation",
        "row_label": "Phương pháp",
        "row_key": "method",
        "metrics": [
            {"key": "dice", "label": "Dice ↑", "direction": "max"},
            {"key": "iou", "label": "IoU ↑", "direction": "max"},
            {"key": "hd95", "label": "HD95 ↓", "direction": "min"},
        ],
        "rows": _FRACATLAS_ABLATION_PLACEHOLDER,
        "note": "Kết quả đang được cập nhật.",
    },
]


def get_dashboard_payload(*, live_models: list[dict[str, object]] | None = None) -> dict[str, Any]:
    live_models = live_models or []
    return {
        "thesis_title": "Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc Transformer",
        "disclaimer": "Kết quả chỉ phục vụ minh họa nghiên cứu và hậu kiểm mô hình; không phải kết luận chẩn đoán lâm sàng.",
        "overview": {
            "dataset_splits": _DATASET_SPLITS,
            "datasets": _dataset_overview_rows(),
            "default_model_id": live_models[0]["model_id"] if live_models else None,
        },
        "experiments": {
            "btxrd_backbones": _BTXRD_BACKBONES,
            "btxrd_ablation": _BTXRD_ABLATION,
            "fracatlas_backbones": _FRACATLAS_BACKBONES,
            "fracatlas_ablation": _FRACATLAS_ABLATION_PLACEHOLDER,
            "sections": _EXPERIMENT_SECTIONS,
        },
        "models": {"available": len(live_models)},
    }
