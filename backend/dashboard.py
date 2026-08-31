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

_DATASETS = [
    {
        "name": "BTXRD",
        "description": "Ảnh X-quang dùng cho thí nghiệm phân đoạn u xương.",
        "total": 3746,
        "positive": 1867,
        "negative": 1879,
        "train": 2247,
        "validation": 375,
        "development_calibration": 375,
        "final_calibration": 375,
        "test": 374,
    },
    {
        "name": "FracAtlas",
        "description": "Ảnh X-quang cơ xương có nhãn gãy xương, dùng làm benchmark bổ sung.",
        "total": 4024,
        "positive": 719,
        "negative": 3305,
        "train": 2414,
        "validation": 403,
        "development_calibration": 403,
        "final_calibration": 402,
        "test": 402,
    },
]

_FRACATLAS_ABLATION_PLACEHOLDER = [
    {"method": "Supervised", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Mean Teacher", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "+ Entropy weighting", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Global CRC", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Adaptive CRC", "dice": "—", "iou": "—", "hd95": "—"},
    {"method": "Boundary-Adaptive CRC", "dice": "—", "iou": "—", "hd95": "—"},
]


def get_dashboard_payload(*, live_models: list[dict[str, object]] | None = None) -> dict[str, Any]:
    live_models = live_models or []
    return {
        "thesis_title": "Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc Transformer",
        "disclaimer": "Kết quả chỉ phục vụ minh họa nghiên cứu và hậu kiểm mô hình; không phải kết luận chẩn đoán lâm sàng.",
        "overview": {
            "datasets": _DATASETS,
            "default_model_id": live_models[0]["model_id"] if live_models else None,
        },
        "experiments": {
            "btxrd_backbones": _BTXRD_BACKBONES,
            "btxrd_ablation": _BTXRD_ABLATION,
            "fracatlas_backbones": _FRACATLAS_BACKBONES,
            "fracatlas_ablation": _FRACATLAS_ABLATION_PLACEHOLDER,
        },
        "models": {"available": len(live_models)},
    }
