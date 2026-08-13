from __future__ import annotations

import csv
import json
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.inference.registry import build_predictor, list_model_specs
from backend.inference.runtime_paths import APP_DIR, RESOURCES_DIR, RUNTIME_STATIC_DIR


UPLOAD_DIR = RUNTIME_STATIC_DIR / "uploads"
RESULT_DIR = RUNTIME_STATIC_DIR / "results"
MANIFEST_PATH = RUNTIME_STATIC_DIR / "upload_manifest.json"
PATIENT_LABELS_PATH = RESOURCES_DIR / "metadata" / "patient_with_labels.csv"
DEFAULT_DEMO_IMAGE_DIR = APP_DIR / "assets" / "cases" / "original"
PREDICTOR_CACHE: dict[str, dict[str, object]] = {}
PREDICTOR_CACHE_LOCK = threading.Lock()


def _resolve_default_model_id() -> str:
    available_ids = {spec.model_id for spec in list_model_specs()}
    for candidate in (
        "thesis_20260809_ce_weights_seg_seed_v2_ce_weights_ce_weights_1_1_seed_42_20260809",
        "thesis_20260809_ce_weights_seg_seed_v2_ce_weights_ce_weights_1_3_seed_42_20260809",
        "thesis_20260809_baseline_seg_seed_v2_baseline_seed_42_20260809",
        "thesis_20260809_baseline_unet_seed_v2_baseline_seed_42_20260809",
    ):
        if candidate in available_ids:
            return candidate
    if available_ids:
        return sorted(available_ids)[0]
    return "thesis_20260809_ce_weights_seg_seed_v2_ce_weights_ce_weights_1_1_seed_42_20260809"


DEFAULT_MODEL_ID = _resolve_default_model_id()
WARMUP_STATE = {
    "enabled": True,
    "started": False,
    "completed": False,
    "model_id": DEFAULT_MODEL_ID,
    "error": None,
}
WARMUP_THREAD: threading.Thread | None = None


for directory in (RUNTIME_STATIC_DIR, UPLOAD_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(items: list[dict]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(items, file, indent=2)


@lru_cache(maxsize=1)
def load_patient_label_index() -> dict[str, dict[str, str]]:
    if not PATIENT_LABELS_PATH.exists():
        return {}

    with PATIENT_LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        index: dict[str, dict[str, str]] = {}
        for row in reader:
            patient_id = (row.get("MaHoSo") or "").strip()
            if not patient_id:
                continue
            index[patient_id] = {
                "MaHoSo": patient_id,
                "TongKetBenhAn_ChanDoanXacDinh": (row.get("TongKetBenhAn_ChanDoanXacDinh") or "").strip(),
                "has_tumor": (row.get("has_tumor") or "").strip(),
                "diagnosis_label": (row.get("diagnosis_label") or "").strip(),
            }
        return index


def extract_patient_code(filename: str) -> str | None:
    stem = Path(filename or "").stem.strip()
    match = re.match(r"^([A-Za-z0-9]+)(?:_.*)?$", stem)
    if not match:
        return None
    return match.group(1)


def build_patient_record(filename: str) -> dict | None:
    patient_id = extract_patient_code(filename)
    if not patient_id:
        return None
    return load_patient_label_index().get(patient_id)


def store_upload(file_storage) -> dict:
    ext = Path(file_storage.filename or "upload.png").suffix.lower() or ".png"
    file_id = uuid.uuid4().hex
    stored_name = f"{file_id}{ext}"
    destination = UPLOAD_DIR / stored_name
    file_storage.save(destination)

    item = {
        "id": file_id,
        "filename": file_storage.filename or stored_name,
        "stored_name": stored_name,
        "image_url": f"/runtime_static/uploads/{stored_name}",
        "source": "custom",
        "patient_record": build_patient_record(file_storage.filename or stored_name),
    }
    manifest = load_manifest()
    manifest.append(item)
    save_manifest(manifest)
    return item


def list_uploaded_images() -> list[dict]:
    return load_manifest()


def list_default_demo_images() -> list[dict]:
    if not DEFAULT_DEMO_IMAGE_DIR.exists():
        return []
    supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return [
        {
            "id": f"default:{path.name}",
            "filename": path.name,
            "stored_name": path.name,
            "image_url": f"/app/assets/cases/original/{path.name}",
            "source": "default",
            "patient_record": build_patient_record(path.name),
        }
        for path in sorted(DEFAULT_DEMO_IMAGE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]


def list_demo_images() -> dict[str, list[dict]]:
    return {
        "custom": list_uploaded_images(),
        "default": list_default_demo_images(),
    }


def delete_uploaded_image(image_id: str) -> dict:
    manifest = load_manifest()
    target_item = next((item for item in manifest if item.get("id") == image_id), None)
    if target_item is None:
        raise FileNotFoundError(f"Uploaded image `{image_id}` was not found.")
    remaining = [item for item in manifest if item.get("id") != image_id]
    save_manifest(remaining)
    image_path = UPLOAD_DIR / target_item["stored_name"]
    if image_path.exists():
        image_path.unlink()
    return {"deleted_image_id": image_id}


def resolve_demo_image(image_id: str | None) -> dict:
    image_groups = list_demo_images()
    all_images = [*image_groups["custom"], *image_groups["default"]]
    if not all_images:
        raise FileNotFoundError("No demo image is available.")
    if image_id is None:
        return all_images[-1]
    for item in all_images:
        if item["id"] == image_id:
            return item
    raise FileNotFoundError(f"Demo image `{image_id}` was not found.")


def resolve_demo_image_path(image_item: dict) -> Path:
    if image_item.get("source") == "default":
        return DEFAULT_DEMO_IMAGE_DIR / image_item["stored_name"]
    return UPLOAD_DIR / image_item["stored_name"]


def available_models() -> list[dict]:
    items = []
    for spec in list_model_specs():
        items.append(
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "predictor_type": spec.predictor_type,
                "checkpoint_path": str(spec.checkpoint_path),
                "config_path": str(spec.config_path),
                "note": spec.note,
                "cache_token": spec.cache_token,
                "extra_metadata": spec.extra_metadata or {},
            }
        )
    source_priority = {
        "thesis_local_batch": 0,
    }
    items.sort(
        key=lambda item: (
            source_priority.get(item.get("extra_metadata", {}).get("source"), 9),
            item.get("extra_metadata", {}).get("run_date", ""),
            item.get("extra_metadata", {}).get("phase", ""),
            item["display_name"].lower(),
        )
    )
    return items


def _save_rgb_png(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    mask_uint8 = mask.astype(np.uint8)
    max_value = int(mask_uint8.max()) if mask_uint8.size else 0
    if max_value <= 0:
        preview = mask_uint8
    elif max_value == 1:
        preview = mask_uint8 * 255
    else:
        preview = np.rint(mask_uint8.astype(np.float32) * (255.0 / max_value)).astype(np.uint8)
    cv2.imwrite(str(path), preview)


def _colorize_heatmap(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    clipped = np.clip(arr, 0.0, 1.0)
    heat = np.rint(clipped * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _save_heatmap_overlay(path: Path, original_rgb: np.ndarray, values: np.ndarray) -> None:
    colored = _colorize_heatmap(values)
    overlay = cv2.addWeighted(original_rgb, 0.45, colored, 0.55, 0.0)
    _save_rgb_png(path, overlay)


def _build_uncertainty_payload(result_id: str, artifacts) -> dict:
    summary = dict(artifacts.extra_metadata.get("uncertainty_summary") or {})
    auxiliary_maps = artifacts.auxiliary_maps or {}
    if not summary or not auxiliary_maps:
        return {
            "available": False,
            "summary": {},
            "heatmaps": {},
            "note": "Backend live hiện chưa sinh được uncertainty hậu kiểm cho ảnh này.",
        }

    heatmap_urls: dict[str, str] = {}
    for map_name, map_values in auxiliary_maps.items():
        heatmap_name = f"{result_id}_{map_name}_heatmap.png"
        heatmap_path = RESULT_DIR / heatmap_name
        _save_heatmap_overlay(heatmap_path, artifacts.original, map_values)
        heatmap_urls[map_name] = f"/runtime_static/results/{heatmap_name}"

    predicted_tumor_present = bool(summary.get("predicted_tumor_present"))
    if predicted_tumor_present:
        note = (
            "Các score này được suy ra hậu kiểm từ logits/softmax của ảnh vừa segment, "
            "không phải là xác suất chẩn đoán lâm sàng."
        )
    else:
        note = (
            "Ảnh này không có vùng tumor dự đoán, nên một số score theo vùng tumor sẽ để trống."
        )

    return {
        "available": True,
        "summary": summary,
        "heatmaps": heatmap_urls,
        "note": note,
    }


def get_predictor(model_id: str):
    current_specs = {spec.model_id: spec for spec in list_model_specs()}
    if model_id not in current_specs:
        raise KeyError(f"Unknown model_id: {model_id}")
    spec = current_specs[model_id]
    cache_token = spec.cache_token or ""
    with PREDICTOR_CACHE_LOCK:
        cached_entry = PREDICTOR_CACHE.get(model_id)
        predictor = cached_entry["predictor"] if cached_entry and cached_entry.get("cache_token") == cache_token else None
        if predictor is None:
            predictor = build_predictor(model_id)
            PREDICTOR_CACHE[model_id] = {
                "cache_token": cache_token,
                "predictor": predictor,
            }
        return predictor


def _eager_load_predictor(model_id: str) -> None:
    predictor = get_predictor(model_id)
    load_model = getattr(predictor, "_load_model", None)
    if callable(load_model):
        load_model()


def _run_default_predictor_warmup() -> None:
    WARMUP_STATE["started"] = True
    WARMUP_STATE["completed"] = False
    WARMUP_STATE["error"] = None
    try:
        _eager_load_predictor(DEFAULT_MODEL_ID)
        WARMUP_STATE["completed"] = True
    except Exception as exc:
        WARMUP_STATE["error"] = str(exc)


def start_default_predictor_warmup() -> None:
    global WARMUP_THREAD
    if WARMUP_STATE["started"]:
        return
    WARMUP_THREAD = threading.Thread(
        target=_run_default_predictor_warmup,
        name="standalone-default-predictor-warmup",
        daemon=True,
    )
    WARMUP_THREAD.start()


def get_warmup_state() -> dict:
    return dict(WARMUP_STATE)


def run_segmentation(*, image_id: str | None, model_id: str | None) -> dict:
    selected_model_id = model_id or DEFAULT_MODEL_ID
    current_specs = {spec.model_id: spec for spec in list_model_specs()}
    selected_spec = current_specs.get(selected_model_id)
    image_item = resolve_demo_image(image_id)
    image_path = resolve_demo_image_path(image_item)
    image = Image.open(image_path).convert("L")

    predictor = get_predictor(selected_model_id)
    artifacts = predictor.predict(image)

    result_id = uuid.uuid4().hex
    overlay_name = f"{result_id}_overlay.png"
    mask_name = f"{result_id}_mask.png"
    overlay_path = RESULT_DIR / overlay_name
    mask_path = RESULT_DIR / mask_name
    _save_rgb_png(overlay_path, artifacts.overlay)
    _save_mask_png(mask_path, artifacts.mask)
    uncertainty_payload = _build_uncertainty_payload(result_id, artifacts)

    return {
        "image_id": image_item["id"],
        "model_id": artifacts.model_id,
        "display_name": selected_spec.display_name if selected_spec else artifacts.model_id,
        "checkpoint_name": artifacts.checkpoint_path.name,
        "checkpoint_path": str(artifacts.checkpoint_path),
        "original_image_url": image_item["image_url"],
        "original_filename": image_item.get("filename") or image_item["stored_name"],
        "patient_record": image_item.get("patient_record"),
        "overlay_image_url": f"/runtime_static/results/{overlay_name}",
        "mask_image_url": f"/runtime_static/results/{mask_name}",
        "note": artifacts.note,
        "metadata": artifacts.extra_metadata,
        "uncertainty": uncertainty_payload,
    }
