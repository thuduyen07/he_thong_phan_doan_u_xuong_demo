from __future__ import annotations

import csv
import json
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from backend.inference.image_ops import blend_rgb, colorize_heatmap, save_grayscale_png, save_rgb_png
from backend.inference.registry import build_predictor, list_model_records, list_model_specs
from backend.inference.runtime_paths import APP_DIR, RESOURCES_DIR, RUNTIME_STATIC_DIR


UPLOAD_DIR = RUNTIME_STATIC_DIR / "uploads"
RESULT_DIR = RUNTIME_STATIC_DIR / "results"
MANIFEST_PATH = RUNTIME_STATIC_DIR / "upload_manifest.json"
PATIENT_LABELS_PATH = RESOURCES_DIR / "metadata" / "patient_with_labels.csv"
DEFAULT_DEMO_IMAGE_DIR = APP_DIR / "assets" / "cases" / "original"
SAMPLE_DIR = RESOURCES_DIR / "samples"
SAMPLE_MANIFEST_PATH = SAMPLE_DIR / "samples.yaml"
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PREDICTOR_CACHE: dict[str, dict[str, object]] = {}
PREDICTOR_CACHE_LOCK = threading.Lock()


def _resolve_default_model_id() -> str | None:
    available_ids = {spec.model_id for spec in list_model_specs()}
    if available_ids:
        return sorted(available_ids)[0]
    return None


DEFAULT_MODEL_ID = _resolve_default_model_id()
WARMUP_STATE = {
    "enabled": DEFAULT_MODEL_ID is not None,
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
    ext = Path(file_storage.filename or "upload.png").suffix.lower()
    if ext not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("Chỉ chấp nhận ảnh PNG, JPEG, BMP, TIFF hoặc WebP.")
    try:
        with Image.open(file_storage.stream) as candidate:
            candidate.verify()
        file_storage.stream.seek(0)
    except Exception as exc:
        raise ValueError("Tệp tải lên không phải ảnh hợp lệ.") from exc
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
    }
    manifest = load_manifest()
    manifest.append(item)
    save_manifest(manifest)
    return item


def list_uploaded_images() -> list[dict]:
    return load_manifest()


def list_default_demo_images() -> list[dict]:
    if not SAMPLE_MANIFEST_PATH.exists():
        return []
    payload = yaml.safe_load(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    entries = payload.get("samples", []) if isinstance(payload, dict) else []
    samples: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        relative_file = str(entry.get("file", "")).strip()
        sample_id = str(entry.get("id", "")).strip()
        if not relative_file or not sample_id:
            continue
        candidate = (SAMPLE_DIR / relative_file).resolve()
        if SAMPLE_DIR.resolve() not in candidate.parents or not candidate.is_file():
            continue
        samples.append(
            {
                "id": f"default:{sample_id}",
                "filename": Path(relative_file).name,
                "display_name": str(entry.get("display_name", sample_id)),
                "stored_name": relative_file,
                "image_url": f"/sample-assets/{relative_file}",
                "source": "default",
            }
        )
    return samples


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
        candidate = (SAMPLE_DIR / image_item["stored_name"]).resolve()
        if SAMPLE_DIR.resolve() not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError("Curated sample asset was not found.")
        return candidate
    return UPLOAD_DIR / image_item["stored_name"]


def available_models() -> list[dict]:
    return list_model_records()


def _save_rgb_png(path: Path, image: np.ndarray) -> None:
    save_rgb_png(path, image)


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    mask_uint8 = mask.astype(np.uint8)
    max_value = int(mask_uint8.max()) if mask_uint8.size else 0
    if max_value <= 0:
        preview = mask_uint8
    elif max_value == 1:
        preview = mask_uint8 * 255
    else:
        preview = np.rint(mask_uint8.astype(np.float32) * (255.0 / max_value)).astype(np.uint8)
    save_grayscale_png(path, preview)


def _save_heatmap_overlay(path: Path, original_rgb: np.ndarray, values: np.ndarray) -> None:
    colored = colorize_heatmap(values)
    overlay = blend_rgb(original_rgb, colored, alpha=0.55)
    _save_rgb_png(path, overlay)


def _build_uncertainty_payload(result_id: str, artifacts) -> dict:
    summary = dict(artifacts.extra_metadata.get("uncertainty_summary") or {})
    conformal_summary = dict(artifacts.extra_metadata.get("conformal_summary") or {})
    auxiliary_maps = artifacts.auxiliary_maps or {}
    if not summary or not auxiliary_maps:
        return {
            "available": False,
            "summary": {},
            "heatmaps": {},
            "conformal": {
                "available": False,
                "summary": {},
                "heatmaps": {},
                "note": "Chưa có conformal output cho ảnh này.",
            },
            "note": "Backend live hiện chưa sinh được uncertainty hậu kiểm cho ảnh này.",
        }

    heatmap_urls: dict[str, str] = {}
    conformal_heatmap_urls: dict[str, str] = {}
    for map_name, map_values in auxiliary_maps.items():
        heatmap_name = f"{result_id}_{map_name}_heatmap.png"
        heatmap_path = RESULT_DIR / heatmap_name
        _save_heatmap_overlay(heatmap_path, artifacts.original, map_values)
        target = conformal_heatmap_urls if map_name.startswith("conformal_") else heatmap_urls
        target[map_name] = f"/runtime_static/results/{heatmap_name}"

    note = (
        "H_G là predictive entropy nhị phân chuẩn hóa của ảnh vừa segment; "
        "không phải xác suất chẩn đoán lâm sàng."
    )
    conformal_available = bool(conformal_summary.get("available"))
    conformal_note = (
        "Checkpoint này không dùng CRC/conformal khi suy luận live; "
        "web không tạo risk-control metadata thay thế."
    )

    return {
        "available": True,
        "summary": summary,
        "heatmaps": heatmap_urls,
        "conformal": {
            "available": conformal_available,
            "summary": conformal_summary,
            "heatmaps": conformal_heatmap_urls,
            "note": conformal_note,
        },
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
            # Render runs one worker; retaining another full PyTorch model is needless on model switches.
            PREDICTOR_CACHE.clear()
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
    if DEFAULT_MODEL_ID is None:
        WARMUP_STATE["completed"] = True
        return
    try:
        _eager_load_predictor(DEFAULT_MODEL_ID)
        WARMUP_STATE["completed"] = True
    except Exception as exc:
        WARMUP_STATE["error"] = str(exc)


def start_default_predictor_warmup() -> None:
    global WARMUP_THREAD
    if WARMUP_STATE["started"] or DEFAULT_MODEL_ID is None:
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
    if selected_model_id is None:
        raise RuntimeError("No checkpoint is currently available in this demo.")
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
        "original_image_url": image_item["image_url"],
        "original_filename": image_item.get("filename") or image_item["stored_name"],
        "overlay_image_url": f"/runtime_static/results/{overlay_name}",
        "mask_image_url": f"/runtime_static/results/{mask_name}",
        "note": artifacts.note,
        "metadata": artifacts.extra_metadata,
        "uncertainty": uncertainty_payload,
    }
