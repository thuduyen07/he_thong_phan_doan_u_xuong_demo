from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.inference.service import (
    APP_DIR,
    RUNTIME_STATIC_DIR,
    available_models,
    delete_uploaded_image,
    get_warmup_state,
    list_demo_images,
    run_segmentation,
    start_default_predictor_warmup,
    store_upload,
)


start_default_predictor_warmup()

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "predictor_warmup": get_warmup_state(),
        }
    )


@app.get("/models")
def models():
    return jsonify(available_models())


@app.get("/get_images")
def get_images():
    return jsonify(list_demo_images())


@app.post("/upload_images")
def upload_images():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files were uploaded under `images`."}), 400
    return jsonify([store_upload(file_storage) for file_storage in files])


@app.delete("/images/<image_id>")
def delete_image(image_id: str):
    return jsonify(delete_uploaded_image(image_id))


@app.post("/segment")
def segment():
    payload = request.get_json(silent=True) or {}
    result = run_segmentation(
        image_id=payload.get("image_id"),
        model_id=payload.get("model_id"),
    )
    return jsonify(result)


@app.get("/runtime_static/<path:filename>")
def runtime_static(filename: str):
    return send_from_directory(RUNTIME_STATIC_DIR, filename)


@app.get("/app/<path:filename>")
def app_files(filename: str):
    return send_from_directory(APP_DIR, filename)


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/<path:filename>")
def app_router(filename: str):
    candidate = APP_DIR / filename
    if candidate.exists() and candidate.is_file():
        return send_from_directory(APP_DIR, filename)
    return send_from_directory(APP_DIR, "index.html")


@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({"error": str(error)}), 500
