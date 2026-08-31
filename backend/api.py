from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory

from backend.deployment import DEPLOYMENT
from backend.dashboard import get_dashboard_payload
from backend.inference.runtime_paths import APP_DIR, RUNTIME_STATIC_DIR
from backend.static_samples import SAMPLE_DIR, get_static_result, list_static_samples


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def _live_service():
    if not DEPLOYMENT.live_enabled:
        return None
    # Import only inside live-capable requests so a static deploy needs no ML runtime.
    from backend.inference import service

    return service


def _feature_disabled_response():
    return jsonify({"error": "FEATURE_DISABLED: Live Demo không được bật trong static deployment."}), 404


@app.get("/health")
def health():
    payload = {"status": "ok", **DEPLOYMENT.as_payload()}
    if DEPLOYMENT.live_enabled:
        payload["predictor_warmup"] = _live_service().get_warmup_state()
    return jsonify(payload)


@app.get("/capabilities")
def capabilities():
    return jsonify(DEPLOYMENT.as_payload())


@app.get("/models")
def models():
    if not DEPLOYMENT.live_enabled:
        return _feature_disabled_response()
    from backend.inference.environment import validate_inference_environment

    return jsonify({"models": _live_service().available_models(), "inference_runtime": validate_inference_environment()})


@app.get("/api/models")
def api_models():
    return models()


@app.get("/dashboard")
def dashboard():
    models = _live_service().available_models() if DEPLOYMENT.live_enabled else []
    return jsonify(get_dashboard_payload(live_models=models))


@app.get("/api/experiments")
def api_experiments():
    return jsonify(get_dashboard_payload()["experiments"])


@app.get("/get_images")
def get_images():
    if not DEPLOYMENT.live_enabled:
        return _feature_disabled_response()
    return jsonify(_live_service().list_demo_images())


@app.get("/static-samples")
def static_samples():
    return jsonify({"samples": list_static_samples()})


@app.get("/static-samples/<sample_id>")
def static_sample_result(sample_id: str):
    try:
        return jsonify(get_static_result(sample_id))
    except FileNotFoundError as exc:
        app.logger.warning("Static sample unavailable: %s", exc)
        return jsonify({"error": "STATIC_SAMPLE_UNAVAILABLE: Không thể tải dữ liệu mẫu này."}), 404


@app.post("/upload_images")
def upload_images():
    if not DEPLOYMENT.live_enabled:
        return _feature_disabled_response()
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files were uploaded under `images`."}), 400
    return jsonify([_live_service().store_upload(file_storage) for file_storage in files])


@app.delete("/images/<image_id>")
def delete_image(image_id: str):
    if not DEPLOYMENT.live_enabled:
        return _feature_disabled_response()
    return jsonify(_live_service().delete_uploaded_image(image_id))


@app.post("/segment")
def segment():
    if not DEPLOYMENT.live_enabled:
        return _feature_disabled_response()
    payload = request.get_json(silent=True) or {}
    from backend.inference.environment import validate_inference_environment

    runtime = validate_inference_environment()
    if not runtime["available"]:
        app.logger.error("Live inference dependency import failed: %s", runtime["missing_dependencies"])
        return jsonify({"error": "DEPENDENCY_MISSING: Server chưa cài đầy đủ dependency cho live inference."}), 503
    try:
        result = _live_service().run_segmentation(
            image_id=payload.get("image_id"),
            model_id=payload.get("model_id"),
        )
    except ModuleNotFoundError as exc:
        app.logger.exception("Live inference dependency import failed: %s", exc.name)
        return jsonify({"error": "DEPENDENCY_MISSING: Server chưa cài đầy đủ dependency cho live inference."}), 503
    except ImportError as exc:
        app.logger.exception("Live inference dependency import failed")
        return jsonify({"error": "DEPENDENCY_MISSING: Server chưa cài đầy đủ dependency cho live inference."}), 503
    except FileNotFoundError:
        app.logger.exception("Live inference artifact was not found")
        return jsonify({"error": "MODEL_UNAVAILABLE: Model đã chọn hiện không khả dụng."}), 503
    except KeyError:
        app.logger.exception("Live inference model was not registered")
        return jsonify({"error": "MODEL_UNAVAILABLE: Model đã chọn hiện không khả dụng."}), 404
    except ValueError:
        app.logger.exception("Live inference model configuration is invalid")
        return jsonify({"error": "INVALID_IMAGE: Không thể đọc hoặc xử lý ảnh đã chọn."}), 400
    except RuntimeError:
        app.logger.exception("Live inference model initialization or execution failed")
        return jsonify({"error": "INFERENCE_FAILED: Không thể hoàn tất phân đoạn cho ảnh này."}), 503
    return jsonify(result)


@app.get("/runtime_static/<path:filename>")
def runtime_static(filename: str):
    return send_from_directory(RUNTIME_STATIC_DIR, filename)


@app.get("/sample-assets/<path:filename>")
def sample_assets(filename: str):
    return send_from_directory(SAMPLE_DIR, filename)


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


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "FILE_TOO_LARGE: Ảnh tải lên vượt giới hạn 10 MB."}), 413


@app.errorhandler(ValueError)
def invalid_request(error):
    app.logger.warning("Invalid request: %s", error)
    return jsonify({"error": "INVALID_IMAGE: Không thể đọc ảnh PNG hoặc JPEG hợp lệ."}), 400


@app.errorhandler(Exception)
def handle_error(error):
    app.logger.exception("Unhandled demo error")
    return jsonify({"error": "Không thể xử lý yêu cầu. Hãy kiểm tra ảnh hoặc cấu hình model."}), 500
