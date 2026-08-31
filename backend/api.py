from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory

from backend.dashboard import get_dashboard_payload
from backend.inference.service import (
    APP_DIR,
    RUNTIME_STATIC_DIR,
    SAMPLE_DIR,
    available_models,
    delete_uploaded_image,
    get_warmup_state,
    list_demo_images,
    run_segmentation,
    store_upload,
)
from backend.inference.environment import validate_inference_environment


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


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
    return jsonify({"models": available_models(), "inference_runtime": validate_inference_environment()})


@app.get("/api/models")
def api_models():
    return jsonify({"models": available_models(), "inference_runtime": validate_inference_environment()})


@app.get("/dashboard")
def dashboard():
    return jsonify(get_dashboard_payload())


@app.get("/api/experiments")
def api_experiments():
    return jsonify(get_dashboard_payload()["experiments"])


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
    runtime = validate_inference_environment()
    if not runtime["available"]:
        app.logger.error("Live inference dependency import failed: %s", runtime["missing_dependencies"])
        return jsonify({"error": "DEPENDENCY_MISSING: Server chưa cài đầy đủ dependency cho live inference."}), 503
    try:
        result = run_segmentation(
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
        return jsonify({"error": "CHECKPOINT_NOT_FOUND: Không tìm thấy artifact model đã đăng ký."}), 503
    except ValueError:
        app.logger.exception("Live inference model configuration is invalid")
        return jsonify({"error": "MODEL_CONFIG_INVALID: Cấu hình model không hợp lệ."}), 503
    except RuntimeError:
        app.logger.exception("Live inference model initialization or execution failed")
        return jsonify({"error": "MODEL_INITIALIZATION_FAILED: Không thể khởi tạo hoặc chạy model."}), 503
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
    return jsonify({"error": "Ảnh tải lên vượt giới hạn 10 MB."}), 413


@app.errorhandler(ValueError)
def invalid_request(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(Exception)
def handle_error(error):
    app.logger.exception("Unhandled demo error")
    return jsonify({"error": "Không thể xử lý yêu cầu. Hãy kiểm tra ảnh hoặc cấu hình model."}), 500
