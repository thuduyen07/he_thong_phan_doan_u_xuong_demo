# Hệ thống phân đoạn u xương trên ảnh X-quang

> Demo nghiên cứu cho luận văn. Nội dung hiển thị chỉ phục vụ hậu kiểm mô hình, không phải kết luận chẩn đoán lâm sàng.

## Demo modes

Một codebase hỗ trợ hai mode rõ ràng qua biến môi trường `DEMO_MODE`:

### Static Demo

- Chỉ hiển thị các ảnh curated cùng segmentation, probability/entropy map và metric tham chiếu đã được chuẩn bị offline.
- Không tải checkpoint, không gọi predictor, không cần `torch` hoặc `transformers`.
- `H_G`, `H_B` và CRC chỉ hiện khi metadata đã được materialize từ pipeline nguồn; giá trị thiếu được ghi `—`, không suy diễn từ heatmap.
- Đây là mode khuyến nghị cho Render Free.

### Live Demo

- Bao gồm toàn bộ Static Demo và một tab Live Demo.
- Cho phép upload ảnh PNG/JPEG, chọn model đăng ký và chạy SegFormer theo validation semantics đã kiểm chứng.
- Cần checkpoint, config và `requirements-inference.txt`. Model chỉ lazy-load khi gọi `/segment`.
- Có thể vượt giới hạn RAM/CPU của Render Free; dùng instance lớn hơn khi cần inference thực.

`DEMO_MODE` chỉ nhận `static` hoặc `live`; giá trị khác làm ứng dụng fail fast khi khởi động.

## Chạy local

### Static

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
DEMO_MODE=static ./dev_scripts/run_local.sh
```

Mở `http://127.0.0.1:4173`. Static mode không gọi `/models` hoặc `/segment` từ frontend.

### Live

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-inference.txt
DEMO_MODE=live ./dev_scripts/run_local.sh
```

Kiểm tra `resources/metadata/models.yaml`, config và checkpoint được đăng ký trước khi dùng tab Live.

## Docker

### Static image

```bash
docker build --build-arg INSTALL_INFERENCE=false -t bone-seg-demo-static .
docker run --rm -p 10000:10000 -e DEMO_MODE=static bone-seg-demo-static
```

### Live image

```bash
docker build --build-arg INSTALL_INFERENCE=true -t bone-seg-demo-live .
docker run --rm -p 10000:10000 -e DEMO_MODE=live bone-seg-demo-live
```

`requirements.txt` là base/static dependency set. `requirements-inference.txt` bao gồm base requirements cùng NumPy, OpenCV, PyTorch và Transformers.

## Render

### Static Demo on Render Free

Tạo một Docker Web Service từ repository này, dùng build mặc định hoặc `--build-arg INSTALL_INFERENCE=false`, đặt environment variable `DEMO_MODE=static`, và start command mặc định của Dockerfile. Static assets nằm trong `resources/samples/`; không upload checkpoint hay full dataset.

### Live Demo on Render

Dùng Docker build arg `INSTALL_INFERENCE=true`, đặt `DEMO_MODE=live`, cung cấp checkpoint/config đúng registry, và giữ Gunicorn một worker. Live inference không được cam kết chạy ổn định trên Render Free vì PyTorch + SegFormer có thể vượt RAM hoặc cold-start budget.

## Curated static samples

`resources/samples/samples.yaml` là nguồn dữ liệu duy nhất cho Static Demo. Mỗi entry ghi case type, metric/provenance và `static_artifact_dir`; browser chỉ nhận URL tương đối dưới `/sample-assets/`.

Tạo hoặc cập nhật danh sách ảnh đã được chấp thuận, de-identified:

```bash
python tools/curate_demo_samples.py \
  --dataset btxrd --method mean_teacher_entropy --seed 42 --split test \
  --metrics-csv /path/to/per_case_metrics.csv \
  --source-root /path/to/research/outputs \
  --confirm-public-deidentified
```

Khi có môi trường inference hợp lệ, materialize hoặc làm giàu artifacts cho tối đa các sample đã curated (không chạy full test set):

```bash
DEMO_MODE=live python tools/curate_demo_samples.py \
  --prepare-static-artifacts \
  --static-model-id btxrd_segformer_b0_entropy
```

Tool dùng một predictor cache và chỉ xử lý unique curated images. Không chạy tool này khi web khởi động.

## API capability

- `/capabilities` trả mode và capability rõ ràng cho frontend.
- Static mode: `/static-samples` và `/static-samples/<id>` hoạt động; `/models`, `/upload_images`, `/get_images`, `/segment` trả `FEATURE_DISABLED`.
- Live mode: giữ Static Demo, đồng thời mở model/image/live endpoints.

## Troubleshooting

- `STATIC_SAMPLE_UNAVAILABLE`: manifest hoặc artifact deployable bị thiếu; kiểm tra `resources/samples/`.
- `MODEL_UNAVAILABLE`: registry/config/checkpoint live không sẵn sàng.
- `DEPENDENCY_MISSING`: cài `requirements-inference.txt` và dùng `DEMO_MODE=live`.
- `FILE_TOO_LARGE` hoặc `INVALID_IMAGE`: dùng PNG/JPEG hợp lệ không quá 10 MB.
- `INFERENCE_FAILED`: thử lại với ảnh hoặc model khác; Static Demo vẫn dùng được.

## Scientific notes

- SegFormer binary inference: grayscale -> bilinear resize 512 -> 3 channels -> ImageNet normalization -> sigmoid -> evaluation threshold.
- Predictive entropy là entropy nhị phân chuẩn hóa, không phải diagnostic confidence.
- `H_B` chỉ có ý nghĩa khi boundary mechanism/config hỗ trợ và biên dự đoán không rỗng.
- CRC/conformal chỉ được hiển thị khi backend/static metadata có artifact checkpoint-specific hợp lệ.
