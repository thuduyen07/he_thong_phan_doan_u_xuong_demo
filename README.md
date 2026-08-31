# Hệ thống phân đoạn u xương trên ảnh X-quang

> **Phiên bản hiện hành:** Đây là web demo nghiên cứu, không phải phần mềm chẩn đoán lâm sàng.
> **Đề tài:** Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc transformer

## 1. Giới thiệu

Demo trình bày bài toán phân đoạn nhị phân vùng tổn thương trên X-quang, các kết quả thực nghiệm đã kiểm chứng, và trạng thái inference khi có artifact phù hợp.

## 2. Phạm vi nghiên cứu

- Không đưa ra chẩn đoán, phân loại ác tính hoặc khuyến nghị điều trị.
- Xác suất foreground dùng sigmoid cho SegFormer binary one-channel.
- Predictive entropy là entropy nhị phân chuẩn hóa; không phải ước lượng epistemic uncertainty.
- CRC là cơ chế điều tiết pseudo-supervision theo rủi ro FNR đã cấu hình, không phải chứng nhận lâm sàng.

## 3. Kiến trúc hệ thống

Flask phục vụ frontend tĩnh, metadata kết quả và registry model. Model chỉ được lazy-load khi registry đánh dấu `enabled: true` và có đủ config/checkpoint tương ứng. Live SegFormer inference dùng adapter mỏng bám theo validation research: grayscale -> bilinear resize 512 -> lặp 3 kênh -> chuẩn hóa ImageNet -> sigmoid một kênh -> threshold evaluation -> resize nearest-neighbor chỉ để hiển thị.

## 4. Phương pháp

SegFormer-B0 là backbone chính. Mean Teacher dùng EMA teacher, pseudo-mask threshold 0.5, entropy weighting và các variants Global CRC, Adaptive CRC, Boundary-Adaptive CRC. H_B là entropy trung bình trên vùng biên của pseudo-mask dự đoán, không dùng ground truth.

## 5. Cấu trúc thư mục

- `backend/`: Flask API, dashboard và inference service.
- `app/`: giao diện HTML/CSS/JS.
- `resources/metadata/models.yaml`: registry model khai báo bằng dữ liệu.
- `resources/checkpoints/`, `resources/configs/`: artifact local, bị gitignore.

## 6. Chuẩn bị checkpoint

Không commit checkpoint. Chỉ enable một model khi config, checkpoint, threshold và calibration artifact (nếu có) cùng checkpoint được xác nhận. Không ghép CRC artifact từ predictor khác.

## 7. Chạy local trên macOS

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt -r requirements-inference.txt
./dev_scripts/run_local.sh
```

Mở `http://127.0.0.1:4173`. Nếu registry không có model hợp lệ, dashboard vẫn hoạt động và live inference được vô hiệu hóa rõ ràng.

## 8. Chạy Docker

```bash
docker compose up --build
```

## 9. Deploy Render Free

Deploy từ Dockerfile với một Gunicorn worker và hai threads. Docker mặc định cài `torch` và `transformers` để hỗ trợ live inference; dùng `--build-arg INSTALL_INFERENCE=false` chỉ khi muốn dashboard-only. Render Free có thể không đủ RAM cho PyTorch + Transformers CPU trong lúc load SegFormer; cần kiểm tra giới hạn instance thực tế hoặc dùng Render paid cho live inference. Health endpoint vẫn khởi động trước khi model được load.

## 10. Thêm model mới

Thêm một entry trong `resources/metadata/models.yaml`, sau đó cung cấp artifact local đúng path. Entry phải chỉ rõ dataset, architecture, method, threshold và calibration artifact checkpoint-specific khi áp dụng.

## Curate demo samples

Built-in X-ray samples are read only from `resources/samples/samples.yaml`. Curate only approved, de-identified source images before deployment:

```bash
python tools/curate_demo_samples.py \
  --dataset btxrd --method mean_teacher_entropy --seed 42 --split test \
  --metrics-csv /path/to/per_case_metrics.csv \
  --source-root /path/to/research/outputs \
  --confirm-public-deidentified
```

The script limits the curation to at most ten unique images and records metric provenance without embedding source filesystem paths.

## 11. Giới hạn

- Không hiển thị Dice, IoU, HD95 hay FNR cho ảnh upload không có mask tham chiếu.
- Kết quả BTXRD ablation là seed 42; CRC variants thể hiện trade-off, không phải cải thiện tuyệt đối.
- FracAtlas là benchmark fracture, không phải dữ liệu u xương.
- Adapter live SegFormer đã được kiểm chứng parity một ảnh với research validation semantics. CRC model chỉ được bật khi có adapter/state artifact nghiên cứu tương ứng; demo không tự fallback sang rule 0.5.

## 1. Mục đích của hệ thống

Đây là bản web demo tập trung riêng vào pipeline thực nghiệm của luận văn:

`Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc transformer`

Hệ thống hỗ trợ hai mục tiêu chính:

- hiển thị pipeline supervised -> mean teacher -> uncertainty -> adaptive CRC
- chạy thử phân đoạn trực tiếp trên ảnh X-quang mới do người dùng tải lên bằng backend cục bộ và các checkpoint đã được giữ lại từ phần thực nghiệm chính thức

## 2. Các nội dung có trên web

Khi mở web, người dùng có thể xem các phần sau:

- phần mô tả kiến trúc pipeline và lịch thực nghiệm
- dashboard đọc phần artifact còn lại trong `resources/outputs` sau khi đã gỡ bộ phương pháp cũ
- chế độ chạy live inference trên ảnh tải lên khi repo được bổ sung checkpoint mới phù hợp

## 3. Chuẩn bị các checkpoint trước khi chạy

Nếu đang dùng một bản clone mới hoặc một máy chưa có sẵn các file pretrained, hãy đảm bảo các checkpoint và artifact cần thiết đã nằm trực tiếp trong thư mục `resources/pretrained/` hoặc được tải xuống từ nguồn cung cấp tương ứng trước khi khởi động hệ thống.

Ví dụ, các file cần có thể nằm trong:

```text
resources/pretrained/segformer_b0_ade_512_512/
```

## 4. Cách khởi động nhanh bằng Docker Compose

```bash
docker compose up --build
```

Sau khi container khởi động, mở trình duyệt tại:

```text
http://127.0.0.1:4173
```

## 5. Cách khởi động trực tiếp bằng Python

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt -r requirements-inference.txt
./dev_scripts/run_local.sh
```

`requirements-inference.txt` là phần dependency bắt buộc cho live inference (không cần cho dashboard-only):

```bash
python3 -m pip install -r requirements-inference.txt
```

## 6. Luồng sử dụng chính

1. Mở web và xem phần `Pipeline` để nắm kiến trúc supervised -> mean teacher -> uncertainty -> adaptive CRC.
2. Xem phần `Experiments` để kiểm tra các artifact còn lại sau khi dọn phương pháp cũ.
3. Khi có checkpoint mới trong repo demo, tải ảnh mới ở phần `Live Lab`, chọn checkpoint, rồi chạy segmentation.
4. Live Lab hiển thị mask, probability map, predictive-entropy map và `H_G` được tính theo semantics đánh giá của checkpoint. `H_B` chỉ hiện giá trị khi boundary mechanism được bật trong config; CRC/conformal chỉ hiện khi có artifact checkpoint-specific hợp lệ. Đây là metadata hậu kiểm của mô hình, không phải độ tin cậy chẩn đoán lâm sàng.

## 7. Cấu trúc thư mục chính

- `server.py`: điểm khởi động của toàn bộ web
- `backend/`: API và backend inference cục bộ
- `app/`: giao diện trình diễn tối giản theo pipeline
- `resources/`: checkpoint, metadata, conformal profile, và artifact thực nghiệm
- `runtime_src/`: mã nguồn tối thiểu cần thiết để backend chạy độc lập
- `runtime_static/`: nơi lưu ảnh tải lên và kết quả phát sinh khi chạy live

## 8. Ghi chú kỹ thuật ngắn

- Endpoint `/dashboard` tổng hợp dữ liệu trực tiếp từ artifact local còn lại sau khi dọn bộ cũ.
- Endpoint `/segment` sẽ báo lỗi rõ ràng nếu repo demo chưa có checkpoint live mới.
- Ảnh người dùng tải lên được lưu tại `runtime_static/uploads` và có thể xóa lại từ giao diện.
- `requirements.txt` là bản nhẹ để deploy dashboard web.
- `requirements-inference.txt` chỉ cần khi muốn bật live inference cục bộ hoặc trên host đủ RAM.
