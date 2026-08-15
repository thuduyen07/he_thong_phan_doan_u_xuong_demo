# Web Demo Pipeline Phân Đoạn U Xương

## 1. Mục đích của hệ thống

Đây là bản web demo tập trung riêng vào pipeline thực nghiệm của luận văn:

`Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc transformer`

Hệ thống hỗ trợ hai mục tiêu chính:

- hiển thị pipeline supervised -> mean teacher -> uncertainty -> adaptive CRC
- chạy thử phân đoạn trực tiếp trên ảnh X-quang mới do người dùng tải lên bằng backend cục bộ và các checkpoint đã được giữ lại từ phần thực nghiệm chính thức

## 2. Các nội dung có trên web

Khi mở web, người dùng có thể xem các phần sau:

- phần mô tả kiến trúc pipeline và lịch thực nghiệm
- dashboard đọc kết quả thực nghiệm trực tiếp từ artifact trong `resources/outputs`
- chế độ chạy live inference trên ảnh tải lên, kèm uncertainty và conformal map hậu kiểm nếu checkpoint có calibration profile

## 3. Pull các file được lưu trữ bằng DVC trước khi chạy

Nếu đang dùng một bản clone mới hoặc một máy chưa có sẵn các file pretrained, hãy kéo các artifact được quản lý bởi DVC trước.

Từ thư mục gốc của repository, chạy:

```bash
dvc pull \
  resources/pretrained/segformer_b0_ade_512_512/pytorch_model.bin.dvc \
  resources/pretrained/segformer_b0_ade_512_512/model.safetensors.dvc \
  resources/pretrained/segformer_b0_ade_512_512/tf_model.h5.dvc
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
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 server.py
```

## 6. Luồng sử dụng chính

1. Mở web và xem phần `Pipeline` để nắm kiến trúc supervised -> mean teacher -> uncertainty -> adaptive CRC.
2. Xem phần `Experiments` để đọc các run đã được đóng gói trong `resources/outputs`.
3. Tải ảnh mới ở phần `Live Lab`, chọn checkpoint, rồi chạy segmentation.
4. Nếu checkpoint có conformal artifact, web sẽ hiển thị thêm prediction set hậu kiểm cùng các heatmap uncertainty.

## 7. Cấu trúc thư mục chính

- `server.py`: điểm khởi động của toàn bộ web
- `backend/`: API và backend inference cục bộ
- `app/`: giao diện trình diễn tối giản theo pipeline
- `resources/`: checkpoint, metadata, conformal profile, và artifact thực nghiệm
- `runtime_src/`: mã nguồn tối thiểu cần thiết để backend chạy độc lập
- `runtime_static/`: nơi lưu ảnh tải lên và kết quả phát sinh khi chạy live

## 8. Ghi chú kỹ thuật ngắn

- Endpoint `/dashboard` tổng hợp dữ liệu thực nghiệm trực tiếp từ artifact local.
- Endpoint `/segment` vẫn dùng backend inference cục bộ như bản demo trước.
- Ảnh người dùng tải lên được lưu tại `runtime_static/uploads` và có thể xóa lại từ giao diện.
