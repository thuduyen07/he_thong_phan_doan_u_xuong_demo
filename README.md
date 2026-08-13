# Hệ Thống Phân Đoạn U Xương Dùng Cho Bảo Vệ Luận Văn

## 1. Mục đích của hệ thống

Đây là bản web demo, được chuẩn bị để phục vụ báo cáo và bảo vệ luận văn:

`Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc transformer`

Hệ thống này được đóng gói trong chính thư mục `he_thong_phan_doan_u_xuong` và có thể chạy độc lập.

Hệ thống hỗ trợ hai mục tiêu chính:

- trình bày trực quan các ca minh họa tiêu biểu của luận văn
- chạy thử phân đoạn trực tiếp trên ảnh X-quang bằng backend cục bộ và các checkpoint đã được giữ lại từ phần thực nghiệm chính thức

## 2. Các nội dung có trên web

Khi mở web, người dùng có thể xem các phần sau:

- phần giới thiệu ngắn về đề tài và quy mô dữ liệu
- phần mô tả pipeline phân đoạn u xương
- các ca minh họa với ảnh gốc, ảnh kết quả dự đoán, và ghi chú diễn giải
- bảng kết quả thực nghiệm chính dùng trong luận văn
- phần so sánh giữa các cấu hình SegFormer và đối chứng U-Net
- chế độ chạy live inference trên ảnh tải lên

## 3. Pull các file được lưu trữ bằng DVC trước khi chạy

Nếu đang dùng một bản clone mới hoặc một máy chưa có sẵn các file pretrained, hãy kéo các artifact được quản lý bởi DVC trước.

Từ thư mục gốc của repository, chạy:

```bash
dvc pull \
  he_thong_phan_doan_u_xuong/resources/pretrained/segformer_b0_ade_512_512/pytorch_model.bin.dvc \
  he_thong_phan_doan_u_xuong/resources/pretrained/segformer_b0_ade_512_512/model.safetensors.dvc \
  he_thong_phan_doan_u_xuong/resources/pretrained/segformer_b0_ade_512_512/tf_model.h5.dvc
```

Nếu máy cục bộ của bạn chưa được cấu hình để truy cập DVC remote, hãy cấu hình credential cho remote trước rồi chạy lại `dvc pull`.

Sau khi pull xong, hãy kiểm tra các file pretrained đã xuất hiện trong:

```text
he_thong_phan_doan_u_xuong/resources/pretrained/segformer_b0_ade_512_512/
```

Bước này đặc biệt quan trọng khi backend live inference không tìm thấy các file pretrained backbone lúc khởi động.

## 4. Cách khởi động nhanh bằng Docker Compose

Đây là hướng dẫn chi tiết cách dùng hệ thống khi dùng bằng docker compose.

Trước hết, hãy bảo đảm `Docker Desktop` đã được mở và Docker daemon đang chạy.

Từ thư mục gốc của hệ thống, tức là thư mục `he_thong_phan_doan_u_xuong`, chạy:

```bash
docker-compose up --build
```

Sau khi container khởi động, mở trình duyệt tại:

```text
http://127.0.0.1:4173
```

Các lệnh Docker Compose hữu ích:

```bash
docker-compose up --build -d
docker-compose logs -f
docker-compose down
docker-compose down -v
```

Ý nghĩa:

- `docker-compose up --build`: build image và chạy hệ thống
- `docker-compose up --build -d`: chạy nền
- `docker-compose logs -f`: xem log trực tiếp
- `docker-compose down`: dừng container nhưng giữ dữ liệu runtime
- `docker-compose down -v`: dừng container và xóa luôn volume runtime để làm sạch ảnh upload, mask

Nếu máy dùng Docker CLI đời mới có tích hợp Compose plugin, có thể dùng các lệnh tương đương sau:

```bash
docker compose up --build
docker compose logs -f
docker compose down
```

Lưu ý:

- image được build theo hướng CPU-only
- container dùng backend cục bộ và các checkpoint đã đóng gói sẵn
- dữ liệu phát sinh khi chạy live được lưu trong volume `he_thong_phan_doan_u_xuong_runtime_static`
- bản triển khai này đã được build và chạy thành công bằng `docker-compose` trên máy học viên

## 5. Cách khởi động trực tiếp bằng Python

Nếu không dùng Docker, có thể chạy trực tiếp như sau:

Từ thư mục gốc của repository, chạy:

```bash
../.venv/bin/python server.py
```

Sau khi chạy, mở trình duyệt tại địa chỉ:

```text
http://127.0.0.1:4173
```

Nếu máy không có sẵn môi trường `../.venv`, có thể cài thư viện bằng:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 server.py
```

## 6. Các chế độ dùng của hệ thống

### Chế độ 1: Xem các ca minh họa đã chuẩn bị sẵn

Đây là chế độ phù hợp nhất để quan sát nhanh kết quả định tính của luận văn.

Các bước thao tác:

1. Mở web.
2. Giữ ở chế độ xem các ca minh họa có sẵn.
3. Chọn từng ca trong danh sách bên trái hoặc khu vực lựa chọn ca.
4. Quan sát ảnh gốc, ảnh kết quả, và phần ghi chú diễn giải đi kèm.

Chế độ này hữu ích để:

- xem các ca đúng điển hình
- xem các ca sai điển hình
- đối chiếu nhanh giữa ảnh gốc và vùng mô hình dự đoán

### Chế độ 2: Chạy live inference trên ảnh tải lên

Đây là chế độ dùng để trình diễn khả năng chạy thật của hệ thống bằng backend cục bộ.

Các bước thao tác:

1. Mở web.
2. Chuyển sang chế độ `Live inference`.
3. Nhấn kiểm tra backend nếu cần.
4. Tải một ảnh X-quang từ máy lên.
5. Chọn model trong danh sách.
6. Nhấn nút chạy phân đoạn.
7. Chờ hệ thống trả về ảnh kết quả, mask, và metadata cơ bản.

Kết quả trả về gồm:

- ảnh gốc
- ảnh kết quả vùng dự đoán lên ảnh gốc
- mask phân đoạn
- thông tin metadata như loại model, số pixel foreground, kích thước ảnh đã xử lý, và đường dẫn checkpoint

## 7. Các model đang được giữ lại trong hệ thống

Hệ thống hiện chỉ giữ các model thuộc lớp bằng chứng thực nghiệm chính được dùng để trình bày trong luận văn.

Bao gồm:

- `SegFormer Binary Baseline [1.0, 2.0]`
- `SegFormer Binary CE [1.0, 1.0]`
- `SegFormer Binary CE [1.0, 3.0]`
- `U-Net nhẹ` dùng làm đối chứng trực tiếp

Model mặc định của chế độ live là:

- `SegFormer Binary CE [1.0, 1.0]`

Đây là checkpoint được chọn theo tiêu chí validation loss trong phần thực nghiệm chính.

## 8. Cấu trúc thư mục chính

- `server.py`: điểm khởi động của toàn bộ web
- `backend/`: API và backend inference cục bộ
- `app/`: giao diện trình diễn
- `resources/`: checkpoint, metadata, và tài nguyên đã đóng gói
- `runtime_src/`: mã nguồn tối thiểu cần thiết để backend chạy độc lập
- `runtime_static/`: nơi lưu ảnh tải lên và kết quả phát sinh khi chạy live
- `Dockerfile`: cấu hình build image Docker cho hệ standalone
- `compose.yaml`: cấu hình Docker Compose để chạy hệ thống bằng một lệnh
- `.dockerignore`: loại các file không cần thiết khỏi build context để image gọn hơn

## 9. Ghi chú kỹ thuật ngắn

- Backend trong thư mục này là backend cục bộ, chạy độc lập với các checkpoint đã được chép sẵn.
- Nếu mở web nhưng chưa thấy kết quả live, chỉ cần kiểm tra lại backend và chạy lại model mong muốn.
- Docker image chạy bằng `gunicorn` với `1` worker để tránh nạp lặp model nặng không cần thiết.
- Healthcheck của container dùng endpoint `/health` để kiểm tra backend đã sẵn sàng hay chưa.
- Nếu lệnh Docker báo không kết nối được daemon, chỉ cần mở `Docker Desktop` rồi chạy lại.