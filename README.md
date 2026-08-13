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

## 3. Cách dùng nhanh trên máy mới

Nếu đây là lần đầu người dùng clone hệ thống về máy mới, cách dùng chuẩn bây giờ là:

1. chép file key service account vào đúng đường dẫn `configs/ggserviceaccount.json`
2. chạy script `dev_scripts/up_system.sh`
3. đợi script tự kiểm tra, tự setup phần cần thiết, tự `dvc pull`, tự dựng web
4. mở web và dùng hệ thống
5. khi không dùng nữa thì chạy `dev_scripts/down_system.sh`

### 3.1. Đặt file key đúng chỗ

Người dùng cần được cung cấp file key DVC dạng `.json`.

Sau khi clone repo, hãy chép file key đó vào đúng đường dẫn sau:

```text
/Users/duyennguyen/Downloads/Master-Thesis/he_thong_phan_doan_u_xuong/configs/ggserviceaccount.json
```

Trong repo độc lập của hệ thống, đây chính là:

```text
he_thong_phan_doan_u_xuong/configs/ggserviceaccount.json
```

Lưu ý:

- file này là credential cục bộ, không được đưa lên Git
- script `up_system.sh` sẽ đọc key ở đúng đường dẫn này
- nếu file rỗng, sai định dạng JSON, hoặc đặt sai vị trí thì script sẽ dừng và báo lỗi

### 3.2. Khởi động hệ thống

Từ thư mục gốc của hệ thống `he_thong_phan_doan_u_xuong`, chạy:

```bash
bash dev_scripts/up_system.sh
```

Hoặc nếu file đã có quyền thực thi:

```bash
./dev_scripts/up_system.sh
```

Script này sẽ tự làm các việc sau:

- kiểm tra `python3`
- kiểm tra file key `configs/ggserviceaccount.json`
- kiểm tra `dvc`, và nếu chưa có thì thử cài `dvc[gdrive]` bằng `pip --user`
- cấu hình `DVC remote` cục bộ bằng key vừa được chép vào
- chạy `dvc pull` cho các file pretrained cần thiết
- kiểm tra `Docker Compose`
- kiểm tra `Docker daemon`
- build và chạy web bằng Docker Compose
- đợi backend sẵn sàng rồi thông báo đường dẫn truy cập

Sau khi script chạy xong thành công, mở trình duyệt tại:

```text
http://127.0.0.1:4173
```

### 3.3. Tắt hệ thống

Khi muốn tắt hệ thống, từ thư mục gốc của hệ thống chạy:

```bash
bash dev_scripts/down_system.sh
```

Hoặc:

```bash
./dev_scripts/down_system.sh
```

Script này sẽ chạy lệnh Docker Compose phù hợp để dừng web.

### 3.4. Những gì `up_system.sh` cần từ máy người dùng

Để script chạy trơn, máy người dùng nên có:

- `python3`
- `pip` đi kèm với `python3`
- `Docker Desktop` hoặc môi trường Docker tương đương
- kết nối mạng để cài `dvc` lần đầu nếu máy chưa có sẵn
- kết nối mạng để `dvc pull` dữ liệu từ remote

Script có thể tự cài `dvc` nếu thiếu, nhưng không tự cài Docker Desktop. Nếu Docker chưa có hoặc Docker daemon chưa chạy, script sẽ dừng và yêu cầu người dùng mở hoặc cài Docker trước.

### 3.5. Lỗi thường gặp

- Nếu script báo thiếu key, hãy kiểm tra lại file `configs/ggserviceaccount.json`.
- Nếu script báo key không hợp lệ, hãy kiểm tra lại nội dung JSON hoặc xin lại key mới.
- Nếu script báo không tìm thấy `dvc` sau khi cài, hãy mở terminal mới rồi chạy lại script.
- Nếu script báo Docker daemon chưa chạy, hãy mở `Docker Desktop` rồi chạy lại.
- Nếu `dvc pull` lỗi quyền truy cập, rất có thể file key được cấp chưa đúng hoặc đã hết hiệu lực.

## 4. Các chế độ dùng của hệ thống

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

## 5. Các model đang được giữ lại trong hệ thống

Hệ thống hiện chỉ giữ các model thuộc lớp bằng chứng thực nghiệm chính được dùng để trình bày trong luận văn.

Bao gồm:

- `SegFormer Binary Baseline [1.0, 2.0]`
- `SegFormer Binary CE [1.0, 1.0]`
- `SegFormer Binary CE [1.0, 3.0]`
- `U-Net nhẹ` dùng làm đối chứng trực tiếp

Model mặc định của chế độ live là:

- `SegFormer Binary CE [1.0, 1.0]`

Đây là checkpoint được chọn theo tiêu chí validation loss trong phần thực nghiệm chính.

## 6. Cấu trúc thư mục chính

- `server.py`: điểm khởi động của toàn bộ web
- `backend/`: API và backend inference cục bộ
- `app/`: giao diện trình diễn
- `resources/`: checkpoint, metadata, và tài nguyên đã đóng gói
- `runtime_src/`: mã nguồn tối thiểu cần thiết để backend chạy độc lập
- `runtime_static/`: nơi lưu ảnh tải lên và kết quả phát sinh khi chạy live
- `configs/ggserviceaccount.json`: nơi người dùng chép key DVC cục bộ
- `dev_scripts/up_system.sh`: script khởi động hệ thống cho người dùng cuối
- `dev_scripts/down_system.sh`: script tắt hệ thống
- `Dockerfile`: cấu hình build image Docker cho hệ standalone
- `compose.yaml`: cấu hình Docker Compose để chạy hệ thống bằng một lệnh
- `.dockerignore`: loại các file không cần thiết khỏi build context để image gọn hơn

## 7. Ghi chú kỹ thuật ngắn

- Backend trong thư mục này là backend cục bộ, chạy độc lập với các checkpoint đã được chép sẵn.
- Nếu mở web nhưng chưa thấy kết quả live, chỉ cần kiểm tra lại backend và chạy lại model mong muốn.
- Docker image chạy bằng `gunicorn` với `1` worker để tránh nạp lặp model nặng không cần thiết.
- Healthcheck của container dùng endpoint `/health` để kiểm tra backend đã sẵn sàng hay chưa.
- Nếu lệnh Docker báo không kết nối được daemon, chỉ cần mở `Docker Desktop` rồi chạy lại.
- Script `up_system.sh` chỉ kéo đúng các artifact DVC cần cho hệ thống demo này.
