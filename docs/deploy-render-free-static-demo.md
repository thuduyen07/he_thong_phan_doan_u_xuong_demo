# Triển khai Render Free: Demo chỉ sử dụng Dashboard

Tùy chọn này triển khai dashboard của luận văn, thư viện ảnh mẫu đã được tuyển chọn và các bảng kết quả mà không thực hiện suy luận trực tiếp bằng PyTorch/Transformers. Đây là phương án được khuyến nghị khi sử dụng Render Free vì có thời gian build ngắn nhất và yêu cầu bộ nhớ thấp nhất.

> Repository này là một ứng dụng Flask, không phải là một Render Static Site thuần túy: trình duyệt sử dụng `/dashboard` và `/get_images`. Hãy triển khai ứng dụng dưới dạng **Docker Web Service** và tắt các thư viện phụ thuộc phục vụ suy luận.

## Các thành phần được bao gồm

* tuyên bố miễn trừ trách nhiệm nghiên cứu, thông tin tập dữ liệu và các bảng kết quả;

* các tài nguyên đã được tuyển chọn trong `resources/samples/` và `samples.yaml`;

* không tải checkpoint, không thực hiện phân đoạn, không lưu trữ lâu dài các tệp được tải lên và không sử dụng `/segment`.

## Danh sách kiểm tra trước khi triển khai

1. Đẩy repository của bản demo lên GitHub/GitLab/Bitbucket.

2. Không thêm `resources/checkpoints/` hoặc cấu hình mô hình dùng cho môi trường production vào nhánh chỉ dành cho dashboard này.

3. Chỉ giữ lại các ảnh mẫu đã được phê duyệt và loại bỏ thông tin định danh trong `resources/samples/`.

4. Xác nhận rằng `resources/samples/samples.yaml` không chứa bất kỳ đường dẫn nguồn tuyệt đối nào.

## Thiết lập trên bảng điều khiển Render

1. Chọn **New > Web Service** và kết nối repository.

2. Đặt **Root Directory** thành thư mục gốc của repository.

3. Đặt **Language** thành `Docker`.

4. Đặt biến môi trường khi build là `INSTALL_INFERENCE=false`.

5. Đặt các biến môi trường khi chạy:

   * `HOST=0.0.0.0`

   * `PORT=10000`

6. Giữ nguyên command trong Dockerfile. Dockerfile sẽ khởi chạy một Gunicorn worker và liên kết dịch vụ với `PORT` của Render.

7. Chỉ bật tính năng triển khai tự động (automatic deploy) cho nhánh dự kiến sử dụng.

## Xác minh sau khi triển khai

```bash
curl -fsS https://YOUR-SERVICE.onrender.com/health

curl -fsS https://YOUR-SERVICE.onrender.com/get_images
```

Danh sách ảnh mẫu đã được tuyển chọn phải tải thành công. `/segment` được chủ đích vô hiệu hóa trong chế độ này. Không được xem dashboard này như một dịch vụ phục vụ mục đích lâm sàng.

## Đặc điểm hoạt động của Render Free

* Các dịch vụ web miễn phí có thể chuyển sang trạng thái tạm ngừng sau một khoảng thời gian không hoạt động; do đó, yêu cầu tiếp theo có thể gặp hiện tượng khởi động nguội (cold start).

* Hệ thống tệp là tạm thời (ephemeral). Các tệp được tải lên và kết quả được tạo ra không được xem là dữ liệu được lưu trữ lâu dài.

* Trong trường hợp này cần sử dụng Web Service vì một Static Site thực sự không thể phục vụ các tuyến API Flask hiện có.

Tài liệu tham khảo: [Render Web Services](https://render.com/docs/web-services), [Render Docker](https://render.com/docs/docker) và [Render deploy filesystem behavior](https://render.com/docs/deploys).
