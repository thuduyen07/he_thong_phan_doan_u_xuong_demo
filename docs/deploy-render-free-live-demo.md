# Triển khai Render Free: Demo SegFormer trực tiếp

Tùy chọn này chạy mô hình SegFormer-B0 đã được đăng ký thông qua endpoint Flask `/segment`. Cách triển khai này vẫn giữ nguyên quy trình tiền xử lý và bộ điều hợp suy luận tương thích với nghiên cứu, nhưng chỉ phù hợp khi các tệp mô hình và dung lượng bộ nhớ khả dụng đáp ứng đủ yêu cầu.

## Các ràng buộc quan trọng

* Render Free có thể bị giới hạn bộ nhớ quá mức đối với quá trình suy luận trên CPU sử dụng PyTorch + Transformers + SegFormer. Vì vậy, chỉ nên xem đây là một tùy chọn để kiểm tra nhanh/demo, không phải là sự đảm bảo về độ tin cậy.

* Các dịch vụ web miễn phí có thể chuyển sang trạng thái tạm ngừng sau một khoảng thời gian không hoạt động, do đó yêu cầu phân đoạn đầu tiên có thể chậm trong lúc mô hình được tải.

* Dịch vụ sử dụng một Gunicorn worker và cơ chế tải mô hình trì hoãn (lazy model loading) nhằm hạn chế mức sử dụng bộ nhớ.

* Không được tải mô hình xuống trong quá trình suy luận: tệp cấu hình/checkpoint cục bộ đã đăng ký phải có sẵn trong build artifact.

## Các tệp cần thiết để triển khai

Trước khi tạo dịch vụ, hãy xác minh rằng repository/build context chứa chính xác các tệp đã đăng ký sau:

```text
resources/metadata/models.yaml

resources/configs/<registered-config>.yaml

resources/checkpoints/<registered-checkpoint>.pt

resources/pretrained/segformer_b0_ade_512_512/config.json
```

`models.yaml` chỉ được sử dụng các đường dẫn tương đối so với repository. Bản ghi mô hình phải có `enabled: true`. Không công khai các tập dữ liệu huấn luyện, kết quả đánh giá hoặc bất kỳ ảnh y khoa nào chưa được phê duyệt.

## Thiết lập trên bảng điều khiển Render

1. Chọn **New > Web Service** và kết nối repository.

2. Đặt **Root Directory** thành thư mục gốc của repository.

3. Đặt **Language** thành `Docker`.

4. Đặt biến môi trường khi build là `INSTALL_INFERENCE=true` (đây cũng là giá trị mặc định của Docker).

5. Đặt các biến môi trường khi chạy:

   * `HOST=0.0.0.0`

   * `PORT=10000`

6. Triển khai bằng Dockerfile của repository. Không ghi đè lệnh (command) của Dockerfile.

## Kiểm tra trước khi chạy và kiểm tra nhanh

Sau khi triển khai:

```bash
curl -fsS https://YOUR-SERVICE.onrender.com/health

curl -fsS https://YOUR-SERVICE.onrender.com/models

curl -fsS https://YOUR-SERVICE.onrender.com/get_images
```

`/models` phải báo cáo `inference_runtime.available: true` và mô hình được chọn phải có `available: true`. Sau đó, sử dụng một ảnh đã được phê duyệt và tuyển chọn trong giao diện người dùng để thực hiện một yêu cầu `/segment` duy nhất. Kiểm tra log của Render để phát hiện các lỗi liên quan đến checkpoint/config; phản hồi trả về trình duyệt được thiết kế để không chứa stack trace.

## Nếu suy luận trực tiếp thất bại

* `DEPENDENCY_MISSING`: quá trình build Docker không cài đặt `requirements-inference.txt` hoặc dịch vụ không sử dụng Docker image như mong đợi.

* `CHECKPOINT_NOT_FOUND`: tệp đã đăng ký không được đưa vào build context.

* `MODEL_CONFIG_INVALID` hoặc `MODEL_INITIALIZATION_FAILED`: kiểm tra log an toàn phía máy chủ; không thay thế mô hình hoặc âm thầm thay đổi cấu hình nghiên cứu của mô hình.

* OOM, SIGKILL hoặc liên tục thất bại khi khởi động nguội (cold start): sử dụng phương án triển khai chỉ có dashboard hoặc chuyển chức năng suy luận trực tiếp sang một instance trả phí có nhiều bộ nhớ hơn.

## Lưu trữ và an toàn

Hệ thống tệp của Render mặc định là tạm thời (ephemeral). Các tệp tải lên và ảnh overlay được tạo ra chỉ tồn tại tạm thời. Không sử dụng bản triển khai này để thu thập hoặc lưu giữ ảnh lâm sàng, đồng thời không để lộ bất kỳ ảnh mẫu hoặc đường dẫn nguồn nào chưa được phê duyệt.

Tài liệu tham khảo: [Render Web Services](https://render.com/docs/web-services), [Render Docker](https://render.com/docs/docker) và [Render troubleshooting](https://render.com/docs/troubleshooting-deploys).
