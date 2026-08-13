window.THESIS_APP_DATA = {
  thesisTitle:
    "Xây dựng hệ thống phân đoạn u xương trên ảnh X-quang dựa trên kiến trúc transformer",
  heroMetrics: [
    { label: "Cohort bệnh nhân", value: "606", note: "5258 ảnh X-quang đã làm sạch" },
    { label: "Ảnh có chú giải thủ công", value: "371", note: "55 bệnh nhân có mặt nạ tham chiếu thủ công" },
    { label: "Model live còn giữ", value: "6", note: "4 checkpoint SegFormer + 2 checkpoint U-Net cho demo live" },
    { label: "Benchmark balanced76", value: "152", note: "1318 ảnh cho benchmark mức bệnh nhân" },
  ],
  datasetSummary: [
    "Cohort tổng quát: 606 bệnh nhân, 5258 ảnh X-quang.",
    "Tập chú giải tham chiếu thủ công của `seg_seed_v2`: 55 bệnh nhân, 371 ảnh có mặt nạ tham chiếu thủ công.",
    "Test split hẹp của `seg_seed_v2`: 57 ảnh positive-reference theo mặt nạ tham chiếu thủ công.",
    "Benchmark `balanced76`: 152 bệnh nhân, 1318 ảnh; headline overlap được đọc trên 535 ảnh positive-reference.",
  ],
  defensePoints: [
    "Hệ thống hiện đủ để chứng minh pipeline học được tín hiệu localization có ý nghĩa trên tập tham chiếu nội bộ.",
    "Bằng chứng hiện tại chưa đủ để khẳng định một hệ thống phân đoạn lâm sàng hoàn chỉnh hay đánh giá ngoài nguồn dữ liệu nội bộ.",
    "Các metric overlap trên web phải được hiểu là mức khớp với mặt nạ tham chiếu thủ công, không phải tumor extent chuẩn vàng đã được bác sĩ xác nhận độc lập.",
    "Web này tách rõ phần demo định tính với phần bảng định lượng để tránh diễn giải quá mức.",
  ],
  pipeline: [
    {
      step: "1",
      title: "Dữ liệu đầu vào",
      detail: "Ảnh X-quang xương, split theo bệnh nhân để giảm data leakage.",
    },
    {
      step: "2",
      title: "Tiền xử lý",
      detail: "Ảnh được đọc về thang xám và resize nhất quán theo đúng protocol inference đang giữ lại cho hệ standalone.",
    },
    {
      step: "3",
      title: "Segmentation model",
      detail: "SegFormer-B0 là kiến trúc trung tâm; live hiện giữ 4 checkpoint SegFormer và 2 run U-Net để so sánh trực tiếp trên web.",
    },
    {
      step: "4",
      title: "Đánh giá",
      detail: "Checkpoint trung tâm được chọn theo validation loss; Dice/IoU/Precision/Recall dùng để phân tích hành vi overlap trên tập tham chiếu thủ công.",
    },
    {
      step: "5",
      title: "Trình diễn",
      detail: "Artifact mode luôn sẵn sàng; live inference hiện chỉ mở đúng các checkpoint thực nghiệm được giữ lại trong Chương 3 và Chương 4 của luận văn.",
    },
  ],
  curatedCases: [
    {
      id: "CH150125393_37394e05_TP",
      title: "Ca đúng điển hình",
      tag: "Ca đúng",
      summary: "Mô hình bắt được vùng bất thường rõ ràng trên một ảnh positive-reference tiêu biểu.",
      note:
        "Phù hợp để trình diễn năng lực localization định tính của mô hình trên ca có tín hiệu bất thường X-quang tương đối rõ trong tập tham chiếu nội bộ.",
      badges: ["Positive-reference", "Định tính mạnh", "An toàn khi bảo vệ"],
      original: "./assets/cases/original/CH150125393_37394e05_TP.jpg",
      preview:
        "./assets/cases/previews/CH150125393_37394e05_TP_segformer_ce_1_1_preview.png",
      overlayPrimary:
        "./assets/cases/overlays/CH150125393_37394e05_TP_segformer_ce_1_1_overlay.png",
      overlaySecondary:
        "./assets/cases/overlays/CH150125393_37394e05_TP_segformer_ce_1_3_overlay.png",
      primaryLabel: "Kết quả run chọn theo validation loss: CE [1.0, 1.0]",
      secondaryLabel: "Kết quả run có Dice test tốt nhất: CE [1.0, 3.0]",
    },
    {
      id: "CH211215058_cec1e195_TP",
      title: "Ca đúng ổn định",
      tag: "Ca đúng",
      summary: "Dùng để minh họa rằng không chỉ một ca riêng lẻ mà nhiều ảnh positive-reference vẫn cho kết quả phân đoạn hợp lý.",
      note:
        "Có thể dùng khi hội đồng hỏi về tính lặp lại của kết quả định tính, thay vì chỉ dựa vào một ví dụ đẹp duy nhất.",
      badges: ["Positive-reference", "Hiển thị ổn định", "SegFormer demo"],
      original: "./assets/cases/original/CH211215058_cec1e195_TP.jpg",
      preview:
        "./assets/cases/previews/CH211215058_cec1e195_TP_segformer_ce_1_1_preview.png",
      overlayPrimary:
        "./assets/cases/overlays/CH211215058_cec1e195_TP_segformer_ce_1_1_overlay.png",
      overlaySecondary:
        "./assets/cases/overlays/CH211215058_cec1e195_TP_segformer_ce_1_3_overlay.png",
      primaryLabel: "Kết quả CE [1.0, 1.0]",
      secondaryLabel: "Kết quả CE [1.0, 3.0]",
    },
    {
      id: "CH101103422_2b5bfde1_TN",
      title: "Ca âm tham chiếu",
      tag: "Ca âm",
      summary: "Mô hình tạo ít foreground hơn trên ca không có vùng tham chiếu dương tính trong benchmark hẹp.",
      note:
        "Hữu ích để chứng minh web không chỉ hiển thị ca dương tính mà còn có ca âm để hội đồng nhìn tính chọn lọc của mô hình.",
      badges: ["Negative-reference", "Ca đối chứng", "Foreground thấp"],
      original: "./assets/cases/original/CH101103422_2b5bfde1_TN.jpg",
      preview:
        "./assets/cases/previews/CH101103422_2b5bfde1_TN_segformer_ce_1_1_preview.png",
      overlayPrimary:
        "./assets/cases/overlays/CH101103422_2b5bfde1_TN_segformer_ce_1_1_overlay.png",
      overlaySecondary:
        "./assets/cases/overlays/CH101103422_2b5bfde1_TN_segformer_ce_1_3_overlay.png",
      primaryLabel: "Kết quả CE [1.0, 1.0]",
      secondaryLabel: "Kết quả CE [1.0, 3.0]",
    },
    {
      id: "CH190712223_1aa8f916_FP",
      title: "Ca báo động giả",
      tag: "Báo động giả",
      summary: "Mô hình vẫn dự đoán foreground trên một ca không có vùng tham chiếu dương tính trong thiết lập đánh giá.",
      note:
        "Nên dùng để trả lời câu hỏi về failure mode và vì sao hệ thống hiện mới dừng ở mức hỗ trợ trực quan hóa nghiên cứu.",
      badges: ["Failure case", "False positive", "Bằng chứng trung thực"],
      original: "./assets/cases/original/CH190712223_1aa8f916_FP.jpg",
      preview:
        "./assets/cases/previews/CH190712223_1aa8f916_FP_segformer_ce_1_1_preview.png",
      overlayPrimary:
        "./assets/cases/overlays/CH190712223_1aa8f916_FP_segformer_ce_1_1_overlay.png",
      overlaySecondary:
        "./assets/cases/overlays/CH190712223_1aa8f916_FP_segformer_ce_1_3_overlay.png",
      primaryLabel: "Kết quả CE [1.0, 1.0]",
      secondaryLabel: "Kết quả CE [1.0, 3.0]",
    },
    {
      id: "CH220313307_f8da7f11_FN",
      title: "Ca bỏ sót",
      tag: "Bỏ sót",
      summary: "Ca khó với tín hiệu bị bỏ sót, giúp minh họa giới hạn hiện tại của hệ thống.",
      note:
        "Đây là ca quan trọng cho defense vì nó cho phép diễn giải trung thực: pipeline có ích nhưng chưa đủ độ tin cậy để thay thế chuyên gia.",
      badges: ["Failure case", "False negative", "Giới hạn quan trọng"],
      original: "./assets/cases/original/CH220313307_f8da7f11_FN.jpg",
      preview:
        "./assets/cases/previews/CH220313307_f8da7f11_FN_segformer_ce_1_1_preview.png",
      overlayPrimary:
        "./assets/cases/overlays/CH220313307_f8da7f11_FN_segformer_ce_1_1_overlay.png",
      overlaySecondary:
        "./assets/cases/overlays/CH220313307_f8da7f11_FN_segformer_ce_1_3_overlay.png",
      primaryLabel: "Kết quả CE [1.0, 1.0]",
      secondaryLabel: "Kết quả CE [1.0, 3.0]",
    },
  ],
  batchSummary: [
    "Web hiện giữ 6 run thực nghiệm cốt lõi để trình diễn live và đối chiếu định tính ngay trên giao diện.",
    "Ba run SegFormer nhị phân ngày `20260809` là lớp bằng chứng chính thức cho phần phân đoạn `background/tumor` trên web.",
    "Checkpoint mặc định của live mode là `CE [1.0, 1.0]` vì đây là run được chọn theo rule validation loss trong binary-refresh.",
    "Checkpoint có Dice test hẹp tốt nhất trong nhóm SegFormer là `CE [1.0, 3.0]` với `test Dice = 0.6506`.",
    "Run `U-Net` nhẹ được giữ lại như một đối chứng CNN tối thiểu, không phải bằng chứng để claim Transformer vượt CNN.",
  ],
  binaryRefresh: [
    {
      run: "Binary baseline",
      ce: "[1.0, 2.0]",
      valLoss: 0.5273122471,
      primaryDice: 0.6480466551,
      testDice: 0.2294666817,
    },
    {
      run: "Binary CE cân bằng",
      ce: "[1.0, 1.0]",
      valLoss: 0.41611411,
      primaryDice: 0.6934726645,
      testDice: 0.515674614,
    },
    {
      run: "Foreground-upweighted mạnh hơn",
      ce: "[1.0, 3.0]",
      valLoss: 0.4360509831,
      primaryDice: 0.7249098788,
      testDice: 0.6506029765,
    },
  ],
  balanced76: [
    {
      run: "Binary baseline",
      dice: 0.3036448085,
      precision: 0.9063908558,
      recall: 0.2284793786,
    },
    {
      run: "CE [1.0, 1.0]",
      dice: 0.2955870608,
      precision: 0.9058563737,
      recall: 0.2210531574,
    },
    {
      run: "CE [1.0, 3.0]",
      dice: 0.39943216,
      precision: 0.9049874443,
      recall: 0.3143388145,
    },
  ],
  headToHead: [
    {
      model: "SegFormer-B0",
      init: "pretrained + warm-start",
      params: "3,714,658",
      valLoss: 0.5273,
      testDice: 0.2295,
      testIou: 0.159,
      precision: 0.3396,
      recall: 0.2431,
    },
    {
      model: "U-Net nhẹ",
      init: "random initialization",
      params: "1,942,594",
      valLoss: 0.6609,
      testDice: 0.5254,
      testIou: 0.3725,
      precision: 0.4852,
      recall: 0.6668,
    },
  ],
  liveInference: {
    defaultModelId: "thesis_20260809_ce_weights_seg_seed_v2_ce_weights_ce_weights_1_1_seed_42_20260809",
  },
};
