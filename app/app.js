(function () {
  const data = window.THESIS_APP_DATA;

  const state = {
    mode: "curated",
    selectedCaseId: data.curatedCases[0].id,
    models: [],
    selectedModelId: data.liveInference.defaultModelId,
    liveResult: null,
    showUncertainty: false,
  };

  const caseMap = new Map(data.curatedCases.map((item) => [item.id, item]));
  const uncertaintyScoreConfig = [
    {
      key: "mean_predictive_entropy_pred_tumor",
      label: "Mean entropy",
      note: "Entropy trung bình trong vùng tumor dự đoán. Thông số này càng nhỏ thì vùng dự đoán càng chắc chắn.",
    },
    {
      key: "p90_predictive_entropy_pred_tumor",
      label: "P90 entropy",
      note: "Ngưỡng 90% của entropy trong vùng tumor dự đoán. Thông số này càng nhỏ thì các điểm bất định cao trong vùng tumor càng ít.",
    },
    {
      key: "mean_predictive_entropy_boundary_tumor",
      label: "Boundary entropy",
      note: "Entropy trung bình trên dải biên của tumor dự đoán. Thông số này càng nhỏ thì biên tumor dự đoán càng rõ và ổn định.",
    },
    {
      key: "mean_one_minus_msp_pred_tumor",
      label: "1 - MSP",
      note: "Mức bất định từ xác suất lớp cao nhất. Thông số này càng nhỏ thì mô hình càng tự tin vào lớp dự đoán chính.",
    },
    {
      key: "mean_one_minus_margin_pred_tumor",
      label: "1 - Margin",
      note: "Mức sát nhau giữa hai lớp top trong vùng dự đoán. Thông số này càng nhỏ thì khoảng cách giữa hai lớp top càng lớn và quyết định càng chắc chắn.",
    },
    {
      key: "low_tumor_probability_fraction_pred_tumor",
      label: "Low prob fraction",
      note: "Tỉ lệ pixel tumor dự đoán có xác suất dưới ngưỡng. Thông số này càng nhỏ thì càng ít pixel tumor bị xem là thiếu chắc chắn.",
    },
    {
      key: "mean_tumor_probability_pred_tumor",
      label: "Mean tumor prob",
      note: "Xác suất tumor trung bình trong vùng tumor dự đoán. Thông số này càng lớn thì mô hình càng tin rằng vùng dự đoán thực sự là tumor.",
    },
  ];
  const uncertaintyHeatmapConfig = [
    {
      key: "predictive_entropy",
      label: "Predictive entropy heatmap",
      note: "Vùng nóng hơn là nơi mô hình phân vân hơn theo entropy.",
    },
    {
      key: "one_minus_msp",
      label: "1 - MSP heatmap",
      note: "Vùng nóng hơn là nơi confidence top-class thấp hơn.",
    },
    {
      key: "tumor_probability",
      label: "Tumor probability heatmap",
      note: "Xác suất tumor hậu kiểm theo từng pixel.",
    },
  ];

  function $(selector) {
    return document.querySelector(selector);
  }

  function formatNumber(value, digits = 4) {
    return Number(value).toFixed(digits);
  }

  function formatMetricValue(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "N/A";
    }
    if (typeof value === "boolean") {
      return value ? "Yes" : "No";
    }
    if (Number.isInteger(Number(value))) {
      return `${Number(value)}`;
    }
    return formatNumber(value);
  }

  function renderStackList(containerId, values) {
    const target = document.getElementById(containerId);
    target.innerHTML = values.map((value) => `<p>${value}</p>`).join("");
  }

  function renderHeroMetrics() {
    $("#hero-metrics").innerHTML = data.heroMetrics
      .map(
        (item) => `
          <article class="metric-card">
            <p class="metric-value">${item.value}</p>
            <p class="metric-label">${item.label}</p>
            <p class="metric-note">${item.note}</p>
          </article>
        `
      )
      .join("");
  }

  function renderPipeline() {
    $("#pipeline-strip").innerHTML = data.pipeline
      .map(
        (item) => `
          <article class="pipeline-card">
            <div class="pipeline-step">${item.step}</div>
            <h3>${item.title}</h3>
            <p>${item.detail}</p>
          </article>
        `
      )
      .join("");
  }

  function renderCases() {
    $("#case-list").innerHTML = data.curatedCases
      .map(
        (item) => `
          <button type="button" class="case-button ${
            item.id === state.selectedCaseId ? "active" : ""
          }" data-case-id="${item.id}">
            <span class="case-button-tag">${item.tag}</span>
            <strong>${item.title}</strong>
            <span>${item.id.replace(/_(TP|TN|FP|FN)$/, "")}</span>
          </button>
        `
      )
      .join("");

    document.querySelectorAll("[data-case-id]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedCaseId = button.dataset.caseId;
        state.liveResult = null;
        renderCases();
        renderStage();
      });
    });
  }

  function renderBadges(items) {
    return items.map((item) => `<span class="pill">${item}</span>`).join("");
  }

  function renderResultGallery(items) {
    $("#result-gallery").innerHTML = items
      .map(
        (item) => `
          <figure class="image-panel">
            <figcaption>${item.label}</figcaption>
            <div class="image-frame ${item.emphasized ? "result-frame" : ""}">
              <img src="${item.src}" alt="${item.alt}" />
            </div>
          </figure>
        `
      )
      .join("");
  }

  function renderCuratedStage() {
    const selected = caseMap.get(state.selectedCaseId);
    $("#stage-tag").textContent = selected.tag;
    $("#stage-title").textContent = selected.title;
    $("#stage-summary").textContent = selected.summary;
    $("#stage-badges").innerHTML = renderBadges(selected.badges);
    $("#original-image").src = selected.original;
    renderResultGallery([
      {
        label: selected.primaryLabel,
        src: selected.overlayPrimary,
        alt: `${selected.title} - ${selected.primaryLabel}`,
        emphasized: true,
      },
      {
        label: selected.secondaryLabel,
        src: selected.overlaySecondary,
        alt: `${selected.title} - ${selected.secondaryLabel}`,
        emphasized: true,
      },
    ]);
    $("#case-notes").innerHTML = `<p>${selected.note}</p>`;
    renderUncertaintyPanel();
  }

  function renderLiveStage() {
    const result = state.liveResult;
    $("#stage-tag").textContent = "Live inference";
    $("#stage-title").textContent = result ? result.display_name || result.model_id : "Chưa có kết quả";
    $("#stage-summary").textContent = result
      ? result.note || "Kết quả được trả về từ backend inference nội bộ."
      : "Tải ảnh X-quang lên, chọn checkpoint cục bộ, rồi chạy segmentation bằng backend độc lập của hệ thống này.";

    $("#stage-badges").innerHTML = renderBadges(
      result
        ? [
            result.metadata?.model_type || "model",
            result.checkpoint_name || "checkpoint",
            `foreground:${result.metadata?.foreground_pixels ?? 0}`,
          ]
        : ["Standalone backend", "Local checkpoints", "Thesis-aligned"]
    );

    $("#original-image").src = result ? result.original_image_url : "";
    renderResultGallery(
      result
        ? [
            {
              label: "Kết quả sau phân đoạn của model đã chọn",
              src: result.overlay_image_url,
              alt: `${result.display_name || result.model_id} - segmentation result`,
              emphasized: true,
            },
            {
              label: "Predict mask của model đã chọn",
              src: result.mask_image_url,
              alt: `${result.display_name || result.model_id} - predict mask`,
              emphasized: false,
            },
          ]
        : []
    );
    $("#case-notes").innerHTML = result
      ? `
        <p><strong>Model:</strong> ${result.model_id}</p>
        <p><strong>Checkpoint:</strong> ${result.checkpoint_path}</p>
        <p><strong>Metadata:</strong> model_type=${result.metadata?.model_type || "N/A"},
        foreground_pixels=${result.metadata?.foreground_pixels ?? "N/A"},
        processed_shape=${result.metadata?.processed_shape_hw?.join("x") || "N/A"}</p>
      `
      : "<p>Chế độ này hiện chạy bằng backend thật của chính thư mục `he_thong_phan_doan_u_xuong`, không còn dùng engine giả lập trong trình duyệt.</p>";
    renderUncertaintyPanel();
  }

  function renderUncertaintyPanel() {
    const panel = $("#uncertainty-panel");
    const toggleButton = $("#toggle-uncertainty");
    const result = state.liveResult;
    const uncertainty = result?.uncertainty;
    const available = Boolean(uncertainty?.available);

    toggleButton.disabled = !available;
    toggleButton.textContent = state.showUncertainty ? "Ẩn uncertainty score" : "Uncertainty score";

    if (state.mode !== "live" || !state.showUncertainty || !available) {
      panel.classList.add("hidden");
      return;
    }

    $("#uncertainty-note").textContent = uncertainty.note || "";
    $("#uncertainty-score-grid").innerHTML = uncertaintyScoreConfig
      .map((item) => {
        const value = uncertainty.summary?.[item.key];
        return `
          <article class="uncertainty-score-card">
            <h4>${item.label}</h4>
            <p class="uncertainty-score-value">${formatMetricValue(value)}</p>
            <p class="uncertainty-score-note">${item.note}</p>
          </article>
        `;
      })
      .join("");

    $("#uncertainty-heatmaps").innerHTML = uncertaintyHeatmapConfig
      .filter((item) => uncertainty.heatmaps?.[item.key])
      .map(
        (item) => `
          <figure class="image-panel">
            <figcaption>${item.label}</figcaption>
            <div class="image-frame">
              <img src="${uncertainty.heatmaps[item.key]}" alt="${item.label}" />
            </div>
            <p class="muted-note">${item.note}</p>
          </figure>
        `
      )
      .join("");

    panel.classList.remove("hidden");
  }

  function renderStage() {
    if (state.mode === "curated") {
      renderCuratedStage();
      return;
    }
    renderLiveStage();
  }

  function renderTable(rows, columns, formatters = {}) {
    const header = columns.map((column) => `<th>${column.label}</th>`).join("");
    const body = rows
      .map((row) => {
        const cells = columns
          .map((column) => {
            const formatter = formatters[column.key];
            const value = formatter ? formatter(row[column.key], row) : row[column.key];
            return `<td>${value}</td>`;
          })
          .join("");
        return `<tr>${cells}</tr>`;
      })
      .join("");

    return `<div class="table-shell"><table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function renderExperiments() {
    renderStackList("batch-summary", data.batchSummary);

    $("#binary-refresh-table").innerHTML = renderTable(
      data.binaryRefresh,
      [
        { key: "run", label: "Run" },
        { key: "ce", label: "CE weights" },
        { key: "valLoss", label: "Best val loss" },
        { key: "primaryDice", label: "Best primary Dice" },
        { key: "testDice", label: "Test Dice" },
      ],
      {
        valLoss: (value) => formatNumber(value),
        primaryDice: (value) => formatNumber(value),
        testDice: (value) => formatNumber(value),
      }
    );

    $("#balanced76-table").innerHTML = renderTable(
      data.balanced76,
      [
        { key: "run", label: "Run" },
        { key: "dice", label: "Dice" },
        { key: "precision", label: "Precision" },
        { key: "recall", label: "Recall" },
      ],
      {
        dice: (value) => formatNumber(value),
        precision: (value) => formatNumber(value),
        recall: (value) => formatNumber(value),
      }
    );
    $("#headtohead-table").innerHTML = renderTable(
      data.headToHead,
      [
        { key: "model", label: "Mô hình" },
        { key: "init", label: "Khởi tạo" },
        { key: "params", label: "Số tham số" },
        { key: "valLoss", label: "Best val loss" },
        { key: "testDice", label: "Dice" },
        { key: "testIou", label: "IoU" },
        { key: "precision", label: "Precision" },
        { key: "recall", label: "Recall" },
      ],
      {
        valLoss: (value) => formatNumber(value),
        testDice: (value) => formatNumber(value),
        testIou: (value) => formatNumber(value),
        precision: (value) => formatNumber(value),
        recall: (value) => formatNumber(value),
      }
    );

    renderCeChart();
  }

  function renderCeChart() {
    const maxDice = Math.max(...data.binaryRefresh.map((item) => item.testDice));
    $("#ce-chart").innerHTML = data.binaryRefresh
      .map(
        (item) => `
          <div class="bar-row">
            <div class="bar-copy">
              <strong>${item.ce}</strong>
              <span>${item.run}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width: ${(item.testDice / maxDice) * 100}%"></div>
            </div>
            <div class="bar-value">${formatNumber(item.testDice)}</div>
          </div>
        `
      )
      .join("");
  }

  async function checkBackendHealth() {
    $("#backend-status").textContent = "Đang kiểm tra backend cục bộ...";
    try {
      const response = await fetch("/health");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      $("#backend-status").textContent = `Backend sẵn sàng: ${payload.status}. Warm-up model: ${
        payload.predictor_warmup?.model_id || "N/A"
      }.`;
      return true;
    } catch (error) {
      $("#backend-status").textContent = `Không kết nối được backend cục bộ: ${error.message}`;
      return false;
    }
  }

  async function refreshModels() {
    $("#backend-status").textContent = "Đang tải model từ backend cục bộ...";
    try {
      const response = await fetch("/models");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const models = await response.json();
      state.models = models;
      if (!models.some((item) => item.model_id === state.selectedModelId) && models[0]) {
        state.selectedModelId = models[0].model_id;
      }
      renderModelSelect();
      $("#backend-status").textContent = `Đã tải ${models.length} model/checkpoint cục bộ.`;
    } catch (error) {
      $("#backend-status").textContent = `Không thể tải model: ${error.message}`;
    }
  }

  function renderModelSelect() {
    const select = $("#model-select");
    if (state.models.length === 0) {
      select.innerHTML = `<option value="${data.liveInference.defaultModelId}">${data.liveInference.defaultModelId}</option>`;
      select.value = data.liveInference.defaultModelId;
      return;
    }

    select.innerHTML = state.models
      .map(
        (item) => `
          <option value="${item.model_id}">
            ${item.display_name || item.model_id}
          </option>
        `
      )
      .join("");
    select.value = state.selectedModelId;
  }

  async function runLiveSegmentation() {
    const fileInput = $("#live-upload");
    const file = fileInput.files[0];
    if (!file) {
      $("#backend-status").textContent = "Hãy chọn một ảnh trước khi chạy live inference.";
      return;
    }

    $("#backend-status").textContent = "Đang upload ảnh lên backend cục bộ...";
    try {
      const uploadForm = new FormData();
      uploadForm.append("images", file);
      const uploadResponse = await fetch("/upload_images", {
        method: "POST",
        body: uploadForm,
      });
      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: HTTP ${uploadResponse.status}`);
      }
      const uploadPayload = await uploadResponse.json();
      const uploadedImage = Array.isArray(uploadPayload) ? uploadPayload[uploadPayload.length - 1] : null;
      if (!uploadedImage?.id) {
        throw new Error("Missing uploaded image id.");
      }

      $("#backend-status").textContent = "Đang chạy inference từ checkpoint cục bộ...";
      const segmentationResponse = await fetch("/segment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_id: uploadedImage.id,
          model_id: state.selectedModelId,
        }),
      });
      if (!segmentationResponse.ok) {
        throw new Error(`Segmentation failed: HTTP ${segmentationResponse.status}`);
      }
      state.liveResult = await segmentationResponse.json();
      state.showUncertainty = false;
      $("#backend-status").textContent = "Inference hoàn tất và đã hiển thị kết quả.";
      renderStage();
    } catch (error) {
      $("#backend-status").textContent = `Lỗi live inference: ${error.message}`;
    }
  }

  function setMode(mode) {
    state.mode = mode;
    if (mode !== "live") {
      state.showUncertainty = false;
    }
    document.querySelectorAll(".mode-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === mode);
    });
    $("#curated-controls").classList.toggle("hidden", mode !== "curated");
    $("#live-controls").classList.toggle("hidden", mode !== "live");
    $("#mode-description").textContent =
      mode === "curated"
        ? "Artifact mode luôn chạy được để bảo vệ luận văn bằng ảnh minh họa và số liệu đã chốt."
        : "Live inference hiện dùng backend cục bộ và checkpoint thật đã được chép vào thư mục này.";
    renderStage();
  }

  function bindEvents() {
    document.querySelectorAll(".mode-button").forEach((button) => {
      button.addEventListener("click", () => setMode(button.dataset.mode));
    });

    $("#model-select").addEventListener("change", (event) => {
      state.selectedModelId = event.target.value;
    });

    $("#check-backend").addEventListener("click", checkBackendHealth);
    $("#refresh-models").addEventListener("click", refreshModels);
    $("#run-live-segmentation").addEventListener("click", runLiveSegmentation);
    $("#toggle-uncertainty").addEventListener("click", () => {
      if (!state.liveResult?.uncertainty?.available) {
        return;
      }
      state.showUncertainty = !state.showUncertainty;
      renderStage();
    });
  }

  async function bootstrap() {
    $("#thesis-title").textContent = data.thesisTitle;
    renderHeroMetrics();
    renderStackList("dataset-summary", data.datasetSummary);
    renderStackList("defense-points", data.defensePoints);
    renderPipeline();
    renderCases();
    renderExperiments();
    renderModelSelect();
    bindEvents();
    await checkBackendHealth();
    await refreshModels();
    setMode("curated");
  }

  bootstrap();
})();
