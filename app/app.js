(function () {
  const state = {
    dashboard: null,
    models: [],
    images: { custom: [], default: [] },
    selectedImageId: null,
    selectedModelId: null,
    liveResult: null,
  };

  const uncertaintyScoreConfig = [
    ["global_entropy", "H_G (global entropy)"],
    ["boundary_entropy", "H_B (boundary entropy)"],
    ["uncertain_pixel_ratio", "Uncertain pixel ratio"],
    ["predicted_tumor_ratio", "Predicted tumor ratio"],
    ["mean_tumor_probability", "Mean tumor probability"],
  ];

  const conformalScoreConfig = [
    ["target_coverage", "Target coverage"],
    ["probability_floor", "Probability floor"],
    ["mean_set_size", "Mean set size"],
    ["sure_tumor_pixels", "Sure tumor pixels"],
    ["outer_tumor_pixels", "Outer tumor pixels"],
    ["uncertain_pixels", "Uncertain pixels"],
    ["uncertain_fraction", "Uncertain fraction"],
  ];

  function $(selector) {
    return document.querySelector(selector);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function formatNumber(value, digits = 4) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits) : "N/A";
  }

  function formatMetricValue(value) {
    if (value === null || value === undefined || value === "") {
      return "N/A";
    }
    if (typeof value === "number") {
      if (Number.isInteger(value)) {
        return `${value}`;
      }
      return formatNumber(value);
    }
    return String(value);
  }

  const metricDefinitions = {
    dice: { direction: "max" },
    iou: { direction: "max" },
    hd95: { direction: "min" },
    precision: { direction: "max" },
    recall: { direction: "max" },
  };

  function getBestMetricValues(columns, rows) {
    return new Map(
      columns
        .filter((column) => column.metric)
        .map((column) => {
          const values = rows.map((row) => Number(row[column.key])).filter(Number.isFinite);
          if (!values.length) {
            return [column.key, null];
          }
          const best = metricDefinitions[column.metric].direction === "min" ? Math.min(...values) : Math.max(...values);
          return [column.key, best];
        })
    );
  }

  function buildTable(columns, rows) {
    if (!rows.length) {
      return '<p class="muted-note">Chưa có dữ liệu.</p>';
    }

    const bestMetricValues = getBestMetricValues(columns, rows);
    const header = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("");
    const body = rows
      .map(
        (row) => `
          <tr>
            ${columns
              .map((column) => {
                const value = column.render ? column.render(row) : row[column.key];
                const isBest = bestMetricValues.get(column.key) === Number(row[column.key]);
                const content = column.html ? value : escapeHtml(value);
                return `<td>${isBest ? `<strong class="metric-best">${content}</strong>` : content}</td>`;
              })
              .join("")}
          </tr>
        `
      )
      .join("");

    return `
      <div class="table-wrap">
        <table>
          <thead><tr>${header}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    `;
  }

  function renderDatasetTable() {
    const datasets = state.dashboard.overview?.datasets || [];
    const columns = [
      {
        label: "Bộ dữ liệu",
        key: "name",
        html: true,
        render: (row) => `<strong>${escapeHtml(row.name)}</strong><span class="table-description">${escapeHtml(row.description)}</span>`,
      },
      { label: "Tổng", key: "total", render: (row) => Number(row.total).toLocaleString("vi-VN") },
      { label: "Dương tính", key: "positive", render: (row) => Number(row.positive).toLocaleString("vi-VN") },
      { label: "Âm tính", key: "negative", render: (row) => Number(row.negative).toLocaleString("vi-VN") },
      { label: "Train", key: "train" },
      { label: "Val", key: "validation" },
      { label: "Hiệu chỉnh phát triển", key: "development_calibration" },
      { label: "Hiệu chỉnh cuối", key: "final_calibration" },
      { label: "Test", key: "test" },
    ];
    $("#dataset-table").innerHTML = buildTable(columns, datasets);
  }

  function renderExperimentTables() {
    const experiments = state.dashboard.experiments || {};
    const formatTableMetric = (value) => (Number.isFinite(Number(value)) ? formatNumber(value, 3) : formatMetricValue(value));
    const backboneColumns = [
      { label: "Model", key: "model" }, { label: "Dice ↑", key: "dice", metric: "dice", render: (row) => formatNumber(row.dice, 3) },
      { label: "IoU ↑", key: "iou", metric: "iou", render: (row) => formatNumber(row.iou, 3) }, { label: "HD95 ↓", key: "hd95", metric: "hd95", render: (row) => formatNumber(row.hd95, 3) },
      { label: "Precision ↑", key: "precision", metric: "precision", render: (row) => formatNumber(row.precision, 3) }, { label: "Recall ↑", key: "recall", metric: "recall", render: (row) => formatNumber(row.recall, 3) },
    ];
    const ablationColumns = [
      { label: "Phương pháp", key: "method" }, { label: "Dice ↑", key: "dice", metric: "dice", render: (row) => formatTableMetric(row.dice) },
      { label: "IoU ↑", key: "iou", metric: "iou", render: (row) => formatTableMetric(row.iou) }, { label: "HD95 ↓", key: "hd95", metric: "hd95", render: (row) => formatTableMetric(row.hd95) },
    ];
    $("#btxrd-backbones-table").innerHTML = buildTable(backboneColumns, experiments.btxrd_backbones || []);
    $("#btxrd-ablation-table").innerHTML = buildTable(ablationColumns, experiments.btxrd_ablation || []);
    $("#fracatlas-backbones-table").innerHTML = buildTable(backboneColumns, experiments.fracatlas_backbones || []);
    $("#fracatlas-ablation-table").innerHTML = buildTable(ablationColumns, experiments.fracatlas_ablation || []);
  }

  function flattenImages() {
    return [...(state.images.default || []), ...(state.images.custom || [])];
  }

  function renderImageOptions() {
    const allImages = flattenImages();
    const select = $("#image-select");

    if (!state.selectedImageId && allImages.length) {
      state.selectedImageId = allImages[0].id;
    }

    select.innerHTML = allImages
      .map((item) => {
        const label = item.source === "custom" ? `[upload] ${item.filename}` : (item.display_name || `[sample] ${item.filename}`);
        const selected = item.id === state.selectedImageId ? "selected" : "";
        return `<option value="${escapeHtml(item.id)}" ${selected}>${escapeHtml(label)}</option>`;
      })
      .join("");

    if (state.selectedImageId) {
      const image = allImages.find((item) => item.id === state.selectedImageId);
      if (image) {
        $("#original-image").src = image.image_url;
      }
    }
  }

  function renderModelOptions() {
    const select = $("#model-select");
    const previousModelId = state.selectedModelId || select.value;
    const available = state.models.filter((model) => model.available === true && model.model_id);
    const selectedModel = available.find((model) => model.model_id === previousModelId) || available[0] || null;
    state.selectedModelId = selectedModel?.model_id || null;
    select.innerHTML = available.length
      ? available
          .map((model) => {
            const selected = model.model_id === selectedModel?.model_id ? "selected" : "";
            return `<option value="${escapeHtml(model.model_id)}" ${selected}>${escapeHtml(model.display_name)}</option>`;
          })
          .join("")
      : '<option value="" disabled selected>Chưa có checkpoint/config tương thích</option>';
    select.disabled = !available.length;
    $("#run-live-segmentation").disabled = !selectedModel;
  }

  function renderResultGallery(items) {
    $("#result-gallery").innerHTML = items
      .map(
        (item) => `
          <figure class="image-panel">
            <figcaption>${escapeHtml(item.label)}</figcaption>
            <div class="image-frame ${item.emphasized ? "result-frame" : ""}">
              <img src="${escapeHtml(item.src)}" alt="${escapeHtml(item.alt)}" />
            </div>
          </figure>
        `
      )
      .join("");
  }

  function renderScoreGrid(containerId, summary, config) {
    const container = document.getElementById(containerId);
    container.innerHTML = config
      .map(([key, label]) => `
        <article class="score-card">
          <p class="metric-label">${escapeHtml(label)}</p>
          <p class="metric-value small">${escapeHtml(formatMetricValue(summary?.[key]))}</p>
        </article>
      `)
      .join("");
  }

  function renderHeatmaps(containerId, maps) {
    const entries = Object.entries(maps || {});
    const container = document.getElementById(containerId);
    if (!entries.length) {
      container.innerHTML = '<p class="muted-note">Không có heatmap cho kết quả này.</p>';
      return;
    }

    container.innerHTML = entries
      .map(
        ([key, value]) => `
          <figure class="image-panel compact">
            <figcaption>${escapeHtml(key)}</figcaption>
            <div class="image-frame result-frame">
              <img src="${escapeHtml(value)}" alt="${escapeHtml(key)}" />
            </div>
          </figure>
        `
      )
      .join("");
  }

  function renderLiveResult() {
    const result = state.liveResult;
    if (!result) {
      return;
    }

    $("#stage-title").textContent = result.display_name || result.model_id;
    $("#stage-summary").textContent = result.note || "Kết quả inference từ backend cục bộ.";
    $("#original-image").src = result.original_image_url;
    $("#stage-badges").innerHTML = [
      result.metadata?.model_type,
      result.checkpoint_name,
      `foreground:${result.metadata?.foreground_pixels ?? 0}`,
      result.uncertainty?.conformal?.available ? "conformal-ready" : "uncertainty-only",
    ]
      .filter(Boolean)
      .map((item) => `<span class="pill">${escapeHtml(item)}</span>`)
      .join("");

    renderResultGallery([
      {
        label: "Overlay",
        src: result.overlay_image_url,
        alt: "Segmentation overlay",
        emphasized: true,
      },
      {
        label: "Mask",
        src: result.mask_image_url,
        alt: "Segmentation mask",
        emphasized: false,
      },
    ]);

    $("#case-notes").innerHTML = `
      <p><strong>Ảnh:</strong> ${escapeHtml(result.original_filename)}</p>
      <p><strong>Model:</strong> ${escapeHtml(result.model_id)}</p>
      <p><strong>Foreground fraction:</strong> ${escapeHtml(formatMetricValue(result.metadata?.foreground_fraction))}</p>
    `;

    const uncertainty = result.uncertainty || {};
    const hasUncertainty = Boolean(uncertainty.available);
    $("#uncertainty-panel").classList.toggle("hidden", !hasUncertainty);
    if (!hasUncertainty) {
      return;
    }

    $("#uncertainty-note").textContent = uncertainty.note || "";
    renderScoreGrid("uncertainty-score-grid", uncertainty.summary, uncertaintyScoreConfig);
    renderHeatmaps("uncertainty-heatmaps", uncertainty.heatmaps || {});

    const conformal = uncertainty.conformal || {};
    $("#conformal-section").classList.toggle("hidden", !conformal.available);
    $("#conformal-note").textContent = conformal.note || "";
    renderScoreGrid("conformal-score-grid", conformal.summary, conformalScoreConfig);
    renderHeatmaps("conformal-heatmaps", conformal.heatmaps || {});
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Request failed: ${response.status}`);
    }
    return payload;
  }

  async function loadDashboard() {
    state.dashboard = await fetchJson("/dashboard");
    $("#research-disclaimer").textContent = state.dashboard.disclaimer;
    renderDatasetTable();
  }

  async function loadModels() {
    const payload = await fetchJson("/models");
    if (!payload || !Array.isArray(payload.models)) {
      throw new Error("Phản hồi danh sách model không hợp lệ.");
    }
    state.models = payload.models;
    renderModelOptions();
    renderExperimentTables();
  }

  async function refreshModels() {
    try {
      await loadModels();
    } catch (error) {
      console.error("Unable to refresh model registry:", error);
      $("#backend-status").textContent = "Không tải được danh sách model.";
    }
  }

  async function loadImages() {
    state.images = await fetchJson("/get_images");
    renderImageOptions();
  }

  async function updateHealth() {
    const health = await fetchJson("/health");
    const warmup = health.predictor_warmup || {};
    const parts = [
      `status: ${health.status}`,
      `warmup started: ${String(Boolean(warmup.started))}`,
      `completed: ${String(Boolean(warmup.completed))}`,
    ];
    if (warmup.error) {
      parts.push(`error: ${warmup.error}`);
    }
    $("#backend-status").textContent = parts.join(" | ");
  }

  async function uploadImage() {
    const input = $("#live-upload");
    const [file] = input.files || [];
    if (!file) {
      window.alert("Hãy chọn một ảnh trước khi tải lên.");
      return;
    }

    const formData = new FormData();
    formData.append("images", file);
    const payload = await fetchJson("/upload_images", {
      method: "POST",
      body: formData,
    });
    const uploaded = payload[0];
    state.selectedImageId = uploaded?.id || state.selectedImageId;
    input.value = "";
    await loadImages();
  }

  async function deleteImage() {
    if (!state.selectedImageId) {
      window.alert("Chưa chọn ảnh để xóa.");
      return;
    }
    if (String(state.selectedImageId).startsWith("default:")) {
      window.alert("Ảnh mẫu mặc định không thể xóa.");
      return;
    }
    await fetchJson(`/images/${encodeURIComponent(state.selectedImageId)}`, { method: "DELETE" });
    state.selectedImageId = null;
    await loadImages();
  }

  async function runSegmentation() {
    if (!state.selectedImageId) {
      window.alert("Hãy chọn ảnh đầu vào trước.");
      return;
    }
    if (!state.selectedModelId) {
      window.alert("Hãy chọn checkpoint trước.");
      return;
    }

    const button = $("#run-live-segmentation");
    button.disabled = true;
    button.textContent = "Đang chạy...";
    try {
      state.liveResult = await fetchJson("/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: state.selectedImageId, model_id: state.selectedModelId }),
      });
      renderLiveResult();
    } finally {
      button.disabled = false;
      button.textContent = "Chạy segmentation";
    }
  }

  function bindEvents() {
    $("#check-backend").addEventListener("click", updateHealth);
    $("#refresh-models").addEventListener("click", refreshModels);
    $("#refresh-library").addEventListener("click", loadImages);
    $("#upload-image").addEventListener("click", uploadImage);
    $("#delete-image").addEventListener("click", deleteImage);
    $("#run-live-segmentation").addEventListener("click", runSegmentation);
    $("#image-select").addEventListener("change", (event) => {
      state.selectedImageId = event.target.value;
      const selected = flattenImages().find((item) => item.id === state.selectedImageId);
      if (selected) {
        $("#original-image").src = selected.image_url;
      }
    });
    $("#model-select").addEventListener("change", (event) => {
      state.selectedModelId = event.target.value;
    });
  }

  async function bootstrap() {
    bindEvents();
    try {
      await loadDashboard();
      await Promise.all([loadModels(), loadImages(), updateHealth()]);
    } catch (error) {
      $("#backend-status").textContent = error.message;
    }
  }

  bootstrap();
})();
