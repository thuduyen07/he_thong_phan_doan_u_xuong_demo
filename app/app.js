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
    ["mean_predictive_entropy_pred_tumor", "Mean entropy"],
    ["p90_predictive_entropy_pred_tumor", "P90 entropy"],
    ["mean_predictive_entropy_boundary_tumor", "Boundary entropy"],
    ["mean_one_minus_msp_pred_tumor", "1 - MSP"],
    ["mean_one_minus_margin_pred_tumor", "1 - Margin"],
    ["low_tumor_probability_fraction_pred_tumor", "Low prob fraction"],
    ["mean_tumor_probability_pred_tumor", "Mean tumor prob"],
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

  function renderHeroMetrics() {
    const metrics = state.dashboard.hero_metrics || [];
    $("#hero-metrics").innerHTML = metrics
      .map(
        (item) => `
          <article class="metric-card">
            <p class="metric-value">${escapeHtml(item.value)}</p>
            <p class="metric-label">${escapeHtml(item.label)}</p>
            <p class="metric-note">${escapeHtml(item.note)}</p>
          </article>
        `
      )
      .join("");
  }

  function renderStackList(containerId, items) {
    document.getElementById(containerId).innerHTML = items
      .map((item) => `<p>${escapeHtml(item)}</p>`)
      .join("");
  }

  function renderPipeline() {
    const pipeline = state.dashboard.pipeline || {};

    $("#pipeline-strip").innerHTML = (pipeline.modules || [])
      .map(
        (item, index) => `
          <article class="pipeline-card">
            <div class="pipeline-step">${index + 1}</div>
            <h3>${escapeHtml(item.title)}</h3>
            <p>${escapeHtml(item.summary)}</p>
            <div class="card-list">
              ${(item.details || []).map((detail) => `<span>${escapeHtml(detail)}</span>`).join("")}
            </div>
          </article>
        `
      )
      .join("");

    $("#schedule-grid").innerHTML = (pipeline.schedule || [])
      .map(
        (item) => `
          <article class="schedule-card">
            <p class="mini-tag">${escapeHtml(item.epoch_range)}</p>
            <h3>${escapeHtml(item.phase)}</h3>
            <p>${escapeHtml(item.purpose)}</p>
            <div class="badge-row">
              ${(item.signals || []).map((signal) => `<span class="pill">${escapeHtml(signal)}</span>`).join("")}
            </div>
          </article>
        `
      )
      .join("");

    $("#formula-list").innerHTML = (pipeline.formulas || [])
      .map(
        (item) => `
          <article class="formula-card">
            <p class="formula-label">${escapeHtml(item.label)}</p>
            <code>${escapeHtml(item.value)}</code>
          </article>
        `
      )
      .join("");
  }

  function buildTable(columns, rows) {
    if (!rows.length) {
      return '<p class="muted-note">Chưa có dữ liệu.</p>';
    }

    const header = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("");
    const body = rows
      .map(
        (row) => `
          <tr>
            ${columns
              .map((column) => `<td>${escapeHtml(column.render ? column.render(row) : row[column.key])}</td>`)
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

  function renderExperimentTables() {
    const experiments = state.dashboard.experiments || {};
    const runColumns = [
      { label: "Run", key: "display_name" },
      { label: "Phase", key: "phase" },
      { label: "Mode", key: "training_mode" },
      { label: "Primary", render: (row) => `${row.primary_metric || "metric"}: ${formatMetricValue(row.primary_value)}` },
      { label: "Test Dice", render: (row) => formatMetricValue(row.test_dice) },
      { label: "Test IoU", render: (row) => formatMetricValue(row.test_iou) },
      { label: "Conformal", render: (row) => (row.conformal?.available ? `yes (${formatMetricValue(row.conformal.target_coverage)})` : "no") },
    ];

    $("#runs-table").innerHTML = buildTable(runColumns, experiments.runs || []);
    renderStackList("artifact-status", [
      `Số artifact thực nghiệm còn lại: ${(experiments.runs || []).length}.`,
      `Số checkpoint live hiện khả dụng: ${state.models.length}.`,
      state.models.length
        ? "Repo demo đang có checkpoint mới để chạy live inference."
        : "Repo demo hiện chưa có checkpoint mới, nên live inference đang ở trạng thái chờ bổ sung artifact mới.",
    ]);
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
        const label = item.source === "custom" ? `[upload] ${item.filename}` : `[sample] ${item.filename}`;
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
    if (!state.selectedModelId && state.models.length) {
      state.selectedModelId = state.models[0].model_id;
    }
    select.innerHTML = state.models.length
      ? state.models
          .map((model) => {
            const selected = model.model_id === state.selectedModelId ? "selected" : "";
            return `<option value="${escapeHtml(model.model_id)}" ${selected}>${escapeHtml(model.display_name)}</option>`;
          })
          .join("")
      : '<option value="">Chưa có checkpoint mới</option>';
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
      <p><strong>Checkpoint:</strong> ${escapeHtml(result.checkpoint_path)}</p>
      <p><strong>Metadata:</strong> ${escapeHtml(JSON.stringify(result.metadata || {}))}</p>
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
    $("#thesis-title").textContent = state.dashboard.thesis_title;
    renderStackList("overview-summary", state.dashboard.overview.summary || []);
    renderHeroMetrics();
    renderPipeline();
  }

  async function loadModels() {
    state.models = await fetchJson("/models");
    if (!state.selectedModelId) {
      state.selectedModelId = state.dashboard?.overview?.default_model_id || state.models[0]?.model_id || null;
    }
    renderModelOptions();
    renderExperimentTables();
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
    $("#refresh-models").addEventListener("click", loadModels);
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
