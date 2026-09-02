(function () {
  const state = {
    capabilities: null,
    dashboard: null,
    staticSamples: [],
    images: { custom: [], default: [] },
    models: [],
    selectedImageId: null,
  };

  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  const formatNumber = (value, digits = 4) => (Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—");
  const formatInteger = (value) => String(value).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  const setError = (target, message) => {
    const el = $(target);
    el.textContent = message || "";
    el.classList.toggle("hidden", !message);
  };

  async function fetchJson(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (_) {
      throw new Error("NETWORK_ERROR");
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP_${response.status}`);
    return payload;
  }

  function friendlyError(error) {
    const message = String(error?.message || error);
    if (message.includes("MODEL_UNAVAILABLE")) return "Mô hình phù hợp với dữ liệu này hiện không khả dụng.";
    if (message.includes("DEPENDENCY_MISSING")) return "Máy chủ Live Demo chưa có đầy đủ thành phần cần thiết để chạy mô hình.";
    if (message.includes("MODEL_LOAD_FAILED")) return "Không thể tải mô hình. Vui lòng thử lại sau.";
    if (message.includes("INVALID_IMAGE")) return "Không thể đọc hoặc xử lý ảnh. Vui lòng chọn PNG hoặc JPEG hợp lệ.";
    if (message.includes("FILE_TOO_LARGE") || message.includes("413")) return "Ảnh vượt quá dung lượng cho phép 10 MB.";
    if (message.includes("NETWORK_ERROR")) return "Không thể kết nối tới máy chủ. Vui lòng thử lại.";
    if (message.includes("STATIC_SAMPLE_UNAVAILABLE")) return "Không thể tải dữ liệu mẫu này.";
    return "Không thể hoàn tất phân đoạn cho ảnh này.";
  }

  function renderDatasetSplitTable(dataset) {
    const columns = ["Tập dữ liệu", "Số ảnh", "Dương tính", "Âm tính", "Tỉ lệ"];
    const rows = dataset.rows || [];
    return `<article class="result-card dataset-split-table" aria-labelledby="${escapeHtml(dataset.id)}-split-title"><h3 id="${escapeHtml(dataset.id)}-split-title">${escapeHtml(dataset.title)}</h3><div class="table-wrap"><table><thead><tr>${columns.map((label) => `<th scope="col">${escapeHtml(label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr${row.is_total ? ' class="is-total"' : ""}><th scope="row">${escapeHtml(row.split)}</th><td>${formatInteger(row.images)}</td><td>${formatInteger(row.positive)}</td><td>${formatInteger(row.negative)}</td><td>${escapeHtml(row.ratio)}</td></tr>`).join("")}</tbody></table></div></article>`;
  }

  function computeBestMetricKeys(section) {
    const rows = section.rows || [];
    const best = {};
    (section.metrics || []).forEach((metric) => {
      const numeric = rows.map((row) => Number(row[metric.key])).filter(Number.isFinite);
      best[metric.key] = numeric.length ? (metric.direction === "min" ? Math.min(...numeric) : Math.max(...numeric)) : null;
    });
    return best;
  }

  function renderExperimentTable(section) {
    const bestMetrics = computeBestMetricKeys(section);
    const columns = [{ key: section.row_key, label: section.row_label }, ...(section.metrics || [])];
    const rows = (section.rows || [])
      .map((row) => {
        const isPrimary = section.primary_value && row[section.row_key] === section.primary_value;
        const cells = columns
          .map((column) => {
            if (column.key === section.row_key) {
              const badge = isPrimary && section.primary_badge ? `<span class="proposed-badge">${escapeHtml(section.primary_badge)}</span>` : "";
              return `<th scope="row"><div class="table-row-label">${escapeHtml(row[column.key])}${badge}</div></th>`;
            }
            const numericValue = Number(row[column.key]);
            const displayValue = Number.isFinite(numericValue) ? formatNumber(numericValue, 3) : escapeHtml(row[column.key]);
            const isBest = Number.isFinite(numericValue) && bestMetrics[column.key] !== null && numericValue === bestMetrics[column.key];
            return `<td>${isBest ? `<strong class="metric-best">${displayValue}</strong>` : displayValue}</td>`;
          })
          .join("");
        return `<tr${isPrimary ? ' class="proposed-row"' : ""}>${cells}</tr>`;
      })
      .join("");
    const note = section.note ? `<p class="muted-note">${escapeHtml(section.note)}</p>` : "";
    return `<article class="result-card experiment-card"><h3>${escapeHtml(section.title)}</h3>${note}<div class="table-wrap"><table><thead><tr>${columns.map((column) => `<th scope="col">${escapeHtml(column.label)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div></article>`;
  }

  function renderDashboard() {
    const datasetSplits = state.dashboard?.overview?.dataset_splits || [];
    $("#research-disclaimer").textContent = state.dashboard?.disclaimer || "";
    $("#dataset-split-tables").innerHTML = datasetSplits.map(renderDatasetSplitTable).join("");
    const sections = state.dashboard?.experiments?.sections || [];
    $("#experiment-sections").innerHTML = sections.map(renderExperimentTable).join("");
  }

  function normalizeStatic(payload) {
    return {
      source: "static",
      displayName: payload.sample?.display_name || "Prepared static result",
      modelName: payload.model?.display_name || "",
      dataset: payload.sample?.dataset || "",
      heading: "Kết quả mẫu tham chiếu",
      artifacts: {
        original: payload.artifacts?.original || null,
        overlay: payload.artifacts?.overlay || null,
        predictionMask: payload.artifacts?.prediction_mask || payload.artifacts?.mask || null,
        referenceMask: payload.artifacts?.reference_mask || null,
        probability: payload.artifacts?.probability_heatmap || null,
        entropy: payload.artifacts?.entropy_heatmap || null,
      },
      uncertainty: payload.uncertainty || {},
      referenceMetrics: payload.reference_metrics || {},
      note: "Kết quả được chuẩn bị offline từ pipeline nghiên cứu đã đăng ký.",
      provenance: payload.provenance || {},
    };
  }

  function normalizeLive(payload) {
    return {
      source: "live",
      displayName: payload.display_name || payload.model_id,
      modelName: payload.display_name || "",
      dataset: payload.dataset || "",
      heading: payload.dataset ? `Kết quả từ mô hình ${String(payload.dataset).toUpperCase()}` : "Kết quả suy luận trực tiếp",
      artifacts: {
        original: payload.original_image_url || null,
        overlay: payload.overlay_image_url || null,
        predictionMask: payload.mask_image_url || null,
        referenceMask: null,
        probability: payload.uncertainty?.tumor_probability?.heatmap_url || null,
        entropy: payload.uncertainty?.pixel_entropy?.heatmap_url || null,
      },
      uncertainty: payload.uncertainty || {},
      referenceMetrics: { available: false },
      note: payload.note || "Kết quả inference từ backend.",
      provenance: {},
    };
  }

  function buildImagePanel(label, url, alt, detail = "", legend = "", placeholder = "") {
    if (!url && !placeholder) return "";
    const body = url
      ? `<div class="image-frame ${label !== "Ảnh gốc" ? "result-frame" : ""}"><img src="${escapeHtml(url)}" alt="${escapeHtml(alt)}" /></div>`
      : `<div class="image-frame image-placeholder"><p>${escapeHtml(placeholder)}</p></div>`;
    return `<figure class="image-panel"><figcaption>${escapeHtml(label)}</figcaption>${body}${detail ? `<p class="score-detail">${escapeHtml(detail)}</p>` : ""}${legend ? `<div class="heatmap-legend"><span>Thấp</span><i></i><span>Cao</span><em>${escapeHtml(legend)}</em></div>` : ""}</figure>`;
  }

  function buildReferenceMetrics(result) {
    if (!result.referenceMetrics?.available) return "";
    const items = [["Dice ↑", result.referenceMetrics.dice], ["IoU ↑", result.referenceMetrics.iou], ["HD95 ↓", result.referenceMetrics.hd95], ["Precision ↑", result.referenceMetrics.precision], ["Recall ↑", result.referenceMetrics.recall]];
    return `<section class="reference-metrics-inline"><p class="section-tag">Reference metrics</p><h4>Hiệu năng trên mẫu tham chiếu</h4><div class="uncertainty-score-grid">${items.map(([label, value]) => `<article class="score-card"><p class="metric-label">${escapeHtml(label)}</p><p class="metric-value small">${formatNumber(value)}</p></article>`).join("")}</div></section>`;
  }

  function buildUncertainty(result) {
    const uncertainty = result.uncertainty || {};
    const global = uncertainty.global || {};
    const boundary = uncertainty.boundary || {};
    const summary = uncertainty.summary || {};
    const hasPanel = Boolean(uncertainty.pixel_entropy?.available || global.available || boundary.available);
    if (!hasPanel) return "";
    const cards = [
      ["Bất định toàn cục (H_G)", global.available ? formatNumber(global.value) : "—", "Tóm tắt mức bất định dự đoán trên toàn ảnh."],
      ["Bất định vùng biên (H_B)", boundary.available ? formatNumber(boundary.value) : "—", boundary.available ? "Tóm tắt bất định quanh vùng biên phân đoạn." : "Chưa có định nghĩa hậu kiểm tương đương cho cấu hình này."],
      ["Tỷ lệ điểm ảnh bất định", uncertainty.pixel_entropy?.available ? formatNumber(summary.uncertain_pixel_ratio) : "—", "Tỷ lệ điểm ảnh có entropy vượt ngưỡng analysis cấu hình trên toàn ảnh."],
    ];
    const facts = [];
    if (global.available) facts.push(`H_G của ảnh này là ${formatNumber(global.value)}.`);
    if (Number.isFinite(Number(summary.uncertain_pixel_ratio))) facts.push(`${(Number(summary.uncertain_pixel_ratio) * 100).toFixed(2)}% điểm ảnh có entropy vượt ngưỡng analysis.`);
    if (boundary.available) facts.push("H_B được tính trên vùng biên phân đoạn của ảnh này.");
    return `<section class="uncertainty-panel-inline"><div class="uncertainty-header"><div><p class="section-tag">Predictive uncertainty</p><h4>Bất định và hậu kiểm</h4></div></div><div class="uncertainty-score-grid">${cards.map(([label, value, detail]) => `<article class="score-card"><p class="metric-label">${escapeHtml(label)}</p><p class="metric-value small">${escapeHtml(value)}</p><p class="score-detail">${escapeHtml(detail)}</p></article>`).join("")}</div>${facts.length ? `<div class="audit-interpretation"><article><h4>Nhận xét hậu kiểm</h4>${facts.map((fact) => `<p>${escapeHtml(fact)}</p>`).join("")}</article></div>` : ""}</section>`;
  }

  function buildResultCard(result) {
    const primary = [buildImagePanel("Ảnh gốc", result.artifacts.original, "Ảnh X-quang gốc"), buildImagePanel("Lớp phủ phân đoạn", result.artifacts.overlay, "Kết quả lớp phủ phân đoạn")]
      .filter(Boolean)
      .join("");
    const maps = [
      buildImagePanel("Bản đồ xác suất vùng u", result.artifacts.probability, "Heatmap xác suất vùng u", "Vị trí mô hình dự đoán thuộc vùng foreground; thang hiển thị 0 đến 1.", "Xác suất vùng u"),
      buildImagePanel("Bản đồ bất định dự đoán", result.artifacts.entropy, "Heatmap bất định dự đoán", "Nơi mô hình ít chắc chắn hơn về dự đoán phân đoạn.", "Bất định dự đoán"),
    ]
      .filter(Boolean)
      .join("");
    return `<article class="model-result-card"><div class="stage-header"><div><p class="stage-tag">${escapeHtml(result.source === "static" ? "KẾT QUẢ MẪU THAM CHIẾU" : "KẾT QUẢ SUY LUẬN TRỰC TIẾP")}</p><h4>${escapeHtml(result.heading)}</h4><p class="score-detail">${escapeHtml(result.modelName)}</p></div>${result.dataset ? `<div class="badge-row"><span class="pill">${escapeHtml(String(result.dataset).toUpperCase())}</span></div>` : ""}</div><p class="stage-summary">${escapeHtml(result.note)}</p><section class="result-image-section"><h4>So sánh phân đoạn</h4><div class="result-image-pair">${primary}</div></section><section class="result-image-section"><h4>Xác suất và bất định</h4><div class="result-image-pair">${maps}</div></section><details class="mask-disclosure"><summary>Xem mặt nạ nhị phân</summary><div class="result-image-pair">${buildImagePanel("Mặt nạ Ground Truth", result.artifacts.referenceMask, "Mặt nạ Ground Truth", "", "", "Không có mặt nạ tham chiếu")}${buildImagePanel("Mặt nạ dự đoán", result.artifacts.predictionMask, "Mặt nạ phân đoạn dự đoán", "", "", "Không có mặt nạ dự đoán")}</div></details>${buildReferenceMetrics(result)}${buildUncertainty(result)}</article>`;
  }

  function renderStaticResult(payload) {
    const result = normalizeStatic(payload);
    $("#stage-source").textContent = "KẾT QUẢ MẪU THAM CHIẾU";
    $("#stage-title").textContent = result.displayName;
    $("#stage-summary").textContent = result.note;
    $("#stage-badges").innerHTML = [result.dataset ? `dataset:${result.dataset}` : null, result.provenance?.split ? `split:${result.provenance.split}` : null]
      .filter(Boolean)
      .map((value) => `<span class="pill">${escapeHtml(value)}</span>`)
      .join("");
    $("#result-gallery").innerHTML = `<div class="model-result-stack">${buildResultCard(result)}</div>`;
    $("#reference-metrics").classList.add("hidden");
    $("#uncertainty-panel").classList.add("hidden");
  }

  function renderResultSet(payload) {
    const results = (payload.results || []).map(normalizeLive);
    const isAlternativeSet = results.length > 1;
    $("#stage-source").textContent = isAlternativeSet ? "KẾT QUẢ THAY THẾ THEO TẬP DỮ LIỆU" : "KẾT QUẢ SUY LUẬN TRỰC TIẾP";
    $("#stage-title").textContent = isAlternativeSet ? "Hai kết quả thay thế từ các mô hình đã đăng ký" : results[0]?.displayName || "Kết quả suy luận";
    $("#stage-summary").textContent =
      payload.routing?.mode === "multi_model_alternatives"
        ? "Ảnh tải lên không có quy tắc chọn mô hình liên tập dữ liệu đã được kiểm chứng trong dự án hiện tại, nên hệ thống hiển thị cả hai kết quả thay thế."
        : results[0]?.note || "";
    $("#stage-badges").innerHTML = [
      payload.image?.source === "default" ? "ảnh tham chiếu tích hợp" : "ảnh tải lên",
      payload.routing?.mode === "trusted_dataset" && payload.image?.dataset ? `dataset:${payload.image.dataset}` : null,
      payload.routing?.mode === "multi_model_alternatives" ? `${results.length} mô hình` : null,
    ]
      .filter(Boolean)
      .map((value) => `<span class="pill">${escapeHtml(value)}</span>`)
      .join("");
    $("#result-gallery").innerHTML = `<div class="model-result-stack">${results.map(buildResultCard).join("")}</div>`;
    $("#reference-metrics").classList.add("hidden");
    $("#uncertainty-panel").classList.add("hidden");
  }

  async function loadStaticResult(id) {
    setError("#static-error", "");
    try {
      renderStaticResult(await fetchJson(`/static-samples/${encodeURIComponent(id)}`));
    } catch (error) {
      setError("#static-error", friendlyError(error));
    }
  }

  function samplesForDataset(datasetId) {
    return state.staticSamples.filter((sample) => sample.available && sample.dataset === datasetId);
  }

  function renderStaticOptions() {
    const datasets = [
      ["btxrd", "#static-sample-select-btxrd"],
      ["fracatlas", "#static-sample-select-fracatlas"],
    ];
    datasets.forEach(([datasetId, selector]) => {
      const select = $(selector);
      const items = samplesForDataset(datasetId);
      select.innerHTML = items.map((sample) => `<option value="${escapeHtml(sample.id)}">${escapeHtml(sample.display_name)}</option>`).join("") || '<option value="">Chưa có mẫu khả dụng</option>';
      select.disabled = !items.length;
    });
    const firstAvailable = datasets.map(([, selector]) => $(selector).value).find(Boolean);
    if (firstAvailable) loadStaticResult(firstAvailable);
  }

  function flattenImages() {
    return [...(state.images.default || []), ...(state.images.custom || [])];
  }

  function renderLiveOptions() {
    const images = flattenImages();
    const image = $("#image-select");
    image.innerHTML = images.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.display_name || item.filename)}</option>`).join("");
    state.selectedImageId = image.value || null;
    $("#live-preview").src = images.find((item) => item.id === state.selectedImageId)?.image_url || "";
    $("#run-live-segmentation").disabled = !state.selectedImageId;
    const datasets = [...new Set(state.models.filter((model) => model.available).map((model) => String(model.dataset || "").toUpperCase()).filter(Boolean))];
    $("#live-routing-note").textContent = datasets.length ? `Ảnh tải lên sẽ chạy qua các mô hình: ${datasets.join(", ")}.` : "Chưa có mô hình live khả dụng.";
  }

  async function ensureLiveData() {
    const [models, images] = await Promise.all([fetchJson("/models"), fetchJson("/get_images")]);
    state.models = models.models || [];
    state.images = images || { custom: [], default: [] };
    renderLiveOptions();
  }

  function setDemoMode(mode) {
    if (mode === "live" && !state.capabilities?.features?.live_demo) return;
    $("#static-demo-panel").classList.toggle("hidden", mode !== "static");
    $("#live-demo-panel").classList.toggle("hidden", mode !== "live");
    ["static", "live"].forEach((name) => {
      const tab = $(`#${name}-tab`);
      const active = name === mode;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    if (mode === "live") ensureLiveData().catch((error) => setError("#live-error", friendlyError(error)));
  }

  async function uploadImage() {
    const [file] = $("#live-upload").files || [];
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      setError("#live-error", "Ảnh vượt quá dung lượng cho phép 10 MB.");
      return;
    }
    const form = new FormData();
    form.append("images", file);
    setError("#live-error", "");
    try {
      const uploaded = await fetchJson("/upload_images", { method: "POST", body: form });
      state.images = await fetchJson("/get_images");
      renderLiveOptions();
      $("#image-select").value = uploaded[0]?.id || $("#image-select").value;
      state.selectedImageId = $("#image-select").value;
      $("#live-preview").src = flattenImages().find((item) => item.id === state.selectedImageId)?.image_url || "";
    } catch (error) {
      setError("#live-error", friendlyError(error));
    }
  }

  async function runLiveSegmentation() {
    if (!state.selectedImageId) return;
    const button = $("#run-live-segmentation");
    button.disabled = true;
    button.textContent = "Đang phân đoạn và tính bất định...";
    setError("#live-error", "");
    try {
      renderResultSet(
        await fetchJson("/segment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_id: state.selectedImageId }),
        }),
      );
    } catch (error) {
      setError("#live-error", friendlyError(error));
    } finally {
      button.disabled = false;
      button.textContent = "Chạy phân đoạn";
    }
  }

  function handleStaticSampleChange(activeDataset, inactiveDataset) {
    const active = $(`#static-sample-select-${activeDataset}`);
    const inactive = $(`#static-sample-select-${inactiveDataset}`);
    if (active.value) {
      inactive.selectedIndex = -1;
      loadStaticResult(active.value);
    }
  }

  function bindEvents() {
    $("#static-tab").addEventListener("click", () => setDemoMode("static"));
    $("#live-tab").addEventListener("click", () => setDemoMode("live"));
    $("#static-sample-select-btxrd").addEventListener("change", () => handleStaticSampleChange("btxrd", "fracatlas"));
    $("#static-sample-select-fracatlas").addEventListener("change", () => handleStaticSampleChange("fracatlas", "btxrd"));
    $("#image-select").addEventListener("change", (event) => {
      state.selectedImageId = event.target.value;
      $("#live-preview").src = flattenImages().find((item) => item.id === state.selectedImageId)?.image_url || "";
    });
    $("#live-upload").addEventListener("change", uploadImage);
    $("#run-live-segmentation").addEventListener("click", runLiveSegmentation);
  }

  async function bootstrap() {
    bindEvents();
    try {
      const [capabilities, dashboard, samples] = await Promise.all([fetchJson("/capabilities"), fetchJson("/dashboard"), fetchJson("/static-samples")]);
      state.capabilities = capabilities;
      state.dashboard = dashboard;
      state.staticSamples = samples.samples || [];
      renderDashboard();
      renderStaticOptions();
      if (capabilities.features.live_demo) $("#live-tab").classList.remove("hidden");
    } catch (error) {
      setError("#static-error", friendlyError(error));
    }
  }

  bootstrap();
})();
