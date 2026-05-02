// Dashboard logic — pure ES6, no bundler
const API_BASE = "/api/v1";
const TIME_LABELS = ["Day 1", "Week 3", "Week 6"];

const chartInstances = {};

// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll(".tabs").forEach((tabs) => {
  tabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    const tabId = btn.dataset.tab;
    tabs.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    // Hide all sibling panels
    const section = tabs.closest(".form-section") || tabs.parentElement;
    section.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    document.getElementById(tabId)?.classList.add("active");
  });
});

// ── Reset ──────────────────────────────────────────────────────────────────
document.getElementById("reset-btn")?.addEventListener("click", (e) => {
  e.preventDefault();
  document.getElementById("patient-form").reset();
  showEmptyState();
});

// ── Form submit ────────────────────────────────────────────────────────────
document.getElementById("patient-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  const payload = collectFormData();
  if (!payload) return;

  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/prediction/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    showError(err.message);
    showEmptyState();
  } finally {
    setLoading(false);
  }
});

// ── Form collection ────────────────────────────────────────────────────────
function num(id) {
  const v = parseFloat(document.getElementById(id)?.value);
  return isNaN(v) ? 0 : v;
}

function collectFormData() {
  const age = parseInt(document.getElementById("age").value);
  if (!age || age < 0 || age > 120) {
    showError("Please enter a valid age (0–120).");
    return null;
  }
  const gender = document.querySelector('input[name="gender"]:checked')?.value || "male";

  return {
    patient_name: document.getElementById("patient_name").value.trim(),
    phone_no: document.getElementById("phone_no").value.trim(),
    age,
    gender,
    fracture_location: document.getElementById("fracture_location").value,
    biomarkers_day1:  { bsap: num("bsap_d1"),  alp: num("alp_d1"),  p1np: num("p1np_d1") },
    biomarkers_week3: { bsap: num("bsap_w3"),  alp: num("alp_w3"),  p1np: num("p1np_w3") },
    biomarkers_week6: { bsap: num("bsap_w6"),  alp: num("alp_w6"),  p1np: num("p1np_w6") },
    minerals_day1:    { calcium: num("ca_d1"),   phosphorus: num("phos_d1") },
    minerals_week3:   { calcium: num("ca_w3"),   phosphorus: num("phos_w3") },
    minerals_week6:   { calcium: num("ca_w6"),   phosphorus: num("phos_w6") },
    callus_d1: num("callus_d1"),
    callus_w3: num("callus_w3"),
    callus_w6: num("callus_w6"),
  };
}

// ── Render results ─────────────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById("empty-state").style.display = "none";
  document.getElementById("result-cards").style.display = "flex";
  document.getElementById("result-cards").style.flexDirection = "column";
  document.getElementById("result-cards").style.gap = "20px";

  renderGauge(data.healing_probability, data.healing_category);
  renderModelScores(data.confidence_scores, data.model_used);
  renderTrendCharts(data.biomarker_trends);
  renderSimilarCases(data.similar_cases);
  renderExplanation(data.clinical_explanation);
  renderFlagsAndRecs(data.risk_flags, data.recommendations);
}

// ── Gauge ──────────────────────────────────────────────────────────────────
function renderGauge(probability, category) {
  const pct = Math.round(probability * 100);
  const circumference = 2 * Math.PI * 64; // r=64
  const fill = (pct / 100) * circumference;

  const colors = { Good: "#22c55e", Moderate: "#f59e0b", Poor: "#ef4444" };
  const color = colors[category] || "#64748b";

  document.getElementById("gauge-pct").textContent = `${pct}%`;
  const arc = document.getElementById("gauge-arc");
  arc.setAttribute("stroke-dasharray", `${fill} ${circumference}`);
  arc.setAttribute("stroke", color);

  document.getElementById("probability-text").textContent =
    `${pct}% probability of successful healing`;

  const badge = document.getElementById("category-badge");
  badge.textContent = category;
  badge.className = `category-badge badge-${category.toLowerCase()}`;
}

// ── Model scores ───────────────────────────────────────────────────────────
function renderModelScores(scores, bestModel) {
  const container = document.getElementById("model-scores");
  container.innerHTML = "";

  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([name, prob]) => {
    const pct = Math.round(prob * 100);
    const isBest = name === bestModel;
    container.insertAdjacentHTML("beforeend", `
      <div class="model-score-row">
        <span class="model-name" style="${isBest ? "color:var(--primary);font-weight:700" : ""}">${name}${isBest ? " ★" : ""}</span>
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:${pct}%;background:${isBest ? "var(--primary)" : "var(--border)"}"></div>
        </div>
        <span class="score-val">${pct}%</span>
      </div>`);
  });
}

// ── Charts ─────────────────────────────────────────────────────────────────
function renderTrendCharts(trends) {
  renderLineChart("chart-bsap",   trends.bsap_trend,   "BSAP (U/L)",    "#1e40af");
  renderLineChart("chart-alp",    trends.alp_trend,    "ALP (U/L)",     "#0ea5e9");
  renderLineChart("chart-p1np",   trends.p1np_trend,   "P1NP (ng/mL)",  "#7c3aed");
  renderLineChart("chart-callus", trends.callus_trend, "Callus (mm²)",  "#22c55e");
  document.getElementById("trend-summary").textContent = trends.trend_summary;
}

function renderLineChart(canvasId, data, label, color) {
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }
  const ctx = document.getElementById(canvasId).getContext("2d");
  chartInstances[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels: TIME_LABELS,
      datasets: [{
        label,
        data,
        borderColor: color,
        backgroundColor: color + "22",
        pointBackgroundColor: color,
        pointRadius: 5,
        tension: 0.35,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true, labels: { font: { size: 11 }, color: "#0f172a" } },
      },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { ticks: { font: { size: 10 } }, beginAtZero: false },
      },
    },
  });
}

// ── Similar cases ──────────────────────────────────────────────────────────
function renderSimilarCases(cases) {
  const tbody = document.getElementById("cases-tbody");
  tbody.innerHTML = "";
  if (!cases || cases.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:16px">No similar cases found</td></tr>';
    return;
  }
  const colors = { Good: "#dcfce7", Moderate: "#fef3c7", Poor: "#fee2e2" };
  const textColors = { Good: "#166534", Moderate: "#92400e", Poor: "#991b1b" };

  cases.forEach((c) => {
    const sim = Math.round(c.similarity_score * 100);
    const bg = colors[c.outcome] || "#f1f5f9";
    const tc = textColors[c.outcome] || "#334155";
    tbody.insertAdjacentHTML("beforeend", `
      <tr>
        <td>${c.patient_name || "—"}</td>
        <td>${c.phone_no || "—"}</td>
        <td>${c.age}</td>
        <td style="text-transform:capitalize">${c.gender}</td>
        <td style="text-transform:capitalize">${c.fracture_location}</td>
        <td>${c.callus_w6.toFixed(0)} mm²</td>
        <td><span class="outcome-pill" style="background:${bg};color:${tc}">${c.outcome}</span></td>
        <td class="sim-score">${sim}%</td>
      </tr>`);
  });
}

// ── Explanation ────────────────────────────────────────────────────────────
function renderExplanation(text) {
  document.getElementById("explanation-text").textContent = text || "No explanation available.";
}

// ── Flags + Recs ───────────────────────────────────────────────────────────
function renderFlagsAndRecs(flags, recs) {
  const flagsEl = document.getElementById("flags-list");
  const recsEl = document.getElementById("recs-list");

  flagsEl.innerHTML = flags && flags.length
    ? flags.map((f) => `<div class="flag-item">⚠️ ${f}</div>`).join("")
    : '<div class="flag-item" style="background:#f0fdf4;border-color:var(--good)">✅ No risk flags identified</div>';

  recsEl.innerHTML = recs && recs.length
    ? recs.map((r) => `<div class="rec-item">💡 ${r}</div>`).join("")
    : '<div class="rec-item">Standard follow-up recommended.</div>';
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function setLoading(on) {
  const btn = document.getElementById("submit-btn");
  const txt = document.getElementById("btn-text");
  const spinner = document.getElementById("btn-spinner");
  btn.disabled = on;
  txt.textContent = on ? "Predicting…" : "Predict Healing";
  spinner.style.display = on ? "inline-block" : "none";
}

function showEmptyState() {
  document.getElementById("empty-state").style.display = "block";
  document.getElementById("result-cards").style.display = "none";
}

function showError(msg) {
  const el = document.getElementById("form-error");
  el.textContent = `Error: ${msg}`;
  el.style.display = "block";
}

function clearError() {
  document.getElementById("form-error").style.display = "none";
}
