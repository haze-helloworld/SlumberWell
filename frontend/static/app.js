// ============================================================
//  SlumberWell – app.js
// ============================================================
const API = (window.location.protocol.startsWith("http")) ? "" : "http://127.0.0.1:8000";

const STAGE_COLOURS = {
  W:   "#64748b",
  N1:  "#818cf8",
  N2:  "#38bdf8",
  N3:  "#2563eb",
  REM: "#f472b6",
};
const STAGE_FULL = {
  W:   "Wake",
  N1:  "NREM Stage 1",
  N2:  "NREM Stage 2",
  N3:  "NREM Stage 3 (Deep)",
  REM: "REM Sleep",
};
const FEATURES = [
  "mean","std","var","rms","min","max","range",
  "delta_power","theta_power","alpha_power","beta_power",
  "rel_delta","rel_theta","rel_alpha","rel_beta","spectral_entropy"
];

// ── Charts ──────────────────────────────────────────────────────
let hypnoChart    = null;
let donutChart    = null;
let compareChart  = null;

// ── Navigation ──────────────────────────────────────────────────
function navigate(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("page-" + page).classList.add("active");
  document.querySelector(`[data-page="${page}"]`).classList.add("active");
  if (page === "metrics") loadMetrics();
  if (page === "about")   loadAbout();
}

// ── API health check ─────────────────────────────────────────────
let activeModelName = "Support Vector Machine (SVM)";

async function checkHealth() {
  const dot  = document.querySelector(".status-dot");
  const text = document.querySelector(".status-text");
  try {
    const r = await fetch(`${API}/api/health`);
    if (r.ok) {
      const d = await r.json();
      dot.classList.add("online");
      if (d.model) {
        activeModelName = d.model === "SVM" ? "Support Vector Machine (SVM)" : d.model;
      }
      text.textContent = d.model || "Online";
    } else throw new Error();
  } catch {
    dot.classList.add("offline");
    text.textContent = "Offline";
  }
}

// ── Demo dashboard ───────────────────────────────────────────────
async function loadDemo() {
  try {
    const r = await fetch(`${API}/api/demo`);
    if (!r.ok) throw new Error("API not available");
    const d = await r.json();
    renderDashboard(d);
  } catch {
    // Use local synthetic data if backend is down
    renderDashboard(syntheticDemo());
  }
}

function syntheticDemo() {
  // Real SC4001 stats used as fallback when API is offline
  const STAGE_NAMES = ["W","N1","N2","N3","REM"];
  const counts = {W:1918, N1:87, N2:281, N3:215, REM:149};
  // Build a compact representative 240-epoch slice for the fallback view
  const cyc = Array(8).fill("W").concat(Array(4).fill("N1"), Array(16).fill("N2"),
    Array(12).fill("N3"), Array(8).fill("N2"), Array(12).fill("REM"));
  let night = Array(5).fill("W");
  for (let i=0;i<5;i++) night = night.concat(cyc).concat(Array(35).fill("W"));
  night = night.slice(0,240);
  const n = 2650; // real epoch count
  return {
    subject: "SC4001",
    description: "Whole-night recording SC4001E0-PSG (Sleep Cassette dataset). Wake periods NOT trimmed.",
    hypnogram: night,         // shortened for offline view
    hypno_encoded: night.map(s=>STAGE_NAMES.indexOf(s)),
    summary: {
      subject: "SC4001",
      n_epochs: 2650,
      total_recording_min: 1325.0,
      total_sleep_min: 366.0,
      sleep_efficiency_pct: 27.62,
      sleep_onset_latency_min: 2.5,
      rem_latency_min: 0.0,
      stage_counts: counts,
      stage_percent: {W:72.38, N1:3.28, N2:10.60, N3:8.11, REM:5.62},
    },
    stage_colours: STAGE_COLOURS,
  };
}

function renderDashboard(d) {
  const s = d.summary;
  // Update subtitle with real subject info
  const sub = d.subject || "";
  const desc = d.description || "";
  const modelLabel = activeModelName === "SVM" ? "Support Vector Machine (SVM)" : activeModelName;
  document.querySelector("#page-dashboard .page-subtitle").textContent =
    sub ? `Subject ${sub} · ${(s.n_epochs||"?").toLocaleString()} epochs · ${modelLabel}` :
          `Demo night analysis · ${modelLabel}`;

  // Stats cards
  el("stat-tst").textContent   = formatMins(s.total_sleep_min);
  el("stat-eff").textContent   = s.sleep_efficiency_pct.toFixed(1) + "%";
  el("stat-onset").textContent = s.sleep_onset_latency_min != null ? formatMins(s.sleep_onset_latency_min) : "—";
  el("stat-rem").textContent   = (s.rem_latency_min != null && s.rem_latency_min > 0) ? formatMins(s.rem_latency_min) : "—";
  if (el("stat-rec")) el("stat-rec").textContent = formatMins(s.total_recording_min);
  document.querySelectorAll(".stat-card").forEach(c => c.classList.remove("shimmer"));

  // Recording info pill
  const infoEl = el("recordingInfo");
  if (infoEl) {
    infoEl.innerHTML = d.description
      ? `<div class="rec-info-pill"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>${d.description}</div>`
      : "";
  }

  // Legend
  const leg = el("hypnoLegend");
  leg.innerHTML = Object.entries(STAGE_COLOURS).map(([k,v]) =>
    `<div class="legend-item"><div class="legend-dot" style="background:${v}"></div>${k}</div>`
  ).join("");

  drawHypno(d.hypnogram, d.hypno_encoded);
  drawDonut(s.stage_percent);
  drawStageTable(s.stage_counts, s.total_sleep_min);
}

function drawHypno(hypno, encoded) {
  const STAGE_ORDER = { W:0, N1:1, N2:2, N3:3, REM:4 };
  const totalMins = hypno.length * 0.5;
  const showHours = totalMins > 120;
  // Label every 60 min (2 epochs/min = 120 epochs) or every 30 min for short recordings
  const labelStep = showHours ? 120 : 60;
  const labels = hypno.map((_,i) => {
    if (i % labelStep === 0) {
      const mins = i * 0.5;
      return showHours ? `${(mins/60).toFixed(0)}h` : `${mins}m`;
    }
    return "";
  });
  const data = hypno.map(s => 4 - STAGE_ORDER[s]);

  if (hypnoChart) hypnoChart.destroy();
  const ctx = document.getElementById("hypnoChart").getContext("2d");
  hypnoChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data,
        fill: true,
        stepped: true,
        borderWidth: 1.5,
        borderColor: "#818cf8",
        backgroundColor: (ctx2) => {
          const g = ctx2.chart.ctx.createLinearGradient(0,0,0,200);
          g.addColorStop(0, "rgba(129,140,248,0.25)");
          g.addColorStop(1, "rgba(56,189,248,0.04)");
          return g;
        },
        tension: 0,
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: hypno.length > 500 ? 0 : 400 },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", maxRotation: 0, autoSkip: false } },
        y: {
          grid: { color: "rgba(255,255,255,0.04)" },
          min: 0, max: 4,
          ticks: { color: "#94a3b8", callback: v => ["REM","N3","N2","N1","W"][v] || "" }
        }
      },
      plugins: { legend: { display: false }, tooltip: {
        callbacks: {
          title: ctx2 => { const i = ctx2[0].dataIndex; return `Epoch ${i} (${(i*0.5).toFixed(0)} min)`; },
          label: ctx2 => { const stages = ["REM","N3","N2","N1","W"]; return " Stage: " + stages[ctx2.raw]; }
        }
      }}
    }
  });
}

function drawDonut(pct) {
  const labels = Object.keys(pct);
  const vals   = Object.values(pct);
  const colors = labels.map(k => STAGE_COLOURS[k]);
  if (donutChart) donutChart.destroy();
  const ctx = document.getElementById("donutChart").getContext("2d");
  donutChart = new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{ data: vals, backgroundColor: colors, borderWidth: 2, borderColor: "#060b18", hoverOffset: 6 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { color: "#94a3b8", font: { size: 12 }, boxWidth: 12, padding: 12 } },
        tooltip: { callbacks: { label: c => ` ${c.label}: ${c.raw.toFixed(1)}%` } }
      },
      cutout: "62%",
    }
  });
}

function drawStageTable(counts, totalSleepMin) {
  const maxCount = Math.max(...Object.values(counts));
  const html = Object.entries(counts).map(([stage, cnt]) => {
    const pct = maxCount > 0 ? (cnt / maxCount * 100).toFixed(0) : 0;
    const mins = (cnt * 0.5).toFixed(0);
    return `
      <div class="stage-row">
        <div class="stage-dot-lg" style="background:${STAGE_COLOURS[stage]}"></div>
        <div class="stage-name-lg" style="color:${STAGE_COLOURS[stage]}">${stage}</div>
        <div class="stage-bar-track">
          <div class="stage-bar-fill" style="width:${pct}%;background:${STAGE_COLOURS[stage]}"></div>
        </div>
        <div class="stage-mins">${mins} min</div>
      </div>`;
  }).join("");
  el("stageTable").innerHTML = html;
}

// ── Predict page ─────────────────────────────────────────────────
function buildFeatureInputs() {
  const grid = el("featureInputs");
  const defaults = {
    mean: "0.12", std: "25.4", var: "645.0", rms: "25.4",
    min: "-80.2", max: "78.5", range: "158.7",
    delta_power: "142.5", theta_power: "38.2", alpha_power: "12.1", beta_power: "8.5",
    rel_delta: "0.62", rel_theta: "0.16", rel_alpha: "0.05", rel_beta: "0.04",
    spectral_entropy: "0.73"
  };
  grid.innerHTML = FEATURES.map(f => `
    <div class="feature-field">
      <label class="feature-label" for="f_${f}">${f.replace(/_/g," ")}</label>
      <input class="feature-input" id="f_${f}" type="number" step="any" value="${defaults[f] || '0'}" />
    </div>`).join("");
}

const STAGE_PRESETS = {
  W:   [-0.0374, 21.3682, 456.5979, 21.3682, -132.776, 118.0559, 250.8319, 181.8718, 46.3638, 47.0383, 105.4418, 0.4777, 0.1218, 0.1236, 0.277, 0.8587],
  N1:  [-0.0445, 9.6395, 92.9203, 9.6396, -41.5, 55.0296, 96.5296, 65.3177, 10.3346, 4.307, 8.0301, 0.7423, 0.1175, 0.0489, 0.0913, 0.7054],
  N2:  [-0.0087, 12.0127, 144.3053, 12.0127, -33.371, 61.7866, 95.1576, 106.9843, 14.1008, 7.6287, 5.5463, 0.7968, 0.105, 0.0568, 0.0413, 0.6935],
  N3:  [0.1447, 26.0474, 678.4665, 26.0478, -68.1168, 76.1163, 144.2332, 547.5702, 16.1157, 20.3799, 4.9425, 0.9296, 0.0274, 0.0346, 0.0084, 0.5083],
  REM: [-0.127, 7.8033, 60.8912, 7.8043, -24.3677, 34.2868, 58.6544, 31.3903, 14.0555, 3.297, 5.1537, 0.5824, 0.2608, 0.0612, 0.0956, 0.7859]
};

function loadPreset(stage) {
  const p = STAGE_PRESETS[stage];
  if (!p) return;
  FEATURES.forEach((f, i) => {
    const inp = el("f_" + f);
    if (inp) inp.value = p[i];
  });
  runPredict();
}

function fillRandom() {
  const keys = Object.keys(STAGE_PRESETS);
  const randomKey = keys[Math.floor(Math.random() * keys.length)];
  loadPreset(randomKey);
}

async function runPredict() {
  const features = FEATURES.map(f => parseFloat(el("f_"+f).value));
  if (features.some(isNaN)) { showToast("Please fill all feature fields"); return; }
  const res = el("predictResult");
  res.innerHTML = `<div class="predict-placeholder"><div class="placeholder-icon-wrap spin"><i data-lucide="loader-circle"></i></div><p>Predicting…</p></div>`;
  lucide.createIcons();
  try {
    const r = await fetch(`${API}/api/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    renderPredictResult(d);
  } catch(e) {
    res.innerHTML = `<div class="predict-placeholder"><div class="placeholder-icon-wrap error"><i data-lucide="circle-x"></i></div><p>${e.message}</p></div>`;
    lucide.createIcons();
  }
}

function renderPredictResult(d) {
  const col = STAGE_COLOURS[d.stage];
  const stageFull = STAGE_FULL[d.stage];
  const probHtml = Object.entries(d.probabilities).map(([s,p]) => `
    <div class="proba-row">
      <div class="proba-label">${s}</div>
      <div class="proba-track"><div class="proba-fill" style="width:${(p*100).toFixed(1)}%;background:${STAGE_COLOURS[s]}"></div></div>
      <div class="proba-val">${(p*100).toFixed(1)}%</div>
    </div>`).join("");
  el("predictResult").innerHTML = `
    <div class="result-stage-card" style="background:${col}18;border:1px solid ${col}44">
      <div class="result-stage-name" style="color:${col}">${d.stage}</div>
      <div class="result-stage-full" style="color:${col}">${stageFull}</div>
    </div>
    <div class="inference-badge">
      <i data-lucide="zap" style="width:12px;height:12px"></i>
      Inference: ${d.inference_ms.toFixed(2)} ms
    </div>
    <div class="proba-bars">${probHtml}</div>`;
  lucide.createIcons();
}

function loadDemoFeatures() {
  // N2-like demo features
  el("batchInput").value = JSON.stringify([
    [0.05,28.3,800,28.3,-90,88,178,180,45,10,7,0.65,0.16,0.04,0.03,0.68],
    [0.08,55.2,3047,55.2,-160,155,315,680,32,6,4,0.82,0.04,0.01,0.005,0.52],
    [0.15,22.1,489,22.1,-65,68,133,60,55,22,18,0.38,0.35,0.14,0.11,0.88],
    [0.2,18.5,342,18.5,-55,52,107,45,22,18,12,0.35,0.17,0.14,0.09,0.81],
  ], null, 2);
}

async function runBatch() {
  let epochs;
  try { epochs = JSON.parse(el("batchInput").value); }
  catch { showToast("Invalid JSON"); return; }
  const res = el("batchResult");
  res.innerHTML = "<p class='text-muted'>Running…</p>";
  try {
    const r = await fetch(`${API}/api/predict/batch`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs })
    });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    renderBatchResult(d);
  } catch(e) {
    res.innerHTML = `<p style='color:#ef4444'>${e.message}</p>`;
  }
}

function renderBatchResult(d) {
  const s = d.summary;
  const statHtml = [
    ["Sleep Efficiency", s.sleep_efficiency_pct.toFixed(1)+"%"],
    ["Total Sleep", formatMins(s.total_sleep_min)],
    ["Sleep Onset", s.sleep_onset_latency_min != null ? formatMins(s.sleep_onset_latency_min) : "—"],
    ["REM Latency", s.rem_latency_min != null ? formatMins(s.rem_latency_min) : "—"],
    ["Inference", d.inference_ms_total.toFixed(1)+" ms"],
  ].map(([l,v]) => `<div class="batch-stat"><div class="batch-stat-label">${l}</div><div class="batch-stat-val">${v}</div></div>`).join("");

  const hypnoHtml = d.hypnogram.map(s2 =>
    `<span style="display:inline-block;padding:0.2rem 0.5rem;border-radius:4px;background:${STAGE_COLOURS[s2]}22;color:${STAGE_COLOURS[s2]};font-size:0.78rem;font-family:monospace;margin:0.15rem">${s2}</span>`
  ).join("");

  el("batchResult").innerHTML = `
    <div class="batch-summary">${statHtml}</div>
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.4rem">Predicted Hypnogram:</div>
    <div>${hypnoHtml}</div>`;
}

// ── Metrics page ─────────────────────────────────────────────────
let metricsLoaded = false;

function syntheticMetrics() {
  return {
    m: {
      model_name: "SVM",
      class_names: ["W", "N1", "N2", "N3", "REM"],
      validation: { macro_f1_of_selection_run: 0.6400858684906844 },
      test: {
        accuracy: 0.6892748181178128,
        macro_f1: 0.6569923506047701,
        n1_recall: 0.5002410800385728,
        per_class: {
          W:   { precision: 0.8981, recall: 0.7573, f1: 0.8217 },
          N1:  { precision: 0.3897, recall: 0.5002, f1: 0.4381 },
          N2:  { precision: 0.8031, recall: 0.6440, f1: 0.7148 },
          N3:  { precision: 0.4819, recall: 0.8945, f1: 0.6263 },
          REM: { precision: 0.6072, recall: 0.7830, f1: 0.6840 }
        },
        confusion_matrix: [
          [6831, 1205,  369,  272,  343],
          [ 355, 2075, 1000,   50,  668],
          [ 363, 1578, 7310, 1279,  821],
          [  33,   11,  126, 1500,    7],
          [  24,  455,  297,   12, 2843]
        ]
      }
    },
    models: [
      { model: "SVM", macro_f1: 0.6400858684906844 },
      { model: "XGBoost", macro_f1: 0.6392867098604869 },
      { model: "HistGradientBoosting", macro_f1: 0.6365228603546311 },
      { model: "Random Forest", macro_f1: 0.6198013223700853 },
      { model: "Extra Trees", macro_f1: 0.5831054018870775 },
      { model: "Majority baseline", macro_f1: 0.0868535154589509 }
    ]
  };
}

async function loadMetrics() {
  if (metricsLoaded) return;
  try {
    const [mR, cR] = await Promise.all([
      fetch(`${API}/api/metrics`),
      fetch(`${API}/api/metrics/comparison`)
    ]);
    if (!mR.ok || !cR.ok) throw new Error("Metrics API returned HTTP error");
    const m = await mR.json();
    const c = await cR.json();
    renderMetrics(m, c.models || []);
    metricsLoaded = true;
  } catch(e) {
    console.warn("Could not load metrics from API, rendering offline benchmark metrics:", e);
    const syn = syntheticMetrics();
    renderMetrics(syn.m, syn.models);
  }
}

function renderMetrics(m, models) {
  const t = m.test;
  const modLabel = m.model_name === "SVM" ? "Support Vector Machine (SVM)" : (m.model_name || activeModelName);
  const metricsSubtitle = document.querySelector("#page-metrics .page-subtitle");
  if (metricsSubtitle) {
    metricsSubtitle.textContent = `Evaluation results for the selected ${modLabel} model`;
  }
  el("m-val-f1").textContent  = (m.validation?.macro_f1_of_selection_run*100 || 0).toFixed(1)+"%";
  el("m-test-acc").textContent = (t.accuracy*100).toFixed(1)+"%";
  el("m-test-f1").textContent  = (t.macro_f1*100).toFixed(1)+"%";
  el("m-n1-recall").textContent = (t.n1_recall*100).toFixed(1)+"%";

  // Confusion matrix
  const cm = t.confusion_matrix;
  const cls = m.class_names || ["W","N1","N2","N3","REM"];
  let tableHtml = `<table class="conf-matrix"><thead><tr><th>Pred →<br>True ↓</th>`;
  cls.forEach(c => tableHtml += `<th style="color:${STAGE_COLOURS[c]}">${c}</th>`);
  tableHtml += `</tr></thead><tbody>`;
  const maxVal = Math.max(...cm.flat());
  cm.forEach((row, ri) => {
    tableHtml += `<tr><th style="color:${STAGE_COLOURS[cls[ri]]}">${cls[ri]}</th>`;
    row.forEach((v, ci) => {
      const intensity = maxVal > 0 ? v/maxVal : 0;
      const bg = ri===ci
        ? `rgba(129,140,248,${0.15+intensity*0.55})`
        : `rgba(239,68,68,${intensity*0.35})`;
      tableHtml += `<td class="conf-cell" style="background:${bg}">${v}</td>`;
    });
    tableHtml += `</tr>`;
  });
  tableHtml += `</tbody></table>`;
  el("confMatContainer").innerHTML = tableHtml;

  // Per-class table
  const pc = t.per_class;
  let pcHtml = `<table class="data-table"><thead><tr><th>Stage</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>`;
  Object.entries(pc).forEach(([s, v]) => {
    pcHtml += `<tr>
      <td><span style="color:${STAGE_COLOURS[s]};font-weight:600">${s}</span></td>
      <td>${(v.precision*100).toFixed(1)}%</td>
      <td>${(v.recall*100).toFixed(1)}%</td>
      <td>${(v.f1*100).toFixed(1)}%</td>
    </tr>`;
  });
  pcHtml += `</tbody></table>`;
  el("perClassTable").innerHTML = pcHtml;

  // Model comparison chart
  if (compareChart) compareChart.destroy();
  const ctx = document.getElementById("comparisonChart").getContext("2d");
  const sortedModels = [...models].sort((a,b) => b.macro_f1 - a.macro_f1);
  compareChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sortedModels.map(m2 => m2.model),
      datasets: [{
        label: "Macro F1",
        data: sortedModels.map(m2 => +(m2.macro_f1*100).toFixed(2)),
        backgroundColor: sortedModels.map((_, i) =>
          i===0 ? "rgba(129,140,248,0.7)" : "rgba(56,189,248,0.4)"),
        borderColor: sortedModels.map((_, i) =>
          i===0 ? "#818cf8" : "#38bdf8"),
        borderWidth: 1,
        borderRadius: 5,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" }, max: 80 },
        y: { grid: { display: false }, ticks: { color: "#94a3b8" } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx2 => ` Macro F1: ${ctx2.raw.toFixed(2)}%` } }
      }
    }
  });
}

// ── About page ───────────────────────────────────────────────────
let aboutLoaded = false;
async function loadAbout() {
  if (aboutLoaded) return;
  aboutLoaded = true;

  // Stage pills
  el("stagePills").innerHTML = Object.entries(STAGE_COLOURS).map(([k,v]) =>
    `<span class="stage-pill" style="color:${v};border-color:${v};background:${v}18">${k} – ${STAGE_FULL[k]}</span>`
  ).join("");

  // Pipeline
  const steps = [
    ["1","Read EDF","Load single-channel EEG (Fpz-Cz) with MNE"],
    ["2","Resample","Resample to 100 Hz using polyphase filter"],
    ["3","Bandpass Filter","4th-order Butterworth 0.3–35 Hz"],
    ["4","Epoch","Cut into 30-second non-overlapping epochs"],
    ["5","Feature Extraction","Compute 16 statistical + spectral features"],
    ["6","Temporal Context","Append ±1 and ±2 epoch context (40 features total)"],
    ["7","Classify","Support Vector Machine (RBF kernel, C=10, γ=0.01)"],
  ];
  el("pipelineSteps").innerHTML = steps.map(([n,t,d]) => `
    <div class="pipeline-step">
      <div class="step-num">${n}</div>
      <div><div class="step-title">${t}</div><div class="step-desc">${d}</div></div>
    </div>`).join("");

  // Feature chips
  el("featureChips").innerHTML = FEATURES.map(f =>
    `<span class="feat-chip">${f}</span>`).join("");

  // Best model
  try {
    const r = await fetch(`${API}/api/info`);
    const d = await r.json();
    const hp = d.hyperparameters || {};
    const isSVM = (d.model_name || "").toUpperCase().includes("SVM");
    const kvs = isSVM ? [
      ["Model", "Support Vector Machine (SVM)"],
      ["Kernel", "RBF"],
      ["C (Regularization)", hp.svc__C ?? hp.C ?? "10"],
      ["Gamma (γ)", hp.svc__gamma ?? hp.gamma ?? "0.01"],
      ["Class Weight", "balanced"],
      ["Features", `${(d.feature_names || []).length || 40} (16 base + temporal context)`],
      ["Val Macro F1", "64.0%"],
      ["Test Macro F1", "65.7%"],
      ["Training Epochs", (d.split?.n_epochs?.train ? d.split.n_epochs.train.toLocaleString() : "133,512")],
    ] : [
      ["Model", d.model_name || "Classical ML"],
      ["Val Macro F1", "64.0%"],
      ["Features", `${(d.feature_names || []).length || 40}`],
      ["Training Epochs", (d.split?.n_epochs?.train ? d.split.n_epochs.train.toLocaleString() : "—")],
    ];
    el("bestModelInfo").innerHTML = `<div class="kv-grid">${kvs.map(([k,v]) =>
      `<div class="kv-item"><div class="kv-key">${k}</div><div class="kv-val">${v}</div></div>`
    ).join("")}</div>`;
  } catch {
    el("bestModelInfo").innerHTML = `<p class="text-muted">API offline – start the backend to view model info.</p>`;
  }
}

// ── Utilities ────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }

function formatMins(mins) {
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m} min`;
}

function showToast(msg) {
  const t = document.createElement("div");
  t.style.cssText = "position:fixed;bottom:2rem;right:2rem;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0.75rem 1.2rem;color:#f1f5f9;font-size:0.88rem;z-index:9999;box-shadow:0 8px 24px rgba(0,0,0,0.5)";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildFeatureInputs();
  checkHealth();
  loadDemo();
});
