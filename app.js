/**
 * app.js — Optimizer Playground main application logic
 */

'use strict';

// ── Optimizer registry ────────────────────────────────────────────────────────
const OPTIMIZERS = [
  { id: 'sgd',      label: 'SGD',      color: '#378ADD', desc: 'vanilla gradient descent' },
  { id: 'momentum', label: 'Momentum', color: '#1D9E75', desc: 'SGD + velocity' },
  { id: 'nesterov', label: 'Nesterov', color: '#D4537E', desc: 'lookahead momentum' },
  { id: 'adagrad',  label: 'AdaGrad',  color: '#BA7517', desc: 'adaptive per-param lr' },
  { id: 'rmsprop',  label: 'RMSProp',  color: '#7F77DD', desc: 'leaky AdaGrad' },
  { id: 'adam',     label: 'Adam',     color: '#D85A30', desc: 'adaptive + momentum' },
  { id: 'adamw',    label: 'AdamW',    color: '#f0a832', desc: 'Adam + weight decay' },
];

// Per-optimizer extra parameters beyond shared learning rate
const OPT_PARAMS = {
  sgd:      [],
  momentum: [{ id: 'beta',  label: 'momentum β',  min: 0.5, max: 0.99, step: 0.01, def: 0.9 }],
  nesterov: [{ id: 'beta',  label: 'momentum β',  min: 0.5, max: 0.99, step: 0.01, def: 0.9 }],
  adagrad:  [{ id: 'eps',   label: 'ε (×10⁻⁸)',   min: 1,   max: 100,  step: 1,    def: 1   }],
  rmsprop:  [
    { id: 'rho',  label: 'decay ρ',     min: 0.5, max: 0.99,   step: 0.01,  def: 0.9   },
    { id: 'eps',  label: 'ε (×10⁻⁸)',  min: 1,   max: 100,    step: 1,     def: 1     },
  ],
  adam: [
    { id: 'beta1', label: 'β₁ (mean)',       min: 0.5,  max: 0.99,   step: 0.01,  def: 0.9   },
    { id: 'beta2', label: 'β₂ (variance)',   min: 0.9,  max: 0.9999, step: 0.001, def: 0.999 },
    { id: 'eps',   label: 'ε (×10⁻⁸)',      min: 1,    max: 100,    step: 1,     def: 1     },
  ],
  adamw: [
    { id: 'beta1', label: 'β₁ (mean)',       min: 0.5,  max: 0.99,   step: 0.01,  def: 0.9   },
    { id: 'beta2', label: 'β₂ (variance)',   min: 0.9,  max: 0.9999, step: 0.001, def: 0.999 },
    { id: 'eps',   label: 'ε (×10⁻⁸)',      min: 1,    max: 100,    step: 1,     def: 1     },
    { id: 'wd',    label: 'weight decay λ',  min: 0,    max: 0.1,    step: 0.001, def: 0.01  },
  ],
};

// ── Application state ─────────────────────────────────────────────────────────
let currentFn = null;
let currentGrad = null;
let currentGmin = null;          // cached global min for current function
const activeOpts = new Set(['adam']);

const SLIDER_HALF = 3;           // ±3 units around global min for start-point sliders

// Update x0/x1 slider ranges to ±SLIDER_HALF around global min.
// Falls back to ±10 centred on 0 when no bounded min exists.
function updateStartSliders(gmin) {
  const cx = gmin ? gmin.x : 0;
  const cy = gmin ? gmin.y : 0;
  const half = SLIDER_HALF;

  const x0El   = document.getElementById('x0-slider');
  const x1El   = document.getElementById('x1-slider');
  const x0Disp = document.getElementById('x0-display');
  const x1Disp = document.getElementById('x1-display');
  const x0Lb   = document.getElementById('x0-lb');
  const x1Lb   = document.getElementById('x1-lb');
  const x0Ub   = document.getElementById('x0-ub');
  const x1Ub   = document.getElementById('x1-ub');

  const x0Min = cx - half, x0Max = cx + half;
  const x1Min = cy - half, x1Max = cy + half;

  const fmt = v => {
    const abs = Math.abs(v);
    if (abs >= 100) return v.toFixed(1);
    if (abs >= 10)  return v.toFixed(2);
    return v.toFixed(3);
  };

  x0El.min = x0Min; x0El.max = x0Max; x0El.step = (2 * half) / 200;
  x1El.min = x1Min; x1El.max = x1Max; x1El.step = (2 * half) / 200;

  // Default start: offset 80% toward the edge so we start meaningfully away from min
  const x0Default = cx + half * 0.8;
  const x1Default = cy + half * 0.8;
  x0El.value = Math.min(Math.max(x0Default, x0Min), x0Max);
  x1El.value = Math.min(Math.max(x1Default, x1Min), x1Max);

  x0Disp.textContent = fmt(parseFloat(x0El.value));
  x1Disp.textContent = fmt(parseFloat(x1El.value));
  if (x0Lb) x0Lb.textContent = fmt(x0Min);
  if (x0Ub) x0Ub.textContent = fmt(x0Max);
  if (x1Lb) x1Lb.textContent = fmt(x1Min);
  if (x1Ub) x1Ub.textContent = fmt(x1Max);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function makeFn(expr) {
  return new Function('x0', 'x1', 'Math', `"use strict"; return (${expr});`);
}

function fmtVal(v, step) {
  if (step < 0.001) return v.toFixed(4);
  if (step < 0.01)  return v.toFixed(3);
  if (step < 0.1)   return v.toFixed(2);
  return v.toFixed(1);
}

// ── UI: Build optimizer toggles ───────────────────────────────────────────────
function buildToggles() {
  const container = document.getElementById('optimizer-toggles');
  container.innerHTML = '';
  OPTIMIZERS.forEach(opt => {
    const btn = document.createElement('button');
    btn.className = 'opt-toggle' + (activeOpts.has(opt.id) ? '' : ' off');
    btn.style.borderColor = opt.color;
    btn.style.color = opt.color;
    btn.innerHTML = `
      <span class="opt-toggle-name">${opt.label}</span>
      <span class="opt-toggle-desc">${opt.desc}</span>
    `;
    btn.title = `Toggle ${opt.label}`;
    btn.onclick = () => {
      if (activeOpts.has(opt.id)) {
        if (activeOpts.size === 1) return; // at least one must be active
        activeOpts.delete(opt.id);
        btn.classList.add('off');
      } else {
        activeOpts.add(opt.id);
        btn.classList.remove('off');
      }
      buildParamCards();
      buildPlot();
    };
    container.appendChild(btn);
  });
}

// ── UI: Build per-optimizer param cards ───────────────────────────────────────
function buildParamCards() {
  const container = document.getElementById('opt-params');
  const card = document.getElementById('opt-params-card');

  // Snapshot current slider values BEFORE clearing the DOM — querying by id
  // after innerHTML='' always returns null, losing user-adjusted values.
  const savedVals = {};
  OPTIMIZERS.forEach(opt => {
    OPT_PARAMS[opt.id].forEach(p => {
      const el = document.getElementById(`p-${opt.id}-${p.id}`);
      if (el) savedVals[`p-${opt.id}-${p.id}`] = el.value;
    });
  });

  container.innerHTML = '';
  let hasParams = false;

  OPTIMIZERS.forEach(opt => {
    if (!activeOpts.has(opt.id)) return;
    OPT_PARAMS[opt.id].forEach(p => {
      hasParams = true;
      const id = `p-${opt.id}-${p.id}`;
      const curVal = (savedVals[id] !== undefined) ? savedVals[id] : p.def;
      const div = document.createElement('div');
      div.className = 'param-card';
      div.innerHTML = `
        <div class="param-card-label" style="color:${opt.color}">${opt.label}</div>
        <div class="param-card-label">${p.label}</div>
        <div class="param-card-val" id="${id}-disp" style="color:${opt.color}">${fmtVal(parseFloat(curVal), p.step)}</div>
        <input type="range" id="${id}" class="slider"
          min="${p.min}" max="${p.max}" step="${p.step}" value="${curVal}">
      `;
      container.appendChild(div);
      div.querySelector('input').addEventListener('input', function () {
        document.getElementById(`${id}-disp`).textContent = fmtVal(parseFloat(this.value), p.step);
        buildPlot();
      });
    });
  });

  card.style.display = hasParams ? 'block' : 'none';
}

// ── UI: Update results table ──────────────────────────────────────────────────
function updateResultsTable(results) {
  const card = document.getElementById('results-card');
  const body = document.getElementById('results-body');
  const head = document.getElementById('results-head');

  if (results.length === 0) { card.style.display = 'none'; return; }
  card.style.display = 'block';

  head.innerHTML = `
    <th>Optimizer</th>
    <th>Final loss</th>
    <th>End x₀</th>
    <th>End x₁</th>
    <th>Steps</th>
  `;

  body.innerHTML = '';
  const sorted = [...results].sort((a, b) =>
    (Number.isFinite(a.finalLoss) ? a.finalLoss : 1e18) - (Number.isFinite(b.finalLoss) ? b.finalLoss : 1e18)
  );

  sorted.forEach((r, i) => {
    const tr = document.createElement('tr');
    const isBest = i === 0 && sorted.length > 1;
    tr.innerHTML = `
      <td>
        <div class="cell-name">
          <span class="opt-color-dot" style="background:${r.color}"></span>
          ${r.label}
          ${isBest ? '<span class="badge-best">best</span>' : ''}
        </div>
      </td>
      <td>${isFinite(r.finalLoss) ? r.finalLoss.toFixed(6) : '<span style="color:#f09898">diverged</span>'}</td>
      <td>${r.endX.toFixed(5)}</td>
      <td>${r.endY.toFixed(5)}</td>
      <td>${r.steps}</td>
    `;
    body.appendChild(tr);
  });
}

// ── UI: Update legend ─────────────────────────────────────────────────────────
function updateLegend(results, gmin) {
  const row = document.getElementById('legend-row');
  row.innerHTML = '';
  results.forEach(r => {
    row.innerHTML += `
      <span class="legend-item">
        <span class="legend-line" style="background:${r.color}"></span>
        ${r.label}
      </span>`;
  });
  row.innerHTML += `
    <span class="legend-item">
      <span class="legend-dot" style="background:#ffffff; border:2px solid #00C8FF"></span>
      start
    </span>`;
  if (gmin) {
    row.innerHTML += `
      <span class="legend-item">
        <span class="legend-dot" style="background:#F0A832; border:2px solid #00C8FF"></span>
        global min
      </span>`;
  }
}

// ── Main plot builder ─────────────────────────────────────────────────────────
function buildPlot() {
  if (!currentFn || !currentGrad) return;

  const lr      = parseInt(document.getElementById('lr-slider').value) / 1000;
  const numIter = parseInt(document.getElementById('iter-slider').value);
  const x0s     = parseFloat(document.getElementById('x0-slider').value);
  const x1s     = parseFloat(document.getElementById('x1-slider').value);

  // Run all active optimizers
  const allPaths = {};
  const results = [];
  OPTIMIZERS.forEach(opt => {
    if (!activeOpts.has(opt.id)) return;
    const path = dispatchOptimizer(opt.id, currentGrad, x0s, x1s, lr, numIter);
    allPaths[opt.id] = path;
    const xs = path.map(p => p[0]), ys = path.map(p => p[1]);
    const zs = path.map(p => { try { return currentFn(p[0], p[1], Math); } catch (e) { return NaN; } });
    results.push({
      label: opt.label, color: opt.color, steps: path.length - 1,
      endX: xs[xs.length - 1], endY: ys[ys.length - 1],
      finalLoss: zs[zs.length - 1],
    });
  });

  // Use cached global min (computed once in applyFn, not on every slider drag)
  const gmin = currentGmin;
  const banner = document.getElementById('global-min-banner');
  if (gmin) {
    banner.className = 'global-min-banner found';
    banner.style.display = 'block';
    banner.textContent = `global min ≈ ${gmin.z.toFixed(5)}   at   (x₀ = ${gmin.x.toFixed(4)}, x₁ = ${gmin.y.toFixed(4)})`;
  } else {
    banner.className = 'global-min-banner notfound';
    banner.style.display = 'block';
    banner.textContent = 'no bounded global minimum found — function appears unbounded in this region';
  }

  // Bounding box for surface
  let allX = [x0s], allY = [x1s];
  Object.values(allPaths).forEach(p => p.forEach(pt => { allX.push(pt[0]); allY.push(pt[1]); }));
  if (gmin) { allX.push(gmin.x); allY.push(gmin.y); }
  const pad = 2.0;
  const rawXMin = allX.reduce((a, b) => Math.min(a, b), Infinity) - pad;
  const rawXMax = allX.reduce((a, b) => Math.max(a, b), -Infinity) + pad;
  const rawYMin = allY.reduce((a, b) => Math.min(a, b), Infinity) - pad;
  const rawYMax = allY.reduce((a, b) => Math.max(a, b), -Infinity) + pad;

  // Cap domain to ±80 around midpoint to prevent enormous surfaces
  const MAX_HALF = 80;
  const xMid = (rawXMin + rawXMax) / 2, yMid = (rawYMin + rawYMax) / 2;
  const xHalf = Math.min((rawXMax - rawXMin) / 2, MAX_HALF);
  const yHalf = Math.min((rawYMax - rawYMin) / 2, MAX_HALF);
  const xMin = xMid - xHalf, xMax = xMid + xHalf;
  const yMin = yMid - yHalf, yMax = yMid + yHalf;

  // Build surface grid
  const GN = 38;
  const gx = [], gy = [], gz = [];
  for (let i = 0; i <= GN; i++) {
    const rx = [], ry = [], rz = [];
    const xi = xMin + (xMax - xMin) * i / GN;
    for (let j = 0; j <= GN; j++) {
      const yj = yMin + (yMax - yMin) * j / GN;
      rx.push(xi); ry.push(yj);
      try {
        const z = currentFn(xi, yj, Math);
        rz.push(isFinite(z) ? Math.max(-300, Math.min(300, z)) : null);
      } catch (e) { rz.push(null); }
    }
    gx.push(rx); gy.push(ry); gz.push(rz);
  }

  // Plotly traces
  const traces = [{
    type: 'surface', x: gx, y: gy, z: gz,
    colorscale: [
      ['0.0', '#080B12'],
      ['0.25', '#0D2040'],
      ['0.5', '#0066AA'],
      ['0.75', '#00C8FF'],
      ['1.0', '#E0F8FF'],
    ],
    opacity: 0.72, showscale: false, hoverinfo: 'none',
    lighting: { ambient: 0.80, diffuse: 0.65, specular: 0.15 },
    contours: { z: { show: false } },
  }];

  // Path traces
  OPTIMIZERS.forEach(opt => {
    if (!activeOpts.has(opt.id)) return;
    const path = allPaths[opt.id];
    const xs = path.map(p => p[0]), ys = path.map(p => p[1]);
    const zs = path.map(p => { try { return currentFn(p[0], p[1], Math); } catch (e) { return NaN; } });

    traces.push({
      type: 'scatter3d', x: xs, y: ys, z: zs, mode: 'lines',
      line: { color: opt.color, width: 5 }, name: opt.label,
      hovertemplate: `<b>${opt.label}</b><br>x₀: %{x:.4f}<br>x₁: %{y:.4f}<br>loss: %{z:.5f}<extra></extra>`,
    });
    // End marker
    traces.push({
      type: 'scatter3d', x: [xs[xs.length - 1]], y: [ys[ys.length - 1]], z: [zs[zs.length - 1]],
      mode: 'markers', marker: { size: 6, color: opt.color, symbol: 'circle' },
      showlegend: false, name: `${opt.label} end`,
      hovertemplate: `${opt.label} end<br>loss: ${isFinite(zs[zs.length-1]) ? zs[zs.length-1].toFixed(5) : 'diverged'}<extra></extra>`,
    });
  });

  const startZ = (() => { try { const z = currentFn(x0s, x1s, Math); return isFinite(z) ? z : 0; } catch (e) { return 0; } })();
  traces.push({
    type: 'scatter3d', x: [x0s], y: [x1s], z: [startZ], mode: 'markers',
    marker: { size: 9, color: '#ffffff', symbol: 'circle', line: { color: '#00C8FF', width: 2 } },
    name: 'start',
    hovertemplate: `start<br>(${x0s.toFixed(2)}, ${x1s.toFixed(2)})<br>loss: ${isFinite(startZ) ? startZ.toFixed(4) : 'N/A'}<extra></extra>`,
  });

  // Global min marker + drop line
  if (gmin) {
    traces.push({
      type: 'scatter3d', x: [gmin.x], y: [gmin.y], z: [gmin.z],
      mode: 'markers+text',
      marker: { size: 11, color: '#F0A832', symbol: 'diamond', line: { color: '#00C8FF', width: 2 } },
      text: ['◆ global min'], textposition: 'top center',
      textfont: { size: 11, color: '#F0A832' }, name: 'global min',
      hovertemplate: `global min<br>x₀: ${gmin.x.toFixed(4)}<br>x₁: ${gmin.y.toFixed(4)}<br>loss: ${gmin.z.toFixed(5)}<extra></extra>`,
    });
    const flatZ = gz.flat().filter(z => z != null);
    const zBase = (flatZ.length ? flatZ.reduce((a, b) => Math.min(a, b), Infinity) : 0) - 1;
    traces.push({
      type: 'scatter3d', x: [gmin.x, gmin.x], y: [gmin.y, gmin.y], z: [gmin.z, zBase],
      mode: 'lines', line: { color: '#F0A832', width: 2, dash: 'dot' },
      hoverinfo: 'none', showlegend: false, name: '',
    });
  }

  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 0, r: 0, t: 10, b: 0 },
    showlegend: false,
    scene: {
      bgcolor: '#080B12',
      xaxis: {
        title: { text: 'x₀', font: { size: 12, color: '#8A9BB8', family: 'JetBrains Mono' } },
        tickfont: { size: 9, color: '#4A5870', family: 'JetBrains Mono' },
        gridcolor: 'rgba(0,200,255,0.08)', zerolinecolor: 'rgba(0,200,255,0.18)',
        backgroundcolor: '#0D1117',
      },
      yaxis: {
        title: { text: 'x₁', font: { size: 12, color: '#8A9BB8', family: 'JetBrains Mono' } },
        tickfont: { size: 9, color: '#4A5870', family: 'JetBrains Mono' },
        gridcolor: 'rgba(0,200,255,0.08)', zerolinecolor: 'rgba(0,200,255,0.18)',
        backgroundcolor: '#0D1117',
      },
      zaxis: {
        title: { text: 'loss', font: { size: 12, color: '#8A9BB8', family: 'JetBrains Mono' } },
        tickfont: { size: 9, color: '#4A5870', family: 'JetBrains Mono' },
        gridcolor: 'rgba(0,200,255,0.08)', zerolinecolor: 'rgba(0,200,255,0.18)',
        backgroundcolor: '#080B12',
      },
      camera: { eye: { x: 1.6, y: -1.8, z: 1.2 } },
    },
  };

  Plotly.react('plot', traces, layout, { responsive: true, displayModeBar: false });
  updateResultsTable(results);
  updateLegend(results, gmin);
}

// ── Error display ─────────────────────────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

// ── Apply function ────────────────────────────────────────────────────────────
function applyFn() {
  const expr = document.getElementById('fn-input').value.trim();
  const gradMode = document.getElementById('grad-mode').value;
  showError('');
  try {
    const f = makeFn(expr);
    f(1, 1, Math); // test evaluation
    currentFn = f;

    if (gradMode === 'auto') {
      currentGrad = (x0, x1) => numericalGrad(currentFn, x0, x1);
    } else {
      const g0expr = document.getElementById('grad0-input').value.trim();
      const g1expr = document.getElementById('grad1-input').value.trim();
      const g0 = makeFn(g0expr), g1 = makeFn(g1expr);
      g0(1, 1, Math); g1(1, 1, Math);
      currentGrad = (x0, x1) => [g0(x0, x1, Math), g1(x0, x1, Math)];
    }

    // Find global min first, then set slider ranges around it
    currentGmin = findGlobalMin(currentFn, currentGrad);
    updateStartSliders(currentGmin);
    buildPlot();
  } catch (e) {
    showError('Expression error: ' + e.message);
  }
}

// ── Preset loader ─────────────────────────────────────────────────────────────
function setPreset(fn, g0, g1, btn) {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  document.getElementById('fn-input').value = fn;
  document.getElementById('grad0-input').value = g0;
  document.getElementById('grad1-input').value = g1;
  document.getElementById('grad-mode').value = 'manual';
  toggleGradMode();
  applyFn();
}

function toggleGradMode() {
  const manual = document.getElementById('grad-mode').value === 'manual';
  document.getElementById('manual-grad').style.display = manual ? 'flex' : 'none';
}

// ── Slider wiring ─────────────────────────────────────────────────────────────
document.getElementById('grad-mode').addEventListener('change', toggleGradMode);

const fmtCoord = v => {
  const abs = Math.abs(parseFloat(v));
  if (abs >= 100) return parseFloat(v).toFixed(1);
  if (abs >= 10)  return parseFloat(v).toFixed(2);
  return parseFloat(v).toFixed(3);
};

[
  { id: 'lr-slider',   dispId: 'lr-display',   fmt: v => (parseInt(v) / 1000).toFixed(3) },
  { id: 'iter-slider', dispId: 'iter-display',  fmt: v => parseInt(v).toString() },
  { id: 'x0-slider',  dispId: 'x0-display',    fmt: fmtCoord },
  { id: 'x1-slider',  dispId: 'x1-display',    fmt: fmtCoord },
].forEach(({ id, dispId, fmt }) => {
  document.getElementById(id).addEventListener('input', function () {
    document.getElementById(dispId).textContent = fmt(this.value);
    buildPlot();
  });
});

document.getElementById('fn-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') applyFn();
});

// ── Init ──────────────────────────────────────────────────────────────────────
buildToggles();
buildParamCards();
applyFn();

// ── Animation engine ──────────────────────────────────────────────────────────
let animState = {
  isPlaying: false,
  frame: 0,
  rafId: null,
  lastTime: 0,
  paths: {},       // { optId: [[x0,x1], ...] }
  zs: {},          // { optId: [z, ...] }
  maxLen: 0,
  surface: null,   // pre-built surface traces
  gmin: null,
  startZ: 0,
  x0s: 0, x1s: 0,
};

function toggleAnimation() {
  if (!currentFn || !currentGrad) return;
  if (animState.isPlaying) {
    stopAnimation();
  } else {
    startAnimation();
  }
}

function startAnimation() {
  // Build all paths fresh
  const lr      = parseInt(document.getElementById('lr-slider').value) / 1000;
  const numIter = parseInt(document.getElementById('iter-slider').value);
  const x0s     = parseFloat(document.getElementById('x0-slider').value);
  const x1s     = parseFloat(document.getElementById('x1-slider').value);

  animState.paths = {};
  animState.zs    = {};
  animState.maxLen = 0;
  animState.x0s = x0s; animState.x1s = x1s;
  animState.gmin = currentGmin;
  animState.startZ = (() => { try { const z = currentFn(x0s, x1s, Math); return isFinite(z) ? z : 0; } catch(e){ return 0; } })();

  OPTIMIZERS.forEach(opt => {
    if (!activeOpts.has(opt.id)) return;
    const path = dispatchOptimizer(opt.id, currentGrad, x0s, x1s, lr, numIter);
    animState.paths[opt.id] = path;
    animState.zs[opt.id] = path.map(p => {
      try { const z = currentFn(p[0], p[1], Math); return isFinite(z) ? z : NaN; } catch(e){ return NaN; }
    });
    animState.maxLen = Math.max(animState.maxLen, path.length);
  });

  // Pre-build static surface (reuse current plot's surface)
  animState.frame = 0;
  animState.isPlaying = true;

  const btn = document.getElementById('btn-play');
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  const label = document.getElementById('play-label');
  btn.classList.add('playing');
  playIcon.style.display = 'none';
  pauseIcon.style.display = '';
  label.textContent = 'Pause';

  animState.lastTime = performance.now();
  animState.rafId = requestAnimationFrame(animTick);
}

function stopAnimation() {
  animState.isPlaying = false;
  if (animState.rafId) { cancelAnimationFrame(animState.rafId); animState.rafId = null; }

  const btn = document.getElementById('btn-play');
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  const label = document.getElementById('play-label');
  btn.classList.remove('playing');
  playIcon.style.display = '';
  pauseIcon.style.display = 'none';
  label.textContent = 'Play';

  // Restore full plot
  buildPlot();
}

function animTick(now) {
  if (!animState.isPlaying) return;

  const speed = parseInt(document.getElementById('anim-speed').value);
  const msPerFrame = Math.max(16, 300 / speed);

  if (now - animState.lastTime >= msPerFrame) {
    animState.lastTime = now;
    animState.frame++;
    if (animState.frame >= animState.maxLen) {
      animState.frame = animState.maxLen - 1;
      stopAnimation();
      return;
    }
    renderAnimFrame(animState.frame);
  }
  animState.rafId = requestAnimationFrame(animTick);
}

function renderAnimFrame(frame) {
  const { paths, zs, maxLen, x0s, x1s, startZ, gmin } = animState;
  const pct = maxLen > 1 ? frame / (maxLen - 1) : 1;

  // Update progress bar + step display
  document.getElementById('anim-progress-bar').style.width = (pct * 100).toFixed(1) + '%';
  document.getElementById('anim-step-display').textContent = `step ${frame} / ${maxLen - 1}`;
  document.getElementById('anim-speed-display').textContent = document.getElementById('anim-speed').value;

  // Build path traces up to current frame
  const traces = [];

  OPTIMIZERS.forEach(opt => {
    if (!paths[opt.id]) return;
    const path = paths[opt.id];
    const end  = Math.min(frame + 1, path.length);
    const xs = path.slice(0, end).map(p => p[0]);
    const ys = path.slice(0, end).map(p => p[1]);
    const zArr = zs[opt.id].slice(0, end);

    traces.push({
      type: 'scatter3d', x: xs, y: ys, z: zArr, mode: 'lines',
      line: { color: opt.color, width: 5 }, name: opt.label,
      hovertemplate: `<b>${opt.label}</b><br>x₀: %{x:.4f}<br>x₁: %{y:.4f}<br>loss: %{z:.5f}<extra></extra>`,
    });

    // Moving head dot
    if (xs.length > 0) {
      traces.push({
        type: 'scatter3d',
        x: [xs[xs.length-1]], y: [ys[ys.length-1]], z: [zArr[zArr.length-1]],
        mode: 'markers',
        marker: { size: 8, color: opt.color, symbol: 'circle',
          line: { color: '#ffffff', width: 1 } },
        showlegend: false, hoverinfo: 'skip', name: '',
      });
    }
  });

  // Start marker
  traces.push({
    type: 'scatter3d', x: [x0s], y: [x1s], z: [startZ], mode: 'markers',
    marker: { size: 9, color: '#ffffff', symbol: 'circle', line: { color: '#00C8FF', width: 2 } },
    name: 'start', hoverinfo: 'skip',
  });

  // Global min marker
  if (gmin) {
    traces.push({
      type: 'scatter3d', x: [gmin.x], y: [gmin.y], z: [gmin.z],
      mode: 'markers+text',
      marker: { size: 11, color: '#F0A832', symbol: 'diamond', line: { color: '#00C8FF', width: 2 } },
      text: ['◆ min'], textposition: 'top center',
      textfont: { size: 10, color: '#F0A832' }, name: 'global min', hoverinfo: 'skip',
    });
  }

  // Use Plotly.react with current layout (no surface rebuild — keep it from buildPlot)
  Plotly.react('plot', traces, window._lastLayout || {}, { responsive: true, displayModeBar: false });
}

// Patch buildPlot to cache layout for animation
const _origBuildPlot = buildPlot;
buildPlot = function() {
  _origBuildPlot();
  // Cache layout after buildPlot sets it
  setTimeout(() => {
    const plotEl = document.getElementById('plot');
    if (plotEl && plotEl.layout) window._lastLayout = plotEl.layout;
  }, 100);
};

// Speed slider display update
document.getElementById('anim-speed').addEventListener('input', function() {
  document.getElementById('anim-speed-display').textContent = this.value;
});
