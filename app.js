/* MLP vs Teachable Machine: client-side demo.
 * Both models run in the browser. Add a model by pushing another object
 * { id, name, cls, info, predict(canvas224) -> Promise<Float32Array(10)> } into state.models. */
(() => {
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const SIZE = 224;

// Numbers from the written report (same 1,500 test images for both models).
const REPORT = {
  perClassCorrect: { mlp: [52, 71, 9, 25, 19, 29, 73, 77, 48, 21], tm: [147, 139, 143, 129, 134, 140, 136, 138, 146, 144] },
  perClassTotal: 150,
};

const state = {
  classes: [],
  models: [],
  base: null,            // 224x224 canvas of the current image
  trueLabel: -1,
  source: '',
  gallery: [],
  cat: 'all', cls: -1, page: 0,
  tokenTry: 0, tokenStress: 0,
  stream: null, liveTimer: null, liveBusy: false,
};

const ico = name => `<img class="ico" src="icons/${String(name).toLowerCase()}.svg" alt="" width="24" height="24">`;
const lab = i => `<span class="lab">${ico(state.classes[i])}${state.classes[i]}</span>`;
const mk = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmtPct = p => (p * 100 < 10 ? (p * 100).toFixed(1) : Math.round(p * 100)) + '%';
const argmax = a => { let m = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[m]) m = i; return m; };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const frame = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));

/* ------------------------------------------------------------------ models */

async function loadMLP() {
  const manifest = await (await fetch('models/mlp/mlp_manifest.json')).json();
  const buf = await (await fetch('models/mlp/mlp_weights.bin')).arrayBuffer();
  const all = new Float32Array(buf);
  let off = 0;
  const layers = manifest.layers.map(L => {
    const W = all.subarray(off, off + L.in * L.out); off += L.in * L.out;
    const b = all.subarray(off, off + L.out); off += L.out;
    return { ...L, W, b };
  });
  if (off !== all.length) throw new Error('MLP weight file does not match its manifest');
  const S = manifest.input_size;
  const tmp = mk(S, S);
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.imageSmoothingEnabled = true; tctx.imageSmoothingQuality = 'high';

  const input = canvas => {
    tctx.clearRect(0, 0, S, S);
    tctx.drawImage(canvas, 0, 0, S, S);
    const d = tctx.getImageData(0, 0, S, S).data;
    const x = new Float32Array(S * S * 3);
    for (let i = 0, j = 0; i < d.length; i += 4) { x[j++] = d[i] / 255; x[j++] = d[i + 1] / 255; x[j++] = d[i + 2] / 255; }
    return x;
  };
  const forward = x => {
    let a = x;
    for (const L of layers) {
      const out = new Float32Array(L.out);
      out.set(L.b);
      for (let i = 0; i < L.in; i++) {
        const ai = a[i];
        if (ai === 0) continue;
        const row = i * L.out;
        for (let j = 0; j < L.out; j++) out[j] += ai * L.W[row + j];
      }
      if (L.activation === 'relu') { for (let j = 0; j < L.out; j++) if (out[j] < 0) out[j] = 0; }
      else if (L.activation === 'softmax') {
        let m = -Infinity; for (let j = 0; j < L.out; j++) if (out[j] > m) m = out[j];
        let s = 0; for (let j = 0; j < L.out; j++) { out[j] = Math.exp(out[j] - m); s += out[j]; }
        for (let j = 0; j < L.out; j++) out[j] /= s;
      }
      a = out;
    }
    return a;
  };
  forward(new Float32Array(S * S * 3));   // warm-up
  const params = layers.reduce((s, L) => s + L.in * L.out + L.out, 0);
  return {
    id: 'mlp', name: 'MLP', nick: 'the pixel counter', cls: 'mlp', classes: manifest.classes,
    info: `${(buf.byteLength / 1048576).toFixed(1)} MB · ${params.toLocaleString()} params`,
    predict: async canvas => forward(input(canvas)),
  };
}

async function loadTM(classes) {
  const tm = await tmImage.load('models/tm/model.json', 'models/tm/metadata.json');
  const labels = tm.getClassLabels();
  const idx = labels.map(l => classes.findIndex(c => c.toLowerCase() === String(l).toLowerCase()));
  if (idx.some(i => i < 0)) throw new Error('Teachable Machine class names do not match the MLP classes');
  await tm.predict(mk(SIZE, SIZE)).catch(() => {});   // warm-up: compiles the GPU shaders once
  const entry = performance.getEntriesByType('resource').find(r => r.name.endsWith('models/tm/weights.bin'));
  const sizeMB = entry ? (entry.decodedBodySize || entry.encodedBodySize || 0) / 1048576 : 0;
  return {
    id: 'tm', name: 'Teachable Machine', nick: 'the shape spotter', cls: 'tm', classes: labels,
    info: `${sizeMB ? sizeMB.toFixed(1) + ' MB · ' : ''}pre-trained CNN`,
    predict: async canvas => {
      const preds = await tm.predict(canvas);
      const out = new Float32Array(classes.length);
      preds.forEach((p, k) => { out[idx[k]] = p.probability; });
      return out;
    },
  };
}

async function predictAll(canvas) {
  const out = {};
  for (const m of state.models) {
    const t0 = performance.now();
    const p = await m.predict(canvas);
    out[m.id] = { p, ms: performance.now() - t0 };
  }
  return out;
}

/* ------------------------------------------------------------------ result cards */

function buildCards(container, prefix) {
  container.innerHTML = '';
  for (const m of state.models) {
    const el = document.createElement('article');
    el.className = `model ${m.cls}`;
    el.innerHTML = `<header><div><h3>${m.name}</h3><p class="nick">${m.nick || ''}</p></div><span class="meta">${m.info}</span></header>
      <div class="top1" id="${prefix}-top-${m.id}">–</div>
      <div class="bars" id="${prefix}-bars-${m.id}"></div>`;
    container.appendChild(el);
  }
}

function renderCard(prefix, m, res, truth) {
  const p = res.p, order = [...p.keys()].sort((a, b) => p[b] - p[a]);
  const top = order[0];
  let badge = '';
  if (truth >= 0) badge = top === truth ? '<span class="badge ok">correct</span>' : '<span class="badge no">wrong</span>';
  $(`#${prefix}-top-${m.id}`).innerHTML =
    `${lab(top)}<small>${fmtPct(p[top])}</small>${badge}`;
  const bars = $(`#${prefix}-bars-${m.id}`);
  bars.style.setProperty('--c', `var(--${m.cls})`);
  bars.innerHTML = order.map((k, r) => `
    <div class="bar ${r === 0 ? 'top' : ''} ${k === truth ? 'truth' : ''}">
      <span class="name">${lab(k)}</span>
      <span class="track"><span class="fill" style="width:${(p[k] * 100).toFixed(1)}%"></span></span>
      <span class="pct">${fmtPct(p[k])}</span>
    </div>`).join('');
  $(`#${prefix}-top-${m.id}`).closest('.model').querySelector('.meta').textContent = `${m.info} · ${res.ms.toFixed(0)} ms`;
}

function renderVerdict(results, truth) {
  const tops = state.models.map(m => argmax(results[m.id].p));
  const v = $('#verdict');
  const names = state.models.map((m, i) => `${m.name}: <b>${lab(tops[i])}</b>`).join(' &nbsp;·&nbsp; ');
  const agree = tops.every(t => t === tops[0]);
  let extra = '';
  if (truth >= 0) extra = ` &nbsp;·&nbsp; it is a <b>${lab(truth)}</b>`;
  v.className = 'verdict ' + (agree ? 'agree' : 'differ');
  v.innerHTML = (agree ? 'Both models agree. ' : 'The models disagree. ') + names + extra;
}

async function runTry() {
  if (!state.base || !state.models.length) return;
  const my = ++state.tokenTry;
  $('#verdict').className = 'verdict'; $('#verdict').textContent = 'Thinking…';
  const res = await predictAll(state.base);
  if (my !== state.tokenTry) return;
  for (const m of state.models) renderCard('try', m, res[m.id], state.trueLabel);
  renderVerdict(res, state.trueLabel);
}

/* ------------------------------------------------------------------ input handling */

function drawCover(srcEl, sw, sh) {
  const c = mk(SIZE, SIZE), x = c.getContext('2d');
  const s = Math.min(sw, sh), sx = (sw - s) / 2, sy = (sh - s) / 2;
  x.imageSmoothingQuality = 'high';
  x.drawImage(srcEl, sx, sy, s, s, 0, 0, SIZE, SIZE);
  return c;
}

function setInput(canvas, source, truth = -1) {
  state.base = canvas;
  state.source = source;
  state.trueLabel = truth;
  const pv = $('#preview');
  pv.getContext('2d').drawImage(canvas, 0, 0, SIZE, SIZE);
  $('#drop').classList.add('has-img');
  $('#source').textContent = source;
  $('#stressCanvas').getContext('2d').drawImage(canvas, 0, 0);
  resetStress(false);
  clearHeat();
  runTry();
  if ($('#tab-stress:not([hidden])')) runStress();
  if ($('#tab-look:not([hidden])') && state.models.length) computeHeat();
}

function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => { setInput(drawCover(img, img.naturalWidth, img.naturalHeight), `Your photo: ${file.name || 'pasted image'}`); URL.revokeObjectURL(url); };
  img.onerror = () => showBanner('That file could not be read as a photo.');
  img.src = url;
}

function loadFromGallery(item) {
  const img = new Image();
  img.onload = () => {
    const c = mk(SIZE, SIZE); c.getContext('2d').drawImage(img, 0, 0, SIZE, SIZE);
    setInput(c, `Test photo · it is a ${state.classes[item.t]}`, item.t);
  };
  img.src = 'gallery/' + item.f;
}

function showBanner(msg) { const b = $('#banner'); b.textContent = msg; b.hidden = false; }

function initInput() {
  const drop = $('#drop'), file = $('#file');
  drop.addEventListener('click', () => file.click());
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); } });
  $('#btnChoose').addEventListener('click', () => file.click());
  file.addEventListener('change', () => { loadFile(file.files[0]); file.value = ''; });
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));
  window.addEventListener('paste', e => { const f = [...(e.clipboardData?.files || [])][0]; if (f) loadFile(f); });

  // webcam
  const video = $('#video');
  $('#btnCam').addEventListener('click', async () => {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      video.srcObject = state.stream; await video.play();
      $('#camBox').hidden = false;
    } catch (err) {
      showBanner('Webcam unavailable: ' + (err.message || err) + ' (browsers only allow it on https:// pages or localhost).');
    }
  });
  const snap = () => { if (video.videoWidth) setInput(drawCover(video, video.videoWidth, video.videoHeight), 'Webcam photo'); };
  $('#btnSnap').addEventListener('click', snap);
  const stopCam = () => {
    $('#live').checked = false; clearInterval(state.liveTimer);
    state.stream?.getTracks().forEach(t => t.stop()); state.stream = null; $('#camBox').hidden = true;
  };
  $('#btnCamOff').addEventListener('click', stopCam);
  $('#live').addEventListener('change', e => {
    clearInterval(state.liveTimer);
    if (e.target.checked) state.liveTimer = setInterval(async () => {
      if (state.liveBusy || !video.videoWidth) return;
      state.liveBusy = true; try { snap(); await sleep(50); } finally { state.liveBusy = false; }
    }, 900);
  });
}

/* ------------------------------------------------------------------ heatmaps */

const heatState = { running: false };

function clearHeat() { $('#heatGrid').innerHTML = ''; $('#lookNote').textContent = ''; $('#lookProgress').textContent = ''; }

function blurMap(m, w, h, r) {
  const tmp = new Float32Array(m.length), out = new Float32Array(m.length);
  const pass = (src, dst, horiz) => {
    const len = horiz ? w : h, lines = horiz ? h : w, step = horiz ? 1 : w, lineStep = horiz ? w : 1;
    for (let l = 0; l < lines; l++) {
      const o = l * lineStep; let sum = 0;
      for (let i = -r; i <= r; i++) sum += src[o + clamp(i, 0, len - 1) * step];
      for (let i = 0; i < len; i++) {
        dst[o + i * step] = sum / (2 * r + 1);
        sum += src[o + clamp(i + r + 1, 0, len - 1) * step] - src[o + clamp(i - r, 0, len - 1) * step];
      }
    }
  };
  pass(m, tmp, true); pass(tmp, out, false);
  pass(out, tmp, true); pass(tmp, out, false);
  return out;
}

async function occlusion(model, canvas, target, onStep) {
  const P = 56, S = 28, n = Math.floor((SIZE - P) / S) + 1, EPS = 1e-4;
  const base = (await model.predict(canvas))[target];
  const accL = new Float32Array(SIZE * SIZE), accP = new Float32Array(SIZE * SIZE), cnt = new Float32Array(SIZE * SIZE);
  const tmp = mk(SIZE, SIZE), t = tmp.getContext('2d');
  let done = 0;
  for (let gy = 0; gy < n; gy++) for (let gx = 0; gx < n; gx++) {
    t.drawImage(canvas, 0, 0);
    t.fillStyle = 'rgb(128,128,128)'; t.fillRect(gx * S, gy * S, P, P);
    const p = (await model.predict(tmp))[target];
    const dLog = Math.log(base + EPS) - Math.log(p + EPS), dProb = base - p;   // log-probability keeps the map informative even when a model is ~100% sure
    for (let y = gy * S; y < gy * S + P; y++) for (let x = gx * S; x < gx * S + P; x++) { const i = y * SIZE + x; accL[i] += dLog; accP[i] += dProb; cnt[i]++; }
    onStep(++done, n * n);
    if (done % 7 === 0) await frame();
  }
  const mapL = new Float32Array(SIZE * SIZE), mapP = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < mapL.length; i++) { mapL[i] = cnt[i] ? Math.max(0, accL[i] / cnt[i]) : 0; mapP[i] = cnt[i] ? accP[i] / cnt[i] : 0; }
  return { map: blurMap(mapL, SIZE, SIZE, 9), mapP, base };
}

function paintHeat(canvasEl, base, map) {
  const ctx = canvasEl.getContext('2d');
  ctx.drawImage(base, 0, 0);
  let mx = 0; for (const v of map) if (v > mx) mx = v;
  const img = ctx.getImageData(0, 0, SIZE, SIZE), d = img.data;
  if (mx > 0.004) {
    for (let i = 0; i < map.length; i++) {
      const v = clamp(map[i] / mx, 0, 1), a = 0.82 * Math.pow(v, 1.15), k = i * 4;
      const g = Math.round(215 - 175 * v);
      d[k] = d[k] * (1 - a) * 0.8 + 255 * a; d[k + 1] = d[k + 1] * (1 - a) * 0.8 + g * a; d[k + 2] = d[k + 2] * (1 - a) * 0.8 + 40 * a;
    }
  }
  ctx.putImageData(img, 0, 0);
  return mx;
}

async function computeHeat() {
  if (!state.base) { $('#lookNote').textContent = 'Pick an image in the “Try it” tab first.'; return; }
  if (heatState.running) return;
  heatState.running = true;
  const btn = $('#btnLook'); btn.disabled = true;
  const grid = $('#heatGrid'); grid.innerHTML = '';
  const canvas = state.base, sel = $('#lookClass').value;
  try {
    for (const m of state.models) {
      const probs = await m.predict(canvas);
      const target = sel === 'own' ? argmax(probs) : +sel;
      const card = document.createElement('div'); card.className = 'heat';
      card.innerHTML = `<h3 style="color:var(--${m.cls}-ink)">${m.name}</h3><canvas width="${SIZE}" height="${SIZE}"></canvas><p></p>`;
      grid.appendChild(card);
      const { map, mapP, base } = await occlusion(m, canvas, target, (i, n) => { $('#lookProgress').textContent = `${m.name}: ${i}/${n}`; });
      paintHeat($('canvas', card), canvas, map);
      let worst = 0; for (const v of mapP) if (v > worst) worst = v;
      $('p', card).innerHTML = `Explaining “<b>${lab(target)}</b>” (confidence ${fmtPct(base)}). ` +
        (worst > 0.02 ? `Hiding the most important region lowers that confidence by ${(worst * 100).toFixed(1)} points.`
                      : 'Hiding any single region barely changes its answer: it is using many cues at once.');
    }
    $('#lookProgress').textContent = '';
    $('#lookNote').textContent = 'Each map is scaled to its own maximum, so compare where the red is, not how strong it is. A model that looks at the animal gives a compact red blob on it; a model that relies on background colours spreads red elsewhere.';
  } catch (e) { showBanner('Heatmap failed: ' + e.message); }
  heatState.running = false; btn.disabled = false;
}

/* ------------------------------------------------------------------ stress test */

const stress = { noise: 0, blur: 0, bright: 100, rot: 0 };
let noiseField = null;

function seededNoise() {
  let s = 0x9e3779b9;
  const rnd = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const f = new Float32Array(SIZE * SIZE * 3);
  for (let i = 0; i < f.length; i += 2) {
    const u = Math.max(rnd(), 1e-9), v = rnd(), r = Math.sqrt(-2 * Math.log(u));
    f[i] = r * Math.cos(2 * Math.PI * v); if (i + 1 < f.length) f[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return f;
}

function boxBlur(d, w, h, r) {
  const pass = (src, dst, horiz) => {
    const len = horiz ? w : h, lines = horiz ? h : w, step = horiz ? 4 : w * 4, lineStep = horiz ? w * 4 : 4;
    for (let l = 0; l < lines; l++) for (let c = 0; c < 3; c++) {
      const o = l * lineStep + c; let sum = 0;
      for (let i = -r; i <= r; i++) sum += src[o + clamp(i, 0, len - 1) * step];
      for (let i = 0; i < len; i++) {
        dst[o + i * step] = sum / (2 * r + 1);
        sum += src[o + clamp(i + r + 1, 0, len - 1) * step] - src[o + clamp(i - r, 0, len - 1) * step];
      }
    }
  };
  const a = Float32Array.from(d), b = new Float32Array(d.length);
  for (let n = 0; n < 2; n++) { pass(a, b, true); pass(b, a, false); }   // two passes approximate a gaussian
  for (let i = 0; i < d.length; i++) if (i % 4 !== 3) d[i] = a[i];
}

function applyPerturb(src, p) {
  const c = mk(SIZE, SIZE), x = c.getContext('2d', { willReadFrequently: true });
  x.fillStyle = 'rgb(128,128,128)'; x.fillRect(0, 0, SIZE, SIZE);
  x.save(); x.translate(SIZE / 2, SIZE / 2); x.rotate(p.rot * Math.PI / 180); x.drawImage(src, -SIZE / 2, -SIZE / 2); x.restore();
  const img = x.getImageData(0, 0, SIZE, SIZE), d = img.data;
  if (p.blur > 0) boxBlur(d, SIZE, SIZE, Math.round(p.blur));
  if (p.noise > 0 && !noiseField) noiseField = seededNoise();
  const k = p.bright / 100;
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
    for (let ch = 0; ch < 3; ch++) {
      let v = d[i + ch] * k;
      if (p.noise > 0) v += noiseField[j + ch] * p.noise * 255;
      d[i + ch] = clamp(v, 0, 255);
    }
  }
  x.putImageData(img, 0, 0);
  return c;
}

let stressTimer = null;
function scheduleStress() { clearTimeout(stressTimer); stressTimer = setTimeout(runStress, 120); }

async function runStress() {
  if (!state.base || !state.models.length) return;
  const my = ++state.tokenStress;
  const c = applyPerturb(state.base, stress);
  $('#stressCanvas').getContext('2d').drawImage(c, 0, 0);
  const res = await predictAll(c);
  if (my !== state.tokenStress) return;
  for (const m of state.models) renderCard('st', m, res[m.id], state.trueLabel);
}

function resetStress(run = true) {
  Object.assign(stress, { noise: 0, blur: 0, bright: 100, rot: 0 });
  $('#s-noise').value = 0; $('#s-blur').value = 0; $('#s-bright').value = 100; $('#s-rot').value = 0;
  syncOutputs();
  if (state.base) $('#stressCanvas').getContext('2d').drawImage(state.base, 0, 0);
  if (run) runStress();
}

function syncOutputs() {
  $('#o-noise').textContent = stress.noise.toFixed(2);
  $('#o-blur').textContent = stress.blur + ' px';
  $('#o-bright').textContent = stress.bright + '%';
  $('#o-rot').textContent = stress.rot + '°';
}

const SWEEPS = {
  noise: { label: 'Noise level', vals: [0, .05, .1, .15, .2, .3, .4, .5, .6], key: 'noise', fmt: v => v.toFixed(2) },
  blur: { label: 'Blur radius (px)', vals: [0, 1, 2, 3, 4, 6, 8, 10, 12], key: 'blur', fmt: v => v },
  bright: { label: 'Brightness (%)', vals: [100, 80, 60, 45, 30, 20, 12, 6, 3], key: 'bright', fmt: v => v },
  rot: { label: 'Rotation (°)', vals: [0, 8, 15, 23, 30, 38, 45, 53, 60], key: 'rot', fmt: v => v },
};

async function runSweep() {
  if (!state.base) { $('#sweepInfo').textContent = 'Pick an image in the “Try it” tab first.'; return; }
  const btn = $('#btnSweep'); btn.disabled = true;
  const kind = $('#sweepKind').value, cfg = SWEEPS[kind];
  const clean = await predictAll(state.base);
  const ref = state.trueLabel >= 0 ? state.trueLabel : argmax(clean.tm ? clean.tm.p : clean[state.models[0].id].p);
  const already = state.models.filter(m => argmax(clean[m.id].p) !== ref).map(m => m.name);
  $('#sweepInfo').textContent = `Tracking the probability of “${state.classes[ref]}” (${state.trueLabel >= 0 ? 'the true class' : 'the answer of the stronger model on the clean image'}).` +
    (already.length ? ` ${already.join(' and ')} already ${already.length > 1 ? 'miss' : 'misses'} it on the clean image; pick one from “Both right” in the explorer to watch both fall.` : '');
  const series = {};
  state.models.forEach(m => series[m.id] = []);
  for (let i = 0; i < cfg.vals.length; i++) {
    const p = { noise: 0, blur: 0, bright: 100, rot: 0, [cfg.key]: cfg.vals[i] };
    const c = applyPerturb(state.base, p);
    for (const m of state.models) series[m.id].push((await m.predict(c))[ref]);
    drawSweep(cfg, series, state.classes[ref]);
    await frame();
  }
  btn.disabled = false;
}

function drawSweep(cfg, series, refName) {
  const cv = $('#sweepChart'), x = cv.getContext('2d'), W = cv.width, H = cv.height;
  const L = 56, R = 18, T = 34, B = 52, pw = W - L - R, ph = H - T - B;
  x.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const col = { text: css.getPropertyValue('--text'), muted: css.getPropertyValue('--muted'), line: css.getPropertyValue('--line'), mlp: css.getPropertyValue('--mlp'), tm: css.getPropertyValue('--tm') };
  x.font = '13px system-ui, sans-serif'; x.textBaseline = 'middle';
  x.fillStyle = col.text; x.textAlign = 'left'; x.fillText(`Probability of “${refName}” as the image gets worse`, L, 14);
  x.strokeStyle = col.line; x.fillStyle = col.muted; x.textAlign = 'right'; x.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = T + ph - ph * g / 4;
    x.beginPath(); x.moveTo(L, y); x.lineTo(L + pw, y); x.stroke();
    x.fillText(`${g * 25}%`, L - 8, y);
  }
  x.textAlign = 'center';
  const n = cfg.vals.length, X = i => L + pw * i / (n - 1), Y = v => T + ph - ph * clamp(v, 0, 1);
  cfg.vals.forEach((v, i) => x.fillText(String(cfg.fmt(v)), X(i), T + ph + 16));
  x.fillStyle = col.text; x.fillText(cfg.label, L + pw / 2, H - 12);
  state.models.forEach((m, k) => {
    const s = series[m.id], c = col[m.cls].trim();
    x.strokeStyle = c; x.fillStyle = c; x.lineWidth = 2.5; x.beginPath();
    s.forEach((v, i) => i ? x.lineTo(X(i), Y(v)) : x.moveTo(X(i), Y(v))); x.stroke();
    s.forEach((v, i) => { x.beginPath(); x.arc(X(i), Y(v), 3.5, 0, 6.3); x.fill(); });
    x.textAlign = 'left'; x.fillText('● ' + m.name, L + pw - 190 + k * 0, T + 10 + k * 18 + 0);
  });
}

function initStress() {
  const bind = (id, key, conv = Number) => $(id).addEventListener('input', e => { stress[key] = conv(e.target.value); syncOutputs(); scheduleStress(); });
  bind('#s-noise', 'noise', v => +v / 100); bind('#s-blur', 'blur'); bind('#s-bright', 'bright'); bind('#s-rot', 'rot');
  $('#btnReset').addEventListener('click', () => resetStress(true));
  $('#btnSweep').addEventListener('click', runSweep);
  syncOutputs();
}

/* ------------------------------------------------------------------ explorer */

const CATS = [
  { id: 'all', label: 'All', test: () => true },
  { id: 'both', label: 'Both right', cls: 'cat-both', test: g => g.m === g.t && g.g === g.t },
  { id: 'tm', label: 'Only Teachable Machine right', cls: 'cat-tm', test: g => g.g === g.t && g.m !== g.t },
  { id: 'mlp', label: 'Only MLP right', cls: 'cat-mlp', test: g => g.m === g.t && g.g !== g.t },
  { id: 'none', label: 'Both wrong', cls: 'cat-none', test: g => g.m !== g.t && g.g !== g.t },
];
const PER_PAGE = 48;

function catOf(g) { return CATS.slice(1).find(c => c.test(g)); }

async function initExplorer() {
  let list;
  try { list = await (await fetch('gallery/gallery.json')).json(); } catch { $('#galleryGrid').textContent = 'gallery.json not found.'; return; }
  state.gallery = list;
  { let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    state.display = list.slice(); for (let i = state.display.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [state.display[i], state.display[j]] = [state.display[j], state.display[i]]; } }
  const sel = $('#classFilter');
  state.classes.forEach((c, i) => sel.insertAdjacentHTML('beforeend', `<option value="${i}">${c}</option>`));
  sel.addEventListener('change', () => { state.cls = +sel.value; state.page = 0; renderGallery(); });
  const seg = $('#catFilter');
  seg.innerHTML = CATS.map(c => `<button data-cat="${c.id}" class="${c.id === 'all' ? 'active' : ''}">${c.label} <span class="n"></span></button>`).join('');
  seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    state.cat = b.dataset.cat; state.page = 0;
    $$('#catFilter button').forEach(x => x.classList.toggle('active', x === b)); renderGallery();
  });
  CATS.forEach(c => { $(`#catFilter [data-cat="${c.id}"] .n`).textContent = `(${list.filter(c.test).length})`; });
  $('#pgPrev').addEventListener('click', () => { state.page = Math.max(0, state.page - 1); renderGallery(true); });
  $('#pgNext').addEventListener('click', () => { state.page++; renderGallery(true); });
  $('#galleryGrid').addEventListener('click', e => { const t = e.target.closest('.tile'); if (t) { loadFromGallery(state.view[+t.dataset.i]); activate('try'); window.scrollTo({ top: 0, behavior: 'smooth' }); } });
  renderGallery();
  renderSamples();
  renderCollage();
  const acc = k => list.filter(g => g[k] === g.t).length / list.length;
  $('#liveCheck').textContent = `Measured live in this browser on the exported test images: MLP ${(acc('m') * 100).toFixed(2)}%, Teachable Machine ${(acc('g') * 100).toFixed(2)}%. ` +
    'The headline numbers above come from the training notebooks; tiny differences are expected because images were resized and JPEG-compressed again for the web.';
}

function renderGallery(scroll) {
  const cat = CATS.find(c => c.id === state.cat);
  state.view = state.display.filter(g => cat.test(g) && (state.cls < 0 || g.t === state.cls));
  const pages = Math.max(1, Math.ceil(state.view.length / PER_PAGE));
  state.page = clamp(state.page, 0, pages - 1);
  const slice = state.view.slice(state.page * PER_PAGE, (state.page + 1) * PER_PAGE);
  $('#galleryGrid').innerHTML = slice.map((g, i) => {
    const c = catOf(g);
    return `<button class="tile ${c.cls}" data-i="${state.page * PER_PAGE + i}" title="Open in Try it">
      <img loading="lazy" src="gallery/${g.f}" alt="${state.classes[g.t]}">
      <div class="cap"><b>${lab(g.t)}</b><br>
      MLP: <span class="${g.m === g.t ? 'ok' : 'no'}">${lab(g.m)}</span><br>
      TM: <span class="${g.g === g.t ? 'ok' : 'no'}">${lab(g.g)}</span></div></button>`;
  }).join('') || '<p class="muted">No images match.</p>';
  $('#pgInfo').textContent = `Page ${state.page + 1} of ${pages} · ${state.view.length} images`;
  $('#pgPrev').disabled = state.page === 0; $('#pgNext').disabled = state.page >= pages - 1;
  if (scroll) $('#tab-explore').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderCollage() {
  const wanted = ['Cat', 'Dog', 'Butterfly', 'Horse'];
  const picks = wanted.map(n => state.gallery.find(g => g.t === state.classes.indexOf(n) && g.g === g.t && g.m !== g.t)).filter(Boolean);
  $('#collage').innerHTML = picks.map((g, i) => `<button class="polaroid" data-i="${i}" aria-label="Open this ${state.classes[g.t]} photo">
      <img src="gallery/${g.f}" alt="A ${state.classes[g.t].toLowerCase()}">
      <span class="guess"><span class="no">MLP: ${lab(g.m)} ✗</span><span class="ok">TM: ${lab(g.g)} ✓</span></span></button>`).join('');
  $('#collage').onclick = e => { const b = e.target.closest('.polaroid'); if (!b) return; loadFromGallery(picks[+b.dataset.i]); activate('try'); $('#tab-try').scrollIntoView({ behavior: 'smooth', block: 'start' }); };
}

function renderSamples() {
  const firsts = state.classes.map((_, i) => state.gallery.find(g => g.t === i)).filter(Boolean);
  $('#samples').innerHTML = firsts.map((g, i) => `<button data-i="${i}" title="${state.classes[g.t]}"><img src="gallery/${g.f}" alt="${state.classes[g.t]}"><span class="em">${ico(state.classes[g.t])}</span></button>`).join('');
  $('#samples').addEventListener('click', e => { const b = e.target.closest('button'); if (b) loadFromGallery(firsts[+b.dataset.i]); });
}

/* ------------------------------------------------------------------ results tab */

function renderResults() {
  const rows = state.classes.map((c, i) => {
    const m = REPORT.perClassCorrect.mlp[i] / REPORT.perClassTotal, t = REPORT.perClassCorrect.tm[i] / REPORT.perClassTotal;
    return `<div class="pc"><span class="lab">${ico(c)}${c}</span><div class="pair">
      <div class="row"><span class="track"><span class="fill mlp" style="width:${m * 100}%"></span></span><span>${(m * 100).toFixed(1)}%</span></div>
      <div class="row"><span class="track"><span class="fill tm" style="width:${t * 100}%"></span></span><span>${(t * 100).toFixed(1)}%</span></div></div></div>`;
  }).join('');
  $('#perClass').innerHTML = rows + '<p class="fine"><b style="color:var(--mlp)">■</b> MLP &nbsp; <b style="color:var(--tm)">■</b> Teachable Machine</p>';
}

/* ------------------------------------------------------------------ tabs */

function activate(name) {
  $$('.tab').forEach(t => { const on = t.dataset.tab === name; t.classList.toggle('active', on); t.setAttribute('aria-selected', on); });
  $$('.panel').forEach(p => { const on = p.id === 'tab-' + name; p.hidden = !on; p.classList.toggle('active', on); });
  if (name === 'stress') { runStress(); }
  if (name === 'look' && state.base && !$('#heatGrid').children.length && state.models.length) computeHeat();
  if (name === 'look' && !state.base) $('#lookNote').textContent = 'Pick an image in the “Try it” tab first.';
}

function initTabs() {
  $$('.tab').forEach(t => t.addEventListener('click', () => {
    activate(t.dataset.tab);
    const top = $('.hero').offsetHeight;                        // bring the section into view instead of leaving the hero on screen
    if (window.scrollY < top) window.scrollTo({ top, behavior: 'smooth' });
  }));
  $('#btnLook').addEventListener('click', computeHeat);
}

/* ------------------------------------------------------------------ boot */

function setChip(id, cls, text) { const c = $(id); c.className = 'chip ' + cls; c.lastChild.textContent = text; }

async function boot() {
  initTabs(); initInput(); initStress();
  let mlp, tm;
  try {
    mlp = await loadMLP(); state.classes = mlp.classes; state.models.push(mlp);
    $('#par-mlp').textContent = mlp.info.split('·')[1].replace('params', '').trim();
    setChip('#chip-mlp', 'ok', `MLP ready · ${mlp.info}`);
  } catch (e) { setChip('#chip-mlp', 'err', 'MLP failed to load'); showBanner('Could not load the MLP: ' + e.message + '. Serve this folder over http:// (not file://).'); return; }
  try {
    tm = await loadTM(state.classes); state.models.push(tm);
    setChip('#chip-tm', 'ok', `Teachable Machine ready · ${tm.info}`);
  } catch (e) { setChip('#chip-tm', 'err', 'Teachable Machine failed to load'); showBanner('Could not load Teachable Machine: ' + e.message); }

  state.classes.forEach((c, i) => $('#lookClass').insertAdjacentHTML('beforeend', `<option value="${i}">${c}</option>`));
  buildCards($('#modelCols'), 'try'); buildCards($('#stressCols'), 'st');
  renderResults();
  await initExplorer();
  window.__app = { state, predictAll, loadFromGallery, applyPerturb, occlusion, setInput, runSweep, computeHeat, activate };
  window.__ready = true;
  if (new URLSearchParams(location.search).has('autostart') && state.gallery.length) loadFromGallery(state.gallery[0]);
}

// offline helper used once to precompute explorer predictions with exactly this pipeline
window.__precompute = async items => {
  const out = [];
  for (const it of items) {
    const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = 'gallery/' + it.f; });
    const c = mk(SIZE, SIZE); c.getContext('2d').drawImage(img, 0, 0, SIZE, SIZE);
    const r = await predictAll(c);
    const m = argmax(r.mlp.p), g = argmax(r.tm.p);
    out.push({ ...it, m, mc: +r.mlp.p[m].toFixed(4), g, gc: +r.tm.p[g].toFixed(4), pm: Array.from(r.mlp.p, v => +v.toFixed(5)), pg: Array.from(r.tm.p, v => +v.toFixed(5)) });
  }
  return out;
};

boot();
})();
