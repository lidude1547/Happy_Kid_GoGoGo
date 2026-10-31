/* ================= 眼睛控制版：鏡頭眼神追蹤＋注視停留選取 ================= */
// 臉部追蹤：Google MediaPipe Face Landmarker（Apache-2.0，免費），影像只在這台裝置上處理，不會上傳
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const EYE_SIM = /[?&]sim=1/.test(location.search);
const SIM_GLARE = /[?&]glare=1/.test(location.search);
const wait = ms => new Promise(r => setTimeout(r, ms));
document.body.classList.add('eye-mode');
S.set.eye = Object.assign({ dwell: 1.8, mode: 'auto', cursor: true, cam: true, steady: false }, S.set.eye || {});
S.set.eye.mode = 'auto';
S.set.capture = false; S.set.voiceRec = false; S.set.tips = false;
const EYE = { x: innerWidth / 2, y: innerHeight / 2, face: false, lastFace: 0, ready: false, model: null, samples: [], collect: null, recent: [], mouse: { x: innerWidth / 2, y: innerHeight / 2 }, err: null };
addEventListener('mousemove', e => { EYE.mouse = { x: e.clientX, y: e.clientY }; });

/* ---------- 聲音被瀏覽器擋住時：字幕照常顯示，請家人點一下就恢復 ---------- */
let MUTED = false;
function audioBlocked() { if (MUTED) return; MUTED = true; $('#eye-mute').hidden = false; }
const _speak = speechSynthesis.speak.bind(speechSynthesis);
speechSynthesis.speak = u => { u.addEventListener('error', e => { if (e.error === 'not-allowed') audioBlocked(); }); _speak(u); };
const _play = HTMLMediaElement.prototype.play;
HTMLMediaElement.prototype.play = function () { const p = _play.call(this); if (p && p.catch) p.catch(e => { if (e && e.name === 'NotAllowedError' && this.id !== 'eye-cam') audioBlocked(); }); return p; };
addEventListener('pointerdown', () => { if (!MUTED) return; MUTED = false; $('#eye-mute').hidden = true; if (LAST_LINE) talk(LAST_LINE); }, true);

/* ---------- 眼神特徵：模型的眼球方向（blendshapes）＋精確的頭部角度（3D 轉換矩陣）＋瞳孔位置 ---------- */
const BS = ['eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeBlinkLeft', 'eyeBlinkRight'];
function eyeFeatures(L, bs, mat) {
  const P = i => L[i];
  const eye = (o, n, up, lo, ir) => { const a = P(o), b = P(n), w = Math.hypot(b.x - a.x, b.y - a.y) || 1e-3, r = P(ir); return [(r.x - (a.x + b.x) / 2) / w, (r.y - (P(up).y + P(lo).y) / 2) / w]; };
  const e1 = eye(33, 133, 159, 145, 468), e2 = eye(263, 362, 386, 374, 473);
  const l = P(234), r = P(454), top = P(10), chin = P(152), nose = P(1);
  const fw = Math.hypot(r.x - l.x, r.y - l.y) || 1e-3, fh = Math.hypot(chin.x - top.x, chin.y - top.y) || 1e-3;
  const f = { ix: (e1[0] + e2[0]) / 2, iy: (e1[1] + e2[1]) / 2, yaw: (nose.x - (l.x + r.x) / 2) / fw, pitch: (nose.y - (top.y + chin.y) / 2) / fh, tx: (l.x + r.x) / 2 - .5, ty: (top.y + chin.y) / 2 - .5, size: fw, bx: 0, by: 0, blink: 0 };
  if (mat && mat.data && mat.data.length >= 16) {
    const m = mat.data; // 4×4，以欄為主：旋轉＋位移（公分）
    f.yaw = Math.asin(Math.max(-1, Math.min(1, -m[2]))); f.pitch = Math.atan2(m[6], m[10]);
    const tz = Math.abs(m[14]) || 50; f.tx = m[12] / tz; f.ty = m[13] / tz;
  }
  if (bs && bs.categories) {
    const s = {}; bs.categories.forEach(c => { if (BS.includes(c.categoryName)) s[c.categoryName] = c.score; });
    f.bx = ((s.eyeLookOutLeft || 0) - (s.eyeLookInLeft || 0) + (s.eyeLookInRight || 0) - (s.eyeLookOutRight || 0)) / 2;
    f.by = ((s.eyeLookUpLeft || 0) + (s.eyeLookUpRight || 0) - (s.eyeLookDownLeft || 0) - (s.eyeLookDownRight || 0)) / 2;
    f.blink = Math.max(s.eyeBlinkLeft || 0, s.eyeBlinkRight || 0);
  }
  return f;
}
// 兩種模型：眼睛＋頭（比較準），只用頭（眼鏡反光、眼睛特徵不穩時比較可靠）；特徵少一點，避免只記住校正點
const FEATS = {
  eye: f => [f.yaw, f.pitch, f.tx, f.ty, f.bx, f.by, f.ix, f.iy, f.yaw * f.yaw, f.pitch * f.pitch],
  head: f => [f.yaw, f.pitch, f.tx, f.ty, f.yaw * f.yaw, f.pitch * f.pitch, f.yaw * f.pitch],
};
function solve(A, b) {
  const n = b.length; A = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; const d = A[c][c] || 1e-9;
    for (let r = 0; r < n; r++) if (r !== c) { const k = A[r][c] / d; for (let j = c; j <= n; j++) A[r][j] -= k * A[c][j]; }
  }
  return A.map((r, i) => r[n] / (r[i] || 1e-9));
}
function fitRaw(samples, mode, lam) {
  const rows = samples.map(s => FEATS[mode](s.f)), m = rows[0].length;
  const mu = Array.from({ length: m }, (_, j) => rows.reduce((a, r) => a + r[j], 0) / rows.length);
  const sd = mu.map((u, j) => Math.sqrt(rows.reduce((a, r) => a + (r[j] - u) ** 2, 0) / rows.length) || 1);
  const Z = rows.map(r => [1, ...r.map((v, j) => (v - mu[j]) / sd[j])]), k = m + 1;
  const A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => Z.reduce((a, z) => a + z[i] * z[j], 0) + (i === j && i ? lam * rows.length : 0)));
  const bx = Array.from({ length: k }, (_, i) => Z.reduce((a, z, n) => a + z[i] * samples[n].tx, 0));
  const by = Array.from({ length: k }, (_, i) => Z.reduce((a, z, n) => a + z[i] * samples[n].ty, 0));
  return { mode, mu, sd, wx: solve(A, bx), wy: solve(A.map(r => r.slice()), by), sx: [1, 0], sy: [1, 0] };
}
const rawPredict = (M, f) => { const z = [1, ...FEATS[M.mode](f).map((v, j) => (v - M.mu[j]) / M.sd[j])]; return [z.reduce((a, v, i) => a + v * M.wx[i], 0), z.reduce((a, v, i) => a + v * M.wy[i], 0)]; };
function predict(M, f) { const [x, y] = rawPredict(M, f); return [M.sx[0] * x + M.sx[1], M.sy[0] * y + M.sy[1]]; }
// 去縮小：迴歸會把預測點往中間拉（看邊邊時點不到邊邊），用校正點把範圍拉回來
const lin = (xs, ys) => { const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n, v = xs.reduce((a, x) => a + (x - mx) ** 2, 0); const a = v > 1e-9 ? xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / v : 1; return [Math.max(.8, Math.min(2.2, a)), my - Math.max(.8, Math.min(2.2, a)) * mx]; };
function fitModel(samples, mode, lam = .15) {
  const M = fitRaw(samples, mode, lam), pts = [...new Set(samples.map(s => s.k))];
  const P = pts.map(k => { const ss = samples.filter(s => s.k === k), p = ss.map(s => rawPredict(M, s.f)); return { tx: ss[0].tx, ty: ss[0].ty, px: p.reduce((a, q) => a + q[0], 0) / p.length, py: p.reduce((a, q) => a + q[1], 0) / p.length }; });
  M.sx = lin(P.map(p => p.px), P.map(p => p.tx)); M.sy = lin(P.map(p => p.py), P.map(p => p.ty));
  return M;
}
// 留一點驗證：拿掉一個校正點來訓練，再預測那一點，算平均誤差（畫面對角線的比例）
function lopoError(samples, mode) {
  const pts = [...new Set(samples.map(s => s.k))]; if (pts.length < 5) return 9;
  let sum = 0;
  for (const k of pts) {
    const tr = samples.filter(s => s.k !== k), te = samples.filter(s => s.k === k), M = fitModel(tr, mode);
    const p = te.map(s => predict(M, s.f)), px = p.reduce((a, q) => a + q[0], 0) / p.length, py = p.reduce((a, q) => a + q[1], 0) / p.length;
    sum += Math.hypot((px - te[0].tx) * innerWidth, (py - te[0].ty) * innerHeight);
  }
  return sum / pts.length / Math.hypot(innerWidth, innerHeight);
}
function chooseModel() {
  const s = EYE.samples; if (s.length < 40 || new Set(s.map(x => x.k)).size < 6) return false;
  const errs = { eye: lopoError(s, 'eye'), head: lopoError(s, 'head') };
  const mode = S.set.eye.mode === 'auto' ? (errs.eye <= errs.head * 1.08 ? 'eye' : 'head') : S.set.eye.mode;
  EYE.model = fitModel(s, mode); EYE.err = errs[mode]; EYE.errs = errs; EYE.ready = true; EYE.bias = [0, 0];
  return true;
}
// 每個校正點去掉眨眼、突然晃動的畫面（取中位數附近的樣本）
function robust(ss) {
  if (ss.length < 6) return ss;
  const med = k => { const v = ss.map(s => s.f[k]).sort((a, b) => a - b); return v[v.length >> 1]; };
  const keys = ['yaw', 'pitch', 'bx', 'by'], m = Object.fromEntries(keys.map(k => [k, med(k)]));
  const mad = Object.fromEntries(keys.map(k => { const v = ss.map(s => Math.abs(s.f[k] - m[k])).sort((a, b) => a - b); return [k, v[v.length >> 1] || 1e-4]; }));
  const keep = ss.filter(s => keys.every(k => Math.abs(s.f[k] - m[k]) <= 3 * mad[k]));
  return keep.length >= ss.length * .5 ? keep : ss;
}
// 平滑（One Euro 濾波）：看著不動時很穩，眼睛移動時跟得上；「加強穩定」給頭部或眼睛會不自主晃動的孩子
class OneEuro {
  constructor() { this.x = null; this.dx = 0; this.t = 0; }
  f(x, t) {
    const [minC, beta] = S.set.eye.steady ? [.25, .0012] : [.5, .003];
    if (this.x === null) { this.x = x; this.t = t; return x; }
    const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
    const a = c => 1 / (1 + 1 / (2 * Math.PI * c * dt));
    const dx = (x - this.x) / dt; this.dx += a(1) * (dx - this.dx);
    this.x += a(minC + beta * Math.abs(this.dx)) * (x - this.x); return this.x;
  }
}
const FX = new OneEuro(), FY = new OneEuro();
EYE.bias = [0, 0];
function onFrame(f) {
  const now = performance.now();
  EYE.face = !!f; $('#eye-cam').classList.toggle('face', !!f);
  if (!f) return;
  EYE.lastFace = now; EYE.last = f;
  if (f.blink > .55) return;               // 眨眼的畫面不用
  EYE.recent.push(f); if (EYE.recent.length > 4) EYE.recent.shift();
  const avg = {}; for (const k in f) avg[k] = EYE.recent.reduce((a, r) => a + r[k], 0) / EYE.recent.length;
  if (EYE.collect) EYE.collect(avg);
  if (!EYE.model) return;
  const [px, py] = predict(EYE.model, avg);
  EYE.x = Math.max(0, Math.min(innerWidth, FX.f((px + EYE.bias[0]) * innerWidth, now)));
  EYE.y = Math.max(0, Math.min(innerHeight, FY.f((py + EYE.bias[1]) * innerHeight, now)));
}

/* ---------- 鏡頭與追蹤模型 ---------- */
let FL = null;
async function startCamera() {
  const st = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  const v = $('#eye-cam'); v.srcObject = st; await v.play().catch(() => {}); return v;
}
async function loadTracker() {
  const { FaceLandmarker, FilesetResolver } = await import(MP + '/vision_bundle.mjs');
  const fs = await FilesetResolver.forVisionTasks(MP + '/wasm');
  const opt = d => ({ baseOptions: { modelAssetPath: MP_MODEL, delegate: d }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
  try { FL = await FaceLandmarker.createFromOptions(fs, opt('GPU')); } catch (e) { FL = await FaceLandmarker.createFromOptions(fs, opt('CPU')); }
}
function trackLoop() {
  const v = $('#eye-cam'); let last = -1;
  const step = () => {
    try {
      if (v.readyState >= 2 && v.currentTime !== last) {
        last = v.currentTime;
        const r = FL.detectForVideo(v, performance.now()), L = r.faceLandmarks && r.faceLandmarks[0];
        onFrame(L && L.length >= 478 ? eyeFeatures(L, r.faceBlendshapes && r.faceBlendshapes[0], r.facialTransformationMatrixes && r.facialTransformationMatrixes[0]) : null);
      }
    } catch (e) {}
    requestAnimationFrame(step);
  };
  step();
}
// 模擬模式（?sim=1）：沒有鏡頭時用滑鼠或測試程式當作眼睛，用來檢查整套流程
function simLoop() {
  setInterval(() => {
    if (window.__noface) return onFrame(null);
    const g = window.__gaze || EYE.mouse, u = g.x / innerWidth - .5, v = g.y / innerHeight - .5, n = s => (Math.random() - .5) * s;
    const g2 = SIM_GLARE ? 40 : 1;  // ?glare=1：模擬眼鏡反光（眼睛特徵很亂）
    onFrame({ ix: .22 * u + n(.02 * g2), iy: .05 * v + n(.03 * g2), yaw: n(.006), pitch: n(.006), tx: n(.002), ty: n(.002), bx: .6 * u + n(.06 * g2), by: -.2 * v + n(.08 * g2), size: .32, blink: 0 });
  }, 33);
}

/* ---------- 擺位檢查：臉在框框中間、距離剛好、光線夠亮 ---------- */
const lumaCv = document.createElement('canvas'); lumaCv.width = 32; lumaCv.height = 24;
function luma() { try { const c = lumaCv.getContext('2d', { willReadFrequently: true }); c.drawImage($('#eye-cam'), 0, 0, 32, 24); const d = c.getImageData(0, 0, 32, 24).data; let s = 0; for (let i = 0; i < d.length; i += 4) s += .3 * d[i] + .59 * d[i + 1] + .11 * d[i + 2]; return s / (d.length / 4); } catch (e) { return 128; } }
async function positionCheck() {
  if (EYE_SIM && !/[?&]pos=1/.test(location.search)) return;
  const v = $('#eye-cam'); v.classList.remove('hide'); v.classList.add('big'); $('#pos-oval').hidden = false; $('#pos-msg').hidden = false;
  $('#eye-boot').classList.add('on', 'pos'); bootStep('face', '擺位檢查：請把臉放進框框裡', '');
  talk('請把臉放進框框裡，坐好。');
  let good = 0; const t0 = performance.now();
  while (performance.now() - t0 < 25000) {
    await wait(250);
    const f = EYE.last, fresh = performance.now() - EYE.lastFace < 400, L = luma();
    let msg = '很好！保持這個姿勢', ok = true;
    if (!fresh || !f) { msg = '看不到臉：請面向鏡頭'; ok = false; }
    else if (L < 55) { msg = '光線太暗：請打開燈，或讓光從前面照到臉'; ok = false; }
    else if (f.size < .16) { msg = '太遠了：請靠近鏡頭一點'; ok = false; }
    else if (f.size > .6) { msg = '太近了：請離鏡頭遠一點'; ok = false; }
    else if (Math.abs(f.tx) > .15 || Math.abs(f.ty) > .18) { msg = '請把臉移到框框中間'; ok = false; }
    $('#pos-msg').textContent = msg; $('#pos-msg').className = ok ? 'ok' : '';
    good = ok ? good + 1 : 0;
    if (good >= 6) break;
  }
  v.classList.remove('big'); $('#pos-oval').hidden = true; $('#pos-msg').hidden = true; v.classList.toggle('hide', !S.set.eye.cam);
  $('#eye-boot').classList.remove('on', 'pos');
}

/* ---------- 校正：星星跑 9 個位置，孩子只要看著 ---------- */
const CAL_PTS = [[.5, .5], [.08, .1], [.5, .08], [.92, .1], [.92, .5], [.92, .9], [.5, .92], [.08, .9], [.08, .5], [.29, .5], [.71, .5]];
async function calibrate(again = false) {
  EYE.ready = false; EYE.model = null; FX.x = FY.x = null; EYE.bias = [0, 0];
  await positionCheck();
  const lay = $('#eye-cal'), star = $('#cal-star'); lay.classList.add('on'); $('#cal-text').hidden = false;
  star.style.transform = `translate(${innerWidth / 2}px,${innerHeight / 2}px)`;
  await Promise.race([talk(again ? '我們再看一次星星。頭不要動，只用眼睛看。' : '頭不要動，只用眼睛看著星星。星星跑到哪裡，眼睛就看哪裡！', 'happy'), wait(9000)]);
  $('#cal-text').hidden = true;
  EYE.samples = [];
  for (const [k, [px, py]] of CAL_PTS.entries()) {
    const t = `translate(${px * innerWidth}px,${py * innerHeight}px)`; star.style.setProperty('--t', t); star.style.transform = t;
    await wait(1200);
    const ss = []; const t0 = performance.now();
    EYE.collect = f => ss.push({ f, tx: px, ty: py, k });
    while (ss.length < 30 && performance.now() - t0 < 8000) await wait(40);
    EYE.collect = null; EYE.samples.push(...robust(ss));
    star.classList.remove('grab'); void star.offsetWidth; star.classList.add('grab');
  }
  lay.classList.remove('on');
  if (!chooseModel()) { await Promise.race([talk('豆豆看不太到你的眼睛。請面向鏡頭，我們再看一次星星。'), wait(8000)]); return calibrate(true); }
  S.set.eye.last = { err: EYE.err, mode: EYE.model.mode, d: Date.now() }; save();
  return calResult();
}
// 校正分數：看「開始玩」或「再校正一次」，10 秒沒選就直接開始
const calScore = () => Math.max(0, Math.min(100, Math.round(100 - EYE.err * 420)));
function calResult() {
  return new Promise(res => {
    const sc = calScore(), lv = sc >= 75 ? '很準 😄' : sc >= 50 ? '可以 🙂' : '不太準 😕';
    $('#calres-score').textContent = `${sc} 分　${lv}`;
    $('#calres-mode').textContent = '';
    $('#calres-tip').textContent = sc >= 75 ? '' : '小提醒：頭靠好不要動、臉正對鏡頭、光線從前面照；筆電螢幕往後仰一點或把筆電墊高，讓鏡頭和眼睛一樣高；眼鏡擦乾淨。';
    openOv('ov-calres');
    talk(sc >= 50 ? `校正好了，${sc}分！看著「開始玩」就可以開始。` : `校正只有${sc}分，不太準。可以看著「再校正一次」，或請家人調整位置。`, 'happy');
    let done = false; const fin = v => { if (done) return; done = true; closeOv('ov-calres'); res(v); };
    $('#calres-go').onclick = () => fin('go');
    $('#calres-again').onclick = () => fin('again');
    setTimeout(() => fin('go'), 12000);
  }).then(v => v === 'again' ? calibrate(true) : v);
}

/* ---------- 注視停留選取 ---------- */
let TG = [], TG_T = 0;
function gazeLayer() { const ovs = [...document.querySelectorAll('.ov.on')]; return ovs[ovs.length - 1] || document.querySelector('.screen.on'); }
function gazeTargets() {
  const L = gazeLayer(); if (!L) return [];
  return [...L.querySelectorAll('button,[data-gaze]')].filter(n => {
    if (n.disabled || n.classList.contains('found') || n.closest('[hidden],.busy,[data-nogaze]')) return false;   // 用過的（找到的、送走的）不再當目標
    const r = n.getBoundingClientRect(); if (r.width < 8 || r.height < 8 || r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) return false;
    const cs = getComputedStyle(n); return cs.visibility !== 'hidden' && cs.pointerEvents !== 'none' && (+cs.opacity > .25 || n.getAnimations().some(x => x.playState === 'running'));   // 淡出完的看不見，不算
  });
}
// 「一定要看準」的按鈕（回首頁、設定、換人、再聽一次）：眼神要在按鈕上才算
const TIGHT = '#g-back,#h-who,#eye-btn-set,#eye-btn-cal,.again,.help-x';
const DW = new Map(); let FOCUS = null, COOL = 0, LAST_T = performance.now();
const dwellOf = n => S.set.eye.dwell * (+n.dataset.dwell || (n.classList.contains('tapme') ? .7 : 1));
function gazeTick(now) {
  const dt = Math.min(.1, (now - LAST_T) / 1000); LAST_T = now;
  const cz = $('#gz-cursor'), ring = $('#gz-ring');
  const blocked = !EYE.ready || $('#eye-cal').classList.contains('on') || $('#eye-boot').classList.contains('on');
  const noFace = EYE.ready && now - EYE.lastFace > 1500;
  $('#eye-noface').hidden = !noFace || blocked;
  cz.classList.toggle('off', blocked || !S.set.eye.cursor || noFace);
  cz.style.transform = `translate(${EYE.x}px,${EYE.y}px)`;
  if (blocked || noFace || now < COOL) { setFocus(null); ring.style.display = 'none'; return; }
  if (now - TG_T > 120) { TG = gazeTargets(); TG_T = now; }
  const loose = TG.filter(n => !n.matches(TIGHT)), few = loose.length <= 4;
  const ys = loose.map(n => { const r = n.getBoundingClientRect(); return r.top + r.height / 2; });
  const VW = loose.length > 1 && Math.max(...ys) - Math.min(...ys) < innerHeight * .2 ? 0 : .5;
  let best = null, bd = 1e9, ba = 1e12;
  for (const n of TG) {
    const tight = n.matches(TIGHT), r = n.getBoundingClientRect(), dx = Math.max(r.left - EYE.x, 0, EYE.x - r.right), dy = Math.max(r.top - EYE.y, 0, EYE.y - r.bottom), d = Math.hypot(dx, dy * (tight ? 1 : VW)), a = r.width * r.height;
    if (tight && d > 24) continue;
    if (d < bd || (d === bd && a < ba)) { bd = d; ba = a; best = n; }
  }
  // 選項不多時，看大概的方向就會選最近的那一個；選項多時要看近一點
  const R = few ? 1e9 : Math.max(90, .15 * Math.min(innerWidth, innerHeight));
  const f = best && bd <= R ? best : null;
  setFocus(f);
  for (const [n, p] of DW) { const q = n === f ? p + dt / dwellOf(n) : p - dt / 1.2; if (q <= 0 || !n.isConnected) DW.delete(n); else DW.set(n, q); }
  if (f && !DW.has(f)) DW.set(f, dt / dwellOf(f));
  const p = f ? DW.get(f) || 0 : 0;
  if (f && p > 0) {
    const r = f.getBoundingClientRect(), br = parseFloat(getComputedStyle(f).borderTopLeftRadius) || 16, g = 14;
    Object.assign(ring.style, { display: 'block', width: r.width + 2 * g + 'px', height: r.height + 2 * g + 'px', left: r.left - g + 'px', top: r.top - g + 'px', borderRadius: Math.min(br + g, (Math.min(r.width, r.height) + 2 * g) / 2) + 'px' });
    ring.style.setProperty('--p', Math.min(1, p).toFixed(3));
  } else ring.style.display = 'none';
  if (f && p >= 1) { DW.clear(); COOL = now + 750; ring.style.display = 'none'; learn(f); activate(f); }
}
function setFocus(n) { if (FOCUS === n) return; if (FOCUS) FOCUS.classList.remove('gz-on'); FOCUS = n; if (n) n.classList.add('gz-on'); }
function activate(n) { n.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window })); }
// 邊玩邊修正：選到一個目標時，孩子其實是看著它的中間，把偏差慢慢修正回來
function learn(n) {
  if (!EYE.model || !(n instanceof HTMLButtonElement)) return;   // 圖形區塊（塗色）不知道孩子看的是哪一點，不拿來修正
  const r = n.getBoundingClientRect(); if (r.width > innerWidth * .6 || r.height > innerHeight * .6) return;
  const ex = (r.left + r.width / 2 - EYE.x) / innerWidth, ey = (r.top + r.height / 2 - EYE.y) / innerHeight;
  EYE.bias = [Math.max(-.15, Math.min(.15, EYE.bias[0] + .25 * ex)), Math.max(-.15, Math.min(.15, EYE.bias[1] + .25 * ey))];
}

/* ---------- 畫面：選人、年齡、首頁、設定 ---------- */
function renderWho() {
  if (!S.profiles.length) return renderNew();
  $('#who-mascot').innerHTML = mascot(130, 'happy');
  const list = $('#who-list'); list.innerHTML = '';
  S.profiles.forEach((p, i) => {
    const b = el(`<button class="who-btn" style="animation-delay:${i * 70}ms"><span class="av">${p.av}</span><span>${esc(p.name)}</span><small>${AGES[p.age].label}</small></button>`);
    b.onclick = () => { S.cur = p.id; save(); goHome(); };
    list.appendChild(b);
  });
  const add = el(`<button class="who-btn"><span class="av" style="opacity:.5">＋</span><span>新增</span><small>小朋友</small></button>`);
  add.onclick = renderNew; list.appendChild(add);
  show('s-who');
  later0(() => talk('誰要來玩呢？用眼睛看著你的頭像。'), 300);
}
function later0(fn, ms) { setTimeout(fn, ms); }
function renderNew() {
  $('#eyenew-mascot').innerHTML = mascot(110, 'happy');
  const box = $('#eyenew-ages'); box.innerHTML = '';
  Object.entries(AGES).forEach(([k, v]) => {
    const b = el(`<button class="opt"><span class="emo">${v.e}</span><small>${v.label}</small></button>`);
    b.onclick = () => {
      const p = { id: Date.now().toString(36) + rnd(1e4).toString(36), name: '小朋友', av: pick(AVATARS), age: k, stars: 0, prog: {} };
      S.profiles.push(p); S.cur = p.id; save(); closeOv('ov-eyenew'); goHome();
    };
    box.appendChild(b);
  });
  openOv('ov-eyenew');
  later0(() => talk('你幾歲呢？用眼睛看著答案。'), 300);
}
let HOME_PAGE = 0;
function goHome() {
  stopGame();
  const p = cur(); if (!p) return renderWho();
  $('#h-av').textContent = p.av; $('#h-name').textContent = p.name; $('#h-stars').textContent = p.stars;
  $('#h-mascot').innerHTML = mascot(100, 'normal');
  $('#h-say').textContent = '今天想玩什麼？用眼睛看著它';
  const tiles = $('#tiles'); tiles.innerHTML = ''; tiles.classList.add('one-row');
  const all = Object.entries(D), page = all.slice(HOME_PAGE * 3, HOME_PAGE * 3 + 3);
  page.forEach(([k, d], i) => {
    const t = el(`<button class="tile" style="--c:${d.color};--cs:var(--${k}-s);animation-delay:${i * 60}ms"><div class="ico">${d.icon}</div><span>${d.title[p.age]}</span><small>${d.name}</small></button>`);
    t.onclick = () => startGame(k);
    tiles.appendChild(t);
  });
  const nx = el(`<button class="tile next-tile" style="--c:var(--soft);--cs:var(--card)"><div class="ico">➡️</div><span>下一頁</span><small>${HOME_PAGE + 1} / 2</small></button>`);
  nx.onclick = () => { HOME_PAGE = (HOME_PAGE + 1) % 2; goHome(); };
  tiles.appendChild(nx);
  if (!$('#eye-bar')) {
    tiles.insertAdjacentHTML('afterend', `<div class="eye-bar" id="eye-bar"><button class="btn" id="eye-btn-set" data-nogaze style="--dc:var(--soc);--dcs:var(--soc-s)">👀 眼睛設定（家長）</button><button class="btn" id="eye-btn-cal" data-nogaze style="--dc:var(--emo);--dcs:var(--emo-s)">⭐ 重新校正（家長）</button></div>`);
    $('#eye-btn-set').onclick = openEyeSet;
    $('#eye-btn-cal').onclick = async () => { await calibrate(); goHome(); };
  }
  show('s-home');
}
function openEyeSet() {
  const E = S.set.eye, box = $('#eye-set');
  const row = (title, key, opts) => `<div class="row"><b>${title}</b>${opts.map(([v, t]) => `<button class="btn ${E[key] === v ? 'sel' : ''}" data-k="${key}" data-v="${v}">${t}</button>`).join('')}</div>`;
  box.innerHTML = row('看多久才選取', 'dwell', [[1.2, '快 1.2 秒'], [1.8, '標準 1.8 秒'], [2.5, '慢 2.5 秒']])
    + row('穩定度', 'steady', [[false, '一般'], [true, '加強（頭或眼睛會晃）']])
    + row('注視點', 'cursor', [[true, '顯示'], [false, '隱藏']])
    + row('鏡頭小視窗', 'cam', [[true, '顯示'], [false, '隱藏']])
    + `<div class="row"><button class="btn" id="eye-set-cal" style="--dc:var(--emo);--dcs:var(--emo-s)">⭐ 重新校正</button><button class="btn" id="eye-set-ok" style="--dc:var(--move);--dcs:var(--move-s)">完成</button></div>`
    + `<p class="muted" style="text-align:center">${EYE.model ? `目前校正分數：${calScore()} 分` : '尚未校正'}</p>`;
  box.querySelectorAll('[data-k]').forEach(b => b.onclick = () => {
    const k = b.dataset.k, raw = b.dataset.v, v = raw === 'true' ? true : raw === 'false' ? false : isNaN(+raw) ? raw : +raw;
    E[k] = v; save();
    if (k === 'mode') chooseModel();
    if (k === 'cam') $('#eye-cam').classList.toggle('hide', !v || EYE_SIM);
    openEyeSet();
  });
  $('#eye-set-cal').onclick = async () => { closeOv('ov-eye'); await calibrate(); goHome(); };
  $('#eye-set-ok').onclick = () => closeOv('ov-eye');
  openOv('ov-eye');
}
// 不讓眼睛誤觸：回首頁要看久一點；家長用的按鈕只能用手點
$('#g-back').dataset.dwell = '2';
$('#h-who').dataset.dwell = '1.6';
['#h-parent', '#break-skip', '#limit-more'].forEach(s => $(s) && ($(s).dataset.nogaze = ''));
document.querySelectorAll('.help-btn').forEach(b => b.dataset.nogaze = '');

/* ---------- 安全：不使用閃光（閃爍可能引發光敏感癲癇） ---------- */
const TH = EVENTS.find(e => e.k === 'thunder');
if (TH) TH.play = async sc => { const c = addProp(sc, '⛈️', '64%', '6%'); for (let k = 0; k < 2; k++) { c.animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-8px)' }, { transform: 'translateX(8px)' }, { transform: 'translateX(0)' }], { duration: 400 }); sfx.rumble(); await sleep(900); } };

/* ---------- 遊戲名稱與陪玩提示 ---------- */
Object.assign(D.cog.title, { '2-3': '找找好朋友', '3-4': '看看數一數', '4-6': '規律小火車' });
Object.assign(D.lang.title, { '2-3': '猜猜我是誰', '3-4': '眼睛翻翻牌', '4-6': '注音找一找' });
Object.assign(D.move, { name: '身體動作與健康（眼睛體操）', title: { '2-3': '豆豆躲貓貓', '3-4': '星星追追追', '4-6': '眼睛聽口令' } });
Object.assign(D.soc.title, { '2-3': '和豆豆輪流', '3-4': '分享小點心', '4-6': '和豆豆合作' });
Object.assign(D.art.title, { '2-3': '眼睛畫畫', '3-4': '星星連連看', '4-6': '眼睛塗顏色' });

/* ---------- 遊戲（全部只要用眼睛看） ---------- */
const eyeOpts = (box, list) => choose(box, list);
// 認知 2–3：用眼睛找指定顏色／形狀的好朋友
GAMES['cog2-3'] = dom => rounds(dom, 5, async (i, lv) => {
  const n = lv === 0 ? 2 : 3, kinds = Object.keys(SHAPE_N);
  const { mode, tc, ts } = fresh(() => { const mode = lv < 2 ? 'color' : lv === 2 ? 'shape' : pick(['color', 'shape', 'both']), tc = pick(COLORS), ts = pick(kinds); return { mode, tc, ts, key: `${mode}|${mode === 'shape' ? '' : tc.n}|${mode === 'color' ? '' : ts}|${n}` }; });
  const isT = b => (mode === 'shape' || b.c === tc) && (mode === 'color' || b.k === ts);
  const nm = b => mode === 'shape' ? SHAPE_N[b.k] : mode === 'color' ? b.c.n : b.c.n + SHAPE_N[b.k];
  const label = mode === 'color' ? tc.n : mode === 'shape' ? SHAPE_N[ts] : tc.n + SHAPE_N[ts];
  const tgt = { c: mode === 'shape' ? pick(COLORS) : tc, k: mode === 'color' ? pick(kinds) : ts }, list = [tgt];
  while (list.length < n) {
    const o = mode === 'color' ? { c: pick(COLORS.filter(c => c !== tc)), k: pick(kinds) } : mode === 'shape' ? { c: pick(COLORS), k: pick(kinds.filter(k => k !== ts)) } : (Math.random() < .5 ? { c: tc, k: pick(kinds.filter(k => k !== ts)) } : { c: pick(COLORS.filter(c => c !== tc)), k: ts });
    if (!list.some(x => nm(x) === nm(o))) list.push(o);
  }
  const ico = mode === 'color' ? `<span class="splash" style="background:${tc.c}"></span>` : buddy(ts, mode === 'shape' ? '#B9B2CC' : tc.c, 56);
  const wrap = body(`<div class="eye-sign">${ico}${label}</div><div class="q-opts"></div>`);
  await prompt(`${label}的好朋友在哪裡？用眼睛看著它！`); setQ(`找${label}的好朋友（${n} 選 1）`);
  await eyeOpts(wrap.querySelector('.q-opts'), shuffle(list).map(b => ({ html: buddy(b.k, b.c.c, 150), ok: isT(b), say: isT(b) ? `找到了！${label}的好朋友！` : `這是${nm(b)}` })));
  wrap.remove();
});
// 認知 3–4：眼睛一個一個看，看到的就亮起來、豆豆數出來，再選數字
GAMES['cog3-4'] = dom => rounds(dom, 5, async (i, lv) => {
  const { n, t } = fresh(() => { const [a, b] = lvl(lv, [[2, 3], [3, 4], [4, 5], [5, 6]]), n = a + rnd(b - a + 1), t = pick(THINGS); return { n, t, key: t.e + n }; });
  const wrap = body(`<div class="scene meadow eye-count one-line" style="min-height:min(40vh,340px)"><div class="hill"></div><div class="count-box"></div></div><div class="q-opts"></div>`);
  const sc = wrap.querySelector('.scene'), box = wrap.querySelector('.count-box'), opts = wrap.querySelector('.q-opts');
  let c = 0;
  const counted = new Promise(res => {
    for (let k = 0; k < n; k++) {
      const it = el(`<button class="cnt" data-dwell=".5" style="animation-delay:${k * 90}ms">${t.e}</button>`);
      it.onclick = () => { if (it.classList.contains('done')) return; c++; it.classList.add('done'); it.disabled = true; it.insertAdjacentHTML('beforeend', `<b>${c}</b>`); jump(it); sayNum(c); bigNum(sc, c); if (c === n) later(res, 1100); };
      box.appendChild(it);
    }
  });
  await prompt(`這裡有好多${t.n}。眼睛一個一個看，看到的會亮起來，我們一起數！`); setQ(`看看數一數：${n} 個 ${t.e}`);
  await counted;
  await prompt(`一共有幾個${t.n}呢？用眼睛看著正確的數字！`);
  const set = new Set([n]); while (set.size < (lv === 0 ? 2 : 3)) set.add(Math.max(1, Math.min(9, n + pick([-2, -1, 1, 2]))));
  await eyeOpts(opts, shuffle([...set]).map(v => ({ html: `<b class="num">${v}</b>`, ok: v === n, say: v === n ? `對！一共有${n}個${t.n}！` : `這是${v}` })));
  wrap.remove();
});
// 認知 4–6：規律小火車，用眼睛選下一節乘客
const EYE_PATS = [['AB', 'AAB', 'ABB'], ['ABC', 'AABB', 'ABB'], ['ABC', 'AABB', 'ABAC', 'AABC'], ['ABBC', 'AABC', 'ABAC', 'ABCC', 'ABC']];
GAMES['cog4-6'] = dom => rounds(dom, 5, async (i, lv) => {
  const { p, it, mid } = fresh(() => { const p = pick(lvl(lv, EYE_PATS)), it = sample(PE, 4), mid = lv >= 3 && Math.random() < .5; return { p, it, mid, key: p + it.slice(0, 3).join('') + (mid ? 'm' : '') }; });
  const m = { A: it[0], B: it[1], C: it[2] }, SW = $('#stage').clientWidth, L = SW < 560 ? Math.max(4, p.length + 2) : Math.min(7, p.length * 2 + 1);
  const full = Array.from({ length: L + 1 }, (_, k) => m[p[k % p.length]]), slotAt = mid ? p.length + rnd(L - p.length) : L, ans = full[slotAt];
  const cw = Math.max(40, Math.min(80, (SW - 34) / (L + 2.3) - 6));
  const used = [...new Set(p.split('').map(x => m[x]))], pool = shuffle(used.length < 3 ? [...used, it[3]] : used);
  const wrap = body(`<div class="scene sky" style="min-height:230px">${SKY}<div class="hill"></div><div class="track"></div><div class="train" style="--cw:${cw}px"><div class="car engine"><i></i><span class="smoke"></span>🙂</div>${full.map((e, k) => k === slotAt ? '<div class="car slot">?</div>' : `<div class="car" style="--cc:${CAR_C[k % 7]}">${e}</div>`).join('')}</div></div><div class="q-opts"></div>`);
  const train = wrap.querySelector('.train'), slot = wrap.querySelector('.slot');
  train.animate([{ transform: 'translateX(110%)' }, { transform: 'translateX(0)' }], { duration: 1200, easing: 'cubic-bezier(.2,.9,.3,1)' });
  await sleep(1200);
  await prompt(mid ? '火車少了一節車廂。問號車廂要坐誰？用眼睛看著它！' : '看看車廂的規律。下一節要坐誰？用眼睛看著它！');
  setQ(`規律 ${full.map((e, k) => k === slotAt ? '❓' : e).join('')}（答案 ${ans}）`);
  await eyeOpts(wrap.querySelector('.q-opts'), pool.map(e => ({ html: `<span class="emo">${e}</span>`, ok: e === ans, say: e === ans ? '坐上車了！' : '不是這個' })));
  slot.classList.remove('slot'); slot.style.setProperty('--cc', CAR_C[slotAt % 7]); slot.textContent = ans; burst(slot);
  await train.animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-130%)' }], { duration: 1300, easing: 'ease-in', fill: 'forwards' }).fin;
  wrap.remove();
});
// 語文 2–3：聽叫聲找動物、聽名字找動物、跟著學叫聲
GAMES['lang2-3'] = dom => rounds(dom, 5, async (i, lv) => {
  const { mode, a, n } = fresh(() => { const mode = lv <= 1 ? 'sound' : lv === 2 ? 'name' : lv === 3 ? 'copy' : pick(['sound', 'name', 'copy']), a = pick(ANI), n = mode === 'copy' ? 0 : lv === 0 ? 2 : 3; return { mode, a, n, key: mode + n + a.k }; });
  const opts = shuffle([a, ...sample(ANI.filter(x => x !== a), Math.max(0, n - 1))]);
  const wrap = body(`<div class="scene meadow" style="min-height:260px">${SKY}<div class="hill"></div><div class="hide-spot ${mode === 'copy' ? 'reveal' : ''}"><div class="animal">${a.e}</div>${BUSH}</div></div><div class="q-opts"></div>`);
  const spot = wrap.querySelector('.hide-spot'), box = wrap.querySelector('.q-opts');
  const call = async () => { spot.classList.add('calling'); await playAnimal(a.k); spot.classList.remove('calling'); };
  if (mode === 'sound') {
    await prompt('草叢裡躲著一隻動物。仔細聽牠的叫聲，猜猜牠是誰？'); setQ(`聽叫聲猜動物（答案：${a.n}）`);
    await call(); await sleep(500); await call(); await sleep(300);
    await prompt('是誰在叫呢？用眼睛看著牠！');
    await eyeOpts(box, opts.map(w => ({ html: `<span class="emo">${w.e}</span><small>${w.n}</small>`, ok: w === a, say: w === a ? `答對了！是${a.n}！` : `這是${w.n}` })));
    spot.classList.add('reveal'); await playAnimal(a.k);
  } else if (mode === 'name') {
    await prompt(`${a.n}在哪裡？用眼睛看著${a.n}！`); setQ(`聽名字找動物（答案：${a.n}）`);
    await eyeOpts(box, opts.map(w => ({ html: `<span class="emo">${w.e}</span>`, ok: w === a, say: w === a ? `對！這是${a.n}！` : `這是${w.n}` })));
    spot.classList.add('reveal'); await playAnimal(a.k);
  } else {
    await prompt(`這是${a.n}。聽聽${a.n}怎麼叫！`); setQ(`聽叫聲：${a.n}`);
    await call(); await sleep(500); await call(); await sleep(300);
    await prompt(`喜歡${a.n}的叫聲嗎？喜歡就看著愛心！`, 'happy');
    await eyeOpts(box, [{ html: '<span class="emo">💗</span><small>喜歡</small>', ok: true, say: `${a.n}也喜歡你！` }, { html: '<span class="emo">🔁</span><small>再聽一次</small>', ok: true, say: '再聽一次！' }]).then(o => o.say === '再聽一次！' ? call() : 0);
  }
  await sleep(500); wrap.remove();
});
// 語文 3–4：眼睛翻翻牌（看著牌就翻開，找出一樣的叫聲）
GAMES['lang3-4'] = dom => rounds(dom, 3, async (i, lv, k) => {
  const pairs = lvl(lv, [[2, 2, 2], [2, 3, 3], [3, 3, 3]])[k];  // 一排最多 6 張
  const { an } = fresh(() => { const an = sample(ANI, pairs); return { an, key: an.map(a => a.k).sort().join() }; });
  const deck = shuffle([...an, ...an]), cols = deck.length;
  const wrap = body(`<div class="scene sky" style="min-height:auto;padding:16px 8px"><div class="cards" style="grid-template-columns:repeat(${cols},auto)"></div></div>`);
  const grid = wrap.querySelector('.cards');
  let open = [], matched = 0, busy = false;
  const done = new Promise(res => deck.forEach((a, j) => {
    const c = el(`<button class="flip" data-k="${a.k}" data-dwell=".7" aria-label="翻牌"><div class="in"><div class="face back"></div><div class="face front">${a.e}</div></div></button>`);
    c.onclick = async () => {
      if (busy || c.classList.contains('open') || c.classList.contains('matched')) return;
      c.classList.add('open'); c.disabled = true; playAnimal(a.k); open.push({ c, a });
      if (open.length < 2) return;
      busy = true; const [x, y] = open; open = [];
      await sleep(1500);
      if (x.a === y.a) { x.c.classList.add('matched'); y.c.classList.add('matched'); burst(x.c); burst(y.c); praise(`一樣！都是${x.a.n}！`); if (++matched === pairs) { await sleep(1600); res(); } }
      else { oops('不一樣'); await sleep(700); [x, y].forEach(o => { o.c.classList.remove('open'); o.c.disabled = false; }); }
      busy = false;
    };
    grid.appendChild(c);
  }));
  await prompt(`用眼睛看著一張牌，牌就會翻開。找出${pairs}對一樣的叫聲！`); setQ(`眼睛翻翻牌 ${pairs} 對：${an.map(a => a.e).join('')}`);
  await done; wrap.remove();
});
// 語文 4–6：注音找一找（哪一個的名字是「ㄇ」開頭）
GAMES['lang4-6'] = dom => rounds(dom, 5, async (i, lv) => {
  const { z, tg, others } = fresh(() => {
    const tg = pick(ZY), z = tg.z, near = lv >= 2 ? sample(ZY.filter(w => w.z === ZY_NEAR[z]), 1) : [];
    return { z, tg, others: [...near, ...sample(ZY.filter(w => w.z !== z && !near.includes(w)), 2 - near.length)], key: z + tg.n };
  });
  const wrap = body(`<div class="eye-sign" style="font-size:56px;padding:4px 34px">${z}</div><div class="q-opts"></div>`);
  await prompt(`哪一個的名字是「${z}」開頭？用眼睛看著它！`); setQ(`「${z}」開頭：${tg.n}`);
  await eyeOpts(wrap.querySelector('.q-opts'), shuffle([tg, ...others]).map(w => ({ html: `<span class="emo">${w.e}</span><small>${w.n}</small>`, ok: w === tg, say: w === tg ? `對！${w.n}，是「${z}」開頭！` : `${w.n}，不是「${z}」開頭` })));
  wrap.remove();
});
// 情緒 2–3、3–4、4–6：沿用原本的玩法（都是點選），眼睛看著就能選；道具加大
// 身體動作（眼睛體操）2–3：豆豆躲貓貓
const PEEK = [.12, .37, .63, .88];
GAMES['move2-3'] = dom => rounds(dom, 4, async (i, lv) => {
  const len = lvl(lv, [3, 4, 5, 6]);
  const { seq } = fresh(() => { const seq = []; while (seq.length < len) { const k = rnd(4); if (k !== seq[seq.length - 1]) seq.push(k); } return { seq, key: seq.join('') }; });
  const wrap = body(`<div class="peek-field" style="height:min(46vh,380px)"></div>`), field = wrap.querySelector('.peek-field');
  await prompt(i === 0 ? '豆豆躲起來了！看到豆豆，就用眼睛看著他！' : '豆豆又要躲起來了，找找看！', 'happy'); setQ(`躲貓貓 ${seq.length} 次`);
  for (let k = 0; k < seq.length; k++) {
    await new Promise(res => {
      const b = el(`<button class="peek" data-dwell=".8" style="left:${PEEK[seq[k]] * 100}%;top:50%">${mascot(120, 'surprised')}</button>`);
      b.onclick = () => { b.onclick = null; b.innerHTML = mascot(120, 'happy'); burst(b, 12, ['💛', '✨']); talk(pick(['找到了！', '被你找到了！', '哇，好厲害！']), 'happy'); later(() => { b.remove(); res(); }, 1300); };
      field.appendChild(b);
    });
  }
  await praise('全部都找到了！'); await sleep(600); wrap.remove();
});
// 身體動作（眼睛體操）3–4：星星追追追（眼睛跟著移動的東西走）
const CHASE_T = [{ e: '⭐', n: '星星' }, { e: '🐝', n: '蜜蜂' }, { e: '🦋', n: '蝴蝶' }, { e: '🎈', n: '氣球' }, { e: '🐟', n: '小魚' }, { e: '🚀', n: '火箭' }, { e: '🐞', n: '瓢蟲' }, { e: '🐤', n: '小鳥' }];
const C2 = Math.PI * 2, ease = t => .5 - .5 * Math.cos(t * C2);
// 只左右移動（頭不動時眼睛左右追視最可靠）
const stepE = t => { const k = Math.floor(t * 4) / 4, f = Math.min(1, (t * 4 - Math.floor(t * 4)) * 2); return ease(k / 2 + f / 8); };
const CHASE_P = { lr: t => [.08 + .84 * ease(t / 2), .5], rl: t => [.92 - .84 * ease(t / 2), .5], back: t => [.08 + .84 * ease(t), .5], wave: t => [.08 + .84 * ease(t), .5 + .05 * Math.sin(t * C2 * 3)], stop: t => [.08 + .84 * stepE(t), .5] };
GAMES['move3-4'] = dom => rounds(dom, 4, async (i, lv) => {
  const { t, path, slow } = fresh(() => { const t = pick(CHASE_T), path = pick(lv === 0 ? ['lr', 'rl'] : Object.keys(CHASE_P)), slow = lv < 2 || Math.random() < .5; return { t, path, slow, key: t.e + path + slow }; });
  const T = lvl(lv, [9, 10, 11, 12]) * 1000, loops = slow ? 1 : 1.5;
  const wrap = body(`<div class="peek-field" style="height:min(40vh,320px)"><span class="chase">${t.e}</span></div><div class="chase-bar"><i></i></div>`);
  const field = wrap.querySelector('.peek-field'), star = wrap.querySelector('.chase'), bar = wrap.querySelector('.chase-bar i');
  const place = u => { const [px, py] = CHASE_P[path](u), r = field.getBoundingClientRect(); star.style.transform = `translate(${px * r.width}px,${py * r.height}px)`; return [r.left + px * r.width, r.top + py * r.height]; };
  place(0);
  await prompt(`${t.n}要動了！眼睛跟著${t.n}走，不要跟丟！`, 'happy'); setQ(`眼睛跟著${t.n}（${path}）`);
  let follow = 0, t0 = performance.now(), last = t0;
  await new Promise(res => { const iv = every(() => {
    const now = performance.now(), el2 = now - t0, [sx, sy] = place(Math.min(1, el2 / T) * loops);
    if (EYE.face && Math.abs(EYE.x - sx) < Math.max(130, .14 * innerWidth)) follow += now - last;
    last = now; bar.style.width = Math.min(100, follow / (T * .55) * 100) + '%';
    if (el2 >= T) { clearInterval(iv); res(); }
  }, 40); });
  if (follow >= T * .5) await praise(`眼睛跟得好棒！`); else await talk('沒關係，眼睛慢慢跟就好！', 'happy');
  await sleep(500); wrap.remove();
});
// 身體動作（眼睛體操）4–6：眼睛聽口令（依照順序看上下左右）
const CMD_OBJ = [{ e: '🌞', n: '太陽' }, { e: '🌼', n: '花' }, { e: '🐶', n: '小狗' }, { e: '🐱', n: '小貓' }, { e: '🚗', n: '汽車' }, { e: '🎈', n: '氣球' }, { e: '🍎', n: '蘋果' }, { e: '⭐', n: '星星' }, { e: '🐟', n: '小魚' }, { e: '🦋', n: '蝴蝶' }, { e: '🍓', n: '草莓' }, { e: '🐘', n: '大象' }];
const ORD = ['先', '再', '然後', '最後'];
GAMES['move4-6'] = dom => rounds(dom, 4, async (i, lv) => {
  const len = lvl(lv, [2, 2, 3, 4]), { objs, seq } = fresh(() => { const objs = sample(CMD_OBJ, 4), seq = sample([0, 1, 2, 3], len); return { objs, seq, key: seq.map(k => objs[k].e).join('') }; });
  const wrap = body(`<div class="cmd-row">${objs.map(o => `<button class="cmd" data-dwell=".9">${o.e}</button>`).join('')}</div>`);
  const bs = [...wrap.querySelectorAll('button.cmd')];
  const say = seq.map((k, j) => `${len > 2 && j === len - 1 ? '最後' : ORD[j]}看${objs[k].n}`).join('，');
  bs.forEach(b => b.disabled = true);
  await prompt(`聽口令：${say}。`); setQ(`口令：${say}`);
  bs.forEach(b => b.disabled = false);
  let step = 0;
  await new Promise(res => bs.forEach((b, k) => b.onclick = () => {
    if (b.classList.contains('done') || step >= seq.length) return;
    if (k !== seq[step]) { shake(b); oops(`要${ORD[step] === '先' ? '先' : ''}看${objs[seq[step]].n}`); return; }
    b.classList.add('done'); b.disabled = true; burst(b, 8); step++;
    if (step === seq.length) later(res, 500); else talk('對！', 'happy');
  }));
  await praise('口令都做到了！'); await sleep(500); wrap.remove();
});
// 社會 2–3、4–6：和豆豆輪流／合作戳泡泡（看著泡泡就會破）
function eyeBubble(dom, age) {
  const coop = age === '4-6';
  return rounds(dom, 3, async (i, lv) => {
    const { r: rule, goal } = fresh(() => {
      const goal = (coop ? lvl(lv, [8, 10, 12]) : lvl(lv, [6, 6, 8])) + 2 * rnd(lv === 0 ? 1 : 3);
      const mk = k => k === 'c' ? { k, ...pick(BUB_C) } : k === 'e' ? { k, ...pick(BUB_E) } : { k: 'any', n: '' };
      if (!coop) { const r = mk(i === 0 ? 'any' : pick(['c', 'e'])); return { r: [r, r], goal, key: r.k + (r.n || '') + goal }; }
      const k = pick(['c', 'e']); let a = mk(k), b = mk(k); while (b.n === a.n) b = mk(k);
      return { r: [a, b], goal, key: k + a.n + b.n };
    });
    const fits = (r, b) => r.k === 'any' || (r.k === 'c' ? b.c === r.c : b.e === r.e);
    const tag = r => r.k === 'c' ? `<span class="dotc" style="background:${r.c}"></span>${r.n}` : r.k === 'e' ? `${r.e}${r.n}` : '泡泡';
    const wrap = body(`<div class="bub"><div class="turn"></div><div class="bub-area"></div><div class="bar"><i></i></div></div>`);
    const turnEl = wrap.querySelector('.turn'), area = wrap.querySelector('.bub-area'), bar = wrap.querySelector('.bar i');
    let turn = 0, popped = 0, fin;
    const allDone = new Promise(r => fin = r);
    const setTurn = () => { turnEl.classList.toggle('dd', turn === 1); turnEl.innerHTML = coop ? `你戳 ${tag(rule[0])}　豆豆戳 ${tag(rule[1])}` : turn === 0 ? `輪到你：看著${tag(rule[0])}` : '輪到豆豆了，等一等'; };
    const pop = b => { burst(b, 8); b.remove(); popped++; bar.style.width = (popped / goal * 100) + '%'; if (popped >= goal) { area.innerHTML = ''; praise(coop ? '我們一起完成了！' : '輪流得真好！'); later(fin, 1800); return true; } return false; };
    const ddPop = () => {
      if (popped >= goal || !area.isConnected) return;
      const b = [...area.querySelectorAll('.bubble')].find(x => fits(rule[coop ? 1 : 0], x.dataset) && x.getBoundingClientRect().top > area.getBoundingClientRect().top + 20);
      if (!b) return later(ddPop, 600);
      jump($('#g-mascot svg'));
      if (pop(b)) return;
      if (!coop) { turn = 0; setTurn(); talk('換你了！'); }
    };
    const spawn = () => {
      if (popped >= goal) return;
      const all = [...area.querySelectorAll('.bubble')], mine = all.filter(x => fits(rule[0], x.dataset)), LANES = 5;
      let free = [...Array(LANES).keys()].filter(l => !all.some(x => +x.dataset.lane === l));
      if (!free.length) { if (mine.length) return; const old = all.find(x => !fits(rule[0], x.dataset)); if (!old) return; free = [+old.dataset.lane]; old.remove(); }
      const lane = pick(free), lw = area.clientWidth / LANES;
      const size = Math.min(160, lw - 16, 120 + rnd(40)), want = !mine.length ? rule[0] : Math.random() < .7 ? rule[coop && Math.random() < .4 ? 1 : 0] : null;
      const c = want && want.k === 'c' ? want.c : pick(BUB_C).c, e = want && want.k === 'e' ? want.e : (rule[0].k === 'e' ? pick(BUB_E).e : '');
      const b = el(`<button class="bubble" data-dwell=".55" aria-label="泡泡">${e ? `<span>${e}</span>` : ''}</button>`);
      b.dataset.c = c; b.dataset.e = e; b.dataset.lane = lane;
      b.style.cssText = `width:${size}px;height:${size}px;left:${lane * lw + (lw - size) / 2}px;--bc:${c};--h:${area.clientHeight}px;animation-duration:${13 + Math.random() * 4}s;font-size:${size * .42}px`;
      b.addEventListener('animationend', () => b.remove());
      b.onclick = () => {
        if (!coop && turn === 1) { talk('等一等，現在輪到豆豆'); return; }
        if (!fits(rule[0], b.dataset)) { shake(b); talk(`你要看${rule[0].n || '泡泡'}`); return; }
        if (pop(b)) return;
        if (!coop) { turn = 1; setTurn(); later(ddPop, 1700); }
      };
      area.appendChild(b);
    };
    setTurn();
    await prompt(coop ? `你戳${rule[0].n}的泡泡，豆豆戳${rule[1].n}的泡泡。我們一起合作！用眼睛看著泡泡，泡泡就會破。` : rule[0].k === 'any' ? '我們輪流戳泡泡！你先，用眼睛看著泡泡，泡泡就會破。' : `這次要戳${rule[0].n}的泡泡。你先，然後換豆豆！`, 'happy');
    setQ(coop ? `合作：你 ${rule[0].n}／豆豆 ${rule[1].n}，共 ${goal} 個` : `輪流戳${rule[0].n || '泡泡'}，共 ${goal} 個`);
    spawn(); const iv = every(spawn, 1300); let iv2 = null;
    if (coop) iv2 = every(ddPop, 3200);
    await allDone; clearInterval(iv); iv2 && clearInterval(iv2); wrap.remove();
  });
}
GAMES['soc2-3'] = eyeBubble; GAMES['soc4-6'] = eyeBubble;
// 社會 3–4：分享小點心（看著好朋友，就把點心送給他）
GAMES['soc3-4'] = dom => rounds(dom, 3, async (i, lv) => {
  const { fr, food, pre, nfood } = fresh(() => {
    const nf = lv === 0 ? 2 : 2 + rnd(2), each = lvl(lv, [[1, 2], [1, 2], [2, 3], [2, 3]])[rnd(2)];
    const pre = lv >= 3 && Math.random() < .6 ? Array.from({ length: nf }, () => rnd(each)) : Array(nf).fill(0); if (pre.every(x => x === pre[0])) pre.fill(0);
    const fr = sample(FRIENDS, nf), food = pick(FOODS), nfood = each * nf - pre.reduce((a, b) => a + b, 0);
    return { fr, food, pre, nfood, key: fr.map(f => f.n).join() + food.n + nfood + pre.join('') };
  });
  const wrap = body(`<div class="scene picnic" style="min-height:min(60vh,460px)"><div class="friends">${fr.map(f => `<button class="friend" data-dwell=".9"><span class="av">${f.e}<span class="feel"></span></span><div class="plate"></div><b>0</b></button>`).join('')}</div><div class="snack-basket"><div class="fruits"></div><div class="bk">🧺</div></div></div><div class="row"></div>`);
  const sc = wrap.querySelector('.scene'), fruits = wrap.querySelector('.fruits'), friends = [...wrap.querySelectorAll('.friend')], row = wrap.querySelector('.row');
  const put = (f, n) => { for (let k = 0; k < n; k++) f.querySelector('.plate').insertAdjacentHTML('beforeend', `<span>${food.e}</span>`); f.querySelector('b').textContent = f.querySelectorAll('.plate span').length; };
  let res; const allDone = new Promise(r => res = r), busy = { v: false };
  const fill = () => { friends.forEach((f, k) => { f.querySelector('.plate').innerHTML = ''; f.querySelector('.feel').textContent = ''; put(f, pre[k]); }); fruits.innerHTML = Array.from({ length: nfood }, () => `<span class="snack">${food.e}</span>`).join(''); };
  fill();
  friends.forEach(f => f.onclick = async () => {
    const s = fruits.querySelector('.snack'); if (!s || busy.v || row.children.length) return;
    busy.v = true; const gh = el(`<span class="fx fly-snack" style="font-size:40px">${food.e}</span>`); document.body.appendChild(gh);
    const a = center(s), b = center(f.querySelector('.plate')); s.remove();
    await gh.animate([{ transform: `translate(${a.x - 20}px,${a.y - 20}px)` }, { transform: `translate(${b.x - 20}px,${b.y - 20}px)` }], { duration: 500, easing: 'ease-in-out' }).fin;
    gh.remove(); put(f, 1); jump(f); busy.v = false;
    if (fruits.children.length) return;
    const counts = friends.map(x => +x.querySelector('b').textContent), max = Math.max(...counts), min = Math.min(...counts);
    await sleep(400);
    if (max === min) { friends.forEach(x => x.querySelector('.feel').textContent = '💕'); burst(sc, 14, ['💗', '✨']); await praise(`一樣多！每個人都有${max}個！`); later(res, 1000); }
    else {
      friends.forEach((x, k) => x.querySelector('.feel').textContent = counts[k] === min ? '😢' : '');
      oops('有人比較少，好像不太公平'); const again = el('<button class="btn">再分一次 🔄</button>'); row.appendChild(again);
      again.onclick = () => { again.remove(); fill(); talk('再分一次，要一樣多！'); };
    }
  });
  const who = fr.map(f => f.n).join('和');
  await prompt(pre.some(x => x) ? `有些好朋友已經有${food.n}了。用眼睛看著好朋友，把${food.n}送給他，讓${who}一樣多！` : `籃子裡有${nfood}個${food.n}。用眼睛看著好朋友，就會把${food.n}送給他。要分得一樣多！`, 'happy');
  setQ(`${nfood} 個${food.e}分給 ${fr.length} 位${pre.some(x => x) ? `（盤子已有 ${pre.join('、')}）` : ''}`);
  await allDone; wrap.remove();
});
// 美感 2–3：眼睛畫畫（看到哪裡就畫到哪裡）
const BRUSH = [{ k: 'rainbow', n: '彩虹筆' }, { k: 'star', n: '星星筆' }, { k: 'bubble', n: '泡泡筆' }, { k: 'heart', n: '愛心筆' }];
const PAPER = [{ k: 'white', c: '#FFFFFF', n: '白紙' }, { k: 'sky', c: 'linear-gradient(#D3EAFF,#FFFFFF)', n: '天空' }, { k: 'sea', c: 'linear-gradient(#BCE3FF,#8FC6F0)', n: '大海' }, { k: 'grass', c: 'linear-gradient(#FFF8D6,#BDEBD2)', n: '草地' }];
const INKS = [['#D94A73', '#FF9F43', '#F2C200'], ['#1F8559', '#2F6CC4', '#7350CC'], ['#FF7EB6', '#4F8DFD', '#2FBF84']];
GAMES['art2-3'] = dom => rounds(dom, 3, async (i, lv) => {
  const { br, pa, ink } = fresh(() => { const br = pick(BRUSH), pa = pick(PAPER), ink = rnd(INKS.length); return { br, pa, ink, key: br.k + pa.k + ink }; });
  const wrap = body(`<div class="paint-box" style="background:${pa.c}"><canvas></canvas></div><div class="row" style="justify-content:center"></div>`);
  const box = wrap.querySelector('.paint-box'), cv = box.querySelector('canvas'), ctx = cv.getContext('2d'), row = wrap.querySelector('.row');
  const r0 = box.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1); cv.width = r0.width * dpr; cv.height = r0.height * dpr; ctx.scale(dpr, dpr);
  await prompt(`這是${br.n}。眼睛看到哪裡，就會畫到哪裡！慢慢看，畫一張漂亮的畫！`, 'happy'); setQ(`眼睛畫畫：${br.n}、${pa.n}`);
  let hue = 0, last = null, t = 0, doneP; const done = new Promise(r => doneP = r);
  const draw = () => {
    const r = box.getBoundingClientRect(), x = EYE.x - r.left, y = EYE.y - r.top;
    if (!EYE.face || x < 0 || y < 0 || x > r.width || y > r.height) { last = null; return; }
    const col = br.k === 'rainbow' ? `hsl(${(hue += 6) % 360} 85% 60%)` : pick(INKS[ink]);
    if (br.k === 'rainbow') { if (last) { ctx.strokeStyle = col; ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(x, y); ctx.stroke(); } }
    else { ctx.fillStyle = col; ctx.font = `${br.k === 'bubble' ? 0 : 34}px sans-serif`; if (br.k === 'bubble') { ctx.globalAlpha = .55; ctx.beginPath(); ctx.arc(x + rnd(16) - 8, y + rnd(16) - 8, 8 + rnd(16), 0, C2); ctx.fill(); ctx.globalAlpha = 1; } else { ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(br.k === 'star' ? '★' : '♥', x + rnd(10) - 5, y + rnd(10) - 5); } }
    last = { x, y };
  };
  const iv = every(() => { draw(); t += 60; if (t === 12000) { const b = el('<button class="btn" data-dwell="1.2">完成 ✓</button>'); b.onclick = () => doneP(); row.appendChild(b); talk('畫好了，就看著「完成」。', 'happy'); } if (t >= 40000) doneP(); }, 60);
  await done; clearInterval(iv);
  try { const p = cur(), sc = Math.min(1, 480 / r0.width), c2 = document.createElement('canvas'); c2.width = r0.width * sc; c2.height = r0.height * sc; const x2 = c2.getContext('2d'); x2.fillStyle = '#fff'; x2.fillRect(0, 0, c2.width, c2.height); x2.drawImage(cv, 0, 0, c2.width, c2.height); const art = getArt(p.id); art.unshift({ d: today(), img: c2.toDataURL('image/jpeg', .72) }); setArt(p.id, art.slice(0, 12)); } catch (e) {}
  burst(box, 14, ['🌈', '✨', '💖']); await praise('好漂亮的畫！存進畫冊了！'); await sleep(600); wrap.remove();
});
// 美感 3–4：星星連連看（看著下一顆星星，就會連起來）
const DOT_SHAPES = [
  { k: 'tri', n: '三角形', p: [[.5, .15], [.85, .85], [.15, .85]] }, { k: 'sq', n: '正方形', p: [[.22, .2], [.78, .2], [.78, .8], [.22, .8]] },
  { k: 'dia', n: '菱形', p: [[.5, .1], [.85, .5], [.5, .9], [.15, .5]] }, { k: 'boat', n: '小船', p: [[.12, .55], [.88, .55], [.72, .85], [.28, .85]] },
  { k: 'house', n: '房子', p: [[.2, .9], [.2, .45], [.5, .12], [.8, .45], [.8, .9]] }, { k: 'kite', n: '風箏', p: [[.5, .1], [.8, .4], [.5, .9], [.2, .4]] },
  { k: 'fish', n: '小魚', p: [[.12, .5], [.35, .25], [.65, .3], [.82, .5], [.65, .7], [.35, .75]] }, { k: 'cup', n: '杯子', p: [[.25, .2], [.75, .2], [.68, .85], [.32, .85]] },
  { k: 'crown', n: '皇冠', p: [[.12, .8], [.12, .3], [.31, .55], [.5, .2], [.69, .55], [.88, .3], [.88, .8]] }, { k: 'tree', n: '大樹', p: [[.5, .08], [.8, .45], [.62, .45], [.85, .78], [.15, .78], [.38, .45], [.2, .45]] },
  { k: 'heart', n: '愛心', p: [[.5, .88], [.15, .5], [.18, .25], [.35, .15], [.5, .3], [.65, .15], [.82, .25], [.85, .5]] },
  { k: 'star', n: '星星', p: Array.from({ length: 10 }, (_, k) => { const a = -Math.PI / 2 + k * Math.PI / 5, r = k % 2 ? .18 : .4; return [.5 + r * Math.cos(a), .52 + r * Math.sin(a)]; }) },
];
const DOT_C = ['#FFD0DE', '#FFDCC4', '#FFF0AE', '#C8F0DC', '#CDE6FF', '#E2D6FF'];
GAMES['art3-4'] = dom => rounds(dom, 3, async (i, lv) => {
  const max = lvl(lv, [4, 5, 7, 10]), { sh, col } = fresh(() => { const sh = pick(DOT_SHAPES.filter(s => s.p.length <= max)), col = rnd(DOT_C.length); return { sh, col, key: sh.k + col }; });
  const wrap = body(`<div class="dots-box"><svg preserveAspectRatio="none"></svg></div>`), box = wrap.querySelector('.dots-box'), svg = box.querySelector('svg');
  const R = box.getBoundingClientRect(), pts = sh.p.map(([x, y]) => [x * R.width, y * R.height]);
  svg.setAttribute('viewBox', `0 0 ${R.width} ${R.height}`);
  const dots = pts.map(([x, y], k) => { const d = el(`<button class="cdot" data-dwell=".75" style="left:${x}px;top:${y}px">${k + 1}</button>`); d.disabled = k > 0; if (!k) d.classList.add('next'); box.appendChild(d); return d; });
  await prompt('從 1 號星星開始。眼睛看著下一顆星星，就會連起來！'); setQ(`連連看：${sh.n}（${pts.length} 點）`);
  await new Promise(res => dots.forEach((d, k) => d.onclick = () => {
    d.classList.remove('next'); d.classList.add('done'); d.disabled = true; sayNum(k + 1);
    if (k > 0) svg.insertAdjacentHTML('beforeend', `<line x1="${pts[k - 1][0]}" y1="${pts[k - 1][1]}" x2="${pts[k][0]}" y2="${pts[k][1]}" stroke="#2F2A44" stroke-width="8" stroke-linecap="round"/>`);
    if (k + 1 < dots.length) { dots[k + 1].disabled = false; dots[k + 1].classList.add('next'); }
    else { svg.insertAdjacentHTML('afterbegin', `<polygon points="${pts.map(p => p.join(',')).join(' ')}" fill="${DOT_C[col]}" stroke="#2F2A44" stroke-width="8" stroke-linejoin="round"/>`); later(res, 700); }
  }));
  burst(box, 14, ['⭐', '✨']); await praise(`連好了！是${sh.n}！`); await sleep(800); wrap.remove();
});
// 美感 4–6：眼睛塗顏色（先看顏色，再看要塗的地方）
const FILL_TPL = [
  { k: 'house', n: '房子', s: '<polygon data-gaze points="35,72 100,22 165,72"/><rect data-gaze x="48" y="72" width="104" height="68"/><rect data-gaze x="86" y="100" width="28" height="40"/><circle data-gaze cx="178" cy="24" r="17"/>' },
  { k: 'fish', n: '小魚', s: '<ellipse data-gaze cx="88" cy="75" rx="62" ry="40"/><polygon data-gaze points="148,75 192,38 192,112"/><polygon data-gaze points="70,38 100,12 110,40"/>' },
  { k: 'car', n: '汽車', s: '<path data-gaze d="M18 108 L18 80 L52 80 L72 50 L130 50 L150 80 L182 80 L182 108 Z"/><g data-gaze><circle cx="56" cy="114" r="18"/><circle cx="146" cy="114" r="18"/></g><rect data-gaze x="82" y="58" width="40" height="22"/>' },
  { k: 'rocket', n: '火箭', s: '<path data-gaze d="M100 8 C130 34 132 92 124 116 L76 116 C68 92 70 34 100 8 Z"/><circle data-gaze cx="100" cy="58" r="14"/><g data-gaze><polygon points="76,88 46,124 78,116"/><polygon points="124,88 154,124 122,116"/></g><polygon data-gaze points="84,118 100,148 116,118"/>' },
  { k: 'bfly', n: '蝴蝶', s: '<g data-gaze><ellipse cx="62" cy="52" rx="40" ry="30"/><ellipse cx="66" cy="104" rx="30" ry="24"/></g><g data-gaze><ellipse cx="138" cy="52" rx="40" ry="30"/><ellipse cx="134" cy="104" rx="30" ry="24"/></g><rect data-gaze x="93" y="30" width="14" height="100" rx="7"/>' },
  { k: 'snow', n: '雪人', s: '<circle data-gaze cx="100" cy="108" r="36"/><circle data-gaze cx="100" cy="52" r="26"/><rect data-gaze x="74" y="12" width="52" height="20" rx="4"/><rect data-gaze x="70" y="74" width="60" height="10" rx="5"/>' },
  { k: 'ice', n: '冰淇淋', s: '<polygon data-gaze points="70,80 130,80 100,146"/><circle data-gaze cx="100" cy="66" r="30"/><circle data-gaze cx="100" cy="28" r="22"/>' },
  { k: 'flower', n: '花', s: '<g data-gaze><circle cx="100" cy="24" r="20"/><circle cx="130" cy="48" r="20"/><circle cx="118" cy="82" r="20"/><circle cx="82" cy="82" r="20"/><circle cx="70" cy="48" r="20"/></g><circle data-gaze cx="100" cy="54" r="16"/><polygon data-gaze points="70,112 130,112 120,148 80,148"/>' },
];
const PALS = ['#FF5C5C', '#FF9F43', '#FFD233', '#2FBF84', '#4F8DFD', '#A06BFF', '#FF7EB6', '#A0704F'];
GAMES['art4-6'] = dom => rounds(dom, 3, async (i, lv) => {
  const { tp, pal } = fresh(() => { const tp = pick(FILL_TPL), pal = sample(PALS, 3); return { tp, pal, key: tp.k + pal.join('') }; });
  const wrap = body(`<div class="fill-box"><svg viewBox="0 0 200 150">${tp.s}</svg></div><div class="q-opts"></div>`);
  const svg = wrap.querySelector('svg'), box = wrap.querySelector('.q-opts'), regions = [...svg.querySelectorAll('[data-gaze]')];
  regions.forEach(r => { r.removeAttribute('data-gaze'); r.setAttribute('fill', '#FFFFFF'); r.querySelectorAll('*').forEach(c => c.setAttribute('fill', 'inherit')); });
  setQ(`眼睛塗顏色：${tp.n}`);
  for (let k = 0; k < regions.length; k++) {
    const r = regions[k]; r.classList.add('now-fill');
    if (k === 0) await prompt(`來塗${tp.n}！虛線圈起來的地方要塗什麼顏色？用眼睛看著顏色！`, 'happy'); else await prompt('這裡要塗什麼顏色？');
    const o = await choose(box, pal.map(c => ({ html: `<span class="swatch" style="background:${c}"></span>`, ok: true, say: pick(['好漂亮！', '塗好了！', '這個顏色好看！']), c })), { wait: 900 });
    r.classList.remove('now-fill'); r.setAttribute('fill', o.c); box.innerHTML = '';
  }
  try { const p = cur(), art = getArt(p.id); art.unshift({ d: today(), img: 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 150" style="background:#fff">${svg.innerHTML.replace(/ class="[^"]*"/g, '')}</svg>`) }); setArt(p.id, art.slice(0, 12)); } catch (e) {}
  burst(svg, 14, ['🎨', '✨']); await praise(`${tp.n}塗好了，好漂亮！`); await sleep(700); wrap.remove();
});
Object.assign(TIPS.cog, { '2-3': '孩子看著選項約 2 秒就會選取，看到圓圈慢慢填滿就對了。', '3-4': '一個一個看，看到的會亮起來並數出來。', '4-6': '請孩子先說出規律，再用眼睛選。' });
Object.assign(TIPS.move, { '2-3': '豆豆會出現在畫面不同位置，練習眼睛快速找目標。', '3-4': '練習眼睛平順追視，跟丟也沒關係。', '4-6': '練習聽指令、記順序，再依序看。' });
Object.assign(TIPS.soc, { '2-3': '和豆豆輪流：等豆豆戳完再換你。', '3-4': '看著好朋友就把點心送過去，要一樣多。', '4-6': '你戳自己的顏色，豆豆戳他的顏色，一起完成。' });
Object.assign(TIPS.art, { '2-3': '看到哪裡就畫到哪裡，12 秒後可以看「完成」。', '3-4': '依照數字順序看星星。', '4-6': '虛線圈起來的地方，看著一個顏色就塗上去。' });

/* ---------- 啟動 ---------- */
function bootStep(s, msg, help) {
  document.querySelectorAll('#eye-boot [data-s]').forEach(x => { const order = ['cam', 'model', 'face', 'cal'], a = order.indexOf(x.dataset.s), b = order.indexOf(s); x.className = a < b ? 'ok' : a === b ? 'now' : ''; });
  $('#boot-msg').textContent = msg; $('#boot-help').textContent = help || '';
}
async function eyeBoot() {
  $('#boot-mascot').innerHTML = mascot(130, 'happy');
  $('#eye-boot').classList.add('on');
  setInterval(() => gazeTick(performance.now()), 33);
  if (EYE_SIM) { simLoop(); bootStep('face', '模擬模式：用滑鼠代替眼睛'); }
  else {
    try {
      bootStep('cam', '正在開啟鏡頭…', '第一次使用時，瀏覽器會詢問能不能使用鏡頭，請家人幫忙按「允許」。影像只在這台裝置上處理，不會上傳。');
      talk('嗨！我是豆豆。第一次使用時，請家人幫忙按「允許」使用鏡頭。');
      await startCamera();
      $('#eye-cam').classList.toggle('hide', !S.set.eye.cam);
      bootStep('model', '正在載入眼睛追蹤…（第一次約需 10 秒，需要網路）');
      await loadTracker(); trackLoop();
    } catch (e) {
      bootStep('cam', '沒辦法開啟鏡頭', '請確認：1. 瀏覽器允許這個網頁使用鏡頭　2. 鏡頭沒有被其他程式占用　3. 有連上網路（第一次載入眼睛追蹤需要）。修好後重新整理頁面。');
      talk('沒辦法開啟鏡頭，請家人幫忙檢查鏡頭權限。'); return;
    }
    bootStep('face', '請面向鏡頭，讓豆豆看到你的臉', '臉在畫面中間、距離鏡頭約 40 到 70 公分最好；戴眼鏡也可以。');
    while (performance.now() - EYE.lastFace > 300) await wait(200);
  }
  bootStep('cal', '找到你了！接下來看星星校正');
  await wait(900);
  $('#eye-boot').classList.remove('on');
  await calibrate();
  cur() ? goHome() : renderWho();
}

// 使用說明（眼睛版）
$('#ov-help .help-sheet').innerHTML = `<button class="help-x" id="help-x" aria-label="關閉">✕</button>
  <h2>豆豆眼睛遊樂園</h2>
  <section><h3>介紹</h3><p>給 2–6 歲、手不方便操作的小朋友（例如腦性麻痺、肢體障礙）使用的互動遊戲。不用手、不用滑鼠，透過鏡頭看眼睛和頭的方向來玩。依照《幼兒園教保活動課程大綱》六大領域設計，三個年齡各有不同玩法。</p></section>
  <section><h3>功能</h3><ul>
    <li>眼睛看著選項約 2 秒（圓圈填滿）就會選取，可調快慢</li>
    <li>開始前頭不動、只用眼睛看星星校正，戴眼鏡也能用</li>
    <li>18 種遊戲，一關一關變難；每個單元約 15 分鐘、題目不重複；結束休息 3 分鐘</li>
    <li>豆豆念出所有說明，畫面同時有大字幕</li>
    <li>影像只在這台裝置上處理，不會上傳</li></ul></section>
  <section><h3>操作方式</h3><ol>
    <li>打開網頁；第一次請家人按「允許」使用鏡頭</li>
    <li>臉對著鏡頭（約 40–70 公分），眼睛跟著星星看，完成校正</li>
    <li>看著年齡或頭像選人，再看著單元開始玩</li>
    <li>看著 🏠 久一點就回首頁；首頁有「眼睛設定」和「重新校正」</li>
    <li>如果聲音被瀏覽器擋住，請家人點一下畫面任何地方</li></ol></section>
  <section><h3>使用規範（照著做會比較準）</h3><ol>
    <li><b>鏡頭位置</b>：鏡頭在螢幕上方正中間，和眼睛差不多高；裝置用支架固定，玩的時候不要移動</li>
    <li><b>距離</b>：臉離鏡頭約 40–70 公分，臉在擺位框框中間</li>
    <li><b>光線</b>：光從前面照亮臉；不要背對窗戶，也不要只有頭頂一盞燈</li>
    <li><b>眼鏡</b>：鏡片擦乾淨；有反光就調整燈光方向或把眼鏡稍微往下調</li>
    <li><b>擺位</b>：頭部一定要有支撐（頭靠、側墊），整個過程頭不動、只用眼睛；換姿勢、換位置就要重新校正</li>
    <li><b>筆電</b>：把筆電墊高或螢幕往後仰，讓鏡頭和眼睛一樣高；選項都排成一排，左右看就能選</li>
    <li><b>校正</b>：看星星時頭盡量不動、只用眼睛看；分數低於 50 分請再校正一次</li>
    <li><b>選取</b>：看著選項直到圓框繞完一圈；太快誤選就把「看多久才選取」調慢，頭或眼睛會晃就開「加強穩定」</li>
    <li><b>休息與安全</b>：每個單元約 15 分鐘，結束會休息 3 分鐘；孩子眼睛疲勞、流淚、頭痛或不舒服就停止。遊戲不使用閃光</li>
    <li><b>隱私</b>：鏡頭影像只在這台裝置上分析，不錄影、不上傳</li></ol></section>
  <button class="btn" id="help-ok" style="--dc:var(--soc);--dcs:var(--soc-s)">知道了</button>`;
$('#help-x').onclick = $('#help-ok').onclick = () => closeOv('ov-help');
