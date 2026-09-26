"""HTML for the identify view: pick a signal, watch the model name it."""

IDENTIFY_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Identify, motor imagery</title>
<style>
  :root {
    --ground:#E9EDEF; --panel:#FFF; --ink:#131C21; --muted:#5A6A72; --faint:#8798A0;
    --rule:#D2DADD; --accent:#0B6E5F; --accent-soft:#D8EAE5; --bad:#9E2B26;
    --left:#2E6E9E; --right:#A85F18; --foot:#67539B; --tongue:#2C7A57; --trace:#8AA3AC;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground:#0E1417; --panel:#161F23; --ink:#E4ECEE; --muted:#9DAEB5; --faint:#74868E;
      --rule:#2A383E; --accent:#58C4AC; --accent-soft:#123A32; --bad:#E0766F;
      --left:#74ADDA; --right:#DFA365; --foot:#A899DD; --tongue:#6EC59C; --trace:#5C7079;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:26px 20px 70px}
  h1{font-size:25px;margin:0 0 4px;letter-spacing:-.02em}
  .sub{color:var(--muted);margin:0 0 18px}
  a{color:var(--accent)}
  .card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:18px}

  .layout{display:grid;grid-template-columns:330px 1fr;gap:18px;align-items:start}
  @media (max-width:880px){.layout{grid-template-columns:1fr}}

  .pickhead{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  select{font:inherit;border-radius:5px;border:1px solid var(--rule);
         padding:7px 10px;background:var(--panel);color:var(--ink);cursor:pointer}
  .chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
  .chip{font:12px ui-monospace,Menlo,monospace;padding:4px 10px;border-radius:99px;
        border:1px solid var(--rule);background:var(--panel);color:var(--muted);cursor:pointer}
  .chip.on{border-color:var(--accent);background:var(--accent-soft);color:var(--accent);font-weight:600}

  .list{max-height:560px;overflow-y:auto;border:1px solid var(--rule);border-radius:6px;
        background:var(--panel)}
  .sig{display:grid;grid-template-columns:34px 1fr 62px;gap:10px;align-items:center;
       padding:9px 12px;border-bottom:1px solid var(--rule);cursor:pointer;background:none;
       border-left:0;border-right:0;border-top:0;width:100%;text-align:left;color:inherit;font:inherit}
  .sig:last-child{border-bottom:0}
  .sig:hover{background:var(--ground)}
  .sig.on{background:var(--accent-soft)}
  .sig .n{font:12px ui-monospace,Menlo,monospace;color:var(--faint)}
  .sig svg{display:block;width:100%;height:26px}
  .sig .lab{font:600 12px ui-monospace,Menlo,monospace;text-align:right}
  .sig .done{font-size:11px;display:block;font-weight:400}

  .scope svg{display:block;width:100%;height:180px}
  .chlabel{font:11px ui-monospace,Menlo,monospace;fill:var(--faint)}
  .clock{font:12px ui-monospace,Menlo,monospace;color:var(--faint);
         display:flex;justify-content:space-between;margin-top:6px}

  .verdict{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
  @media (max-width:640px){.verdict{grid-template-columns:1fr}}
  .vbox{text-align:center;padding:18px}
  .vbox .cap{font:11px ui-monospace,Menlo,monospace;letter-spacing:.09em;
             text-transform:uppercase;color:var(--faint)}
  .vbox .big{font-size:34px;font-weight:700;margin:6px 0 2px;line-height:1.1}
  .vbox .meta{font:13px ui-monospace,Menlo,monospace;color:var(--muted)}
  .vbox.truth{border-style:dashed}

  .ring{display:block;margin:4px auto 2px}
  .ring text{font:700 19px ui-monospace,Menlo,monospace;fill:var(--ink)}

  .bars{display:grid;gap:8px;margin-top:18px}
  .bar{display:grid;grid-template-columns:58px 1fr 44px;gap:10px;align-items:center}
  .bar .nm{font-size:13px;color:var(--muted)}
  .track{height:20px;background:var(--ground);border:1px solid var(--rule);
         border-radius:3px;position:relative;overflow:hidden}
  .fill{position:absolute;inset:0 auto 0 0;transition:width .3s ease}
  .chance{position:absolute;top:-1px;bottom:-1px;left:25%;border-left:2px dashed var(--bad)}
  .bar .pc{font:600 13px ui-monospace,Menlo,monospace;text-align:right}

  .hint{color:var(--muted);font-size:13px}
  .err{color:var(--bad);border:1px solid var(--bad);border-radius:5px;padding:13px;margin-top:14px}
  .tally{font:13px ui-monospace,Menlo,monospace;color:var(--muted);margin-left:auto}
  .ok{color:var(--accent)} .miss{color:var(--bad)}
</style>
</head>
<body>
<div class="wrap">
  <h1>Identify a signal</h1>
  <p class="sub">Pick any EEG signal on the left. The model has never seen these, it was trained on a
  separate part of the recording. <a href="/">Test bench</a> · <a href="/live">Live stream</a></p>

  <div class="layout">
    <div>
      <div class="pickhead">
        <select id="subject"></select>
        <span class="tally" id="tally">no runs yet</span>
      </div>
      <div class="chips" id="chips"></div>
      <div class="list" id="list"><div style="padding:16px" class="hint">loading…</div></div>
      <p class="hint" style="margin-top:10px">The label on each row is the real answer. The model is
      given the signal alone.</p>
    </div>

    <div>
      <div class="card scope">
        <svg id="scope" viewBox="0 0 900 180" preserveAspectRatio="none" aria-label="EEG traces"></svg>
        <div class="clock">
          <span id="clockL">choose a signal to identify</span>
          <span id="clockR"></span>
        </div>
      </div>

      <div class="verdict">
        <div class="card vbox">
          <div class="cap">Model identifies</div>
          <div class="big" id="call">waiting</div>
          <svg class="ring" id="ring" width="96" height="96" viewBox="0 0 96 96"></svg>
          <div class="meta" id="conf">confidence</div>
        </div>
        <div class="card vbox truth">
          <div class="cap">Really was</div>
          <div class="big" id="truth">hidden</div>
          <div class="meta" id="verdictNote">pick a signal</div>
        </div>
      </div>

      <div class="card" style="margin-top:16px">
        <div style="font-size:14px;font-weight:600">Confidence in each class, updating live</div>
        <div class="hint">Dashed line is the 25% chance level.</div>
        <div class="bars" id="bars"></div>
      </div>
    </div>
  </div>
  <div id="error"></div>
</div>

<script>
const CONFIG = __CONFIG__;
const CLASSES = CONFIG.classes;
const COLORS = {left:'var(--left)', right:'var(--right)', foot:'var(--foot)', tongue:'var(--tongue)'};
const el = id => document.getElementById(id);

let catalog = [], filter = null, playing = false, token = 0;
const results = new Map();   // trial index -> {call, conf, ok}

el('subject').innerHTML = CONFIG.subjects.map(s => `<option>${s}</option>`).join('');
el('bars').innerHTML = CLASSES.map(c => `
  <div class="bar">
    <span class="nm">${c}</span>
    <span class="track"><span class="fill" id="f_${c}" style="width:0%;background:${COLORS[c]}"></span><span class="chance"></span></span>
    <span class="pc" id="p_${c}">0%</span>
  </div>`).join('');

el('chips').innerHTML = ['all'].concat(CLASSES).map(c =>
  `<button class="chip${c === 'all' ? ' on' : ''}" data-c="${c}">${c}</button>`).join('');

async function post(url, body) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'request failed');
  return data;
}
function setError(m){ el('error').innerHTML = m ? `<div class="err">${m}</div>` : ''; }
const sleep = ms => new Promise(r => setTimeout(r, ms));

function spark(points) {
  const W = 200, H = 26;
  const d = points.map((v, i) => {
    const x = (i / (points.length - 1)) * W;
    const y = H / 2 - v * (H * 0.38);
    return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><path d="${d}" fill="none"
          stroke="var(--trace)" stroke-width="1" vector-effect="non-scaling-stroke"/></svg>`;
}

function renderList() {
  const shown = catalog.filter(e => !filter || e.truth === filter);
  if (!shown.length) { el('list').innerHTML = '<div style="padding:16px" class="hint">none of those</div>'; return; }

  el('list').innerHTML = shown.map(e => {
    const r = results.get(e.index);
    const mark = r ? `<span class="done ${r.ok ? 'ok' : 'miss'}">${r.ok ? 'identified' : 'missed'}</span>` : '';
    return `<button class="sig" data-i="${e.index}">
      <span class="n">${e.index + 1}</span>
      ${spark(e.preview)}
      <span class="lab" style="color:${COLORS[e.truth]}">${e.truth}${mark}</span>
    </button>`;
  }).join('');

  el('list').querySelectorAll('.sig').forEach(node =>
    node.addEventListener('click', () => identify(parseInt(node.dataset.i, 10))));
}

function drawTraces(traces) {
  const W = 900, H = 180, lane = H / traces.length;
  el('scope').innerHTML = traces.map((pts, i) => {
    const mid = lane * i + lane / 2, amp = lane * 0.38;
    const d = pts.map((v, j) => {
      const x = (j / (pts.length - 1)) * W;
      return `${j ? 'L' : 'M'}${x.toFixed(1)},${(mid - v * amp).toFixed(1)}`;
    }).join(' ');
    return `<path d="${d}" fill="none" stroke="var(--trace)" stroke-width="1.1"
             stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
            <text class="chlabel" x="6" y="${(mid - lane * 0.32).toFixed(1)}">${CONFIG.channels[i] || ''}</text>`;
  }).join('');
}

function drawRing(value, colour) {
  const r = 38, c = 2 * Math.PI * r, on = c * Math.max(0, Math.min(1, value));
  el('ring').innerHTML = `
    <circle cx="48" cy="48" r="${r}" fill="none" stroke="var(--rule)" stroke-width="9"/>
    <circle cx="48" cy="48" r="${r}" fill="none" stroke="${colour}" stroke-width="9"
            stroke-linecap="round" stroke-dasharray="${on.toFixed(1)} ${(c - on).toFixed(1)}"
            transform="rotate(-90 48 48)"/>
    <text x="48" y="55" text-anchor="middle">${Math.round(value * 100)}%</text>`;
}

function showFrame(f, i) {
  drawTraces(f.traces);
  CLASSES.forEach(c => {
    const p = f.probabilities[c] ?? 0;
    el('f_' + c).style.width = (p * 100).toFixed(1) + '%';
    el('p_' + c).textContent = Math.round(p * 100) + '%';
  });
  el('call').textContent = f.label;
  el('call').style.color = COLORS[f.label] || 'var(--ink)';
  drawRing(f.confidence, COLORS[f.label] || 'var(--accent)');
  el('conf').textContent = 'confidence, window ' + (i + 1);
  el('clockL').textContent = `${f.t.toFixed(2)} s into the signal`;
  el('clockR').textContent = i === 0 ? 'first full window, the decision point'
                                     : 'sliding past the imagery';
}

function updateTally() {
  if (!results.size) { el('tally').textContent = 'no runs yet'; return; }
  const hits = [...results.values()].filter(r => r.ok).length;
  el('tally').textContent = `${hits}/${results.size} identified (${Math.round(hits / results.size * 100)}%)`;
}

async function identify(index) {
  if (playing) return;
  playing = true;
  const mine = ++token;
  setError('');

  el('list').querySelectorAll('.sig').forEach(n =>
    n.classList.toggle('on', parseInt(n.dataset.i, 10) === index));

  el('truth').textContent = 'hidden';
  el('verdictNote').textContent = 'identifying…';
  el('clockL').textContent = 'reading the signal…';

  try {
    const data = await post('/api/live/trial', {subject: el('subject').value, index});
    if (mine !== token) return;

    for (let i = 0; i < data.frames.length; i++) {
      if (mine !== token) return;
      showFrame(data.frames[i], i);
      await sleep(420);
    }

    // The first complete window is where the model is most accurate; later ones
    // slide past the imagery into the rest period. Redraw that frame in full so
    // the bars, the ring and the verdict all describe the same moment -- showing
    // the verdict from one window and the bars from another reads as a bug.
    const decision = data.frames[0];
    const ok = decision.label === data.truth;

    showFrame(decision, 0);
    el('conf').textContent = `${Math.round(decision.confidence * 100)}% confident`;
    el('clockR').textContent = 'decision taken at the first full window';

    el('truth').textContent = data.truth;
    el('truth').style.color = COLORS[data.truth] || 'var(--ink)';
    el('verdictNote').innerHTML = ok
      ? '<span class="ok">the model got it right</span>'
      : `<span class="miss">wrong, it said ${decision.label}</span>`;

    results.set(index, {call: decision.label, conf: decision.confidence, ok});
    renderList();
    el('list').querySelectorAll('.sig').forEach(n =>
      n.classList.toggle('on', parseInt(n.dataset.i, 10) === index));
    updateTally();
  } catch (err) {
    setError(err.message);
  } finally {
    playing = false;
  }
}

async function load() {
  setError('');
  el('list').innerHTML = '<div style="padding:16px" class="hint">training the detector on the other split…</div>';
  try {
    const summary = await post('/api/live/start', {subject: el('subject').value});
    CONFIG.channels = summary.channels;
    catalog = await post('/api/live/list', {subject: el('subject').value});
    results.clear();
    updateTally();
    renderList();
    el('clockL').textContent = `${catalog.length} signals the model has never seen, pick one`;
  } catch (err) {
    setError(err.message);
    el('list').innerHTML = '';
  }
}

el('chips').addEventListener('click', ev => {
  const btn = ev.target.closest('.chip');
  if (!btn) return;
  filter = btn.dataset.c === 'all' ? null : btn.dataset.c;
  el('chips').querySelectorAll('.chip').forEach(c => c.classList.toggle('on', c === btn));
  renderList();
});
el('subject').addEventListener('change', load);

// ?subject=A01T&signal=3 opens straight onto one signal, so a particular result
// can be linked to. The number matches the row label, which starts at 1.
async function boot() {
  const q = new URLSearchParams(location.search);
  if (q.has('subject') && CONFIG.subjects.includes(q.get('subject'))) {
    el('subject').value = q.get('subject');
  }
  await load();
  if (q.has('signal')) {
    const n = parseInt(q.get('signal'), 10) - 1;
    if (catalog.some(e => e.index === n)) identify(n);
  }
}
boot();
</script>
</body>
</html>
"""
