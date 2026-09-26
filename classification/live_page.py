"""The HTML for the live view, kept out of server.py so both stay readable."""

LIVE_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live — motor imagery</title>
<style>
  :root {
    --ground:#E9EDEF; --panel:#FFF; --ink:#131C21; --muted:#5A6A72; --faint:#8798A0;
    --rule:#D2DADD; --accent:#0B6E5F; --accent-soft:#D8EAE5; --bad:#9E2B26;
    --secret:#7A4B00; --secret-bg:#FBEFD8; --secret-rule:#D8A94A;
    --left:#2E6E9E; --right:#A85F18; --foot:#67539B; --tongue:#2C7A57;
    --trace:#8AA3AC;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground:#0E1417; --panel:#161F23; --ink:#E4ECEE; --muted:#9DAEB5; --faint:#74868E;
      --rule:#2A383E; --accent:#58C4AC; --accent-soft:#123A32; --bad:#E0766F;
      --secret:#E8B84B; --secret-bg:#31280F; --secret-rule:#6B551C;
      --left:#74ADDA; --right:#DFA365; --foot:#A899DD; --tongue:#6EC59C;
      --trace:#5C7079;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:26px 20px 70px}
  h1{font-size:25px;margin:0 0 4px;letter-spacing:-.02em}
  .sub{color:var(--muted);margin:0 0 8px}
  a{color:var(--accent)}
  .card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:18px}

  .promise{border-left:4px solid var(--accent);background:var(--panel);
           border:1px solid var(--rule);border-left-width:4px;border-radius:5px;
           padding:13px 16px;margin:18px 0;font-size:14px;color:var(--muted)}
  .promise b{color:var(--ink)}

  .bar-controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:16px 0}
  select,button{font:inherit;border-radius:5px;border:1px solid var(--rule);padding:8px 12px;
                background:var(--panel);color:var(--ink);cursor:pointer}
  button.primary{background:var(--accent);color:var(--panel);border-color:var(--accent);font-weight:600}
  button:disabled{opacity:.5;cursor:not-allowed}
  .tally{margin-left:auto;font:13px ui-monospace,Menlo,monospace;color:var(--muted)}

  .scope{margin-top:4px}
  .scope svg{display:block;width:100%;height:190px}
  .chlabel{font:11px ui-monospace,Menlo,monospace;fill:var(--faint)}
  .clock{font:12px ui-monospace,Menlo,monospace;color:var(--faint);
         display:flex;justify-content:space-between;margin-top:6px}

  .two{display:grid;grid-template-columns:1fr 330px;gap:18px;margin-top:18px;align-items:start}
  @media (max-width:800px){.two{grid-template-columns:1fr}}

  .bars{display:grid;gap:8px}
  .bar{display:grid;grid-template-columns:58px 1fr 44px;gap:10px;align-items:center}
  .bar .nm{font-size:13px;color:var(--muted)}
  .track{height:20px;background:var(--ground);border:1px solid var(--rule);
         border-radius:3px;position:relative;overflow:hidden}
  .fill{position:absolute;inset:0 auto 0 0;transition:width .28s ease}
  .chance{position:absolute;top:-1px;bottom:-1px;left:25%;border-left:2px dashed var(--bad)}
  .bar .pc{font:600 13px ui-monospace,Menlo,monospace;text-align:right}

  .verdict{text-align:center;padding:20px 16px}
  .verdict .cap{font:11px ui-monospace,Menlo,monospace;letter-spacing:.1em;
                text-transform:uppercase;color:var(--faint)}
  .verdict .big{font-size:38px;font-weight:700;letter-spacing:-.02em;margin:6px 0 2px;line-height:1.1}
  .verdict .conf{font:13px ui-monospace,Menlo,monospace;color:var(--muted)}

  .secret{margin-top:14px;background:var(--secret-bg);border:1px dashed var(--secret-rule);
          border-radius:6px;padding:16px;text-align:center}
  .secret .cap{font:11px ui-monospace,Menlo,monospace;letter-spacing:.09em;
               text-transform:uppercase;color:var(--secret)}
  .secret .big{font-size:30px;font-weight:700;margin:6px 0 2px;line-height:1.1}
  .secret .note{font-size:12px;color:var(--secret)}
  .secret.hidden .big{filter:blur(11px);user-select:none}

  .log{margin-top:22px}
  table{border-collapse:collapse;width:100%;font-size:13.5px}
  th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.07em;
     text-transform:uppercase;color:var(--faint)}
  td{font-family:ui-monospace,Menlo,monospace}
  .ok{color:var(--accent);font-weight:700}
  .miss{color:var(--bad);font-weight:700}
  .err{color:var(--bad);border:1px solid var(--bad);border-radius:5px;padding:13px;margin-top:14px}
  .muted{color:var(--muted)}
  label.chk{display:inline-flex;align-items:center;gap:7px;font-size:14px;color:var(--muted);cursor:pointer}
  label.chk input{accent-color:var(--accent);width:15px;height:15px;cursor:pointer}
</style>
</head>
<body>
<div class="wrap">
  <h1>Live view</h1>
  <p class="sub">Trials stream past a trained detector in real time. <a href="/">Back to the test bench</a></p>

  <div class="promise">
    <b>The model cannot see the answer.</b> The detector was trained on one part of
    the recording; everything streamed here is from the part it was held out of. The
    true label is attached to the response <i>after</i> the model has produced its
    probabilities, so it reaches your screen and never the classifier.
  </div>

  <div class="bar-controls">
    <select id="subject"></select>
    <select id="speed">
      <option value="1">real time</option>
      <option value="2" selected>2× speed</option>
      <option value="4">4× speed</option>
      <option value="0">as fast as possible</option>
    </select>
    <button class="primary" id="play">Start</button>
    <button id="next" disabled>Skip trial</button>
    <label class="chk"><input type="checkbox" id="peek"> show the answer straight away</label>
    <span class="tally" id="tally">—</span>
  </div>

  <div class="card scope">
    <svg id="scope" viewBox="0 0 900 190" preserveAspectRatio="none" aria-label="EEG traces"></svg>
    <div class="clock">
      <span id="clockL">waiting</span>
      <span id="clockR"></span>
    </div>
  </div>

  <div class="two">
    <div class="card">
      <div style="font-size:14px;font-weight:600;margin-bottom:3px">What the model thinks, as it watches</div>
      <div class="muted" style="font-size:13px;margin-bottom:14px">Updates every window. Dashed line is the 25% chance level.</div>
      <div class="bars" id="bars"></div>
    </div>

    <div>
      <div class="card verdict">
        <div class="cap">Model says</div>
        <div class="big" id="call">—</div>
        <div class="conf" id="conf">waiting for the first full window</div>
      </div>

      <div class="secret hidden" id="secret">
        <div class="cap">For your eyes only</div>
        <div class="big" id="truth">—</div>
        <div class="note" id="truthNote">hidden until the trial ends</div>
      </div>
    </div>
  </div>

  <div class="log">
    <table id="log"><tbody><tr><td class="muted">no trials yet</td></tr></tbody></table>
  </div>
  <div id="error"></div>
</div>

<script>
const CONFIG = __CONFIG__;
const CLASSES = CONFIG.classes;
const COLORS = {left:'var(--left)', right:'var(--right)', foot:'var(--foot)', tongue:'var(--tongue)'};

const el = id => document.getElementById(id);
const subjectSel = el('subject'), speedSel = el('speed');
const playBtn = el('play'), nextBtn = el('next'), peek = el('peek');

subjectSel.innerHTML = CONFIG.subjects.map(s => `<option>${s}</option>`).join('');

el('bars').innerHTML = CLASSES.map(c => `
  <div class="bar">
    <span class="nm">${c}</span>
    <span class="track"><span class="fill" id="f_${c}" style="width:0%;background:${COLORS[c]}"></span><span class="chance"></span></span>
    <span class="pc" id="p_${c}">0%</span>
  </div>`).join('');

let session = null;     // description from the server
let order = [];         // trial indices still to play
let running = false;
let cancelled = false;
let log = [];           // {index, truth, call, conf, ok}

function setError(msg) {
  el('error').innerHTML = msg ? `<div class="err">${msg}</div>` : '';
}

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'request failed');
  return data;
}

function drawTraces(traces) {
  const W = 900, H = 190, n = traces.length;
  const lane = H / n;
  const parts = [];

  traces.forEach((points, i) => {
    const mid = lane * i + lane / 2;
    const amp = lane * 0.38;
    const d = points.map((v, j) => {
      const x = (j / (points.length - 1)) * W;
      const y = mid - v * amp;
      return `${j ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    parts.push(`<path d="${d}" fill="none" stroke="var(--trace)" stroke-width="1.1"
                 stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`);
    parts.push(`<text class="chlabel" x="6" y="${(mid - lane * 0.32).toFixed(1)}">${CONFIG.channels[i] || ''}</text>`);
  });

  el('scope').innerHTML = parts.join('');
}

function showFrame(frame, frameIndex) {
  drawTraces(frame.traces);

  CLASSES.forEach(c => {
    const p = frame.probabilities[c] ?? 0;
    el('f_' + c).style.width = (p * 100).toFixed(1) + '%';
    el('p_' + c).textContent = Math.round(p * 100) + '%';
  });

  el('call').textContent = frame.label;
  el('call').style.color = COLORS[frame.label] || 'var(--ink)';
  el('conf').textContent = `${Math.round(frame.confidence * 100)}% confident`;

  el('clockL').textContent = `${frame.t.toFixed(2)} s into the trial`;
  el('clockR').textContent = frameIndex === 0
    ? 'first full window — the model is at its most accurate here'
    : `window ${frameIndex + 1}, sliding past the imagery`;
}

function revealTruth(truth, call) {
  const box = el('secret');
  box.classList.remove('hidden');
  el('truth').textContent = truth;
  el('truth').style.color = COLORS[truth] || 'var(--ink)';
  el('truthNote').textContent = truth === call
    ? 'the model got this one right'
    : `the model said ${call}`;
}

function hideTruth() {
  el('secret').classList.add('hidden');
  el('truth').textContent = '—';
  el('truthNote').textContent = peek.checked ? 'revealing…' : 'hidden until the trial ends';
}

function renderLog() {
  const body = document.querySelector('#log tbody');
  if (!log.length) { body.innerHTML = '<tr><td class="muted">no trials yet</td></tr>'; return; }
  const head = `<tr><th>trial</th><th>really was</th><th>model said</th><th>confidence</th><th></th></tr>`;
  body.innerHTML = head + log.slice().reverse().slice(0, 15).map(r => `
    <tr>
      <td>${r.index + 1}</td>
      <td style="color:${COLORS[r.truth]}">${r.truth}</td>
      <td>${r.call}</td>
      <td>${Math.round(r.conf * 100)}%</td>
      <td class="${r.ok ? 'ok' : 'miss'}">${r.ok ? 'ok' : 'miss'}</td>
    </tr>`).join('');
}

function updateTally() {
  if (!log.length) { el('tally').textContent = '—'; return; }
  const hits = log.filter(r => r.ok).length;
  el('tally').textContent = `${hits}/${log.length} correct (${Math.round(hits / log.length * 100)}%) · chance 25%`;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function playTrial(index) {
  const data = await post('/api/live/trial', {subject: subjectSel.value, index});
  const frames = data.frames;
  if (!frames.length) return;

  hideTruth();
  if (peek.checked) revealTruth(data.truth, '…');

  const speed = parseFloat(speedSel.value);
  const gap = speed === 0 ? 120 : (CONFIG.step_s * 1000) / speed;

  for (let i = 0; i < frames.length; i++) {
    if (cancelled) return;
    showFrame(frames[i], i);
    await sleep(gap);
  }

  // The first full window is the model's best read; later windows slide past the
  // imagery into the rest period and get steadily worse.
  const decision = frames[0];
  const ok = decision.label === data.truth;

  // Redraw the decision frame so the bars agree with the verdict above them.
  showFrame(decision, 0);
  revealTruth(data.truth, decision.label);
  el('conf').textContent = `${Math.round(decision.confidence * 100)}% confident at the first full window`;
  el('clockR').textContent = 'decision taken at the first full window';

  log.push({index, truth: data.truth, call: decision.label, conf: decision.confidence, ok});
  renderLog();
  updateTally();

  await sleep(speed === 0 ? 200 : 1100);
}

async function run() {
  running = true; cancelled = false;
  playBtn.textContent = 'Stop'; nextBtn.disabled = false;
  setError('');

  try {
    if (!session || session.subject !== subjectSel.value) {
      el('clockL').textContent = 'training the detector on the other split…';
      session = await post('/api/live/start', {subject: subjectSel.value});
      CONFIG.step_s = session.step_s;
      CONFIG.channels = session.channels;
      order = Array.from({length: session.n_trials}, (_, i) => i);
      log = []; renderLog(); updateTally();
    }

    while (running && order.length) {
      if (cancelled) break;
      await playTrial(order.shift());
    }

    if (!order.length) {
      el('clockL').textContent = 'end of the held-out trials';
      playBtn.textContent = 'Start';
      running = false;
    }
  } catch (err) {
    setError(err.message);
    playBtn.textContent = 'Start';
    running = false;
  }
}

playBtn.addEventListener('click', () => {
  if (running) {
    running = false; cancelled = true;
    playBtn.textContent = 'Start';
    nextBtn.disabled = true;
  } else {
    run();
  }
});

nextBtn.addEventListener('click', () => { cancelled = true; setTimeout(() => { cancelled = false; if (running) run(); }, 60); });
peek.addEventListener('change', () => { if (!peek.checked) hideTruth(); });
subjectSel.addEventListener('change', () => { session = null; running = false; cancelled = true; playBtn.textContent = 'Start'; });

// Settings can come from the URL, so a view worth showing someone can be linked.
// ?auto=1 starts it on load.
(function applyQuery() {
  const q = new URLSearchParams(location.search);
  if (q.has('subject') && CONFIG.subjects.includes(q.get('subject'))) subjectSel.value = q.get('subject');
  if (q.has('speed')) speedSel.value = q.get('speed');
  if (q.get('peek') === '1') peek.checked = true;
  if (q.get('auto') === '1') run();
})();
</script>
</body>
</html>
"""
