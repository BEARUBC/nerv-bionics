"""HTML for the test bench: choose a configuration, evaluate it, read the scores."""

BENCH_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motor imagery test bench</title>
<style>
  :root {
    --ground:#EEF2F3; --panel:#FFF; --ink:#131C21; --muted:#5A6A72; --faint:#8798A0;
    --rule:#DCE3E6; --accent:#0B6E5F; --accent-soft:#D8EAE5; --bad:#9E2B26;
    --train:#0B6E5F; --val:#8798A0; --test:#2E6E9E;
    --left:#2E6E9E; --right:#A85F18; --foot:#67539B; --tongue:#2C7A57;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground:#0E1417; --panel:#161F23; --ink:#E4ECEE; --muted:#9DAEB5; --faint:#74868E;
      --rule:#2A383E; --accent:#58C4AC; --accent-soft:#123A32; --bad:#E0766F;
      --train:#58C4AC; --val:#74868E; --test:#74ADDA;
      --left:#74ADDA; --right:#DFA365; --foot:#A899DD; --tongue:#6EC59C;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  .mono{font-family:ui-monospace,Menlo,monospace}

  header{background:var(--panel);border-bottom:1px solid var(--rule)}
  header .in{max-width:1180px;margin:0 auto;padding:0 16px;display:flex;align-items:center;gap:16px;height:56px}
  @media (max-width:520px){.brand span{display:none} nav a{padding:7px 9px;font-size:13px}}
  .brand{font-weight:700;letter-spacing:-.01em}
  .brand span{color:var(--faint);font-weight:400}
  nav{display:flex;gap:4px;margin-left:auto}
  nav a{color:var(--muted);text-decoration:none;padding:7px 12px;border-radius:6px;font-size:14px}
  nav a:hover{background:var(--ground)}
  nav a.on{color:var(--accent);background:var(--accent-soft);font-weight:600}

  .wrap{max-width:1180px;margin:0 auto;padding:26px 16px 70px}
  .layout{display:grid;grid-template-columns:300px 1fr;gap:20px;align-items:start}
  @media (max-width:880px){.layout{grid-template-columns:1fr}}
  .layout>*,.grid2>*{min-width:0}
  .scroll{overflow-x:auto}
  .card{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:20px}
  aside.card{position:sticky;top:16px}
  @media (max-width:880px){aside.card{position:static}}

  .field{margin-bottom:20px}
  .label{display:block;font:600 11px/1 ui-monospace,Menlo,monospace;letter-spacing:.09em;
         text-transform:uppercase;color:var(--faint);margin-bottom:9px}
  .seg{display:flex;background:var(--ground);border-radius:8px;padding:3px;gap:3px}
  .seg button{flex:1;border:0;background:none;color:var(--muted);font:500 13px inherit;
              padding:7px 6px;border-radius:6px;cursor:pointer}
  .seg button.on{background:var(--panel);color:var(--ink);font-weight:600;
                 box-shadow:0 1px 2px rgba(0,0,0,.12)}
  select{width:100%;font:inherit;font-size:14px;border-radius:8px;border:1px solid var(--rule);
         padding:8px 10px;background:var(--panel);color:var(--ink);cursor:pointer}
  .chips{display:flex;gap:6px;flex-wrap:wrap}
  .chip{font:12px ui-monospace,Menlo,monospace;padding:4px 10px;border-radius:99px;
        border:1px solid var(--rule);background:var(--panel);color:var(--muted);cursor:pointer}
  .chip.on{border-color:var(--accent);background:var(--accent-soft);color:var(--accent);font-weight:600}
  .chip i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}

  .timeline{position:relative;height:30px;margin:4px 0 2px}
  .timeline .axis{position:absolute;left:0;right:0;top:13px;height:4px;background:var(--ground);border-radius:2px}
  .timeline .win{position:absolute;top:9px;height:12px;background:var(--accent);border-radius:3px;opacity:.85}
  .timeline .cue{position:absolute;top:2px;bottom:2px;border-left:2px solid var(--ink)}
  .ticks{display:flex;justify-content:space-between;font:10px ui-monospace,Menlo,monospace;color:var(--faint)}
  .range{display:grid;grid-template-columns:44px 1fr 52px;gap:8px;align-items:center;font-size:13px;color:var(--muted)}
  .range b{font:600 12px ui-monospace,Menlo,monospace;color:var(--ink);text-align:right}
  input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer}
  .note{font-size:12px;color:var(--muted);margin-top:6px}
  .note.bad{color:var(--bad)}
  .check{display:flex;gap:8px;align-items:flex-start;font-size:13px;color:var(--muted);margin-top:10px;cursor:pointer}
  .check input{accent-color:var(--accent);margin-top:3px}

  .run{width:100%;padding:12px;font:600 15px inherit;cursor:pointer;background:var(--accent);
       color:var(--panel);border:0;border-radius:8px}
  .run:disabled{opacity:.5;cursor:wait}

  .empty{color:var(--muted);text-align:center;padding:60px 20px}
  .empty b{color:var(--ink)}
  .spin{color:var(--muted);padding:40px 0;text-align:center}
  .err{color:var(--bad);border:1px solid var(--bad);border-radius:8px;padding:14px}

  .ctx{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--muted);margin-bottom:16px}
  .ctx b{color:var(--ink)}
  .tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
  @media (max-width:640px){.tiles{grid-template-columns:repeat(2,1fr)}}
  .tile{border:1px solid var(--rule);border-radius:8px;padding:14px}
  .tile .k{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
  .tile .v{font-size:32px;font-weight:700;letter-spacing:-.02em;line-height:1.15;margin-top:4px}
  .tile .s{font-size:12px;color:var(--muted)}
  .tile.lead{border-color:var(--accent);background:var(--accent-soft)}
  .tile.lead .v{color:var(--accent)}

  h3{font-size:15px;margin:28px 0 2px}
  .hint{color:var(--muted);font-size:13px;margin:0 0 12px}

  .split{display:flex;height:30px;border-radius:6px;overflow:hidden;font:600 12px ui-monospace,Menlo,monospace}
  .split div{display:flex;align-items:center;justify-content:center;color:var(--panel);white-space:nowrap}
  .splitlegend{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px;font-size:13px;color:var(--muted)}
  @media (max-width:520px){.splitlegend{grid-template-columns:1fr}.tile .v{font-size:26px}}
  .splitlegend b{display:block;color:var(--ink);font-size:18px}

  table{border-collapse:collapse;width:100%;font-size:13.5px}
  th,td{padding:8px 10px;border-bottom:1px solid var(--rule);text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{font:600 11px ui-monospace,Menlo,monospace;letter-spacing:.07em;text-transform:uppercase;color:var(--faint)}
  td{font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
  tr.focus td{background:var(--accent-soft)}
  .minibar{display:inline-block;height:8px;border-radius:2px;vertical-align:middle;margin-right:8px;opacity:.85}

  .grid2{display:grid;grid-template-columns:1.1fr .9fr;gap:24px}
  @media (max-width:760px){.grid2{grid-template-columns:1fr}}
  .cm{display:grid;grid-template-columns:56px repeat(4,1fr);gap:3px;font:12px ui-monospace,Menlo,monospace;max-width:420px}
  .cm .h{color:var(--faint);text-align:center;padding:4px 0;font-size:11px}
  .cm .r{color:var(--muted);display:flex;align-items:center}
  .cm .c{min-height:40px;display:flex;align-items:center;justify-content:center;border-radius:4px;font-weight:600}

  .dots{position:relative;height:56px;margin-top:6px;border-bottom:1px solid var(--rule)}
  .dots .d{position:absolute;width:10px;height:10px;border-radius:50%;background:var(--test);opacity:.75;transform:translate(-50%,0)}
  .dots .m{position:absolute;top:0;bottom:-1px;border-left:2px solid var(--ink)}
  .dots .ch{position:absolute;top:0;bottom:-1px;border-left:2px dashed var(--bad)}
  .dotaxis{display:flex;justify-content:space-between;font:10px ui-monospace,Menlo,monospace;color:var(--faint);margin-top:3px}

  .history td.best{color:var(--accent);font-weight:700}
</style>
</head>
<body>
<header><div class="in">
  <div class="brand">NERV <span>motor imagery</span></div>
  <nav>
    <a href="/" class="on">Test bench</a>
    <a href="/live">Live stream</a>
    <a href="/identify">Identify</a>
  </nav>
</div></header>

<div class="wrap">
  <div class="layout">
    <aside class="card">
      <form id="controls">
        <div class="field">
          <span class="label">Recording</span>
          <div class="seg" id="subjects"></div>
        </div>

        <div class="field">
          <span class="label">Model</span>
          <select id="model"></select>
          <div class="note">★ recommended: the most accurate on held-out trials</div>
        </div>

        <div class="field">
          <span class="label">Time window after cue</span>
          <div class="timeline"><div class="axis"></div><div class="cue" style="left:0"></div><div class="win" id="win"></div></div>
          <div class="ticks"><span>cue</span><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span><span>6 s</span></div>
          <div class="range" style="margin-top:10px"><span>start</span><input type="range" id="tmin" min="0" max="2" step="0.25" value="0.5"><b id="tminV"></b></div>
          <div class="range"><span>end</span><input type="range" id="tmax" min="1.5" max="6" step="0.25" value="4.5"><b id="tmaxV"></b></div>
          <div class="note" id="dur"></div>
        </div>

        <div class="field">
          <span class="label">How to test</span>
          <div class="seg" id="protocol">
            <button type="button" data-v="holdout" class="on">50 / 25 / 25</button>
            <button type="button" data-v="cv">5-fold CV</button>
          </div>
          <div class="note" id="protocolNote"></div>
          <label class="check" id="repeatRow"><input type="checkbox" id="repeat">
            <span>Also average over 20 random splits (about a minute)</span></label>
        </div>

        <div class="field">
          <span class="label">Focus class</span>
          <div class="chips" id="classes"></div>
        </div>

        <button type="submit" class="run" id="run">Run test</button>
      </form>
    </aside>

    <main>
      <div class="card" id="result">
        <div class="empty">
          <p><b>Choose a recording and model, then press Run test.</b></p>
          <p>The default trains on half the trials, checks on a quarter, and reports accuracy
          and precision on the last quarter, which the model never sees during training.</p>
        </div>
      </div>

      <h3>Runs this session</h3>
      <p class="hint">Every run is kept here so you can compare settings.</p>
      <div class="card scroll" style="padding:8px 20px 12px">
        <table class="history" id="history"><tbody><tr><td class="hint" style="font-family:inherit">nothing yet</td></tr></tbody></table>
      </div>
    </main>
  </div>
</div>

<script>
const CONFIG = __CONFIG__;
const CLASSES = CONFIG.classes;
const LABELS = Object.fromEntries(CONFIG.models.map(m => [m.key, m.label]));
const history = [];
const state = {subject: CONFIG.subjects[0], protocol: 'holdout', focus: ''};

const $ = id => document.getElementById(id);
const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';

function segmented(host, items, key) {
  $(host).innerHTML = items.map(v =>
    `<button type="button" data-v="${v}" class="${v === state[key] ? 'on' : ''}">${v}</button>`).join('');
  bindSeg(host, key);
}
function bindSeg(host, key) {
  $(host).querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    state[key] = b.dataset.v;
    $(host).querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    refresh();
  }));
}
segmented('subjects', CONFIG.subjects, 'subject');
bindSeg('protocol', 'protocol');

$('model').innerHTML = CONFIG.models.map((m, i) =>
  `<option value="${m.key}">${m.label}${i === 0 ? ' ★' : ''}</option>`).join('');

function renderChips() {
  const items = [{v: '', l: 'all'}].concat(CLASSES.map(c => ({v: c, l: c})));
  $('classes').innerHTML = items.map(it =>
    `<button type="button" class="chip ${it.v === state.focus ? 'on' : ''}" data-v="${it.v}">` +
    (it.v ? `<i style="background:var(--${it.v})"></i>` : '') + `${it.l}</button>`).join('');
  $('classes').querySelectorAll('.chip').forEach(b => b.addEventListener('click', () => {
    state.focus = b.dataset.v; renderChips();
  }));
}
renderChips();

const tmin = $('tmin'), tmax = $('tmax'), runBtn = $('run');

function refresh() {
  const a = parseFloat(tmin.value), b = parseFloat(tmax.value), len = b - a;
  $('tminV').textContent = a.toFixed(2) + ' s';
  $('tmaxV').textContent = b.toFixed(2) + ' s';
  const win = $('win');
  win.style.left = (Math.max(a, 0) / 6 * 100) + '%';
  win.style.width = (Math.max(len, 0) / 6 * 100) + '%';
  win.style.background = len < 1 ? 'var(--bad)' : 'var(--accent)';
  const dur = $('dur');
  if (len < 1) {
    dur.className = 'note bad';
    dur.textContent = 'Window is under 1 s. The models need at least 1 s of signal.';
    runBtn.disabled = true;
  } else {
    dur.className = 'note';
    dur.textContent = `${len.toFixed(2)} s per trial, ${Math.round(len * 250)} samples`;
    runBtn.disabled = false;
  }
  const holdout = state.protocol === 'holdout';
  $('repeatRow').style.display = holdout ? 'flex' : 'none';
  $('protocolNote').textContent = holdout
    ? 'Train on 50 %, check on 25 %, final score on the last 25 %.'
    : 'Train on 80 % and test on 20 %, five times, so every trial is tested once.';
}
tmin.addEventListener('input', refresh);
tmax.addEventListener('input', refresh);
refresh();

function tile(k, v, s, lead) {
  return `<div class="tile ${lead ? 'lead' : ''}"><div class="k">${k}</div><div class="v">${pct(v)}</div><div class="s">${s}</div></div>`;
}

function heat(value, colour) {
  return `background:color-mix(in srgb, ${colour} ${Math.round(value * 85 + 4)}%, var(--panel))`;
}

function renderResult(r) {
  const m = r.metrics;
  const holdout = r.protocol === 'holdout';
  const where = holdout ? 'on the test quarter' : 'over all 5 folds';

  let html = `<div class="ctx"><span><b>${r.subject}</b></span><span>${LABELS[r.model] || r.model}</span>
    <span>${r.tmin.toFixed(2)}–${r.tmax.toFixed(2)} s</span>
    <span>${holdout ? '50 / 25 / 25 hold-out' : '5-fold cross-validation'}</span>
    <span>${r.seconds.toFixed(1)} s to run</span></div>`;

  html += `<div class="tiles">
    ${tile('Accuracy', m.accuracy, `${where} · chance is 25%`, true)}
    ${tile('Precision', m.precision, 'when it names a class, how often right')}
    ${tile('Recall', m.recall, 'of each real class, how much it caught')}
    ${tile('F1 score', m.f1, `balance of the two · kappa ${m.kappa.toFixed(2)}`)}
  </div>`;

  if (holdout) {
    const n = r.split;
    const total = n.train + n.validation + n.test;
    html += `<h3>How the trials were split</h3>
      <p class="hint">Stratified, so every part keeps the four classes in balance.
      ${r.n_dropped ? r.n_dropped + ' noisy training trials were dropped as artifacts.' : ''}</p>
      <div class="split">
        <div style="flex:${n.train};background:var(--train)">train ${n.train}</div>
        <div style="flex:${n.validation};background:var(--val)">check ${n.validation}</div>
        <div style="flex:${n.test};background:var(--test)">test ${n.test}</div>
      </div>
      <div class="splitlegend">
        <div><b>${n.train} trials</b>the model learns from these</div>
        <div><b>${pct(r.validation.accuracy)}</b>accuracy on the check set</div>
        <div><b>${pct(m.accuracy)}</b>accuracy on the test set</div>
      </div>`;
  }

  if (r.repeats) {
    const rp = r.repeats;
    const lo = 0.25, hi = 1.0, x = v => ((v - lo) / (hi - lo) * 100);
    const counts = {};
    const dots = rp.accuracies.map(a => {
      const key = a.toFixed(3); counts[key] = (counts[key] || 0) + 1;
      return `<span class="d" style="left:${x(a)}%;bottom:${(counts[key] - 1) * 11 + 2}px" title="${pct(a)}"></span>`;
    }).join('');
    html += `<h3>Over ${rp.repeats} random splits</h3>
      <p class="hint">One test quarter is about ${r.split.test} trials, so a single run can be lucky or unlucky.
      Each dot is one split; the solid line is the average.</p>
      <div class="tiles" style="grid-template-columns:repeat(2,1fr);margin-bottom:12px">
        ${tile('Average accuracy', rp.accuracy.mean, `± ${pct(rp.accuracy.std)} between splits`, true)}
        ${tile('Average precision', rp.precision.mean, `± ${pct(rp.precision.std)} between splits`)}
      </div>
      <div class="dots">${dots}<span class="m" style="left:${x(rp.accuracy.mean)}%"></span>
        <span class="ch" style="left:0"></span></div>
      <div class="dotaxis"><span>25% chance</span><span>50%</span><span>75%</span><span>100%</span></div>`;
  }

  const rows = CLASSES.map(c => {
    const p = m.per_class[c];
    return `<tr class="${state.focus === c ? 'focus' : ''}">
      <td><span class="minibar" style="width:10px;background:var(--${c})"></span>${c}</td>
      <td><span class="minibar" style="width:${p.precision * 60}px;background:var(--${c})"></span>${pct(p.precision, 0)}</td>
      <td><span class="minibar" style="width:${p.recall * 60}px;background:var(--${c})"></span>${pct(p.recall, 0)}</td>
      <td>${pct(p.f1, 0)}</td><td>${p.support}</td></tr>`;
  }).join('');

  const cells = m.matrix.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return `<div class="r">${CLASSES[i]}</div>` + row.map((n, j) => {
      const share = n / total;
      const colour = i === j ? 'var(--accent)' : 'var(--bad)';
      const ink = share > 0.5 ? 'color:var(--panel)' : '';
      return `<div class="c" style="${heat(share, colour)};${ink}" title="${n} of ${total}">${n}</div>`;
    }).join('');
  }).join('');

  html += `<div class="grid2">
    <div><h3>Per class</h3><p class="hint">Precision and recall for each imagined movement.</p>
      <div class="scroll"><table><thead><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>trials</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div>
    <div><h3>What it confused with what</h3><p class="hint">Rows are the real class, columns what it said.</p>
      <div class="cm"><div></div>${CLASSES.map(c => `<div class="h">${c}</div>`).join('')}${cells}</div></div>
  </div>`;

  $('result').innerHTML = html;
}

function renderHistory() {
  const body = document.querySelector('#history tbody');
  if (!history.length) {
    body.innerHTML = '<tr><td class="hint" style="font-family:inherit">nothing yet</td></tr>';
    return;
  }
  const best = Math.max(...history.map(h => h.metrics.accuracy));
  body.innerHTML = `<tr><th>recording</th><th>model</th><th>window</th><th>test</th><th>accuracy</th><th>precision</th></tr>` +
    history.slice().reverse().map(h => `<tr>
      <td>${h.subject}</td><td>${LABELS[h.model] || h.model}</td>
      <td>${h.tmin.toFixed(2)}–${h.tmax.toFixed(2)} s</td>
      <td>${h.protocol === 'holdout' ? '50/25/25' : '5-fold'}${h.repeats ? ' ×' + h.repeats.repeats : ''}</td>
      <td class="${h.metrics.accuracy === best ? 'best' : ''}">${pct(h.metrics.accuracy)}</td>
      <td>${pct(h.metrics.precision)}</td></tr>`).join('');
}

$('controls').addEventListener('submit', async ev => {
  ev.preventDefault();
  const repeat = state.protocol === 'holdout' && $('repeat').checked;
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';
  $('result').innerHTML = `<p class="spin">Training and testing${repeat ? ' over 20 splits, this takes a little while' : ''}...</p>`;

  const body = {
    subject: state.subject, model: $('model').value, protocol: state.protocol,
    tmin: parseFloat(tmin.value), tmax: parseFloat(tmax.value), repeats: repeat ? 20 : 0,
  };

  try {
    const res = await fetch('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'run failed');
    renderResult(data);
    history.push(data);
    renderHistory();
  } catch (err) {
    $('result').innerHTML = `<div class="err"><b>That run failed.</b><br>${err.message}</div>`;
  } finally {
    runBtn.textContent = 'Run test';
    refresh();
  }
});

// Settings can go in the URL so a configuration can be bookmarked or shared:
// ?subject=A04T&model=riemann_ts_lr&tmin=1&tmax=4&test=cv&run=1
(function applyQuery() {
  const q = new URLSearchParams(location.search);
  if (![...q.keys()].length) return;
  const click = (host, v) => {
    const b = $(host).querySelector(`button[data-v="${v}"]`);
    if (b) b.click();
  };
  if (q.has('subject')) click('subjects', q.get('subject'));
  if (q.has('test')) click('protocol', q.get('test'));
  if (q.has('model') && LABELS[q.get('model')]) $('model').value = q.get('model');
  if (q.has('class')) { state.focus = q.get('class') === 'all' ? '' : q.get('class'); renderChips(); }
  if (q.has('tmin')) tmin.value = q.get('tmin');
  if (q.has('tmax')) tmax.value = q.get('tmax');
  if (q.get('repeat') === '1') $('repeat').checked = true;
  refresh();
  if (q.get('run') === '1' && !runBtn.disabled) $('controls').requestSubmit();
})();
</script>
</body>
</html>
"""
