"""HTML for the review screen: choose a configuration, evaluate it, read the scores."""

from classification.monitor_theme import MONITOR_JS, THEME_CSS, status_bar

BENCH_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review, NERV motor imagery</title>
<style>
__THEME_CSS__
  .layout{display:grid;grid-template-columns:300px minmax(0,1fr);gap:10px;align-items:start}
  @media (max-width:880px){.layout{grid-template-columns:1fr}}
  .setup .sec{padding:10px 12px;border-bottom:1px solid var(--line);display:grid;gap:8px}
  .setup .sec:last-child{border-bottom:0}
  .setup .lbl{font:600 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
  .seg{display:flex;gap:3px;background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:3px}
  .seg button{flex:1;border:0;background:none;color:var(--ink-2);font:600 12px var(--mono);letter-spacing:.04em;
              padding:7px 4px;border-radius:3px;cursor:pointer}
  .seg button.on{background:var(--trace);color:#000}
  select.full{width:100%;background:var(--bg);border:1px solid var(--line-2);border-radius:4px;padding:8px;color:var(--ink);cursor:pointer}
  .timeline{position:relative;height:26px}
  .timeline .axis{position:absolute;left:0;right:0;top:11px;height:4px;background:var(--panel-2);border-radius:2px}
  .timeline .win{position:absolute;top:7px;height:12px;background:var(--trace);border-radius:2px}
  .timeline .rest{position:absolute;top:3px;bottom:3px;border-left:1px dashed var(--ink-3)}
  .timeline .cue{position:absolute;top:1px;bottom:1px;left:0;border-left:2px solid var(--advisory)}
  .ticks{display:flex;justify-content:space-between;font:10px var(--mono);color:var(--ink-3)}
  .range{display:grid;grid-template-columns:40px 1fr 54px;gap:8px;align-items:center;font:12px var(--mono);color:var(--ink-2)}
  .range b{color:var(--ink);text-align:right}
  input[type=range]{width:100%;accent-color:var(--trace);cursor:pointer}
  .note{font:12px/1.45 var(--mono);color:var(--ink-3)}
  .note.bad{color:var(--alarm)}
  .check{display:flex;gap:8px;align-items:flex-start;font:12px var(--mono);color:var(--ink-2);cursor:pointer}
  .check input{accent-color:var(--trace);margin-top:2px}
  .chips{display:flex;gap:4px;flex-wrap:wrap}
  .chips button{background:var(--panel-2);border:1px solid var(--line-2);border-radius:3px;padding:3px 8px;cursor:pointer;
                font:600 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}
  .chips button.on{background:var(--trace);border-color:var(--trace);color:#000}

  .readouts{display:grid;grid-template-columns:repeat(5,minmax(0,1fr))}
  @media (max-width:760px){.readouts{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .readouts .param{border-right:1px solid var(--line);border-bottom:0}
  .readouts .param:last-child{border-right:0}
  .readouts .param.lead .v{color:var(--trace)}
  .empty{padding:48px 16px;text-align:center;color:var(--ink-2);font:13px/1.6 var(--mono)}
  .empty b{color:var(--ink)}
  .splitbar{display:flex;gap:2px;height:28px;border-radius:3px;overflow:hidden;font:600 12px var(--mono)}
  .splitbar div{display:flex;align-items:center;justify-content:center;color:#000;white-space:nowrap}
  .splitkey{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:10px;font:12px var(--mono);color:var(--ink-2)}
  @media (max-width:560px){.splitkey{grid-template-columns:1fr}}
  .splitkey b{display:block;color:var(--ink);font:700 20px var(--num)}
  .two{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,.9fr);gap:10px}
  @media (max-width:980px){.two{grid-template-columns:1fr}}
  .scroll{overflow-x:auto}
  table.data{border-collapse:collapse;width:100%;font:13px var(--mono)}
  table.data th,table.data td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
  table.data th:first-child,table.data td:first-child{text-align:left}
  table.data th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
  table.data tr.focus td{background:#10251D}
  .bar{display:inline-block;height:8px;border-radius:1px;vertical-align:middle;margin-right:8px}
  .cm{display:grid;grid-template-columns:58px repeat(4,minmax(0,1fr));gap:2px;font:600 12px var(--mono);max-width:440px}
  .cm .h{color:var(--ink-3);text-align:center;font-size:11px;padding:4px 0;text-transform:uppercase}
  .cm .r{display:flex;align-items:center;font-size:11px;text-transform:uppercase}
  .cm .c{min-height:38px;display:flex;align-items:center;justify-content:center;border-radius:2px}
  .dots{position:relative;height:60px;border-bottom:1px solid var(--line-2);margin-top:8px}
  .dots .d{position:absolute;width:10px;height:10px;border-radius:50%;background:var(--left);transform:translateX(-50%)}
  .dots .m{position:absolute;top:0;bottom:-1px;border-left:2px solid var(--ink)}
  .dots .ch{position:absolute;top:0;bottom:-1px;left:0;border-left:1px dashed var(--alarm)}
  .dotaxis{display:flex;justify-content:space-between;font:10px var(--mono);color:var(--ink-3);margin-top:4px}
  td.best{color:var(--trace);font-weight:700}
</style>
</head>
<body>
__STATUS__

<main class="screen">
  <div class="layout">
    <form class="panel setup" id="controls" aria-label="Test setup">
      <div class="ph">Setup</div>
      <div class="sec"><span class="lbl">Recording</span><div class="seg" id="subjects"></div></div>
      <div class="sec"><span class="lbl">Model</span><select class="full" id="model"></select>
        <span class="note">★ recommended: most accurate on held-out trials</span></div>
      <div class="sec"><span class="lbl">Window after cue</span>
        <div class="timeline"><div class="axis"></div><div class="rest" style="left:66.67%"></div><div class="win" id="win"></div><div class="cue"></div></div>
        <div class="ticks"><span>cue</span><span>1</span><span>2</span><span>3</span><span>4</span><span>5</span><span>6 s</span></div>
        <div class="range"><span>start</span><input type="range" id="tmin" min="0" max="2" step="0.25" value="0.5"><b id="tminV"></b></div>
        <div class="range"><span>end</span><input type="range" id="tmax" min="1.5" max="6" step="0.25" value="4.5"><b id="tmaxV"></b></div>
        <span class="note" id="dur"></span></div>
      <div class="sec"><span class="lbl">Test</span>
        <div class="seg" id="protocol"><button type="button" data-v="holdout" class="on">50/25/25</button><button type="button" data-v="cv">5-fold CV</button></div>
        <span class="note" id="protocolNote"></span>
        <label class="check" id="repeatRow"><input type="checkbox" id="repeat"><span>Average over 20 random splits (about a minute)</span></label></div>
      <div class="sec"><span class="lbl">Focus class</span><div class="chips" id="classes"></div></div>
    </form>

    <div style="display:grid;gap:10px;min-width:0" id="results">
      <section class="panel"><div class="ph">Result<span class="r" id="ctx">no run yet</span></div>
        <div id="headline"><div class="empty"><b>Choose the setup, then press RUN TEST.</b><br>
          The default trains on half the trials, checks on a quarter,<br>and scores the last quarter, which it never sees in training.</div></div>
      </section>
      <div id="detail" style="display:grid;gap:10px"></div>
      <section class="panel"><div class="ph">Run log<span class="r">every run this session</span></div>
        <div class="events scroll"><table class="data" id="history"><tbody><tr><td class="muted">no runs yet</td></tr></tbody></table></div>
      </section>
    </div>
  </div>
</main>

<div class="softkeys" role="toolbar" aria-label="Controls">
  <button class="key primary" id="run" form="controls" type="submit"><span id="runTxt">Run test</span><kbd>Enter</kbd></button>
  <span class="spacer"></span>
  <span class="readout">scores are on trials the model did not train on · chance 25 %</span>
</div>

<script>
const CONFIG = __CONFIG__;
const CLASSES = CONFIG.classes;
__MONITOR_JS__

const LABELS = Object.fromEntries(CONFIG.models.map(m => [m.key, m.label]));
const history = [];
const state = {subject: CONFIG.subjects[0], protocol: 'holdout', focus: ''};
const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
setMode('review', 'REVIEW');
$('st_filter').textContent = '8-30 Hz';

function bindSeg(host, key) {
  $(host).querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    state[key] = b.dataset.v;
    $(host).querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    refresh();
  }));
}
$('subjects').innerHTML = CONFIG.subjects.map(s => `<button type="button" data-v="${s}" class="${s === state.subject ? 'on' : ''}">${s}</button>`).join('');
bindSeg('subjects', 'subject');
bindSeg('protocol', 'protocol');
$('model').innerHTML = CONFIG.models.map((m, i) => `<option value="${m.key}">${m.label}${i === 0 ? ' ★' : ''}</option>`).join('');

function renderChips() {
  $('classes').innerHTML = [''].concat(CLASSES).map(c => {
    const colour = c && c !== state.focus ? `style="color:var(${CLASS_VAR[c]})"` : '';
    return `<button type="button" data-v="${c}" class="${c === state.focus ? 'on' : ''}" ${colour}>${c || 'all'}</button>`;
  }).join('');
  $('classes').querySelectorAll('button').forEach(b => b.addEventListener('click', () => { state.focus = b.dataset.v; renderChips(); }));
}
renderChips();

const tmin = $('tmin'), tmax = $('tmax'), runBtn = $('run');
function refresh() {
  const a = parseFloat(tmin.value), b = parseFloat(tmax.value), len = b - a;
  $('tminV').textContent = a.toFixed(2) + ' s';
  $('tmaxV').textContent = b.toFixed(2) + ' s';
  $('win').style.left = (a / 6 * 100) + '%';
  $('win').style.width = (Math.max(len, 0) / 6 * 100) + '%';
  $('win').style.background = len < 1 ? 'var(--alarm)' : 'var(--trace)';
  if (len < 1) {
    $('dur').className = 'note bad'; $('dur').textContent = 'Window under 1 s. The models need at least 1 s of signal.';
    runBtn.disabled = true;
  } else {
    $('dur').className = 'note'; $('dur').textContent = `${len.toFixed(2)} s per trial · ${Math.round(len * 250)} samples · dashed line = imagery ends`;
    runBtn.disabled = false;
  }
  const holdout = state.protocol === 'holdout';
  $('repeatRow').style.display = holdout ? 'flex' : 'none';
  $('protocolNote').textContent = holdout ? 'Train 50 %, check 25 %, final score on the last 25 %.'
                                          : 'Train 80 %, test 20 %, five times; every trial tested once.';
  $('st_rec').textContent = state.subject;
  $('st_model').textContent = LABELS[$('model').value];
}
tmin.addEventListener('input', refresh); tmax.addEventListener('input', refresh);
$('model').addEventListener('change', refresh);
refresh();

function readout(k, v, s, lead) {
  return `<div class="param ${lead ? 'lead' : ''}"><div class="k">${k}</div><div class="v">${(v * 100).toFixed(1)}<small>%</small></div><div class="s">${s}</div></div>`;
}
const heat = (share, colour) => `background:color-mix(in srgb, ${colour} ${Math.round(share * 80 + 6)}%, var(--panel))`;

function renderResult(r) {
  const m = r.metrics, holdout = r.protocol === 'holdout';
  $('ctx').textContent = `${r.subject} · ${LABELS[r.model] || r.model} · ${r.tmin.toFixed(2)}–${r.tmax.toFixed(2)} s · ${holdout ? '50/25/25' : '5-fold CV'} · ${r.seconds.toFixed(1)} s`;
  $('headline').innerHTML = `<div class="readouts">
    ${readout('Accuracy', m.accuracy, holdout ? 'test quarter' : 'all 5 folds', true)}
    ${readout('Precision', m.precision, 'right when it names a class')}
    ${readout('Recall', m.recall, 'real trials it caught')}
    ${readout('F1', m.f1, 'balance of the two')}
    <div class="param"><div class="k">Kappa</div><div class="v">${m.kappa.toFixed(2)}</div><div class="s">0 = chance · 1 = perfect</div></div>
  </div>`;

  let html = '';
  if (holdout) {
    const n = r.split;
    html += `<section class="panel"><div class="ph">Trial split<span class="r">stratified · ${r.n_dropped} noisy training trials dropped</span></div><div class="pb">
      <div class="splitbar"><div style="flex:${n.train};background:var(--trace)">TRAIN ${n.train}</div>
        <div style="flex:${n.validation};background:var(--ink-3)">CHECK ${n.validation}</div>
        <div style="flex:${n.test};background:var(--left)">TEST ${n.test}</div></div>
      <div class="splitkey"><div><b>${n.train}</b>trials the model learns from</div>
        <div><b>${pct(r.validation.accuracy)}</b>accuracy on the check set</div>
        <div><b>${pct(m.accuracy)}</b>accuracy on the test set</div></div></div></section>`;
  }
  if (r.repeats) {
    const rp = r.repeats, x = v => (v - 0.25) / 0.75 * 100, stack = {};
    const dots = rp.accuracies.map(a => {
      const k = a.toFixed(3); stack[k] = (stack[k] || 0) + 1;
      return `<span class="d" style="left:${x(a)}%;bottom:${(stack[k] - 1) * 11 + 2}px" title="${pct(a)}"></span>`;
    }).join('');
    html += `<section class="panel"><div class="ph">${rp.repeats} random splits<span class="r">each dot is one split · white line = average</span></div>
      <div class="readouts" style="grid-template-columns:repeat(2,minmax(0,1fr));border-bottom:1px solid var(--line)">
        ${readout('Mean accuracy', rp.accuracy.mean, '± ' + pct(rp.accuracy.std) + ' between splits', true)}
        ${readout('Mean precision', rp.precision.mean, '± ' + pct(rp.precision.std) + ' between splits')}</div>
      <div class="pb"><div class="dots">${dots}<span class="m" style="left:${x(rp.accuracy.mean)}%"></span><span class="ch"></span></div>
      <div class="dotaxis"><span>25% chance</span><span>50%</span><span>75%</span><span>100%</span></div></div></section>`;
  }
  const rows = CLASSES.map(c => {
    const p = m.per_class[c];
    return `<tr class="${state.focus === c ? 'focus' : ''}"><td style="color:var(${CLASS_VAR[c]})">${c.toUpperCase()}</td>
      <td><span class="bar" style="width:${p.precision * 60}px;background:var(${CLASS_VAR[c]})"></span>${pct(p.precision, 0)}</td>
      <td><span class="bar" style="width:${p.recall * 60}px;background:var(${CLASS_VAR[c]})"></span>${pct(p.recall, 0)}</td>
      <td>${pct(p.f1, 0)}</td><td>${p.support}</td></tr>`;
  }).join('');
  const cells = m.matrix.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return `<div class="r" style="color:var(${CLASS_VAR[CLASSES[i]]})">${CLASSES[i]}</div>` + row.map((n, j) => {
      const share = n / total;
      return `<div class="c" style="${heat(share, i === j ? 'var(--trace)' : 'var(--alarm)')};${share > .5 ? 'color:#000' : ''}" title="${n} of ${total}">${n}</div>`;
    }).join('');
  }).join('');
  html += `<div class="two">
    <section class="panel"><div class="ph">Per class</div><div class="scroll"><table class="data">
      <thead><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>trials</th></tr></thead><tbody>${rows}</tbody></table></div></section>
    <section class="panel"><div class="ph">Confusion<span class="r">rows real · columns called</span></div><div class="pb">
      <div class="cm"><div></div>${CLASSES.map(c => `<div class="h">${c}</div>`).join('')}${cells}</div></div></section></div>`;
  $('detail').innerHTML = html;
}

function renderHistory() {
  const best = Math.max(...history.map(h => h.metrics.accuracy));
  $('history').innerHTML = `<thead><tr><th>time</th><th>rec</th><th>model</th><th>window</th><th>test</th><th>accuracy</th><th>precision</th></tr></thead><tbody>` +
    history.slice().reverse().map(h => `<tr><td>${h.time}</td><td>${h.subject}</td><td>${LABELS[h.model] || h.model}</td>
      <td>${h.tmin.toFixed(2)}–${h.tmax.toFixed(2)} s</td><td>${h.protocol === 'holdout' ? '50/25/25' : '5-fold'}${h.repeats ? ' ×' + h.repeats.repeats : ''}</td>
      <td class="${h.metrics.accuracy === best ? 'best' : ''}">${pct(h.metrics.accuracy)}</td><td>${pct(h.metrics.precision)}</td></tr>`).join('') + '</tbody>';
}

$('controls').addEventListener('submit', async ev => {
  ev.preventDefault();
  if (runBtn.disabled) return;
  const repeat = state.protocol === 'holdout' && $('repeat').checked;
  runBtn.disabled = true; $('runTxt').textContent = 'Running…';
  setMode('live', 'RUNNING');
  $('headline').innerHTML = `<div class="empty">Training and testing${repeat ? ' over 20 splits, about a minute' : ''}…</div>`;
  const body = {subject: state.subject, model: $('model').value, protocol: state.protocol,
                tmin: parseFloat(tmin.value), tmax: parseFloat(tmax.value), repeats: repeat ? 20 : 0};
  try {
    const data = await post('/api/run', body);
    data.time = new Date().toLocaleTimeString([], {hour12: false});
    renderResult(data); history.push(data); renderHistory();
  } catch (err) {
    $('headline').innerHTML = `<div class="err" style="margin:12px 0">Run failed: ${err.message}</div>`;
  } finally {
    $('runTxt').textContent = 'Run test';
    setMode('review', 'REVIEW');
    refresh();
  }
});
document.addEventListener('keydown', ev => {
  if (ev.key === 'Enter' && !['SELECT', 'BUTTON', 'INPUT'].includes(ev.target.tagName)) $('controls').requestSubmit();
});

// ?subject=A04T&model=riemann_ts_lr&tmin=1&tmax=4&test=cv&class=foot&repeat=1&run=1
(function applyQuery() {
  const q = new URLSearchParams(location.search);
  if (![...q.keys()].length) return;
  const click = (host, v) => { const b = $(host).querySelector(`button[data-v="${v}"]`); if (b) b.click(); };
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
""".replace('__THEME_CSS__', THEME_CSS).replace('__STATUS__', status_bar('review')).replace('__MONITOR_JS__', MONITOR_JS)
