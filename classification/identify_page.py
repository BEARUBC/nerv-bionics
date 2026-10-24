"""HTML for the identify screen: pick a signal from the queue, watch the model name it."""

from classification.monitor_theme import MONITOR_JS, THEME_CSS, status_bar

IDENTIFY_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Identify, NERV motor imagery</title>
<style>
__THEME_CSS__
  .layout{display:grid;grid-template-columns:290px minmax(0,1fr) 280px;gap:10px;align-items:start}
  @media (max-width:1180px){.layout{grid-template-columns:290px minmax(0,1fr)} .side{grid-column:1 / -1}}
  @media (max-width:760px){.layout{grid-template-columns:1fr}}
  .filters{display:flex;gap:4px;flex-wrap:wrap;padding:8px 10px;border-bottom:1px solid var(--line)}
  .filters button{background:var(--panel-2);border:1px solid var(--line-2);border-radius:3px;padding:3px 8px;
                  font:600 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);cursor:pointer}
  .filters button.on{color:#000;background:var(--trace);border-color:var(--trace)}
  .queue{max-height:calc(100vh - 210px);min-height:240px;overflow-y:auto}
  .row{display:grid;grid-template-columns:34px 1fr 74px;gap:8px;align-items:center;width:100%;padding:7px 10px;
       background:none;border:0;border-bottom:1px solid var(--line);cursor:pointer;text-align:left}
  .row:hover{background:var(--panel-2)}
  .row.on{background:#10251D;box-shadow:inset 3px 0 0 var(--trace)}
  .row .n{font:12px var(--mono);color:var(--ink-3)}
  .row svg{display:block;width:100%;height:24px}
  .row .lab{font:600 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;text-align:right;display:grid;gap:2px;justify-items:end}
  .side{display:grid;gap:10px}
  .param .v.mid{font-size:28px}
  .truth.hidden .v{filter:blur(10px);user-select:none}
</style>
</head>
<body>
__STATUS__

<main class="screen">
  <div class="layout">
    <section class="panel" aria-label="Trial queue">
      <div class="ph">Queue · Cz<span class="r" id="tally">--</span></div>
      <div class="filters" id="filters"></div>
      <div class="queue" id="queue"><div class="pb muted">loading…</div></div>
    </section>

    <div style="display:grid;gap:10px;min-width:0">
      <section class="panel" aria-label="EEG waveforms">
        <div class="ph">EEG · 22 ch · referential<span class="r" id="scopeInfo">choose a trial from the queue</span></div>
        <div class="scope"><canvas id="scope"></canvas></div>
        <div class="banner"><span class="chip wait" id="phase">STANDBY</span>
          <span id="banner">Each trial in the queue was held out from training. Its label is the real answer; the model gets the signal only.</span></div>
      </section>
      <section class="panel" aria-label="Probability trend">
        <div class="ph">Class probability trend<span class="r">dashed line = 25 % chance</span></div>
        <div style="padding:6px 0 4px"><canvas id="trend"></canvas></div>
      </section>
      <div id="error"></div>
    </div>

    <aside class="side">
      <section class="panel" aria-label="Detector output">
        <div class="param">
          <div class="k"><span>Identified as</span><span id="win">--</span></div>
          <div class="v" id="call" style="text-transform:uppercase">--</div>
        </div>
        <div class="param"><div class="k">Confidence</div><div class="v" id="conf">--<small>%</small></div></div>
        <div class="param"><div class="k">Probability</div><div id="probs"></div></div>
        <div class="param truth hidden" id="truthBox">
          <div class="k"><span>Really was</span><span id="match" class="chip wait">hidden</span></div>
          <div class="v mid" id="truth" style="text-transform:uppercase">hidden</div>
          <div class="s" id="truthNote">pick a trial</div>
        </div>
      </section>
    </aside>
  </div>
</main>

<div class="softkeys" role="toolbar" aria-label="Controls">
  <button class="key primary" id="nextBtn">Identify next<kbd>N</kbd></button>
  <button class="key" id="replay" disabled>Replay<kbd>R</kbd></button>
  <label class="key">Rec <select id="subject"></select></label>
  <span class="spacer"></span>
  <span class="readout">↑ ↓ move through the queue · Enter identifies</span>
</div>

<script>
const CONFIG = __CONFIG__;
const CLASSES = CONFIG.classes;
__MONITOR_JS__

const LABELS = Object.fromEntries(CONFIG.models.map(m => [m.key, m.label]));
const scope = new Scope($('scope'));
const trend = new Trend($('trend'), CLASSES);
probRows('probs', CLASSES);
$('subject').innerHTML = CONFIG.subjects.map(s => `<option>${s}</option>`).join('');
$('st_filter').textContent = '0.5-100 Hz · N50';
scope.clear(CONFIG.channel_names);

let session = null, catalog = [], filter = null, playing = false, token = 0, current = null;
const results = new Map();

function setError(m) { $('error').innerHTML = m ? `<div class="err">${m}</div>` : ''; }

$('filters').innerHTML = ['all'].concat(CLASSES).map(c =>
  `<button data-c="${c}" class="${c === 'all' ? 'on' : ''}" ${c !== 'all' ? `style="color:var(${CLASS_VAR[c]})"` : ''}>${c}</button>`).join('');

function spark(points) {
  const W = 200, H = 24;
  const d = points.map((v, i) => `${i ? 'L' : 'M'}${(i / (points.length - 1) * W).toFixed(1)},${(H / 2 - v * H * 0.4).toFixed(1)}`).join(' ');
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><path d="${d}" fill="none" stroke="var(--trace)" stroke-width="1" vector-effect="non-scaling-stroke"/></svg>`;
}

function shown() { return catalog.filter(e => !filter || e.truth === filter); }

function renderQueue() {
  const list = shown();
  if (!list.length) { $('queue').innerHTML = '<div class="pb muted">no trials of that class</div>'; return; }
  $('queue').innerHTML = list.map(e => {
    const r = results.get(e.index);
    const mark = r ? `<span class="chip ${r.ok ? 'ok' : 'miss'}">${r.ok ? 'match' : 'miss'}</span>` : '';
    return `<button class="row ${e.index === current ? 'on' : ''}" data-i="${e.index}">
      <span class="n">#${e.index + 1}</span>${spark(e.preview)}
      <span class="lab" style="color:var(${CLASS_VAR[e.truth]})">${e.truth}${mark}</span></button>`;
  }).join('');
  $('queue').querySelectorAll('.row').forEach(n => n.addEventListener('click', () => identify(parseInt(n.dataset.i, 10))));
  const on = $('queue').querySelector('.row.on');
  if (on) on.scrollIntoView({block: 'nearest'});
}

function updateTally() {
  if (!results.size) { $('tally').textContent = `${catalog.length} held out`; return; }
  const hits = [...results.values()].filter(r => r.ok).length;
  $('tally').textContent = `${hits}/${results.size} correct`;
}

function showFrame(f, i) {
  const t0 = session.tmin + f.t - session.window_s;
  scope.draw({channels: session.channels, focus: session.focus_channels, traces: f.traces,
              uv: f.uv, t0, dur: session.window_s, restFrom: 4.0});
  $('scopeInfo').textContent = `trial #${current + 1} · ${t0.toFixed(1)}–${(t0 + session.window_s).toFixed(1)} s after cue`;
  $('call').textContent = f.label;
  $('call').style.color = `var(${CLASS_VAR[f.label]})`;
  $('conf').innerHTML = Math.round(f.confidence * 100) + '<small>%</small>';
  $('win').textContent = 'window ' + (i + 1);
  setProbs('probs', CLASSES, f.probabilities);
  trend.push(t0 + session.window_s, f.probabilities);
  $('phase').className = 'chip ' + (i === 0 ? 'ok' : 'wait');
  $('phase').textContent = i === 0 ? 'decision' : 'sliding';
}

async function identify(index) {
  if (playing || !session) return;
  playing = true; current = index;
  const mine = ++token;
  setError(''); setMode('live', 'IDENTIFYING');
  renderQueue();
  $('truthBox').classList.add('hidden');
  $('truth').textContent = 'hidden'; $('match').className = 'chip wait'; $('match').textContent = 'hidden';
  $('truthNote').textContent = 'identifying…';
  $('banner').textContent = `Reading trial #${index + 1}…`;
  trend.reset();

  try {
    const data = await post('/api/live/trial', {subject: $('subject').value, index});
    if (mine !== token) return;
    for (let i = 0; i < data.frames.length; i++) {
      if (mine !== token) return;
      showFrame(data.frames[i], i);
      await sleep(420);
    }
    // The first complete window is the decision; redraw it so every readout
    // describes the same moment.
    const decision = data.frames[0];
    const ok = decision.label === data.truth;
    const keep = trend.points.slice();
    showFrame(decision, 0);
    trend.points = keep; trend.render();

    $('truthBox').classList.remove('hidden');
    $('truth').textContent = data.truth;
    $('truth').style.color = `var(${CLASS_VAR[data.truth]})`;
    $('match').className = 'chip ' + (ok ? 'ok' : 'miss');
    $('match').textContent = ok ? 'match' : 'miss';
    $('truthNote').textContent = ok ? 'the model got it right' : `it said ${decision.label}`;
    $('banner').textContent = `Trial #${index + 1}: decision taken at the first full window.`;

    results.set(index, {call: decision.label, ok});
    renderQueue(); updateTally();
    $('replay').disabled = false;
  } catch (err) {
    setError(err.message);
  } finally {
    playing = false;
    setMode('review', 'READY');
  }
}

function step(dir) {
  const list = shown();
  if (!list.length) return null;
  const at = list.findIndex(e => e.index === current);
  return list[Math.max(0, Math.min(list.length - 1, at + dir))].index;
}
function nextUnseen() {
  const list = shown();
  const after = list.filter(e => current === null || e.index > current);
  const pick = after.find(e => !results.has(e.index)) || list.find(e => !results.has(e.index)) || after[0] || list[0];
  return pick ? pick.index : null;
}

async function load() {
  setError(''); setMode('standby', 'TRAINING');
  $('st_rec').textContent = $('subject').value;
  $('queue').innerHTML = '<div class="pb muted">training the detector on the other part of the recording…</div>';
  try {
    session = await post('/api/live/start', {subject: $('subject').value});
    catalog = await post('/api/live/list', {subject: $('subject').value});
    results.clear(); current = null;
    $('st_model').textContent = LABELS[session.model] || session.model;
    $('st_fs').textContent = Math.round(session.fs) + ' Hz';
    updateTally(); renderQueue();
    setMode('review', 'READY');
  } catch (err) { setError(err.message); $('queue').innerHTML = ''; }
}

$('filters').addEventListener('click', ev => {
  const b = ev.target.closest('button'); if (!b) return;
  filter = b.dataset.c === 'all' ? null : b.dataset.c;
  $('filters').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  renderQueue();
});
$('nextBtn').addEventListener('click', () => { const i = nextUnseen(); if (i !== null) identify(i); });
$('replay').addEventListener('click', () => { if (current !== null) identify(current); });
$('subject').addEventListener('change', load);
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'SELECT') return;
  if (ev.key === 'n' || ev.key === 'N') $('nextBtn').click();
  else if (ev.key === 'r' || ev.key === 'R') $('replay').click();
  else if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
    ev.preventDefault();
    const i = step(ev.key === 'ArrowDown' ? 1 : -1);
    if (i !== null && !playing) { current = i; renderQueue(); }
  } else if (ev.key === 'Enter' && current !== null && !ev.target.closest('button')) identify(current);
});

// ?subject=A01T&signal=3 opens straight onto one trial; the number matches the
// row label, which starts at 1.
(async function boot() {
  const q = new URLSearchParams(location.search);
  if (q.has('subject') && CONFIG.subjects.includes(q.get('subject'))) $('subject').value = q.get('subject');
  await load();
  if (q.has('signal')) {
    const n = parseInt(q.get('signal'), 10) - 1;
    if (catalog.some(e => e.index === n)) identify(n);
  }
})();
</script>
</body>
</html>
""".replace('__THEME_CSS__', THEME_CSS).replace('__STATUS__', status_bar('identify')).replace('__MONITOR_JS__', MONITOR_JS)
