"""The HTML for the monitor screen: held-out trials stream past the detector live."""

from classification.monitor_theme import MONITOR_JS, THEME_CSS, status_bar

LIVE_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monitor, NERV motor imagery</title>
<style>
__THEME_CSS__
  .layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:10px;align-items:start}
  @media (max-width:900px){.layout{grid-template-columns:1fr}}
  .side{display:grid;gap:10px}
  .truth.hidden .v{filter:blur(10px);user-select:none}
  .split2{display:grid;grid-template-columns:1fr 1fr}
  .split2 .param:first-child{border-right:1px solid var(--line)}
  .param .v.mid{font-size:28px}
</style>
</head>
<body>
__STATUS__

<main class="screen">
  <div class="layout">
    <div style="display:grid;gap:10px;min-width:0">
      <section class="panel" aria-label="EEG waveforms">
        <div class="ph">EEG · 22 ch · referential<span class="r" id="scopeInfo">time from cue · motor channels highlighted</span></div>
        <div class="scope"><canvas id="scope"></canvas></div>
        <div class="banner"><span class="chip wait" id="phase">STANDBY</span>
          <span id="banner">Press START to stream trials the model has never seen.</span></div>
      </section>

      <section class="panel" aria-label="Probability trend">
        <div class="ph">Class probability trend<span class="r">dashed line = 25 % chance</span></div>
        <div style="padding:6px 0 4px"><canvas id="trend"></canvas></div>
      </section>

      <section class="panel" aria-label="Event log">
        <div class="ph">Event log<span class="r" id="tally">no trials yet</span></div>
        <div class="events"><table><thead><tr><th>time</th><th>trial</th><th>true</th><th>called</th><th class="num">conf</th><th>result</th></tr></thead>
          <tbody id="log"><tr><td colspan="6" class="muted">no trials yet</td></tr></tbody></table></div>
      </section>
      <div id="error"></div>
    </div>

    <aside class="side">
      <section class="panel" aria-label="Detector output">
        <div class="param">
          <div class="k"><span>MI class</span><span id="called" class="chip wait">--</span></div>
          <div class="v" id="call" style="text-transform:uppercase">--</div>
          <div class="s" id="callNote">waiting for the first full window</div>
        </div>
        <div class="param">
          <div class="k"><span>Confidence</span><span id="win">--</span></div>
          <div class="v" id="conf">--<small>%</small></div>
        </div>
        <div class="param"><div class="k"><span>Probability</span></div><div id="probs"></div></div>
        <div class="param truth hidden" id="truthBox">
          <div class="k"><span>True label</span><span id="match" class="chip wait">hidden</span></div>
          <div class="v mid" id="truth" style="text-transform:uppercase">hidden</div>
          <div class="s" id="truthNote">revealed when the trial ends</div>
        </div>
      </section>

      <section class="panel" aria-label="Session">
        <div class="ph">Session</div>
        <div class="split2">
          <div class="param"><div class="k">Accuracy</div><div class="v mid" id="acc">--</div><div class="s" id="accN">0 / 0</div></div>
          <div class="param"><div class="k">Trial</div><div class="v mid" id="trialN">--</div><div class="s" id="trialOf">of --</div></div>
        </div>
        <div class="param"><div class="k">Detector</div>
          <div class="s" id="detInfo">not trained yet</div></div>
      </section>
    </aside>
  </div>
</main>

<div class="softkeys" role="toolbar" aria-label="Controls">
  <button class="key primary" id="play"><span id="playTxt">Start</span><kbd>Space</kbd></button>
  <button class="key" id="next" disabled>Next trial<kbd>N</kbd></button>
  <button class="key" id="peek">Show truth<kbd>T</kbd></button>
  <label class="key">Speed <select id="speed">
    <option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="0">max</option>
  </select></label>
  <label class="key">Rec <select id="subject"></select></label>
  <span class="spacer"></span>
  <span class="readout">held-out trials only · label attached after the model has answered</span>
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
$('st_rec').textContent = $('subject').value;
$('st_filter').textContent = '0.5-100 Hz · N50';
scope.clear(CONFIG.channel_names);

let session = null, order = [], running = false, cancelled = false, peek = false, log = [];

function setError(m) { $('error').innerHTML = m ? `<div class="err">${m}</div>` : ''; }

function windowStart(f) { return session.tmin + f.t - session.window_s; }

function showFrame(f, i) {
  const t0 = windowStart(f);
  scope.draw({channels: session.channels, focus: session.focus_channels, traces: f.traces,
              uv: f.uv, t0, dur: session.window_s, restFrom: 4.0});
  $('scopeInfo').textContent = `${t0.toFixed(1)}–${(t0 + session.window_s).toFixed(1)} s after cue · motor channels highlighted`;
  $('call').textContent = f.label;
  $('call').style.color = `var(${CLASS_VAR[f.label]})`;
  $('called').className = 'chip ' + (f.called ? 'ok' : 'wait');
  $('called').textContent = f.called ? 'called' : 'low conf';
  $('conf').innerHTML = Math.round(f.confidence * 100) + '<small>%</small>';
  $('win').textContent = 'window ' + (i + 1);
  setProbs('probs', CLASSES, f.probabilities);
  trend.push(t0 + session.window_s, f.probabilities);
  $('phase').className = 'chip ' + (i === 0 ? 'ok' : 'wait');
  $('phase').textContent = i === 0 ? 'decision' : 'sliding';
  $('banner').textContent = i === 0
    ? 'First full window: this is the decision point.'
    : 'Window slides on into the rest period; later windows are for display only.';
  $('callNote').textContent = i === 0 ? 'decision window' : 'tracking, decision already taken';
}

function showTruth(truth, call, final) {
  $('truthBox').classList.toggle('hidden', !(final || peek));
  if (!(final || peek)) { $('truth').textContent = 'hidden'; $('match').className = 'chip wait'; $('match').textContent = 'hidden'; return; }
  $('truth').textContent = truth;
  $('truth').style.color = `var(${CLASS_VAR[truth]})`;
  if (final) {
    const ok = truth === call;
    $('match').className = 'chip ' + (ok ? 'ok' : 'miss');
    $('match').textContent = ok ? 'match' : 'miss';
    $('truthNote').textContent = ok ? 'the model got this one right' : `the model said ${call}`;
  } else {
    $('match').className = 'chip wait'; $('match').textContent = 'peeking';
    $('truthNote').textContent = 'shown early because Show truth is on';
  }
}

function renderLog() {
  if (!log.length) { $('log').innerHTML = '<tr><td colspan="6" class="muted">no trials yet</td></tr>'; $('tally').textContent = 'no trials yet'; return; }
  $('log').innerHTML = log.slice().reverse().map(r => `<tr>
    <td>${r.time}</td><td>#${r.index + 1}</td>
    <td style="color:var(${CLASS_VAR[r.truth]})">${r.truth.toUpperCase()}</td>
    <td style="color:var(${CLASS_VAR[r.call]})">${r.call.toUpperCase()}</td>
    <td class="num">${Math.round(r.conf * 100)}%</td>
    <td><span class="chip ${r.ok ? 'ok' : 'miss'}">${r.ok ? 'match' : 'miss'}</span></td></tr>`).join('');
  const hits = log.filter(r => r.ok).length;
  $('tally').textContent = `${hits}/${log.length} correct · chance 25 %`;
  $('acc').textContent = Math.round(hits / log.length * 100) + '%';
  $('accN').textContent = `${hits} / ${log.length} correct`;
}

async function playTrial(index) {
  const data = await post('/api/live/trial', {subject: $('subject').value, index});
  if (!data.frames.length) return;
  $('trialN').textContent = '#' + (index + 1);
  trend.reset();
  showTruth(data.truth, null, false);

  const speed = parseFloat($('speed').value);
  const gap = speed === 0 ? 120 : (session.step_s * 1000) / speed;
  for (let i = 0; i < data.frames.length; i++) {
    if (cancelled) return;
    showFrame(data.frames[i], i);
    await sleep(gap);
  }

  // The first full window is the model's best read; later windows slide past
  // the imagery into the rest period. Redraw it so every readout agrees.
  const decision = data.frames[0];
  const ok = decision.label === data.truth;
  const keep = trend.points.slice();
  showFrame(decision, 0);
  trend.points = keep; trend.render();
  showTruth(data.truth, decision.label, true);
  $('banner').textContent = `Trial #${index + 1} done: decision taken at the first full window.`;

  log.push({index, truth: data.truth, call: decision.label, conf: decision.confidence, ok,
            time: new Date().toLocaleTimeString([], {hour12: false})});
  renderLog();
  await sleep(speed === 0 ? 250 : 1200);
}

async function run() {
  running = true; cancelled = false;
  $('playTxt').textContent = 'Freeze'; $('next').disabled = false;
  setMode('live', 'LIVE'); setError('');
  try {
    if (!session || session.subject !== $('subject').value) {
      setMode('standby', 'TRAINING');
      $('banner').textContent = 'Training the detector on the other part of the recording…';
      session = await post('/api/live/start', {subject: $('subject').value});
      order = Array.from({length: session.n_trials}, (_, i) => i);
      log = []; renderLog();
      $('st_model').textContent = LABELS[session.model] || session.model;
      $('st_fs').textContent = Math.round(session.fs) + ' Hz';
      $('trialOf').textContent = `of ${session.n_trials} held out`;
      $('detInfo').innerHTML = `trained on ${session.n_train} trials<br>window ${session.window_s.toFixed(1)} s · step ${session.step_s} s · smoothing ${session.smoothing}<br>model band ${session.band ? session.band.join('-') + ' Hz' : 'none'}`;
      setMode('live', 'LIVE');
    }
    while (running && order.length) {
      if (cancelled) break;
      await playTrial(order.shift());
    }
    if (!order.length) { $('banner').textContent = 'End of the held-out trials.'; stop(); }
  } catch (err) { setError(err.message); stop(); }
}

function stop() {
  running = false; cancelled = true;
  $('playTxt').textContent = 'Start'; $('next').disabled = true;
  setMode('standby', 'FROZEN');
}

$('play').addEventListener('click', () => running ? stop() : run());
$('next').addEventListener('click', () => { cancelled = true; setTimeout(() => { cancelled = false; if (running) run(); }, 60); });
$('peek').addEventListener('click', () => {
  peek = !peek; $('peek').classList.toggle('on', peek);
  if (!peek && running) showTruth(null, null, false);
});
$('subject').addEventListener('change', () => { $('st_rec').textContent = $('subject').value; session = null; stop(); setMode('standby', 'STANDBY'); });
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'SELECT') return;
  if (ev.code === 'Space') { ev.preventDefault(); $('play').click(); }
  else if (ev.key === 'n' || ev.key === 'N') { if (!$('next').disabled) $('next').click(); }
  else if (ev.key === 't' || ev.key === 'T') { $('peek').click(); }
});

// ?subject=A04T&speed=4&peek=1&auto=1 opens a configured view; auto=1 starts it.
(function applyQuery() {
  const q = new URLSearchParams(location.search);
  if (q.has('subject') && CONFIG.subjects.includes(q.get('subject'))) { $('subject').value = q.get('subject'); $('st_rec').textContent = q.get('subject'); }
  if (q.has('speed')) $('speed').value = q.get('speed');
  if (q.get('peek') === '1') $('peek').click();
  if (q.get('auto') === '1') run();
})();
</script>
</body>
</html>
""".replace('__THEME_CSS__', THEME_CSS).replace('__STATUS__', status_bar('monitor')).replace('__MONITOR_JS__', MONITOR_JS)
