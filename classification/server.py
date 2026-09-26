"""A browser UI for running tests, served from your own machine.

`app.py` does the same job with matplotlib widgets. This serves a web page
instead, which gives real form controls, keeps a table of every run so you can
compare settings, and does not depend on which matplotlib backend your Python
happens to have.

It uses only the standard library for the serving. Nothing leaves your machine:
the server binds to localhost, and the page talks to it over plain JSON.

Run it
------
    python -m classification.server

Then open http://127.0.0.1:8000 (it tries to open your browser for you).
Press Ctrl+C in the terminal to stop it.

Settings can also go in the URL, which makes a configuration shareable:

    http://127.0.0.1:8000/?subject=A04T&tmin=1.0&tmax=4.0&class=foot&run=1
"""

# Check the interpreter before importing the scientific stack, so a Python
# without it gets an explanation instead of a bare ModuleNotFoundError.
from classification._deps import require as _require
_require('classification.server')

import argparse
import json
import sys
import threading
import traceback
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from classification.app import CLASS_ORDER, OFFERED_MODELS, RecordingCache, run_test
from classification.live import LiveSession
from classification.live_page import LIVE_PAGE
from classification.identify_page import IDENTIFY_PAGE
from classification.data import available_subjects
from classification.models import MODEL_REGISTRY

DEFAULT_PORT = 8000

# Loading a recording is slow and cross-validation is CPU-bound, so one run at a
# time. The lock also stops two browser tabs from thrashing the same cache.
RUN_LOCK = threading.Lock()

# Live sessions are expensive to build (they train a detector), so keep one per
# recording once it exists.
LIVE_SESSIONS = {}
LIVE_LOCK = threading.Lock()

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motor imagery test bench</title>
<style>
  :root {
    --ground: #E9EDEF; --panel: #FFFFFF; --ink: #131C21; --muted: #5A6A72;
    --faint: #8798A0; --rule: #D2DADD; --accent: #0B6E5F; --accent-soft: #D8EAE5;
    --bad: #9E2B26; --warn: #8A5300;
    --left: #2E6E9E; --right: #A85F18; --foot: #67539B; --tongue: #2C7A57;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground: #0E1417; --panel: #161F23; --ink: #E4ECEE; --muted: #9DAEB5;
      --faint: #74868E; --rule: #2A383E; --accent: #58C4AC; --accent-soft: #123A32;
      --bad: #E0766F; --warn: #DBA53A;
      --left: #74ADDA; --right: #DFA365; --foot: #A899DD; --tongue: #6EC59C;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ground); color: var(--ink);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 28px 20px 70px; }
  h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -.02em; }
  .sub { color: var(--muted); margin: 0 0 26px; }
  .layout { display: grid; grid-template-columns: 290px 1fr; gap: 22px; align-items: start; }
  @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } }

  .card { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 20px; }
  fieldset { border: 0; padding: 0; margin: 0 0 20px; }
  legend {
    font: 600 11px/1 ui-monospace, Menlo, monospace; letter-spacing: .1em;
    text-transform: uppercase; color: var(--faint); padding: 0 0 9px;
  }
  label.opt { display: flex; align-items: center; gap: 9px; padding: 5px 0; cursor: pointer; }
  label.opt input { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }

  .slider { margin-bottom: 15px; }
  .slider .row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
  .slider .name { font-size: 13px; color: var(--muted); }
  .slider .val { font: 600 13px ui-monospace, Menlo, monospace; color: var(--ink); }
  input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
  .dur {
    font: 12px ui-monospace, Menlo, monospace; color: var(--muted);
    background: var(--ground); border: 1px solid var(--rule); border-radius: 4px;
    padding: 7px 10px; margin-bottom: 16px;
  }
  .dur.bad { color: var(--bad); border-color: var(--bad); }

  button {
    width: 100%; padding: 12px; font: 600 15px inherit; cursor: pointer;
    background: var(--accent); color: var(--panel); border: 0; border-radius: 5px;
  }
  button:disabled { opacity: .55; cursor: wait; }
  button.ghost { background: transparent; color: var(--muted); border: 1px solid var(--rule); margin-top: 9px; font-weight: 400; }

  .headline { display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap; }
  .big { font-size: 52px; font-weight: 700; letter-spacing: -.03em; line-height: 1; }
  .meta { color: var(--muted); font-size: 14px; }
  .meta b { color: var(--ink); }

  .bars { margin-top: 26px; display: grid; gap: 7px; }
  .bar { display: grid; grid-template-columns: 62px 1fr 46px; gap: 11px; align-items: center; }
  .bar .nm { font-size: 13px; color: var(--muted); }
  .track { height: 22px; background: var(--ground); border: 1px solid var(--rule); border-radius: 3px; position: relative; overflow: hidden; }
  .fill { position: absolute; inset: 0 auto 0 0; }
  .chance { position: absolute; top: -1px; bottom: -1px; left: 25%; border-left: 2px dashed var(--bad); }
  .bar .pc { font: 600 13px ui-monospace, Menlo, monospace; text-align: right; }

  table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 10px; }
  th, td { padding: 8px 10px; border-bottom: 1px solid var(--rule); text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { font: 600 11px ui-monospace, Menlo, monospace; letter-spacing: .07em; text-transform: uppercase; color: var(--faint); }
  td { font-family: ui-monospace, Menlo, monospace; }
  .cm td.d { font-weight: 700; color: var(--accent); }
  .cm td.z { color: var(--faint); }

  h3 { font-size: 15px; margin: 30px 0 2px; }
  h3 + .hint { color: var(--muted); font-size: 13px; margin: 0; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
  @media (max-width: 720px) { .grid2 { grid-template-columns: 1fr; } }
  .err { color: var(--bad); background: var(--panel); border: 1px solid var(--bad); border-radius: 5px; padding: 14px; }
  .muted { color: var(--muted); }
  .spin { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Motor imagery test bench</h1>
  <p class="sub">Pick a recording, a class and a time window, then run a real 5-fold cross-validation.</p>

  <div class="layout">
    <form class="card" id="controls">
      <fieldset>
        <legend>Recording</legend>
        <div id="subjects"></div>
      </fieldset>

      <fieldset>
        <legend>Type of data</legend>
        <div id="classes"></div>
      </fieldset>

      <fieldset>
        <legend>Model</legend>
        <div id="models"></div>
      </fieldset>

      <fieldset>
        <legend>Duration</legend>
        <div class="slider">
          <div class="row"><span class="name">start after cue</span><span class="val" id="tminV">0.50 s</span></div>
          <input type="range" id="tmin" min="0" max="2" step="0.25" value="0.5">
        </div>
        <div class="slider">
          <div class="row"><span class="name">end after cue</span><span class="val" id="tmaxV">3.50 s</span></div>
          <input type="range" id="tmax" min="1.5" max="6" step="0.25" value="3.5">
        </div>
        <div class="dur" id="dur"></div>
      </fieldset>

      <button type="submit" id="run">Run test</button>
      <button type="button" class="ghost" id="clear">Clear history</button>
    </form>

    <div>
      <div class="card" id="result">
        <p class="muted">Press <b>Run test</b> to start. The first run loads the recording and takes a few seconds; later runs are about a second.</p>
      </div>

      <h3>Runs this session</h3>
      <p class="hint">Every run is kept so you can compare settings.</p>
      <div class="card" style="padding:10px 20px 14px">
        <table id="history"><tbody><tr><td class="muted" style="text-align:left">nothing yet</td></tr></tbody></table>
      </div>
    </div>
  </div>
</div>

<script>
const CONFIG = __CONFIG__;
const CLASSES = CONFIG.classes;
const history = [];

function radios(host, name, items, checked) {
  document.getElementById(host).innerHTML = items.map(it => `
    <label class="opt">
      <input type="radio" name="${name}" value="${it.value}" ${it.value === checked ? 'checked' : ''}>
      ${it.color ? `<span class="swatch" style="background:${it.color}"></span>` : ''}
      <span>${it.label}</span>
    </label>`).join('');
}

radios('subjects', 'subject', CONFIG.subjects.map(s => ({value: s, label: s})), CONFIG.subjects[0]);
radios('classes', 'klass',
  [{value: '', label: 'all classes'}].concat(CLASSES.map(c => ({value: c, label: c, color: `var(--${c})`}))), '');
radios('models', 'model', CONFIG.models.map(m => ({value: m.key, label: m.label})), CONFIG.models[0].key);

const tmin = document.getElementById('tmin'), tmax = document.getElementById('tmax');
const runBtn = document.getElementById('run');

function refreshDuration() {
  const a = parseFloat(tmin.value), b = parseFloat(tmax.value);
  document.getElementById('tminV').textContent = a.toFixed(2) + ' s';
  document.getElementById('tmaxV').textContent = b.toFixed(2) + ' s';
  const box = document.getElementById('dur');
  const len = b - a;
  if (len < 1) {
    box.className = 'dur bad';
    box.textContent = `${len.toFixed(2)} s window — too short, covariance needs at least 1 s`;
    runBtn.disabled = true;
  } else {
    box.className = 'dur';
    box.textContent = `${len.toFixed(2)} s of signal per trial (${Math.round(len * 250)} samples)`;
    runBtn.disabled = false;
  }
}
tmin.addEventListener('input', refreshDuration);
tmax.addEventListener('input', refreshDuration);
refreshDuration();

function picked(name) {
  return document.querySelector(`input[name="${name}"]:checked`).value;
}

function pct(x) { return (x * 100).toFixed(1) + '%'; }

function renderResult(r) {
  const versus = r.accuracy - 0.25;
  const colour = r.accuracy >= 0.5 ? 'var(--accent)' : (r.accuracy >= 0.25 ? 'var(--ink)' : 'var(--bad)');
  const scope = r.class_filter || 'all four classes';

  const bars = CLASSES.map(c => {
    const v = r.per_class[c];
    const dim = r.class_filter && r.class_filter !== c ? 'opacity:.3;' : '';
    return `<div class="bar">
      <span class="nm">${c}</span>
      <span class="track"><span class="fill" style="width:${(v * 100).toFixed(1)}%;background:var(--${c});${dim}"></span><span class="chance"></span></span>
      <span class="pc">${(v * 100).toFixed(0)}%</span>
    </div>`;
  }).join('');

  const rows = r.matrix.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    const cells = row.map((n, j) =>
      `<td class="${i === j ? 'd' : (n === 0 ? 'z' : '')}">${n}</td>`).join('');
    return `<tr><td>${CLASSES[i]}</td>${cells}<td class="muted">${(row[i] / total * 100).toFixed(0)}%</td></tr>`;
  }).join('');

  document.getElementById('result').innerHTML = `
    <div class="headline">
      <div class="big" style="color:${colour}">${pct(r.accuracy)}</div>
      <div class="meta">
        <b>${r.subject}</b> · ${r.model} · ${r.tmin.toFixed(2)}–${r.tmax.toFixed(2)} s
        (${(r.tmax - r.tmin).toFixed(2)} s window)<br>
        scored on ${scope}: ${r.n_scored} of ${r.n_total} trials<br>
        ${versus >= 0 ? '+' : ''}${pct(versus)} vs the 25% chance level ·
        ${r.seconds.toFixed(1)} s to run · ${r.n_dropped} dropped as artifacts
      </div>
    </div>
    <div class="grid2">
      <div>
        <h3>Correct per class</h3>
        <p class="hint">Dashed line is the 25% chance level.</p>
        <div class="bars">${bars}</div>
      </div>
      <div>
        <h3>What it confused with what</h3>
        <p class="hint">Rows are the real class, columns what it detected.</p>
        <table class="cm">
          <thead><tr><th>actual \\ detected</th>${CLASSES.map(c => `<th>${c}</th>`).join('')}<th>ok</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
}

function renderHistory() {
  const body = document.querySelector('#history tbody');
  if (!history.length) {
    body.innerHTML = '<tr><td class="muted" style="text-align:left">nothing yet</td></tr>';
    return;
  }
  const head = `<tr><th>subject</th><th>class</th><th>model</th><th>window</th><th>accuracy</th></tr>`;
  const best = Math.max(...history.map(h => h.accuracy));
  body.innerHTML = head + history.slice().reverse().map(h => `
    <tr>
      <td style="text-align:left">${h.subject}</td>
      <td>${h.class_filter || 'all'}</td>
      <td>${h.model}</td>
      <td>${h.tmin.toFixed(2)}–${h.tmax.toFixed(2)} s</td>
      <td style="color:${h.accuracy === best ? 'var(--accent)' : 'inherit'};font-weight:${h.accuracy === best ? 700 : 400}">${pct(h.accuracy)}</td>
    </tr>`).join('');
}

document.getElementById('controls').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';
  document.getElementById('result').innerHTML = '<p class="spin">Cutting epochs and cross-validating...</p>';

  const body = {
    subject: picked('subject'),
    class_filter: picked('klass') || null,
    model: picked('model'),
    tmin: parseFloat(tmin.value),
    tmax: parseFloat(tmax.value),
  };

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'run failed');
    renderResult(data);
    history.push(data);
    renderHistory();
  } catch (err) {
    document.getElementById('result').innerHTML =
      `<div class="err"><b>That run failed.</b><br>${err.message}</div>`;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run test';
    refreshDuration();
  }
});

document.getElementById('clear').addEventListener('click', () => {
  history.length = 0;
  renderHistory();
});

// Settings can be put in the URL, so a configuration worth showing someone can be
// bookmarked or pasted. ?run=1 also presses the button on load.
(function applyQuery() {
  const q = new URLSearchParams(location.search);
  if (![...q.keys()].length) return;

  const pick = (name, value) => {
    if (value === null) return;
    const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
    if (el) el.checked = true;
  };

  pick('subject', q.get('subject'));
  pick('model', q.get('model'));
  if (q.has('class')) pick('klass', q.get('class') === 'all' ? '' : q.get('class'));
  if (q.has('tmin')) tmin.value = q.get('tmin');
  if (q.has('tmax')) tmax.value = q.get('tmax');
  refreshDuration();

  if (q.get('run') === '1' && !runBtn.disabled) {
    document.getElementById('controls').requestSubmit();
  }
})();
</script>
</body>
</html>
"""


def make_handler(cache, subjects):
    """Build the request handler, closing over the shared recording cache.

    Parameters
    ----------
    cache : classification.app.RecordingCache
        Parsed recordings, shared across requests.
    subjects : list[str]
        Recordings offered in the page.

    Returns
    -------
    type
        A BaseHTTPRequestHandler subclass.
    """

    config = {
        'subjects': subjects,
        'classes': list(CLASS_ORDER),
        'models': [{'key': k, 'label': MODEL_REGISTRY[k]['label']} for k in OFFERED_MODELS],
    }
    serialised = json.dumps(config)
    page = PAGE.replace('__CONFIG__', serialised)
    live_page = LIVE_PAGE.replace('__CONFIG__', serialised)
    identify_page = IDENTIFY_PAGE.replace('__CONFIG__', serialised)

    def live_session(subject):
        """Return the live session for a recording, building it once.

        Parameters
        ----------
        subject : str
            Dataset stem.

        Returns
        -------
        classification.live.LiveSession
            A session whose detector was trained on the other split.
        """

        with LIVE_LOCK:
            if subject not in LIVE_SESSIONS:
                print(f'  building live session for {subject} (training the detector)...')
                LIVE_SESSIONS[subject] = LiveSession(subject)
                summary = LIVE_SESSIONS[subject].describe()
                print(f'       trained on {summary["n_train"]} trials, '
                      f'streaming {summary["n_trials"]} it has never seen')
            return LIVE_SESSIONS[subject]

    class Handler(BaseHTTPRequestHandler):
        """Serve the page and answer run requests."""

        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            """Keep the terminal readable by dropping the per-request access log."""

        def _send(self, code, body, content_type):
            """Write one response.

            Parameters
            ----------
            code : int
                HTTP status code.
            body : bytes
                Response body.
            content_type : str
                Value for the Content-Type header.

            Returns
            -------
            None
            """

            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """Serve the page itself.

            The path is split first: settings arrive as a query string, and
            comparing the raw path would 404 on every configured link.
            """

            path = urlsplit(self.path).path

            if path in ('/', '/index.html'):
                self._send(200, page.encode('utf-8'), 'text/html; charset=utf-8')
                return

            if path in ('/live', '/live/'):
                self._send(200, live_page.encode('utf-8'), 'text/html; charset=utf-8')
                return

            if path in ('/identify', '/identify/'):
                self._send(200, identify_page.encode('utf-8'), 'text/html; charset=utf-8')
                return

            self._send(404, b'not found', 'text/plain; charset=utf-8')

        def _read_json(self):
            """Parse the request body, or None if it is not valid JSON.

            Returns
            -------
            dict or None
                The decoded body.
            """

            length = int(self.headers.get('Content-Length', 0))
            try:
                return json.loads(self.rfile.read(length) or b'{}')
            except json.JSONDecodeError:
                return None

        def _fail(self, code, message):
            """Send a JSON error.

            Parameters
            ----------
            code : int
                HTTP status code.
            message : str
                Message for the page to display.

            Returns
            -------
            None
            """

            self._send(code, json.dumps({'error': message}).encode(), 'application/json')

        def do_POST(self):
            """Route the JSON endpoints."""

            path = urlsplit(self.path).path

            if path == '/api/live/start':
                self._live_start()
                return
            if path == '/api/live/trial':
                self._live_trial()
                return
            if path == '/api/live/list':
                self._live_list()
                return
            if path != '/api/run':
                self._fail(404, 'not found')
                return

            request = self._read_json()
            if request is None:
                self._fail(400, 'malformed request')
                return

            subject = request.get('subject')
            model = request.get('model')
            tmin = float(request.get('tmin', 0.5))
            tmax = float(request.get('tmax', 3.5))
            class_filter = request.get('class_filter') or None

            # Validate here too: the page checks these, but the endpoint should
            # not depend on the page being the only caller.
            if subject not in subjects:
                self._fail(400, f'unknown recording {subject!r}')
                return
            if model not in OFFERED_MODELS:
                self._fail(400, f'unknown model {model!r}')
                return
            if tmax - tmin < 1.0:
                self._fail(400, 'window must be at least 1 second')
                return

            print(f'  run: {subject} {model} {tmin:.2f}-{tmax:.2f}s '
                  f'{class_filter or "all classes"}')

            try:
                with RUN_LOCK:
                    result = run_test(cache, subject, model, tmin, tmax, class_filter)
            except Exception as exc:
                traceback.print_exc()
                self._fail(500, f'{type(exc).__name__}: {exc}')
                return

            result['matrix'] = result['matrix'].tolist()
            print(f'       -> {result["accuracy"]:.1%} in {result["seconds"]:.1f}s')

            self._send(200, json.dumps(result).encode('utf-8'), 'application/json')

        def _live_start(self):
            """Build (or reuse) a live session and describe it."""

            request = self._read_json()
            if request is None:
                self._fail(400, 'malformed request')
                return

            subject = request.get('subject')
            if subject not in subjects:
                self._fail(400, f'unknown recording {subject!r}')
                return

            try:
                summary = live_session(subject).describe()
            except Exception as exc:
                traceback.print_exc()
                self._fail(500, f'{type(exc).__name__}: {exc}')
                return

            self._send(200, json.dumps(summary).encode('utf-8'), 'application/json')

        def _live_list(self):
            """Return every streamable signal with its label and a thumbnail."""

            request = self._read_json()
            if request is None:
                self._fail(400, 'malformed request')
                return

            subject = request.get('subject')
            if subject not in subjects:
                self._fail(400, f'unknown recording {subject!r}')
                return

            try:
                entries = live_session(subject).catalog()
            except Exception as exc:
                traceback.print_exc()
                self._fail(500, f'{type(exc).__name__}: {exc}')
                return

            self._send(200, json.dumps(entries).encode('utf-8'), 'application/json')

        def _live_trial(self):
            """Stream one held-out trial: frames for the page, truth for the viewer."""

            request = self._read_json()
            if request is None:
                self._fail(400, 'malformed request')
                return

            subject = request.get('subject')
            if subject not in subjects:
                self._fail(400, f'unknown recording {subject!r}')
                return

            try:
                index = int(request.get('index', 0))
            except (TypeError, ValueError):
                self._fail(400, 'index must be a whole number')
                return

            try:
                session = live_session(subject)
                with RUN_LOCK:
                    payload = session.trial(index)
            except IndexError as exc:
                self._fail(400, str(exc))
                return
            except Exception as exc:
                traceback.print_exc()
                self._fail(500, f'{type(exc).__name__}: {exc}')
                return

            self._send(200, json.dumps(payload).encode('utf-8'), 'application/json')

    return Handler


def main(argv=None):
    """Start the server and open a browser at it.

    Parameters
    ----------
    argv : list[str], optional
        Argument list, defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port to listen on.')
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser.')
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')
    try:
        import mne
        mne.set_log_level('ERROR')
    except ImportError:
        pass

    subjects = available_subjects()
    if not subjects:
        print('No .mat recordings found in acquisition/data.', file=sys.stderr)
        return 1

    cache = RecordingCache()
    handler = make_handler(cache, subjects)

    try:
        # Bind to loopback only: this runs an arbitrary-ish workload and has no
        # authentication, so it has no business being reachable from the network.
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    except OSError as exc:
        print(f'Could not listen on port {args.port}: {exc}\n'
              f'Try:  python -m classification.server --port {args.port + 1}', file=sys.stderr)
        return 1

    url = f'http://127.0.0.1:{args.port}'
    print(f'Test bench running at {url}')
    print(f'Live stream at        {url}/live')
    print(f'Identify a signal at  {url}/identify')
    print(f'Recordings available: {", ".join(subjects)}')
    print('Press Ctrl+C to stop.\n')

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        server.server_close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
