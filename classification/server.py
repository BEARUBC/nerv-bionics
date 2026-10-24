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
import time
import traceback
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from classification.app import RecordingCache, run_test
from classification.bench_page import BENCH_PAGE
from classification.holdout import holdout, repeated_holdout
from classification.live import LiveSession
from classification.live_page import LIVE_PAGE
from classification.identify_page import IDENTIFY_PAGE
from classification.data import CLASSES, available_subjects
from classification.models import MODEL_REGISTRY, SELECTED_MODELS

DEFAULT_PORT = 8000

# Loading a recording is slow and cross-validation is CPU-bound, so one run at a
# time. The lock also stops two browser tabs from thrashing the same cache.
RUN_LOCK = threading.Lock()

# Live sessions are expensive to build (they train a detector), so keep one per
# recording once it exists.
LIVE_SESSIONS = {}
LIVE_LOCK = threading.Lock()


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
        'classes': list(CLASSES),
        'models': [{'key': k, 'label': MODEL_REGISTRY[k]['label']} for k in SELECTED_MODELS],
    }
    serialised = json.dumps(config)
    page = BENCH_PAGE.replace('__CONFIG__', serialised)
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
            protocol = request.get('protocol', 'holdout')
            try:
                tmin = float(request.get('tmin', 0.5))
                tmax = float(request.get('tmax', 4.5))
                repeats = int(request.get('repeats', 0))
            except (TypeError, ValueError):
                self._fail(400, 'tmin, tmax and repeats must be numbers')
                return

            # Validate here too: the page checks these, but the endpoint should
            # not depend on the page being the only caller.
            if subject not in subjects:
                self._fail(400, f'unknown recording {subject!r}')
                return
            if model not in SELECTED_MODELS:
                self._fail(400, f'unknown model {model!r}')
                return
            if tmax - tmin < 1.0:
                self._fail(400, 'window must be at least 1 second')
                return
            if protocol not in ('holdout', 'cv'):
                self._fail(400, f'unknown test protocol {protocol!r}')
                return
            if not 0 <= repeats <= 100:
                self._fail(400, 'repeats must be between 0 and 100')
                return

            print(f'  run: {subject} {model} {tmin:.2f}-{tmax:.2f}s {protocol}'
                  f'{f" x{repeats}" if repeats else ""}')

            try:
                with RUN_LOCK:
                    result = self._evaluate(subject, model, tmin, tmax, protocol, repeats)
            except Exception as exc:
                traceback.print_exc()
                self._fail(500, f'{type(exc).__name__}: {exc}')
                return

            print(f'       -> {result["metrics"]["accuracy"]:.1%} accuracy, '
                  f'{result["metrics"]["precision"]:.1%} precision in {result["seconds"]:.1f}s')

            self._send(200, json.dumps(result).encode('utf-8'), 'application/json')

        def _evaluate(self, subject, model, tmin, tmax, protocol, repeats):
            """Run one evaluation and shape it for the page.

            Parameters
            ----------
            subject, model : str
                Recording and registry key.
            tmin, tmax : float
                Epoch window in seconds relative to cue onset.
            protocol : str
                ``'holdout'`` for the 50/25/25 split, ``'cv'`` for 5-fold.
            repeats : int
                Extra hold-out splits to average over; 0 for none.

            Returns
            -------
            dict
                ``metrics`` (the headline scores), ``split`` and ``validation``
                for hold-out, ``repeats`` when requested, and the settings.
            """

            common = {'subject': subject, 'model': model, 'tmin': tmin, 'tmax': tmax,
                      'protocol': protocol, 'repeats': None}

            if protocol == 'cv':
                result = run_test(cache, subject, model, tmin, tmax)
                return {**common, 'metrics': result['metrics'], 'seconds': result['seconds'],
                        'n_dropped': result['n_dropped']}

            result = holdout(cache, subject, model, tmin, tmax)
            out = {
                **common,
                'metrics': result['test'],
                'validation': result['validation'],
                'split': {'train': result['n_train'] + result['n_dropped'],
                          'validation': result['validation']['n'], 'test': result['test']['n']},
                'n_dropped': result['n_dropped'],
                'seconds': result['seconds'],
            }
            if repeats:
                started = time.perf_counter()
                out['repeats'] = repeated_holdout(cache, subject, model, tmin, tmax, repeats)
                out['seconds'] += time.perf_counter() - started
            return out

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
