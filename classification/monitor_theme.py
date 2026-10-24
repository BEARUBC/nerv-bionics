"""The shared look of every screen: a clinical EEG monitor.

Bedside and EEG acquisition monitors share a vocabulary that clinicians read at a
glance, and the three pages borrow it rather than inventing their own:

- a black screen, so bright traces and numerics carry from across a room;
- a status bar with the recording, the mode, acquisition settings and a clock;
- waveforms on a fixed time grid (major line per second, minor per 0.2 s) with
  channel labels on the left and a calibration bar in microvolts;
- a parameter column of large numeric readouts, each in its own colour, with the
  label printed beside the colour so colour is never the only cue;
- soft keys along the bottom edge, each with a keyboard shortcut.

Each page module pulls in :data:`THEME_CSS`, :func:`status_bar` and
:data:`MONITOR_JS` so the screens stay consistent and the drawing code lives once.
"""

# Monitors are dark by design, so this theme is a single deliberate dark look
# rather than a light theme with a dark variant.
THEME_CSS = r"""
  :root {
    --bg:#05080A; --panel:#0A1013; --panel-2:#0F181C; --line:#1B282D; --line-2:#26363C;
    --ink:#E9F1F2; --ink-2:#9DB0B6; --ink-3:#5E747B;
    --grid-major:#163A2E; --grid-minor:#0D1F19;
    --trace:#4FC97F; --trace-focus:#C8FFDC; --rest:rgba(255,255,255,.035);
    --left:#3DC8E6; --right:#F0B43C; --foot:#B18CFF; --tongue:#FF6FA8;
    --ok:#3FD18A; --alarm:#FF5A52; --advisory:#FFC233; --live:#FF5A52;
    --num:"SF Pro Display","Segoe UI","Roboto","Helvetica Neue",Arial,sans-serif;
    --mono:"SF Mono",ui-monospace,Menlo,Consolas,monospace;
    color-scheme: dark;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.45 var(--num);-webkit-font-smoothing:antialiased;
       font-variant-numeric:tabular-nums}
  button,select,input{font:inherit;color:inherit}
  :focus-visible{outline:2px solid var(--advisory);outline-offset:2px}

  /* ---- status bar ------------------------------------------------------- */
  .status{display:flex;align-items:stretch;gap:0;background:#000;border-bottom:1px solid var(--line-2);
          height:44px;font:12px/1 var(--mono);color:var(--ink-2);overflow-x:auto;white-space:nowrap}
  .status > *{display:flex;align-items:center;padding:0 14px;border-right:1px solid var(--line)}
  .status .brand{gap:8px;color:var(--ink);font:700 13px var(--num);letter-spacing:.06em}
  .status .brand svg{width:22px;height:22px}
  .status .tabs{padding:0;gap:0}
  .status .tabs a{display:flex;align-items:center;height:100%;padding:0 16px;color:var(--ink-2);
                  text-decoration:none;font:600 12px var(--mono);letter-spacing:.08em;text-transform:uppercase}
  .status .tabs a:hover{color:var(--ink);background:var(--panel)}
  .status .tabs a.on{color:#000;background:var(--trace)}
  .status .field b{color:var(--ink);font-weight:600;margin-left:6px}
  .status .grow{flex:1;border-right:0}
  .status .clock{color:var(--ink);font:600 15px var(--mono);border-right:0;border-left:1px solid var(--line)}
  .mode{display:inline-flex;align-items:center;gap:7px;font:700 11px var(--mono);letter-spacing:.12em;
        padding:4px 8px;border-radius:3px;border:1px solid currentColor}
  .mode i{width:8px;height:8px;border-radius:50%;background:currentColor}
  .mode.live{color:var(--live)} .mode.live i{animation:blink 1s steps(2,start) infinite}
  .mode.standby{color:var(--ink-3)}
  .mode.review{color:var(--advisory)}
  @keyframes blink{to{visibility:hidden}}
  @media (prefers-reduced-motion:reduce){.mode.live i{animation:none}}

  /* ---- panels ----------------------------------------------------------- */
  .screen{padding:10px 12px 76px;display:grid;gap:10px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;min-width:0}
  .panel > .ph{display:flex;align-items:center;gap:10px;padding:7px 12px;border-bottom:1px solid var(--line);
               font:600 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
  .panel > .ph .r{margin-left:auto;letter-spacing:.04em;text-transform:none;font-weight:400}
  .panel > .pb{padding:12px}

  /* ---- parameter readouts ---------------------------------------------- */
  .param{border-bottom:1px solid var(--line);padding:10px 12px;display:grid;gap:2px}
  .param:last-child{border-bottom:0}
  .param .k{display:flex;justify-content:space-between;font:600 11px var(--mono);letter-spacing:.1em;
            text-transform:uppercase;color:var(--ink-3)}
  .param .v{font:700 40px/1 var(--num);letter-spacing:-.01em}
  .param .v small{font-size:16px;font-weight:600;margin-left:3px;color:var(--ink-2)}
  .param .s{font:12px var(--mono);color:var(--ink-2)}
  .prob{display:grid;grid-template-columns:62px 1fr 44px;gap:8px;align-items:center;padding:3px 0}
  .prob .nm{font:600 12px var(--mono);letter-spacing:.06em;text-transform:uppercase}
  .prob .tr{height:10px;background:var(--panel-2);border-radius:2px;position:relative;overflow:hidden}
  .prob .tr i{position:absolute;inset:0 auto 0 0;transition:width .25s ease}
  .prob .tr .ch{position:absolute;top:0;bottom:0;left:25%;border-left:1px dashed var(--ink-3)}
  .prob .pc{font:600 14px var(--mono);text-align:right}

  .chip{display:inline-flex;align-items:center;gap:6px;font:700 11px var(--mono);letter-spacing:.1em;
        padding:3px 8px;border-radius:3px;text-transform:uppercase}
  .chip.ok{background:var(--ok);color:#000}
  .chip.miss{background:var(--alarm);color:#000}
  .chip.wait{border:1px solid var(--line-2);color:var(--ink-3)}

  /* ---- event log -------------------------------------------------------- */
  .events{font:12px/1.5 var(--mono);max-height:170px;overflow-y:auto}
  .events table{border-collapse:collapse;width:100%}
  .events td,.events th{padding:5px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
  .events th{color:var(--ink-3);font-weight:600;letter-spacing:.08em;text-transform:uppercase;font-size:11px;
             position:sticky;top:0;background:var(--panel)}
  .events td.num{text-align:right}

  /* ---- soft keys -------------------------------------------------------- */
  .softkeys{position:fixed;left:0;right:0;bottom:0;z-index:5;display:flex;gap:6px;flex-wrap:wrap;
            padding:8px max(12px, env(safe-area-inset-right,0px)) calc(8px + env(safe-area-inset-bottom,0px)) max(12px, env(safe-area-inset-left,0px));
            background:#000;border-top:1px solid var(--line-2)}
  .key{display:inline-flex;align-items:center;gap:8px;min-height:40px;padding:0 14px;cursor:pointer;
       background:linear-gradient(#18252A,#0F181C);border:1px solid var(--line-2);border-radius:4px;
       font:600 12px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink)}
  .key:hover{border-color:var(--ink-3)}
  .key:disabled{opacity:.4;cursor:not-allowed}
  .key kbd{font:600 10px var(--mono);color:var(--ink-3);border:1px solid var(--line-2);border-radius:3px;padding:1px 5px}
  .key.primary{background:var(--trace);border-color:var(--trace);color:#000}
  .key.primary kbd{color:#000;border-color:rgba(0,0,0,.35)}
  .key.on{border-color:var(--advisory);color:var(--advisory)}
  .key select{background:transparent;border:0;color:var(--ink);cursor:pointer;font:600 12px var(--mono);text-transform:uppercase}
  .key select option{background:var(--panel);color:var(--ink)}
  .softkeys .spacer{flex:1}
  .softkeys .readout{display:flex;align-items:center;font:12px var(--mono);color:var(--ink-2);padding:0 6px}
  /* On a phone the keys wrap to several rows; pinned, they would cover the readouts. */
  @media (max-width:640px){.softkeys{position:static}.screen{padding-bottom:12px}.softkeys .readout{display:none}}

  /* ---- scope ------------------------------------------------------------ */
  .scope{position:relative;display:block;width:100%}
  .scope canvas{display:block;width:100%}
  .banner{display:flex;align-items:center;gap:10px;padding:8px 12px;font:12px var(--mono);color:var(--ink-2);
          border-top:1px solid var(--line)}
  .err{margin:0 12px;color:var(--alarm);border:1px solid var(--alarm);border-radius:4px;padding:10px 12px;font:13px var(--mono)}
  .muted{color:var(--ink-3)}
"""

# Monogram for the status bar: a stylised EEG deflection inside a ring.
_MARK = ('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10.5" fill="none" '
         'stroke="#4FC97F" stroke-width="1.5"/><path d="M4 13h3l2-5 3 9 2-6 1.5 2H20" fill="none" '
         'stroke="#4FC97F" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/></svg>')

_TABS = (('monitor', '/live', 'Monitor'), ('identify', '/identify', 'Identify'), ('review', '/', 'Review'))


def status_bar(active, mode_id='mode'):
    """Return the top status bar with the screen tabs and a running clock.

    Parameters
    ----------
    active : str
        ``'monitor'``, ``'identify'`` or ``'review'``.
    mode_id : str, optional
        Element id of the mode pill, so the page can switch it between
        standby, live and review.

    Returns
    -------
    str
        HTML for the bar. Fields with ids ``st_rec``, ``st_model``, ``st_fs`` and
        ``st_filter`` are filled in by the page.
    """

    tabs = ''.join(
        f'<a href="{href}" class="{"on" if key == active else ""}">{label}</a>'
        for key, href, label in _TABS
    )
    return f"""<div class="status" role="banner">
  <div class="brand">{_MARK}NERV&nbsp;MI</div>
  <nav class="tabs" aria-label="Screens">{tabs}</nav>
  <div><span class="mode standby" id="{mode_id}"><i></i>STANDBY</span></div>
  <div class="field">REC<b id="st_rec">-</b></div>
  <div class="field">MODEL<b id="st_model">-</b></div>
  <div class="field">FS<b id="st_fs">250 Hz</b></div>
  <div class="field">FILTER<b id="st_filter">8-30 Hz</b></div>
  <div class="grow"></div>
  <div class="clock" id="st_clock">--:--:--</div>
</div>"""


MONITOR_JS = r"""
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const CLASS_VAR = {left:'--left', right:'--right', foot:'--foot', tongue:'--tongue'};
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const classColor = c => css(CLASS_VAR[c] || '--ink');

async function post(url, body) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'request failed');
  return data;
}

function setMode(kind, text) {
  const m = $('mode');
  m.className = 'mode ' + kind;
  m.innerHTML = '<i></i>' + text;
}

(function clock() {
  const tick = () => { $('st_clock').textContent = new Date().toLocaleTimeString([], {hour12:false}); };
  tick(); setInterval(tick, 1000);
})();

// Multichannel EEG display on a canvas: time grid, channel labels, the motor
// channels highlighted, the post-imagery rest period shaded, and a calibration
// bar in microvolts. Redraws itself on resize with the last data it was given.
class Scope {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.lane = opts.lane || 22;
    this.labelW = 46;
    this.ruler = 22;
    this.data = null;
    new ResizeObserver(() => this.render()).observe(canvas.parentElement);
  }
  draw(data) { this.data = data; this.render(); }
  clear(channels) { this.data = {channels, traces: null}; this.render(); }
  render() {
    const d = this.data;
    const n = d ? d.channels.length : 22;
    const W = this.canvas.parentElement.clientWidth;
    const H = n * this.lane + this.ruler;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = W * dpr; this.canvas.height = H * dpr;
    this.canvas.style.height = H + 'px';
    const g = this.canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = css('--bg'); g.fillRect(0, 0, W, H);
    if (!d) return;

    const x0 = this.labelW, plotW = W - x0 - 8, plotH = n * this.lane;
    const t0 = d.t0 ?? 0, dur = d.dur ?? 4;
    const X = t => x0 + (t - t0) / dur * plotW;

    // rest period after the imagery ends
    if (d.restFrom !== undefined && d.restFrom < t0 + dur) {
      const xs = Math.max(x0, X(d.restFrom));
      g.fillStyle = css('--rest'); g.fillRect(xs, 0, x0 + plotW - xs, plotH);
      g.fillStyle = css('--ink-3'); g.font = '10px ' + css('--mono');
      g.fillText('REST', xs + 6, 12);
    }
    // time grid: minor every 0.2 s, major every second
    for (let t = Math.ceil(t0 / 0.2) * 0.2; t <= t0 + dur + 1e-6; t += 0.2) {
      const major = Math.abs(t - Math.round(t)) < 1e-6;
      g.strokeStyle = css(major ? '--grid-major' : '--grid-minor');
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(Math.round(X(t)) + .5, 0); g.lineTo(Math.round(X(t)) + .5, plotH); g.stroke();
      if (major) {
        g.fillStyle = css('--ink-3'); g.font = '11px ' + css('--mono'); g.textAlign = 'center';
        g.fillText(t.toFixed(0) + ' s', X(t), plotH + 15);
      }
    }
    g.textAlign = 'left';

    const focus = new Set(d.focus || []);
    d.channels.forEach((name, i) => {
      const mid = i * this.lane + this.lane / 2;
      const isFocus = focus.has(name);
      if (isFocus) { g.fillStyle = 'rgba(200,255,220,.045)'; g.fillRect(0, i * this.lane, W, this.lane); }
      g.fillStyle = css(isFocus ? '--trace-focus' : '--ink-3');
      g.font = (isFocus ? '600 ' : '') + '11px ' + css('--mono');
      g.fillText(name, 8, mid + 4);
      const pts = d.traces && d.traces[i];
      if (!pts) return;
      g.strokeStyle = css(isFocus ? '--trace-focus' : '--trace');
      g.lineWidth = isFocus ? 1.3 : 1;
      g.lineJoin = 'round';
      g.beginPath();
      const amp = this.lane * 0.42;
      pts.forEach((v, j) => {
        const x = x0 + j / (pts.length - 1) * plotW;
        const y = mid - Math.max(-1.6, Math.min(1.6, v)) * amp;
        j ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
    });

    // calibration bar: one lane-amplitude unit, labelled in microvolts
    if (d.uv) {
      const h = this.lane * 0.42, x = x0 + plotW - 4, y = plotH - 8;
      g.strokeStyle = css('--ink'); g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x, y - h); g.moveTo(x - 4, y); g.lineTo(x + 0, y); g.stroke();
      g.fillStyle = css('--ink'); g.font = '600 11px ' + css('--mono'); g.textAlign = 'right';
      g.fillText(Math.round(d.uv) + ' µV', x - 8, y - h / 2 + 4);
      g.textAlign = 'left';
    }
  }
}

// Probability trend: one line per class across the decisions made so far.
class Trend {
  constructor(canvas, classes) {
    this.canvas = canvas; this.classes = classes; this.points = [];
    new ResizeObserver(() => this.render()).observe(canvas.parentElement);
  }
  reset() { this.points = []; this.render(); }
  push(t, probabilities) { this.points.push({t, p: probabilities}); this.render(); }
  render() {
    const W = this.canvas.parentElement.clientWidth, H = 120, dpr = window.devicePixelRatio || 1;
    this.canvas.width = W * dpr; this.canvas.height = H * dpr; this.canvas.style.height = H + 'px';
    const g = this.canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = css('--panel'); g.fillRect(0, 0, W, H);
    const x0 = 44, x1 = W - 26, y0 = 10, y1 = H - 20;
    const Y = p => y1 - p * (y1 - y0);
    g.font = '10px ' + css('--mono'); g.fillStyle = css('--ink-3');
    [0, .25, .5, 1].forEach(p => {
      g.strokeStyle = css(p === .25 ? '--ink-3' : '--line');
      g.setLineDash(p === .25 ? [3, 3] : []);
      g.beginPath(); g.moveTo(x0, Y(p) + .5); g.lineTo(x1, Y(p) + .5); g.stroke();
      g.fillText(Math.round(p * 100) + '%', 6, Y(p) + 3);
    });
    g.setLineDash([]);
    const pts = this.points;
    if (!pts.length) { g.fillText('waiting for the first full window', x0 + 8, (y0 + y1) / 2); return; }
    const tMin = pts[0].t, tMax = Math.max(pts[pts.length - 1].t, tMin + 2);
    const X = t => x0 + (t - tMin) / (tMax - tMin) * (x1 - x0);
    g.textAlign = 'center';
    pts.forEach(q => g.fillText(q.t.toFixed(1) + 's', X(q.t), H - 5));
    g.textAlign = 'left';
    this.classes.forEach(c => {
      g.strokeStyle = classColor(c); g.fillStyle = classColor(c); g.lineWidth = 2;
      g.beginPath();
      pts.forEach((q, i) => { const x = X(q.t), y = Y(q.p[c] ?? 0); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
      const last = pts[pts.length - 1];
      g.beginPath(); g.arc(X(last.t), Y(last.p[c] ?? 0), 3.5, 0, Math.PI * 2); g.fill();
    });
  }
}

function probRows(host, classes) {
  $(host).innerHTML = classes.map(c => `
    <div class="prob"><span class="nm" style="color:var(${CLASS_VAR[c]})">${c}</span>
      <span class="tr"><i id="${host}_f_${c}" style="width:0;background:var(${CLASS_VAR[c]})"></i><span class="ch"></span></span>
      <span class="pc" id="${host}_p_${c}">--</span></div>`).join('');
}
function setProbs(host, classes, p) {
  classes.forEach(c => {
    const v = p ? (p[c] ?? 0) : 0;
    $(`${host}_f_${c}`).style.width = (v * 100).toFixed(1) + '%';
    $(`${host}_p_${c}`).textContent = p ? Math.round(v * 100) + '%' : '--';
  });
}
"""
