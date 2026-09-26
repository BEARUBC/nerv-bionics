"""An interactive panel for running tests and reading the results.

`ui.py` replays a recording and animates the detector working. This is the other
half: a control panel where you choose the recording, the class, the epoch window
and the model, press Run, and read what came out.

It is a real evaluation, not a demo. Pressing Run cuts fresh epochs at the window
you chose, cleans them, and does a 5-fold cross-validation, so the accuracy shown
is measured on trials the model did not train on.

Run it
------
    python -m classification.app

Save a result straight to an image without opening a window:

    python -m classification.app --snapshot result.png --subject A04T --tmin 1.0 --tmax 4.0
"""

# Check the interpreter before importing the scientific stack, so a Python
# without it gets an explanation instead of a bare ModuleNotFoundError.
from classification._deps import require as _require
_require('classification.app')

import argparse
import sys
import time
import warnings

import numpy as np
import scipy.io
from sklearn.model_selection import StratifiedKFold

from classification.data import (CUE_CODES, DEFAULT_DATA_DIR, EOG_COLUMNS, N_EEG_CHANNELS,
                                 REJECTED_TRIAL_CODE, TRIAL_START_CODE, available_subjects,
                                 interpolate_nans)
from classification.models import MODEL_REGISTRY, build_model
from classification.preprocessing import prepare

CLASS_ORDER = ('left', 'right', 'foot', 'tongue')
CLASS_COLORS = {
    'left': '#2E6E9E',
    'right': '#A85F18',
    'foot': '#67539B',
    'tongue': '#2C7A57',
}

BACKGROUND = '#F3F6F7'
PANEL = '#FFFFFF'
INK = '#131C21'
MUTED = '#5A6A72'
GRID = '#D2DADD'
GOOD = '#0B6E5F'
BAD = '#9E2B26'

CHANCE = 0.25
N_FOLDS = 5

# Models worth offering in the panel: the three selected ones.
OFFERED_MODELS = ('riemann_ts_lr', 'riemann_mdm', 'csp_lda')


class RecordingCache:
    """Hold each recording's continuous signal so window changes stay fast.

    Reading a 40 MB ``.mat`` takes a few seconds. The panel re-cuts epochs every
    time the window sliders move, so the file is parsed once and kept.
    """

    def __init__(self):
        self._store = {}

    def get(self, subject):
        """Return the continuous signal, cue list and sampling rate for a subject.

        Parameters
        ----------
        subject : str
            Dataset stem such as ``'A01T'``.

        Returns
        -------
        tuple
            ``(signal, cues, fs)`` where ``cues`` is a list of
            ``(sample_position, label)``.
        """

        if subject in self._store:
            return self._store[subject]

        mat = scipy.io.loadmat(str(DEFAULT_DATA_DIR / f'{subject}.mat'))
        fs = float(np.asarray(mat['SampleRate']).ravel()[0])
        signal = interpolate_nans(np.delete(np.asarray(mat['s'], dtype=float), EOG_COLUMNS, axis=1))

        event_type = np.asarray(mat['EVENTTYP']).ravel().astype(int)
        event_pos = np.asarray(mat['EVENTPOS']).ravel().astype(int)

        rejected = set()
        for i, code in enumerate(event_type):
            if code != REJECTED_TRIAL_CODE:
                continue
            for j in range(i, -1, -1):
                if event_type[j] == TRIAL_START_CODE:
                    rejected.add(int(event_pos[j]))
                    break

        cues = []
        for i, code in enumerate(event_type):
            if code not in CUE_CODES:
                continue
            start = None
            for j in range(i, -1, -1):
                if event_type[j] == TRIAL_START_CODE:
                    start = int(event_pos[j])
                    break
            if start in rejected:
                continue
            cues.append((int(event_pos[i]), CUE_CODES[code]))

        self._store[subject] = (signal, cues, fs)
        return self._store[subject]

    def epochs(self, subject, tmin, tmax):
        """Cut epochs at the requested window.

        Parameters
        ----------
        subject : str
            Dataset stem.
        tmin, tmax : float
            Window bounds in seconds relative to cue onset.

        Returns
        -------
        X, y, fs
            Raw epochs, labels, and the sampling rate.
        """

        signal, cues, fs = self.get(subject)
        start_offset = int(round(tmin * fs))
        n_samples = int(round(tmax * fs)) - start_offset

        X, y = [], []
        for position, label in cues:
            start = position + start_offset
            if start < 0 or start + n_samples > signal.shape[0]:
                continue
            X.append(signal[start:start + n_samples, :N_EEG_CHANNELS].T)
            y.append(label)

        return np.stack(X), np.asarray(y), fs


def run_test(cache, subject, model_name, tmin, tmax, class_filter=None, folds=N_FOLDS):
    """Cross-validate one configuration and return everything worth showing.

    Parameters
    ----------
    cache : RecordingCache
        Where the recordings come from.
    subject : str
        Dataset stem.
    model_name : str
        Registry key of the pipeline to test.
    tmin, tmax : float
        Epoch window in seconds relative to cue onset.
    class_filter : str or None
        Score only trials of this class. Training always uses all four, because
        a four-class model needs all four.
    folds : int, optional
        Cross-validation folds.

    Returns
    -------
    dict
        Accuracy, per-class recall, confusion matrix, counts and timing.
    """

    X, y, fs = cache.epochs(subject, tmin, tmax)
    X, keep = prepare(X, fs)
    X, y = X[keep], y[keep]

    started = time.perf_counter()

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    true_all, pred_all = [], []

    for train_idx, test_idx in splitter.split(X, y):
        model = build_model(model_name)
        model.fit(X[train_idx], y[train_idx])
        pred_all.extend(model.predict(X[test_idx]).tolist())
        true_all.extend(y[test_idx].tolist())

    elapsed = time.perf_counter() - started

    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)

    if class_filter:
        mask = true_all == class_filter
        scored_true, scored_pred = true_all[mask], pred_all[mask]
    else:
        scored_true, scored_pred = true_all, pred_all

    per_class = {}
    for name in CLASS_ORDER:
        mask = true_all == name
        per_class[name] = float(np.mean(pred_all[mask] == name)) if mask.any() else float('nan')

    matrix = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
    for t, p in zip(true_all, pred_all):
        matrix[CLASS_ORDER.index(t), CLASS_ORDER.index(p)] += 1

    return {
        'accuracy': float(np.mean(scored_pred == scored_true)) if len(scored_true) else float('nan'),
        'per_class': per_class,
        'matrix': matrix,
        'n_scored': int(len(scored_true)),
        'n_total': int(len(true_all)),
        'n_dropped': int((~keep).sum()),
        'seconds': elapsed,
        'window_samples': X.shape[2],
        'subject': subject,
        'model': model_name,
        'tmin': tmin,
        'tmax': tmax,
        'class_filter': class_filter,
    }


def render(axes, result):
    """Draw a completed result into the three result panels.

    Parameters
    ----------
    axes : dict
        ``'headline'``, ``'bars'`` and ``'matrix'`` axes.
    result : dict
        Output of :func:`run_test`.

    Returns
    -------
    None
    """

    # ---- headline --------------------------------------------------------
    ax = axes['headline']
    ax.clear()
    ax.axis('off')

    accuracy = result['accuracy']
    color = GOOD if accuracy >= 0.5 else (INK if accuracy >= CHANCE else BAD)
    scope = result['class_filter'] or 'all four classes'

    ax.text(0.0, 0.80, 'ACCURACY', fontsize=9, color=MUTED, family='monospace',
            transform=ax.transAxes)
    ax.text(0.0, 0.34, f'{accuracy:.1%}', fontsize=46, color=color, fontweight='bold',
            va='center', transform=ax.transAxes)

    detail = (f'{result["subject"]}  ·  {result["model"]}  ·  '
              f'{result["tmin"]:.1f}–{result["tmax"]:.1f} s '
              f'({result["tmax"] - result["tmin"]:.1f} s window)')
    ax.text(0.30, 0.46, detail, fontsize=11, color=INK, transform=ax.transAxes)
    ax.text(0.30, 0.30, f'scored on {scope}: {result["n_scored"]} of {result["n_total"]} trials',
            fontsize=10, color=MUTED, transform=ax.transAxes)

    versus = accuracy - CHANCE
    ax.text(0.30, 0.14,
            f'{versus:+.1%} vs the {CHANCE:.0%} chance level  ·  '
            f'{result["seconds"]:.1f} s to run  ·  {result["n_dropped"]} trials dropped as artifacts',
            fontsize=10, color=MUTED, transform=ax.transAxes)

    # ---- per-class bars --------------------------------------------------
    ax = axes['bars']
    ax.clear()

    values = [result['per_class'][c] for c in CLASS_ORDER]
    positions = np.arange(len(CLASS_ORDER))

    for pos, name, value in zip(positions, CLASS_ORDER, values):
        highlight = result['class_filter'] in (None, name)
        ax.barh(pos, value, height=0.6, color=CLASS_COLORS[name],
                alpha=1.0 if highlight else 0.25)
        ax.text(value + 0.02, pos, f'{value:.0%}', va='center', fontsize=9,
                color=INK, family='monospace')

    ax.axvline(CHANCE, color=BAD, linestyle='--', linewidth=1)
    ax.text(CHANCE + 0.01, len(CLASS_ORDER) - 0.42, 'chance', fontsize=8, color=BAD)

    ax.set_yticks(positions)
    ax.set_yticklabels(CLASS_ORDER, fontsize=10)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['0', '50%', '100%'], fontsize=8)
    ax.invert_yaxis()
    ax.set_title('correct per class', fontsize=10, color=INK, loc='left')
    ax.tick_params(colors=MUTED)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(GRID)

    # ---- confusion matrix -------------------------------------------------
    ax = axes['matrix']
    ax.clear()

    matrix = result['matrix']
    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(normalised, cmap='BuGn', vmin=0, vmax=1)

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=9,
                    color=(PANEL if normalised[i, j] > 0.55 else INK))

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, fontsize=8, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_ORDER, fontsize=8)
    ax.set_xlabel('detected', fontsize=9, color=MUTED)
    ax.set_ylabel('actual', fontsize=9, color=MUTED)
    ax.set_title('what it confused with what', fontsize=10, color=INK, loc='left')
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_panel(cache, subjects, initial):
    """Create the figure, the controls and the wiring between them.

    Parameters
    ----------
    cache : RecordingCache
        Recording source shared by every run.
    subjects : list[str]
        Recordings offered in the subject selector.
    initial : dict
        Starting values for subject, model, tmin, tmax.

    Returns
    -------
    tuple
        ``(figure, widgets)``; the widgets must be kept alive by the caller or
        matplotlib garbage-collects them and the controls stop responding.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    fig = plt.figure(figsize=(14, 8), facecolor=BACKGROUND)
    # Two fig.text calls rather than suptitle: suptitle centres on its y and its
    # descenders then run into the line below.
    fig.text(0.03, 0.955, 'Motor imagery test bench', fontsize=15, fontweight='bold',
             color=INK, va='center')
    fig.text(0.03, 0.918, 'pick a recording, a class and a time window, then press Run test',
             fontsize=10, color=MUTED, va='center')

    def control_axes(rect, title=None):
        ax = fig.add_axes(rect, facecolor=PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if title:
            ax.set_title(title, fontsize=9, color=MUTED, loc='left', pad=6,
                         fontfamily='monospace')
        return ax

    ax_subject = control_axes([0.03, 0.74, 0.17, 0.13], 'RECORDING')
    ax_class = control_axes([0.03, 0.47, 0.17, 0.22], 'TYPE OF DATA')
    ax_model = control_axes([0.03, 0.26, 0.17, 0.16], 'MODEL')
    ax_start = fig.add_axes([0.055, 0.185, 0.125, 0.025], facecolor=PANEL)
    ax_end = fig.add_axes([0.055, 0.135, 0.125, 0.025], facecolor=PANEL)
    ax_run = fig.add_axes([0.03, 0.045, 0.17, 0.055])

    radio_subject = RadioButtons(ax_subject, subjects, active=subjects.index(initial['subject']))
    radio_class = RadioButtons(ax_class, ('all classes',) + CLASS_ORDER, active=0)
    model_labels = [MODEL_REGISTRY[m]['label'] for m in OFFERED_MODELS]
    radio_model = RadioButtons(ax_model, model_labels,
                               active=OFFERED_MODELS.index(initial['model']))

    for radio in (radio_subject, radio_class, radio_model):
        for label in radio.labels:
            label.set_fontsize(9)
            label.set_color(INK)

    slider_start = Slider(ax_start, 'start (s)', 0.0, 2.0, valinit=initial['tmin'],
                          valstep=0.25, color=GOOD)
    slider_end = Slider(ax_end, 'end (s)', 1.5, 6.0, valinit=initial['tmax'],
                        valstep=0.25, color=GOOD)
    for slider in (slider_start, slider_end):
        slider.label.set_fontsize(9)
        slider.label.set_color(MUTED)
        slider.valtext.set_fontsize(9)
        slider.valtext.set_color(INK)
        slider.valtext.set_family('monospace')

    button_run = Button(ax_run, 'Run test', color='#D8EAE5', hovercolor='#BFE0D8')
    button_run.label.set_fontsize(11)
    button_run.label.set_color(GOOD)
    button_run.label.set_fontweight('bold')

    axes = {
        'headline': fig.add_axes([0.26, 0.70, 0.70, 0.19]),
        'bars': fig.add_axes([0.30, 0.34, 0.26, 0.26]),
        'matrix': fig.add_axes([0.69, 0.34, 0.22, 0.26]),
        'status': fig.add_axes([0.26, 0.04, 0.70, 0.22]),
    }
    for ax in axes.values():
        ax.set_facecolor(BACKGROUND)
    axes['headline'].axis('off')
    axes['status'].axis('off')

    def duration_note():
        """Describe the currently selected window in words.

        Returns
        -------
        str
            A short sentence about the window length.
        """

        length = slider_end.val - slider_start.val
        return (f'window {slider_start.val:.2f}–{slider_end.val:.2f} s '
                f'= {length:.2f} s of signal per trial')

    def set_status(lines, color=MUTED):
        """Write the status block under the results.

        Parameters
        ----------
        lines : list[str]
            Lines to show.
        color : str, optional
            Text colour.

        Returns
        -------
        None
        """

        ax = axes['status']
        ax.clear()
        ax.axis('off')
        for i, line in enumerate(lines):
            ax.text(0.0, 0.88 - i * 0.17, line, fontsize=10, color=color,
                    family='monospace', transform=ax.transAxes)

    def on_run(_event):
        """Run the selected configuration and draw the result.

        Parameters
        ----------
        _event : matplotlib event
            Unused; supplied by the Button callback.

        Returns
        -------
        None
        """

        tmin, tmax = float(slider_start.val), float(slider_end.val)

        if tmax - tmin < 1.0:
            set_status(['The window is shorter than 1 second.',
                        'Covariance needs more samples than that, widen it and run again.'], BAD)
            fig.canvas.draw_idle()
            return

        chosen_class = radio_class.value_selected
        class_filter = None if chosen_class == 'all classes' else chosen_class
        model_name = OFFERED_MODELS[model_labels.index(radio_model.value_selected)]

        set_status([f'Running {N_FOLDS}-fold cross-validation...', duration_note()], INK)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        try:
            result = run_test(cache, radio_subject.value_selected, model_name,
                              tmin, tmax, class_filter)
        except Exception as exc:
            set_status([f'That configuration failed: {type(exc).__name__}', str(exc)[:90]], BAD)
            fig.canvas.draw_idle()
            return

        render(axes, result)

        best = max(result['per_class'], key=lambda c: result['per_class'][c])
        worst = min(result['per_class'], key=lambda c: result['per_class'][c])
        set_status([
            duration_note() + f'  ({result["window_samples"]} samples per trial)',
            f'best class: {best} at {result["per_class"][best]:.0%}   '
            f'weakest: {worst} at {result["per_class"][worst]:.0%}',
            f'{N_FOLDS}-fold cross-validated, so every trial was scored by a model '
            f'that had not seen it.',
        ])
        fig.canvas.draw_idle()

    button_run.on_clicked(on_run)
    set_status(['Ready. Choose your settings and press Run test.',
                duration_note()])

    widgets = (radio_subject, radio_class, radio_model, slider_start, slider_end, button_run)
    return fig, widgets, axes, on_run


def main(argv=None):
    """Open the panel, or render one configuration to an image.

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
    parser.add_argument('--subject', default=None, help='Recording selected at startup.')
    parser.add_argument('--model', default='riemann_ts_lr', choices=OFFERED_MODELS,
                        help='Model selected at startup.')
    parser.add_argument('--tmin', type=float, default=0.5, help='Window start in seconds.')
    parser.add_argument('--tmax', type=float, default=3.5, help='Window end in seconds.')
    parser.add_argument('--snapshot', default=None,
                        help='Run once with these settings, save an image, and exit.')
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

    subject = args.subject or subjects[0]
    if subject not in subjects:
        print(f'{subject} is not in acquisition/data. Available: {", ".join(subjects)}',
              file=sys.stderr)
        return 1

    import matplotlib
    if args.snapshot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cache = RecordingCache()
    initial = {'subject': subject, 'model': args.model, 'tmin': args.tmin, 'tmax': args.tmax}

    print(f'Loading {subject}...')
    fig, widgets, axes, on_run = build_panel(cache, subjects, initial)

    # Run once at startup so the panel opens showing a result rather than blanks.
    on_run(None)

    if args.snapshot:
        fig.savefig(args.snapshot, dpi=120, facecolor=BACKGROUND)
        print(f'Saved {args.snapshot}')
        return 0

    fig._nerv_widgets = widgets  # keep the controls alive
    plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
