"""A small live window showing the detector working.

The CLIs print numbers. This shows the same run as a picture: the EEG going in,
the four class probabilities updating as the window slides, and the call the
detector is currently making.

It replays a recording rather than reading a headset, so you can watch it without
hardware. Because it uses the same rolling-buffer decoder the live path uses, what
you see is what a real session would look like.

Run it
------
    python -m classification.ui --model models/a01t.joblib --subject A01T
    python -m classification.ui --model models/a01t.joblib --subject A01T --only tongue
    python -m classification.ui --model models/a01t.joblib --subject A04T --only foot --speed 2

Save a still image instead of opening a window (useful over SSH):

    python -m classification.ui --model models/a01t.joblib --subject A01T --snapshot look.png
"""

# Check the interpreter before importing the scientific stack, so a Python
# without it gets an explanation instead of a bare ModuleNotFoundError.
from classification._deps import require as _require
_require('classification.ui')

import argparse
import sys
import warnings

import numpy as np

from classification.data import CHANNEL_NAMES, CLASSES, DEFAULT_DATA_DIR, load_mat_epochs
from classification.detector import MotorImageryDetector
from classification.realtime import OnlineDecoder

# Channels over the motor cortex. These are the ones where imagined movement is
# visible by eye, so they are the ones worth drawing.
DISPLAY_CHANNELS = ('C3', 'Cz', 'C4')

# One colour per class, reused by the bars and the verdict text.
CLASS_COLORS = {
    'left': '#2E6E9E',
    'right': '#A85F18',
    'foot': '#67539B',
    'tongue': '#2C7A57',
}

BACKGROUND = '#F3F6F7'
INK = '#131C21'
MUTED = '#5A6A72'
GRID = '#D2DADD'


def build_frames(detector, X, y, step_s, smoothing):
    """Replay each trial through the online decoder and collect what to draw.

    Precomputing is deliberate: the animation then just steps through a list, so a
    slow model can never stall the window or desynchronise the display from the
    numbers underneath it.

    Parameters
    ----------
    detector : MotorImageryDetector
        A trained detector.
    X : numpy.ndarray
        Raw epochs of shape (n_trials, n_channels, n_samples).
    y : numpy.ndarray
        True label per epoch.
    step_s : float
        Seconds between decisions.
    smoothing : int
        Windows averaged before deciding.

    Returns
    -------
    list[dict]
        One entry per decision, each holding the window drawn, the probabilities,
        the truth, and the running score at that point.
    """

    decoder = OnlineDecoder.from_detector(detector, step_s=step_s, smoothing=smoothing)
    window = detector.window_samples

    frames = []
    correct = 0
    answered = 0

    for trial_index, (trial, truth) in enumerate(zip(X, y), start=1):
        decoder.reset()
        decisions = []

        # Walk the trial one sample at a time, exactly as push() would be fed live.
        for sample_index in range(trial.shape[1]):
            decision = decoder.push(trial[:, sample_index])
            if decision is None:
                continue

            decisions.append(decision)
            frames.append({
                'trial': trial_index,
                'truth': truth,
                'window': trial[:, sample_index - window + 1:sample_index + 1],
                'probabilities': decision.probabilities,
                'label': decision.top_label if decision.label is None else decision.label,
                'confidence': decision.confidence,
                'declined': decision.label is None,
                'step': len(decisions),
                'final': False,
            })

        if not decisions:
            continue

        final = decisions[-1]
        called = final.label if final.label is not None else None
        if called is not None:
            answered += 1
            correct += int(called == truth)

        frames[-1]['final'] = True
        frames[-1]['running'] = (correct, answered)

    # Carry the running score forward so every frame can display it.
    running = (0, 0)
    for frame in frames:
        running = frame.get('running', running)
        frame['running'] = running

    return frames


def draw(fig, axes, frame, detector, channel_rows, fs):
    """Redraw the window for one frame.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure being animated.
    axes : dict
        The three axes: ``'signal'``, ``'bars'``, ``'verdict'``.
    frame : dict
        One entry from :func:`build_frames`.
    detector : MotorImageryDetector
        Used for the class order.
    channel_rows : list[tuple[str, int]]
        Channel name and row index into the window, for the traces drawn.
    fs : float
        Sampling rate, for the time axis.

    Returns
    -------
    None
    """

    truth = frame['truth']
    label = frame['label']
    hit = label == truth

    # ---- EEG traces -------------------------------------------------------
    ax = axes['signal']
    ax.clear()

    window = frame['window']
    time = np.arange(window.shape[1]) / fs
    # Spread the traces vertically so they do not overlap.
    spacing = float(np.percentile(np.abs(window), 99)) * 3.0 or 1.0

    for row, (name, index) in enumerate(channel_rows):
        offset = (len(channel_rows) - 1 - row) * spacing
        ax.plot(time, window[index] + offset, linewidth=0.9, color=MUTED)
        ax.text(-0.02, offset, name, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=9, color=INK, family='monospace')

    ax.set_xlim(0, time[-1] if len(time) else 1)
    ax.set_yticks([])
    ax.set_xlabel('seconds inside the current window', fontsize=9, color=MUTED)
    ax.set_title(f'Trial {frame["trial"]}  ·  window {frame["step"]}  ·  '
                 f'{len(channel_rows)} of {window.shape[0]} channels shown',
                 fontsize=10, color=INK, loc='left')
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(GRID)

    # ---- probability bars -------------------------------------------------
    ax = axes['bars']
    ax.clear()

    classes = [str(c) for c in detector.classes_]
    values = [frame['probabilities'].get(c, 0.0) for c in classes]
    positions = np.arange(len(classes))

    colors = [CLASS_COLORS.get(c, MUTED) for c in classes]
    alphas = [1.0 if c == label else 0.35 for c in classes]

    for pos, value, color, alpha in zip(positions, values, colors, alphas):
        ax.barh(pos, value, color=color, alpha=alpha, height=0.62)
        ax.text(value + 0.02, pos, f'{value:.0%}', va='center',
                fontsize=9, color=INK, family='monospace')

    ax.set_yticks(positions)
    ax.set_yticklabels(classes, fontsize=10)
    for tick, name in zip(ax.get_yticklabels(), classes):
        tick.set_color(INK if name == truth else MUTED)
        if name == truth:
            tick.set_fontweight('bold')

    # Mark the true class at the right edge rather than in the tick label, which
    # would be long enough to run off the side of the figure.
    if truth in classes:
        ax.text(1.30, classes.index(truth), 'actual', ha='right', va='center',
                fontsize=9, color=INK, style='italic')

    ax.set_xlim(0, 1.34)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'], fontsize=8)
    ax.axvline(0.25, color='#9E2B26', linestyle='--', linewidth=1)
    # The y-axis is inverted, so a larger value sits below the bottom bar.
    ax.text(0.26, len(classes) - 0.48, 'chance', fontsize=8, color='#9E2B26', va='center')
    ax.invert_yaxis()
    ax.tick_params(colors=MUTED)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.set_title('how sure it is about each class', fontsize=10, color=INK, loc='left')

    # ---- verdict ----------------------------------------------------------
    ax = axes['verdict']
    ax.clear()
    ax.axis('off')

    color = CLASS_COLORS.get(label, MUTED)
    if frame['declined']:
        headline, sub = 'no call', f'leaning {label}, below the confidence floor'
    else:
        headline, sub = label, ('correct' if hit else f'wrong — actually {truth}')

    ax.text(0.5, 0.78, 'DETECTED', ha='center', fontsize=9, color=MUTED,
            family='monospace', transform=ax.transAxes)
    ax.text(0.5, 0.52, headline, ha='center', va='center', fontsize=30,
            color=color, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.30, f'{frame["confidence"]:.0%} confident', ha='center',
            fontsize=12, color=INK, family='monospace', transform=ax.transAxes)
    ax.text(0.5, 0.17, sub, ha='center', fontsize=10,
            color=('#2C7A57' if hit and not frame['declined'] else '#9E2B26'),
            transform=ax.transAxes)

    correct, answered = frame['running']
    score = f'{correct}/{answered} correct so far' if answered else 'first trial'
    ax.text(0.5, 0.02, score, ha='center', fontsize=10, color=MUTED,
            family='monospace', transform=ax.transAxes)


def main(argv=None):
    """Replay a recording through the detector and show it.

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
    parser.add_argument('--model', required=True, help='Detector saved by classification.train.')
    parser.add_argument('--subject', required=True, help='Dataset stem to replay, e.g. A01T.')
    parser.add_argument('--only', choices=sorted(CLASSES), default=None,
                        help='Only replay trials of this class: left, right, foot or tongue.')
    parser.add_argument('--limit', type=int, default=8, help='How many trials to replay.')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier. 2 is twice as fast.')
    parser.add_argument('--step', type=float, default=0.5, help='Seconds between decisions.')
    parser.add_argument('--smoothing', type=int, default=3, help='Windows averaged per decision.')
    parser.add_argument('--min-confidence', type=float, default=None,
                        help='Override the confidence floor stored in the model.')
    parser.add_argument('--snapshot', default=None,
                        help='Save a single still frame to this path instead of opening a window.')
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')

    import matplotlib
    if args.snapshot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    detector = MotorImageryDetector.load(args.model)
    if args.min_confidence is not None:
        detector.min_confidence = args.min_confidence

    mat_path = DEFAULT_DATA_DIR / f'{args.subject}.mat'
    if not mat_path.exists():
        print(f'No recording at {mat_path}.', file=sys.stderr)
        return 1

    # Read further past the cue than the window, so the buffer has room to slide.
    stream_tmax = detector.tmax + 2.0
    X, y, fs = load_mat_epochs(mat_path, tmin=detector.tmin, tmax=stream_tmax)

    if args.only:
        mask = y == args.only
        if not mask.any():
            print(f'No "{args.only}" trials in {args.subject}.', file=sys.stderr)
            return 1
        X, y = X[mask], y[mask]
        print(f'Replaying {args.only} trials only: {mask.sum()} available.')

    X, y = X[:args.limit], y[:args.limit]

    print(f'Building {len(y)} trials...')
    frames = build_frames(detector, X, y, step_s=args.step, smoothing=args.smoothing)
    if not frames:
        print('No decisions produced — the trials are shorter than the model window.', file=sys.stderr)
        return 1

    channel_rows = [
        (name, CHANNEL_NAMES.index(name))
        for name in DISPLAY_CHANNELS
        if name in CHANNEL_NAMES
    ]

    fig = plt.figure(figsize=(12, 6.5), facecolor=BACKGROUND)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.45, 1],
                            hspace=0.42, wspace=0.28,
                            left=0.08, right=0.96, top=0.88, bottom=0.10)

    axes = {
        'signal': fig.add_subplot(grid[0, :]),
        'bars': fig.add_subplot(grid[1, 0]),
        'verdict': fig.add_subplot(grid[1, 1]),
    }
    for ax in axes.values():
        ax.set_facecolor(BACKGROUND)

    title = f'{args.subject} through {detector.model_name}'
    if args.only:
        title += f'  ·  {args.only} trials only'
    fig.suptitle(title, fontsize=13, color=INK, x=0.08, ha='left', fontweight='bold')

    if args.snapshot:
        # Draw the last frame of the first trial: a complete, representative view.
        chosen = next((f for f in frames if f['final']), frames[-1])
        draw(fig, axes, chosen, detector, channel_rows, fs)
        fig.savefig(args.snapshot, dpi=130, facecolor=BACKGROUND)
        print(f'Saved {args.snapshot}')
        return 0

    interval_ms = max(60, int(args.step * 1000 / max(args.speed, 0.01)))

    def update(index):
        """Draw frame ``index``.

        Parameters
        ----------
        index : int
            Frame number supplied by FuncAnimation.

        Returns
        -------
        list
            Empty; the axes are redrawn in place rather than blitted.
        """

        draw(fig, axes, frames[index], detector, channel_rows, fs)
        return []

    print(f'{len(frames)} frames at {interval_ms} ms each. Close the window to stop.')
    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=interval_ms, repeat=False, blit=False
    )
    fig._nerv_animation = animation  # keep a reference so it is not garbage collected
    plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
