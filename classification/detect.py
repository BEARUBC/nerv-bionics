"""Run a trained detector over a recording and show what it detects.

This is step two. It loads a detector saved by ``classification.train`` and runs
it over a ``.mat`` recording, printing one line per trial: what the person was
actually imagining, what the detector said, and how confident it was.

Two modes:

``--mode trials`` (default)
    Cue-aligned. Cut the same 3-second window the detector was trained on from
    each cue and classify it. This is the clean case and gives the highest
    accuracy.

``--mode stream``
    Replay the recording sample by sample through the rolling-buffer decoder, the
    same code path a live headset would use. It proves the online path works
    rather than only the offline maths.

    Stream mode deliberately cuts a *longer* slice than the detector's window so
    the window has somewhere to slide. With a 3 s window and a 3 s trial the
    buffer only fills on the very last sample, which yields exactly one decision
    and tells you nothing about rolling behaviour. ``--stream-tmax`` controls how
    far past the cue to read; the default gives two extra seconds of travel.

Usage
-----
    python -m classification.detect --model models/a01t.joblib --subject A04T
    python -m classification.detect --model models/a01t.joblib --subject A01T --only foot
    python -m classification.detect --model models/a01t.joblib --subject A01T --limit 20
    python -m classification.detect --model models/a01t.joblib --subject A01T --mode stream
"""

# Check the interpreter before importing the scientific stack, so a Python
# without it gets an explanation instead of a bare ModuleNotFoundError.
from classification._deps import require as _require
_require('classification.detect')

import argparse
import sys
import time
import warnings

import numpy as np

from classification.data import CLASSES, DEFAULT_DATA_DIR, load_mat_epochs
from classification.detector import MotorImageryDetector
from classification.preprocessing import prepare_trial
from classification.realtime import OnlineDecoder

# Bar drawn next to each detection so confidence is readable at a glance.
BAR_WIDTH = 16


def confidence_bar(value, width=BAR_WIDTH):
    """Draw a small text bar for a 0-1 value.

    Parameters
    ----------
    value : float
        Confidence between 0 and 1.
    width : int, optional
        Bar width in characters.

    Returns
    -------
    str
        A bar such as ``'#########-------'``.
    """

    filled = int(round(max(0.0, min(1.0, value)) * width))
    return '#' * filled + '-' * (width - filled)


def run_trials(detector, X, y, limit=None, delay=0.0):
    """Classify cue-aligned trials and print a line for each.

    Parameters
    ----------
    detector : MotorImageryDetector
        A trained detector.
    X : numpy.ndarray
        Raw epochs of shape (n_trials, n_channels, n_samples).
    y : numpy.ndarray
        True labels.
    limit : int, optional
        Only process the first N trials.
    delay : float, optional
        Seconds to pause between lines, to make the output readable as it scrolls.

    Returns
    -------
    dict
        ``{'accuracy', 'n_scored', 'n_declined', 'median_latency_ms'}``.
    """

    if limit:
        X, y = X[:limit], y[:limit]

    print(f"\n{'#':>4}  {'actual':<8} {'detected':<8} {'conf':>6}  {'':<{BAR_WIDTH}}  result")
    print('-' * (4 + 2 + 9 + 9 + 7 + 2 + BAR_WIDTH + 9))

    correct = 0
    scored = 0
    declined = 0
    latencies = []

    for i, (trial, truth) in enumerate(zip(X, y), start=1):
        start = time.perf_counter()
        result = detector.detect(trial)
        latencies.append((time.perf_counter() - start) * 1000.0)

        if result.label is None:
            declined += 1
            mark = 'no call'
        else:
            scored += 1
            hit = result.label == truth
            correct += int(hit)
            mark = 'ok' if hit else 'MISS'

        shown = result.label if result.label is not None else result.top_label
        print(f'{i:>4}  {truth:<8} {shown:<8} {result.confidence:>5.0%}  '
              f'{confidence_bar(result.confidence)}  {mark}')

        if delay:
            time.sleep(delay)

    accuracy = correct / scored if scored else float('nan')

    return {
        'accuracy': accuracy,
        'n_scored': scored,
        'n_declined': declined,
        'median_latency_ms': float(np.median(latencies)) if latencies else float('nan'),
    }


def run_stream(detector, X, y, limit=None, step_s=0.5, smoothing=3):
    """Replay trials sample by sample through the online decoder.

    Parameters
    ----------
    detector : MotorImageryDetector
        A trained detector; its pipeline is reused by the online decoder.
    X : numpy.ndarray
        Raw epochs of shape (n_trials, n_channels, n_samples).
    y : numpy.ndarray
        True labels.
    limit : int, optional
        Only process the first N trials.
    step_s : float, optional
        How often the decoder emits a decision, in seconds.
    smoothing : int, optional
        Number of consecutive windows averaged before deciding.

    Returns
    -------
    dict
        ``{'accuracy', 'n_scored', 'median_latency_ms', 'decisions_per_trial'}``.
    """

    if limit:
        X, y = X[:limit], y[:limit]

    # The pipeline is already trained, so reuse it rather than refitting.
    decoder = OnlineDecoder.from_detector(detector, step_s=step_s, smoothing=smoothing)

    print(f'\nReplaying {len(y)} trials sample by sample '
          f'({step_s:g}s between decisions, smoothed over {smoothing} windows)\n')
    print(f"{'#':>4}  {'actual':<8} {'final call':<11} {'conf':>6}  decisions  result")
    print('-' * 62)

    correct = 0
    scored = 0
    latencies = []
    decision_counts = []

    for i, (trial, truth) in enumerate(zip(X, y), start=1):
        decoder.reset()
        decisions = decoder.push_chunk(trial)

        if not decisions:
            print(f'{i:>4}  {truth:<8} {"(too short)":<11}')
            continue

        final = decisions[-1]
        latencies.extend(d.latency_ms for d in decisions)
        decision_counts.append(len(decisions))

        if final.label is None:
            mark = 'no call'
            shown = '-'
        else:
            scored += 1
            hit = final.label == truth
            correct += int(hit)
            mark = 'ok' if hit else 'MISS'
            shown = final.label

        print(f'{i:>4}  {truth:<8} {shown:<11} {final.confidence:>5.0%}  '
              f'{len(decisions):>9}  {mark}')

    return {
        'accuracy': correct / scored if scored else float('nan'),
        'n_scored': scored,
        'median_latency_ms': float(np.median(latencies)) if latencies else float('nan'),
        'decisions_per_trial': float(np.mean(decision_counts)) if decision_counts else 0.0,
    }


def _filter_class(X, y, only):
    """Keep only the trials of one class, if the caller asked for one.

    Running a single class at a time is the quickest way to see which movement
    the detector struggles with, rather than reading it out of a confusion matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Epochs of shape (n_trials, n_channels, n_samples).
    y : numpy.ndarray
        True labels.
    only : str or None
        Class to keep, or None to keep everything.

    Returns
    -------
    X, y : numpy.ndarray
        The filtered epochs and labels.
    """

    if only is None:
        return X, y

    mask = y == only
    if not mask.any():
        raise SystemExit(f'No "{only}" trials in this recording.')

    print(f'Filtering to {only} trials only: {int(mask.sum())} of {len(y)}.')
    return X[mask], y[mask]


def main(argv=None):
    """Load a detector and run it over a recording.

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
    parser.add_argument('--model', required=True, help='Path to a detector saved by classification.train.')
    parser.add_argument('--subject', required=True, help='Dataset stem to run over, for example A04T.')
    parser.add_argument('--mode', choices=['trials', 'stream'], default='trials',
                        help='trials = cue-aligned windows; stream = sample-by-sample replay.')
    parser.add_argument('--only', choices=sorted(CLASSES), default=None,
                        help='Only run trials of this class: left, right, foot or tongue.')
    parser.add_argument('--limit', type=int, default=None, help='Only process the first N trials.')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Seconds between printed lines, to watch it scroll (trials mode).')
    parser.add_argument('--min-confidence', type=float, default=None,
                        help='Override the confidence floor stored in the model.')
    parser.add_argument('--step', type=float, default=0.5,
                        help='Seconds between decisions in stream mode.')
    parser.add_argument('--stream-tmax', type=float, default=None,
                        help='How many seconds after the cue to replay in stream mode. '
                             'Must exceed the detector window for the decoder to slide. '
                             'Defaults to the detector window plus 2 s.')
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')

    detector = MotorImageryDetector.load(args.model)
    if args.min_confidence is not None:
        detector.min_confidence = args.min_confidence

    print('=' * 62)
    print('  DETECTOR')
    print('=' * 62)
    print(detector.describe())

    mat_path = DEFAULT_DATA_DIR / f'{args.subject}.mat'
    if not mat_path.exists():
        print(f'\nNo recording at {mat_path}.', file=sys.stderr)
        return 1

    trained_on = detector.metadata.get('trained_on', [])
    if args.subject in trained_on:
        print(f'\nNote: {args.subject} was part of this detector\'s training data, so these numbers '
              f'flatter it.\n      Run it on a subject it has not seen for an honest read.')
    elif trained_on:
        print(f'\n{args.subject} was NOT in the training set ({", ".join(trained_on)}), '
              f'so this is a genuine cross-subject test - expect a much lower score.')

    print()

    if args.mode == 'trials':
        X, y, fs = load_mat_epochs(mat_path, tmin=detector.tmin, tmax=detector.tmax)
        X, y = _filter_class(X, y, args.only)
        stats = run_trials(detector, X, y, limit=args.limit, delay=args.delay)
    else:
        # Read further past the cue than the detector's window, so the rolling
        # buffer has room to produce more than a single decision per trial.
        stream_tmax = args.stream_tmax if args.stream_tmax is not None else detector.tmax + 2.0
        window_s = detector.window_samples / detector.fs

        if stream_tmax - detector.tmin < window_s:
            print(f'--stream-tmax {stream_tmax:g} gives less than the {window_s:.2f} s the detector '
                  f'needs; raising it to {detector.tmin + window_s:.2f}.', file=sys.stderr)
            stream_tmax = detector.tmin + window_s

        X, y, fs = load_mat_epochs(mat_path, tmin=detector.tmin, tmax=stream_tmax)
        X, y = _filter_class(X, y, args.only)
        stats = run_stream(detector, X, y, limit=args.limit, step_s=args.step)

    chance = 1.0 / len(detector.classes_)

    print('\n' + '=' * 62)
    print('  SUMMARY')
    print('=' * 62)
    print(f'  Trials answered   : {stats["n_scored"]}')
    if stats.get('n_declined'):
        print(f'  Declined to call  : {stats["n_declined"]} (below the {detector.min_confidence:.0%} confidence floor)')
    print(f'  Accuracy          : {stats["accuracy"]:.1%}   (random guessing = {chance:.0%})')
    print(f'  Median latency    : {stats["median_latency_ms"]:.2f} ms per decision')
    if stats.get('decisions_per_trial'):
        print(f'  Decisions / trial : {stats["decisions_per_trial"]:.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
