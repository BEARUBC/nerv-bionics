"""Score every registered model on every axis we care about.

Accuracy alone does not decide which model ships. A model that needs ten minutes
of calibration per user, or 80 ms to classify one window, is worse for a
real-time prosthetic than a slightly less accurate model that fits in a second.
So this script measures six axes and prints them side by side:

- ``acc``          : mean accuracy, stratified k-fold within each subject
- ``kappa``        : Cohen's kappa, chance-corrected so 4-class 25% maps to 0
- ``macro_f1``     : macro F1, catches a model that ignores one class
- ``fit_s``        : seconds to calibrate on one subject (per fold)
- ``infer_ms``     : milliseconds to classify one epoch (the real-time budget)
- ``size_kb``      : pickled model size
- ``loso_acc``     : accuracy when trained on other subjects only (no calibration)

Usage
-----
    python -m classification.benchmark
    python -m classification.benchmark --subjects A01T A04T --tmin 0.5 --tmax 3.5
    python -m classification.benchmark --models riemann_ts_lr riemann_mdm --folds 5
"""

# Check the interpreter before importing the scientific stack, so a Python
# without it gets an explanation instead of a bare ModuleNotFoundError.
from classification._deps import require as _require
_require('classification.benchmark')

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold

from classification.data import available_subjects, load_subjects
from classification.models import MODEL_REGISTRY
from classification.preprocessing import prepare

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'acquisition' / 'data'

# How each axis is turned into a 0-100 score.
#
# Quality axes are scored against a fixed floor and the best model observed, so a
# model that is genuinely close to the leader scores close to it. Using the worst
# model as the floor (plain min-max) would have made the last-placed model score
# 0 even when it sat only a few points behind, which is misleading.
#
# Cost axes are scored against a fixed budget instead, because what matters is
# whether a model fits the device's timing budget, not whether it beat the other
# models by a microsecond.

# A user will not sit through more than a minute of calibration per session.
CALIBRATION_BUDGET_S = 60.0

# 50 ms per decision supports a 20 Hz control loop, which is comfortably faster
# than a prosthetic hand can act on a command.
LATENCY_BUDGET_MS = 50.0

# Quality axes: (floor, whether the floor is the chance level).
QUALITY_AXES = {
    'acc': 'chance',
    'loso_acc': 'chance',
    'kappa': 'zero',
    'macro_f1': 'zero',
}

COST_AXES = {
    'fit_s': CALIBRATION_BUDGET_S,
    'infer_ms': LATENCY_BUDGET_MS,
}

# Weights for the composite score. Accuracy and chance-corrected agreement
# dominate, but calibration and inference cost still count because this runs on a
# wearable device.
COMPOSITE_WEIGHTS = {
    'acc': 0.30,
    'kappa': 0.25,
    'macro_f1': 0.10,
    'loso_acc': 0.15,
    'fit_s': 0.10,
    'infer_ms': 0.10,
}


def measure_inference_ms(model, X, n_repeats=20):
    """Time single-epoch inference, which is what online decoding pays.

    Parameters
    ----------
    model : object
        A fitted estimator with ``predict``.
    X : numpy.ndarray
        Epochs to sample from, shape (n_trials, n_channels, n_samples).
    n_repeats : int, optional
        Number of single-epoch predictions to time.

    Returns
    -------
    float
        Median milliseconds per single-epoch prediction.
    """

    timings = []

    for i in range(min(n_repeats, X.shape[0])):
        single = X[i:i + 1]
        start = time.perf_counter()
        model.predict(single)
        timings.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(timings))


def model_size_kb(model):
    """Measure the serialised footprint of a fitted model.

    Parameters
    ----------
    model : object
        A fitted estimator.

    Returns
    -------
    float
        Pickled size in kilobytes, or NaN when the model cannot be pickled.
    """

    try:
        return len(pickle.dumps(model)) / 1024.0
    except Exception:
        return float('nan')


def evaluate_within_subject(name, X, y, subjects, n_folds=5, model_kwargs=None):
    """Stratified k-fold cross-validation inside each subject.

    This is the calibrated case: the user trains the decoder on their own data
    before using it, which is how most BCI sessions actually run.

    Parameters
    ----------
    name : str
        Registry key of the model to evaluate.
    X : numpy.ndarray
        Preprocessed epochs.
    y : numpy.ndarray
        Class labels.
    subjects : numpy.ndarray
        Subject id per epoch.
    n_folds : int, optional
        Folds per subject.
    model_kwargs : dict, optional
        Extra arguments for the model builder.

    Returns
    -------
    dict
        Per-axis aggregates plus a ``per_subject`` breakdown.
    """

    builder = MODEL_REGISTRY[name]['builder']
    model_kwargs = model_kwargs or {}

    per_subject = []
    fit_times, infer_times, sizes = [], [], []
    pooled_true, pooled_pred = [], []

    for subject_id in sorted(np.unique(subjects)):
        mask = subjects == subject_id
        X_subject, y_subject = X[mask], y[mask]

        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_accuracies, fold_kappas = [], []

        for train_idx, test_idx in splitter.split(X_subject, y_subject):
            model = builder(**model_kwargs) if model_kwargs else builder()

            start = time.perf_counter()
            model.fit(X_subject[train_idx], y_subject[train_idx])
            fit_times.append(time.perf_counter() - start)

            y_pred = model.predict(X_subject[test_idx])
            y_true = y_subject[test_idx]

            fold_accuracies.append(float(np.mean(y_pred == y_true)))
            fold_kappas.append(cohen_kappa_score(y_true, y_pred))
            pooled_true.extend(y_true.tolist())
            pooled_pred.extend(y_pred.tolist())

            infer_times.append(measure_inference_ms(model, X_subject[test_idx]))
            sizes.append(model_size_kb(model))

        per_subject.append({
            'subject': subject_id,
            'acc': float(np.mean(fold_accuracies)),
            'acc_std': float(np.std(fold_accuracies)),
            'kappa': float(np.mean(fold_kappas)),
        })

        print(
            f"    {subject_id}: acc {per_subject[-1]['acc']:.3f} "
            f"+/- {per_subject[-1]['acc_std']:.3f}  kappa {per_subject[-1]['kappa']:.3f}"
        )

    return {
        'acc': float(np.mean([r['acc'] for r in per_subject])),
        'acc_std': float(np.std([r['acc'] for r in per_subject])),
        'kappa': float(np.mean([r['kappa'] for r in per_subject])),
        'macro_f1': float(f1_score(pooled_true, pooled_pred, average='macro')),
        'fit_s': float(np.mean(fit_times)),
        'infer_ms': float(np.median(infer_times)),
        'size_kb': float(np.nanmean(sizes)),
        'per_subject': per_subject,
    }


def evaluate_cross_subject(name, X, y, subjects, model_kwargs=None):
    """Leave-one-subject-out: train on everyone else, test on the held-out user.

    This is the zero-calibration case. It is a strictly harder problem than
    within-subject CV, so the numbers are lower and are not comparable to ``acc``.

    Parameters
    ----------
    name : str
        Registry key of the model to evaluate.
    X : numpy.ndarray
        Preprocessed epochs.
    y : numpy.ndarray
        Class labels.
    subjects : numpy.ndarray
        Subject id per epoch.
    model_kwargs : dict, optional
        Extra arguments for the model builder.

    Returns
    -------
    dict or None
        ``{'loso_acc', 'loso_kappa', 'per_subject'}``, or None when fewer than
        two subjects are available.
    """

    unique_subjects = sorted(np.unique(subjects))
    if len(unique_subjects) < 2:
        return None

    builder = MODEL_REGISTRY[name]['builder']
    model_kwargs = model_kwargs or {}

    accuracies, kappas, per_subject = [], [], []

    for held_out in unique_subjects:
        test_mask = subjects == held_out
        model = builder(**model_kwargs) if model_kwargs else builder()
        model.fit(X[~test_mask], y[~test_mask])

        y_pred = model.predict(X[test_mask])
        y_true = y[test_mask]

        accuracy = float(np.mean(y_pred == y_true))
        accuracies.append(accuracy)
        kappas.append(cohen_kappa_score(y_true, y_pred))
        per_subject.append({'subject': held_out, 'acc': accuracy})
        print(f'    held out {held_out}: acc {accuracy:.3f}')

    return {
        'loso_acc': float(np.mean(accuracies)),
        'loso_kappa': float(np.mean(kappas)),
        'per_subject': per_subject,
    }


def score_quality(values, floor):
    """Score a quality axis from a fixed floor up to the best model observed.

    Parameters
    ----------
    values : list[float]
        Raw values for one axis, one per model.
    floor : float
        The value that scores 0, typically chance level or zero.

    Returns
    -------
    list[float]
        Scores in 0-100, where 100 is the best model on this axis.
    """

    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return [float('nan')] * len(values)

    best = max(finite)
    span = best - floor
    if span <= 0:
        return [100.0 if np.isfinite(v) else float('nan') for v in values]

    return [
        float(np.clip(100.0 * (v - floor) / span, 0.0, 100.0)) if np.isfinite(v) else float('nan')
        for v in values
    ]


def score_cost(values, budget):
    """Score a cost axis against a fixed budget: on budget is 0, free is 100.

    Parameters
    ----------
    values : list[float]
        Raw costs for one axis, one per model.
    budget : float
        The cost that scores 0. Anything above it also scores 0.

    Returns
    -------
    list[float]
        Scores in 0-100, where 100 means the cost is negligible.
    """

    return [
        float(np.clip(100.0 * (1.0 - v / budget), 0.0, 100.0)) if np.isfinite(v) else float('nan')
        for v in values
    ]


def build_scorecard(results, chance_level):
    """Turn raw per-model metrics into a 0-100 scorecard.

    Parameters
    ----------
    results : dict
        Mapping of model name to its measured metrics.
    chance_level : float
        Chance accuracy, used as the floor for the accuracy axes.

    Returns
    -------
    dict
        Mapping of model name to ``{axis: 0-100 score}`` plus ``'composite'``.
    """

    names = list(results)
    scorecard = {name: {} for name in names}

    for axis, floor_kind in QUALITY_AXES.items():
        raw = [results[name].get(axis, float('nan')) for name in names]
        floor = chance_level if floor_kind == 'chance' else 0.0
        for name, score in zip(names, score_quality(raw, floor)):
            scorecard[name][axis] = score

    for axis, budget in COST_AXES.items():
        raw = [results[name].get(axis, float('nan')) for name in names]
        for name, score in zip(names, score_cost(raw, budget)):
            scorecard[name][axis] = score

    for name in names:
        weighted_sum, weight_total = 0.0, 0.0
        for axis, weight in COMPOSITE_WEIGHTS.items():
            score = scorecard[name][axis]
            if np.isfinite(score):
                weighted_sum += weight * score
                weight_total += weight
        scorecard[name]['composite'] = weighted_sum / weight_total if weight_total else float('nan')

    return scorecard


def print_report(results, scorecard, chance_level):
    """Print the raw metric table, the normalised scorecard, and the ranking.

    Parameters
    ----------
    results : dict
        Raw per-model metrics.
    scorecard : dict
        Normalised per-axis scores from :func:`build_scorecard`.
    chance_level : float
        Chance accuracy for the number of classes present.

    Returns
    -------
    None
    """

    order = sorted(results, key=lambda n: -scorecard[n]['composite'])
    width = 96

    print('\n' + '=' * width)
    print('  RAW METRICS')
    print('=' * width)
    header = (
        f"{'Model':<24}{'acc':>14}{'kappa':>8}{'macroF1':>9}"
        f"{'fit_s':>8}{'infer_ms':>10}{'size_kb':>9}{'loso_acc':>10}"
    )
    print(header)
    print('-' * width)

    for name in order:
        r = results[name]
        loso = r.get('loso_acc', float('nan'))
        print(
            f"{MODEL_REGISTRY[name]['label']:<24}"
            f"{r['acc']:.3f} +/- {r['acc_std']:.3f}"
            f"{r['kappa']:>8.3f}{r['macro_f1']:>9.3f}"
            f"{r['fit_s']:>8.2f}{r['infer_ms']:>10.2f}{r['size_kb']:>9.1f}"
            + (f'{loso:>10.3f}' if np.isfinite(loso) else f"{'n/a':>10}")
        )

    print('-' * width)
    print(f'Chance accuracy: {chance_level:.3f}')

    print('\n' + '=' * width)
    print('  SCORECARD  (0-100 per axis)')
    print(f'  quality: 0 = chance/no-skill, 100 = best model here | '
          f'cost: 0 = full budget ({CALIBRATION_BUDGET_S:g}s calib, {LATENCY_BUDGET_MS:g}ms latency)')
    print('=' * width)
    print(
        f"{'Model':<24}{'acc':>8}{'kappa':>8}{'macroF1':>9}"
        f"{'loso':>8}{'calib':>8}{'latency':>9}{'COMPOSITE':>12}"
    )
    print('-' * width)

    for name in order:
        s = scorecard[name]
        cells = []
        for axis in ('acc', 'kappa', 'macro_f1', 'loso_acc', 'fit_s', 'infer_ms'):
            value = s[axis]
            cells.append(f'{value:>8.0f}' if np.isfinite(value) else f"{'n/a':>8}")
        print(f"{MODEL_REGISTRY[name]['label']:<24}" + ''.join(cells) + f"{s['composite']:>12.1f}")

    print('-' * width)
    print('\nRanking by composite score:')
    for rank, name in enumerate(order, start=1):
        print(f"  {rank}. {MODEL_REGISTRY[name]['label']:<24} {scorecard[name]['composite']:.1f}")


def save_results(results, scorecard, path):
    """Write the scorecard to CSV next to the datasets.

    Parameters
    ----------
    results : dict
        Raw per-model metrics.
    scorecard : dict
        Normalised per-axis scores.
    path : pathlib.Path
        Destination CSV file.

    Returns
    -------
    None
    """

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['acc', 'acc_std', 'kappa', 'macro_f1', 'fit_s', 'infer_ms', 'size_kb', 'loso_acc']

    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model'] + columns + [f'score_{c}' for c in COMPOSITE_WEIGHTS] + ['composite'])
        for name in sorted(results, key=lambda n: -scorecard[n]['composite']):
            row = [MODEL_REGISTRY[name]['label']]
            row += [f"{results[name].get(c, float('nan')):.4f}" for c in columns]
            row += [f"{scorecard[name][c]:.1f}" for c in COMPOSITE_WEIGHTS]
            row.append(f"{scorecard[name]['composite']:.1f}")
            writer.writerow(row)

    print(f'\nSaved scorecard to {path}')


def parse_args(argv=None):
    """Parse command line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument list, defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Dataset stems to load. Defaults to every .mat in acquisition/data.')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Models to score. Defaults to all available. Known: {sorted(MODEL_REGISTRY)}')
    parser.add_argument('--tmin', type=float, default=0.5, help='Epoch start in seconds after cue onset.')
    parser.add_argument('--tmax', type=float, default=3.5, help='Epoch end in seconds after cue onset.')
    parser.add_argument('--folds', type=int, default=5, help='Folds for within-subject cross-validation.')
    parser.add_argument('--no-loso', action='store_true', help='Skip the cross-subject evaluation.')
    parser.add_argument('--out', default=str(RESULTS_DIR / 'model_scorecard.csv'),
                        help='Where to write the scorecard CSV.')
    return parser.parse_args(argv)


def main(argv=None):
    """Run the benchmark and print the scorecard.

    Parameters
    ----------
    argv : list[str], optional
        Argument list, defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """

    args = parse_args(argv)
    warnings.filterwarnings('ignore')

    # MNE's CSP logs a rank report per fold, which buries the results table.
    try:
        import mne
        mne.set_log_level('ERROR')
    except ImportError:
        pass

    subject_ids = args.subjects or available_subjects()
    if not subject_ids:
        print('No .mat recordings found in acquisition/data.', file=sys.stderr)
        return 1

    print(f'Loading {len(subject_ids)} subject(s): {", ".join(subject_ids)}')
    X_raw, y, subjects, fs = load_subjects(subject_ids, tmin=args.tmin, tmax=args.tmax)

    X, keep = prepare(X_raw, fs)
    X, y, subjects = X[keep], y[keep], subjects[keep]
    print(f'Preprocessed: {X.shape[0]} epochs kept of {keep.shape[0]} '
          f'({X.shape[1]} channels x {X.shape[2]} samples)')

    requested = args.models or list(MODEL_REGISTRY)
    results = {}

    for name in requested:
        if name not in MODEL_REGISTRY:
            print(f'Skipping unknown model {name!r}', file=sys.stderr)
            continue

        spec = MODEL_REGISTRY[name]
        if spec['needs_torch']:
            try:
                import torch  # noqa: F401
            except ImportError:
                print(f"\n[skip] {spec['label']}: PyTorch is not installed in this interpreter.")
                continue

        print(f"\n[{spec['label']}] within-subject {args.folds}-fold CV")
        metrics = evaluate_within_subject(name, X, y, subjects, n_folds=args.folds)

        if not args.no_loso:
            print(f"[{spec['label']}] cross-subject (leave-one-subject-out)")
            cross = evaluate_cross_subject(name, X, y, subjects)
            if cross:
                metrics.update(cross)

        results[name] = metrics

    if not results:
        print('No models were evaluated.', file=sys.stderr)
        return 1

    chance_level = 1.0 / len(np.unique(y))
    scorecard = build_scorecard(results, chance_level)
    print_report(results, scorecard, chance_level)
    save_results(results, scorecard, Path(args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
