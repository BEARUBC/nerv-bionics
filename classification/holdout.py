"""Train on half a recording, tune on a quarter, test once on the last quarter.

Cross-validation (``app.run_test``) answers "how good is this configuration on
average". This answers the stricter question "if we commit to a model now, how
good is it on trials nobody has looked at":

    train       50 %   the model learns from these
    validation  25 %   used to compare settings and pick a model
    test        25 %   scored once, after the choice is made

Every split is stratified, so each part keeps the four classes balanced. Artifact
rejection is applied to the training part only. Validation and test trials are
all kept, because a live system has to answer on noisy trials too.

One split of ~68 test trials is noisy: a single trial moves accuracy by 1.5
points. :func:`repeated_holdout` re-draws the split many times and reports the
spread, which is the number to trust when comparing two models.

Usage
-----
    python -m classification.holdout
    python -m classification.holdout --subjects A04T --models riemann_ensemble riemann_ts_lr
    python -m classification.holdout --repeats 50 --tmax 4.5
"""

# Check the interpreter before importing the scientific stack, so a Python
# without it gets an explanation instead of a bare ModuleNotFoundError.
from classification._deps import require as _require
_require('classification.holdout')

import argparse
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from classification.data import CLASSES, available_subjects
from classification.models import MODEL_REGISTRY, SELECTED_MODELS, build_model
from classification.preprocessing import prepare

TRAIN_FRACTION = 0.5
SEED = 42
DEFAULT_REPEATS = 20


def split_indices(y, seed=SEED):
    """Stratified 50/25/25 split of trial indices.

    Parameters
    ----------
    y : numpy.ndarray
        Class label per trial.
    seed : int, optional
        Shuffle seed. The same seed and labels always give the same split.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(train, validation, test)`` indices into ``y``.
    """

    idx = np.arange(len(y))
    train, rest = train_test_split(idx, train_size=TRAIN_FRACTION, stratify=y, random_state=seed)
    validation, test = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
    return train, validation, test


def summarise(y_true, y_pred):
    """Accuracy, precision, recall, F1 and kappa, overall and per class.

    Precision answers "when it says *left*, how often is it right"; recall
    answers "of the real *left* trials, how many did it catch". Macro averages
    weight the four classes equally.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.

    Returns
    -------
    dict
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``kappa`` (macro),
        ``per_class`` mapping class to its four numbers, ``matrix`` (rows =
        actual, columns = predicted, in :data:`CLASSES` order) and ``n``.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(CLASSES)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    return {
        'accuracy': float(np.mean(y_true == y_pred)),
        'precision': float(precision.mean()),
        'recall': float(recall.mean()),
        'f1': float(f1.mean()),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'per_class': {
            name: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
            for name, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        'matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'n': int(len(y_true)),
    }


def _fit_and_score(X, y, fs, model_name, seed):
    """Fit on the training part of one split and score validation and test."""

    train, validation, test = split_indices(y, seed)

    X_train, keep = prepare(X[train], fs)
    X_val, _ = prepare(X[validation], fs, reject_uv=None)
    X_test, _ = prepare(X[test], fs, reject_uv=None)

    model = build_model(model_name)
    model.fit(X_train[keep], y[train][keep])

    return {
        'validation': summarise(y[validation], model.predict(X_val)),
        'test': summarise(y[test], model.predict(X_test)),
        'n_train': int(keep.sum()),
        'n_dropped': int((~keep).sum()),
    }


def holdout(cache, subject, model_name, tmin, tmax, seed=SEED):
    """One 50/25/25 run: train, then score validation and test.

    Parameters
    ----------
    cache : classification.app.RecordingCache
        Where the recordings come from.
    subject : str
        Dataset stem.
    model_name : str
        Registry key of the pipeline.
    tmin, tmax : float
        Epoch window in seconds relative to cue onset.
    seed : int, optional
        Split seed.

    Returns
    -------
    dict
        ``validation`` and ``test`` summaries from :func:`summarise`, plus
        ``n_train``, ``n_dropped``, ``seconds`` and the settings used.
    """

    X, y, fs = cache.epochs(subject, tmin, tmax)

    started = time.perf_counter()
    result = _fit_and_score(X, y, fs, model_name, seed)
    result['seconds'] = time.perf_counter() - started

    result.update({'subject': subject, 'model': model_name, 'tmin': tmin, 'tmax': tmax, 'seed': seed})
    return result


def repeated_holdout(cache, subject, model_name, tmin, tmax, repeats=DEFAULT_REPEATS):
    """Re-draw the 50/25/25 split ``repeats`` times and summarise the test scores.

    Parameters
    ----------
    cache : classification.app.RecordingCache
        Where the recordings come from.
    subject : str
        Dataset stem.
    model_name : str
        Registry key of the pipeline.
    tmin, tmax : float
        Epoch window in seconds relative to cue onset.
    repeats : int, optional
        Number of splits, seeded 0 to ``repeats - 1``.

    Returns
    -------
    dict
        Mean and standard deviation of test accuracy, precision, recall and F1,
        plus the per-split accuracies in ``accuracies``.
    """

    X, y, fs = cache.epochs(subject, tmin, tmax)
    runs = [_fit_and_score(X, y, fs, model_name, seed)['test'] for seed in range(repeats)]

    out = {'repeats': repeats, 'accuracies': [r['accuracy'] for r in runs]}
    for key in ('accuracy', 'precision', 'recall', 'f1'):
        values = np.array([r[key] for r in runs])
        out[key] = {'mean': float(values.mean()), 'std': float(values.std())}
    return out


def main(argv=None):
    """Compare models with repeated 50/25/25 hold-out and print a table.

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
    parser.add_argument('--subjects', nargs='+', default=None, help='Recordings (default: all).')
    parser.add_argument('--models', nargs='+', default=list(SELECTED_MODELS),
                        choices=sorted(MODEL_REGISTRY), help='Models to compare.')
    parser.add_argument('--tmin', type=float, default=0.5, help='Window start after cue (s).')
    parser.add_argument('--tmax', type=float, default=4.5, help='Window end after cue (s).')
    parser.add_argument('--repeats', type=int, default=DEFAULT_REPEATS, help='Splits per model.')
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')
    try:
        import mne
        mne.set_log_level('ERROR')
    except ImportError:
        pass

    from classification.app import RecordingCache

    subjects = args.subjects or available_subjects()
    if not subjects:
        print('No .mat recordings found in acquisition/data.', file=sys.stderr)
        return 1

    cache = RecordingCache()
    print(f'50/25/25 hold-out, {args.repeats} splits, window {args.tmin:g}-{args.tmax:g} s\n')
    print(f'  {"model":<24}{"subject":<9}{"accuracy":>16}{"precision":>16}{"F1":>16}')
    print('  ' + '-' * 81)

    for model_name in args.models:
        for subject in subjects:
            r = repeated_holdout(cache, subject, model_name, args.tmin, args.tmax, args.repeats)
            cells = ''.join(f'{r[k]["mean"]:>10.3f} ±{r[k]["std"]:.3f}' for k in ('accuracy', 'precision', 'f1'))
            print(f'  {model_name:<24}{subject:<9}{cells}', flush=True)

    print('\n  Test-set scores, mean ± standard deviation over the splits. Chance is 0.250.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
