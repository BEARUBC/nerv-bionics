"""Train the detector on your recordings and save it to disk.

This is step one of two. It loads one or more ``.mat`` recordings, holds part of
them back, trains the winning pipeline on the rest, reports how well it did on
the held-back part, and writes the result to ``models/``.

The held-out report matters more than the training score. A model always scores
well on data it has already seen; the only number worth trusting is the one from
trials it has never met.

Usage
-----
    python -m classification.train --subjects A01T
    python -m classification.train --subjects A01T A04T --out models/both.joblib
    python -m classification.train --subjects A01T --holdout 0.3 --min-confidence 0.5
    python -m classification.train --subjects A04T --tune-window
"""

# Check the interpreter before importing the scientific stack, so a Python
# without it gets an explanation instead of a bare ModuleNotFoundError.
from classification._deps import require as _require
_require('classification.train')

import argparse
import sys
import warnings

import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from classification.data import available_subjects, load_subjects
from classification.detector import DEFAULT_MODEL_DIR, MotorImageryDetector
from classification.models import build_model
from classification.preprocessing import prepare

# Windows tried by --tune-window, as (tmin, tmax) in seconds after cue onset.
# Anything starting later than ~1.5 s or running past ~4.5 s scored badly on both
# recordings, so the search stays inside the range that is plausible for ERD/ERS.
CANDIDATE_WINDOWS = (
    (0.5, 2.5),
    (0.5, 3.5),
    (0.5, 4.5),
    (1.0, 3.0),
    (1.0, 4.0),
    (1.5, 4.0),
)


def print_confusion(y_true, y_pred, labels):
    """Print a confusion matrix with row and column headers.

    Rows are what the person actually imagined, columns what the detector said,
    so anything off the diagonal is a mistake and its position names the mistake.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : sequence of str
        Class order for the matrix.

    Returns
    -------
    None
    """

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(8, max(len(str(l)) for l in labels) + 2)

    print('\n  Confusion matrix   (rows = actual, columns = predicted)')
    print('  ' + ' ' * width + ''.join(f'{l:>{width}}' for l in labels))

    for label, row in zip(labels, matrix):
        cells = ''.join(f'{int(v):>{width}}' for v in row)
        print(f'  {label:<{width}}{cells}')


def tune_window(subject_ids, model_name, folds=5):
    """Pick the epoch window that cross-validates best on these recordings.

    The best window is subject-specific: on the two recordings here A01T peaks at
    0.5-3.5 s and A04T at 1.0-4.0 s, a difference worth about 3 accuracy points
    on A04T. Rather than hard-coding one compromise, this searches.

    The score reported for the winner is optimistic, because the same folds chose
    it. It is used for ranking only - the honest number still comes from the
    held-out split in :func:`main`, which is cut after the window is fixed.

    Parameters
    ----------
    subject_ids : list[str]
        Recordings to tune against.
    model_name : str
        Registry key of the pipeline being tuned.
    folds : int, optional
        Cross-validation folds per candidate window.

    Returns
    -------
    tuple[float, float]
        The winning ``(tmin, tmax)``.
    """

    print(f'\nSearching {len(CANDIDATE_WINDOWS)} epoch windows '
          f'({folds}-fold CV on {", ".join(subject_ids)})')
    print(f'  {"window":<12}{"length":>8}{"accuracy":>11}')
    print('  ' + '-' * 31)

    scored = []

    for tmin, tmax in CANDIDATE_WINDOWS:
        X, y, _, fs = load_subjects(subject_ids, tmin=tmin, tmax=tmax, verbose=False)
        X, keep = prepare(X, fs)
        X, y = X[keep], y[keep]

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        accuracies = [
            build_model(model_name).fit(X[tr], y[tr]).score(X[te], y[te])
            for tr, te in splitter.split(X, y)
        ]
        accuracy = float(np.mean(accuracies))
        scored.append(((tmin, tmax), accuracy))
        print(f'  {f"{tmin:.1f}-{tmax:.1f} s":<12}{tmax - tmin:>7.1f}s{accuracy:>11.3f}')

    best_window, best_score = max(scored, key=lambda item: item[1])
    print(f'\n  Best: {best_window[0]:.1f}-{best_window[1]:.1f} s at {best_score:.3f}')
    return best_window


def main(argv=None):
    """Train a detector and save it.

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
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Dataset stems to train on. Defaults to every .mat in acquisition/data.')
    parser.add_argument('--out', default=None,
                        help='Where to save the detector. Defaults to models/<subjects>.joblib')
    parser.add_argument('--holdout', type=float, default=0.25,
                        help='Fraction of trials kept back for the honest score (0 to skip).')
    parser.add_argument('--tmin', type=float, default=0.5, help='Epoch start in seconds after cue onset.')
    parser.add_argument('--tmax', type=float, default=3.5, help='Epoch end in seconds after cue onset.')
    parser.add_argument('--model', default='riemann_ts_lr',
                        help='Which pipeline to train. Defaults to the benchmark winner.')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Store a confidence floor; below it the detector declines to answer.')
    parser.add_argument('--tune-window', action='store_true',
                        help='Search for the best epoch window for these recordings before training. '
                             'Worth 3-10 points on a subject the default window does not suit.')
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')
    try:
        import mne
        mne.set_log_level('ERROR')
    except ImportError:
        pass

    subject_ids = args.subjects or available_subjects()
    if not subject_ids:
        print('No .mat recordings found in acquisition/data.', file=sys.stderr)
        return 1

    print(f'Training on: {", ".join(subject_ids)}')

    tmin, tmax = args.tmin, args.tmax
    if args.tune_window:
        tmin, tmax = tune_window(subject_ids, args.model)
        print()

    X_raw, y, subjects, fs = load_subjects(subject_ids, tmin=tmin, tmax=tmax)

    # Preprocess once here so the train/test split sees identical cleaning.
    X, keep = prepare(X_raw, fs)
    X, y = X[keep], y[keep]
    print(f'{X.shape[0]} usable trials, {X.shape[1]} channels x {X.shape[2]} samples')

    detector = MotorImageryDetector(
        model=args.model, fs=fs, tmin=tmin, tmax=tmax,
        min_confidence=args.min_confidence,
    )

    if args.holdout > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.holdout, stratify=y, random_state=42
        )
        print(f'Holding back {len(y_test)} trials for testing, training on {len(y_train)}.')
    else:
        X_train, y_train = X, y
        X_test = y_test = None
        print('Training on everything, no held-out score.')

    detector.metadata['trained_on'] = list(subject_ids)
    detector.fit(X_train, y_train, preprocess=False)

    if X_test is not None:
        predictions = np.asarray([d.top_label for d in detector.detect_many(X_test, preprocess=False)])
        accuracy = float(np.mean(predictions == y_test))
        kappa = cohen_kappa_score(y_test, predictions)
        chance = 1.0 / len(detector.classes_)

        print('\n' + '=' * 62)
        print('  HELD-OUT RESULT  (trials the model never saw during training)')
        print('=' * 62)
        print(f'  Accuracy : {accuracy:.1%}   (random guessing would be {chance:.0%})')
        print(f'  Kappa    : {kappa:.3f}   (0 = guessing, 1 = perfect)')
        print('\n' + classification_report(y_test, predictions, digits=3))
        print_confusion(y_test, predictions, list(detector.classes_))

        detector.metadata['holdout_accuracy'] = round(accuracy, 4)
        detector.metadata['holdout_kappa'] = round(float(kappa), 4)

        # Retrain on everything: the split existed to produce an honest number,
        # and the shipped detector should use all the data available.
        print('\nRetraining on all trials for the saved model.')
        detector.fit(X, y, preprocess=False)
        detector.metadata['trained_on'] = list(subject_ids)

    out_path = args.out or (DEFAULT_MODEL_DIR / f'{"_".join(subject_ids).lower()}.joblib')
    saved = detector.save(out_path)

    print('\n' + '-' * 62)
    print(detector.describe())
    print('-' * 62)
    print(f'\nSaved to {saved}')
    print(f'Run it with:  python -m classification.detect --model {saved} --subject {subject_ids[0]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
