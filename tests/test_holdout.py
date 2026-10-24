"""The 50/25/25 hold-out split and the scores it reports.

    python -m pytest tests/test_holdout.py -v
"""

import numpy as np
import pytest

from classification.data import CLASSES, available_subjects
from classification.holdout import split_indices, summarise

pytestmark = pytest.mark.skipif(
    not available_subjects(), reason='no .mat recordings in acquisition/data'
)


def test_split_is_half_quarter_quarter_and_disjoint():
    """Every trial lands in exactly one part, in roughly 50/25/25 proportions."""

    y = np.repeat(CLASSES, 68)
    train, validation, test = split_indices(y)

    assert len(train) + len(validation) + len(test) == len(y)
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    assert len(train) == len(y) // 2
    assert abs(len(validation) - len(test)) <= 1


def test_split_keeps_classes_balanced_and_is_repeatable():
    """Stratified: each part holds the same share of every class, and a seed fixes it."""

    y = np.repeat(CLASSES, 68)
    train, validation, test = split_indices(y, seed=7)

    for part in (train, validation, test):
        counts = [np.sum(y[part] == c) for c in CLASSES]
        assert max(counts) - min(counts) <= 1

    again = split_indices(y, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip((train, validation, test), again))


def test_summarise_matches_hand_counts():
    """Precision and recall agree with counting the confusion matrix by hand."""

    y_true = ['left', 'left', 'right', 'right', 'foot', 'tongue']
    y_pred = ['left', 'right', 'right', 'right', 'foot', 'foot']
    scores = summarise(y_true, y_pred)

    assert scores['accuracy'] == pytest.approx(4 / 6)
    assert scores['per_class']['right']['precision'] == pytest.approx(2 / 3)
    assert scores['per_class']['left']['recall'] == pytest.approx(1 / 2)
    assert scores['per_class']['tongue']['precision'] == 0.0
    assert sum(map(sum, scores['matrix'])) == len(y_true)
