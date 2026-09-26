"""Checks that the detector works on the recordings in this repository.

These are not unit tests in the strict sense: they load a real ``.mat`` file and
train a real model, which takes a few seconds. That is deliberate. The failure
mode worth catching here is not "does this function return a float" but "does the
whole chain, from raw recording to label, still produce a usable decoder".

Run them with:

    pytest tests/ -v

If no recordings are present the tests skip rather than fail, so the suite is
still safe to run on a machine without the datasets.
"""

import numpy as np
import pytest

from classification.data import CLASSES, available_subjects, load_subjects
from classification.detector import MotorImageryDetector
from classification.models import SELECTED_MODELS, build_model
from classification.preprocessing import prepare

CHANCE = 0.25

# Training on every trial makes the suite slow for no extra signal.
TRAIN_TRIALS = 180


pytestmark = pytest.mark.skipif(
    not available_subjects(),
    reason='No .mat recordings in acquisition/data to test against.',
)


@pytest.fixture(scope='module')
def epochs():
    """Load and clean one subject once for the whole module.

    Returns
    -------
    tuple
        ``(X, y, fs)`` with X preprocessed and artifact epochs removed.
    """

    subject = available_subjects()[0]
    X, y, _, fs = load_subjects([subject])
    X, keep = prepare(X, fs)
    return X[keep], y[keep], fs


@pytest.fixture(scope='module')
def trained(epochs):
    """Train one detector for the tests that only need a fitted model.

    Returns
    -------
    tuple
        ``(detector, X_test, y_test)``.
    """

    X, y, fs = epochs
    detector = MotorImageryDetector(fs=fs)
    detector.fit(X[:TRAIN_TRIALS], y[:TRAIN_TRIALS], preprocess=False)
    return detector, X[TRAIN_TRIALS:], y[TRAIN_TRIALS:]


def test_epochs_have_the_expected_shape(epochs):
    """Loading should give 22-channel, 3-second epochs with known labels."""

    X, y, fs = epochs

    assert X.ndim == 3
    assert X.shape[1] == 22, 'EOG channels should already be dropped'
    assert X.shape[2] == int(3.0 * fs), 'default window is 0.5-3.5 s'
    assert X.shape[0] == len(y)
    assert set(y) <= set(CLASSES)
    assert not np.isnan(X).any(), 'dropped samples should be interpolated'


def test_every_class_is_present(epochs):
    """A four-class problem needs all four classes, or the scores are meaningless."""

    _, y, _ = epochs
    assert set(y) == set(CLASSES)


def test_detector_beats_chance_on_unseen_trials(trained):
    """The whole point: better than guessing on trials it never saw."""

    detector, X_test, y_test = trained
    accuracy = detector.score(X_test, y_test, preprocess=False)

    assert accuracy > CHANCE + 0.15, (
        f'Held-out accuracy {accuracy:.3f} is barely above chance ({CHANCE}). '
        'Something in the pipeline is broken.'
    )


def test_detect_returns_calibrated_probabilities(trained):
    """One trial in, one label plus a probability per class out."""

    detector, X_test, _ = trained
    result = detector.detect(X_test[0], preprocess=False)

    assert result.top_label in CLASSES
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == set(CLASSES)
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_detect_many_matches_detect_one_by_one(trained):
    """The batch path and the single path must not disagree."""

    detector, X_test, _ = trained
    batch = [d.top_label for d in detector.detect_many(X_test[:12], preprocess=False)]
    singly = [detector.detect(trial, preprocess=False).top_label for trial in X_test[:12]]

    assert batch == singly


def test_confidence_floor_makes_it_decline(trained):
    """Above a 100% floor nothing should ever be confident enough to answer."""

    detector, X_test, _ = trained
    original = detector.min_confidence

    try:
        detector.min_confidence = 1.01
        results = detector.detect_many(X_test[:8], preprocess=False)
        assert all(r.label is None for r in results)
        assert all(r.top_label in CLASSES for r in results), 'top_label should survive the floor'
    finally:
        detector.min_confidence = original


def test_save_and_load_round_trip(trained, tmp_path):
    """A reloaded detector must predict exactly what the original predicted."""

    detector, X_test, y_test = trained
    path = detector.save(tmp_path / 'detector.joblib')

    reloaded = MotorImageryDetector.load(path)

    before = [d.top_label for d in detector.detect_many(X_test, preprocess=False)]
    after = [d.top_label for d in reloaded.detect_many(X_test, preprocess=False)]

    assert before == after
    assert reloaded.window_samples == detector.window_samples
    assert reloaded.fs == detector.fs
    assert reloaded.band == detector.band
    assert list(reloaded.classes_) == list(detector.classes_)


def test_wrong_window_length_is_rejected(trained):
    """Feeding a short window must fail loudly, not return a confident guess."""

    detector, X_test, _ = trained

    with pytest.raises(ValueError, match='samples'):
        detector.detect(X_test[0][:, :400], preprocess=False)


def test_wrong_channel_count_is_rejected(trained):
    """Same for the wrong montage."""

    detector, X_test, _ = trained

    with pytest.raises(ValueError, match='channels'):
        detector.detect(X_test[0][:8, :], preprocess=False)


def test_untrained_detector_explains_itself():
    """Using a fresh detector should say what to do, not raise something cryptic."""

    with pytest.raises(RuntimeError, match='not trained'):
        MotorImageryDetector().detect(np.zeros((22, 750)))


@pytest.mark.parametrize('name', SELECTED_MODELS)
def test_each_selected_model_trains_and_beats_chance(epochs, name):
    """All three shortlisted models should still work, not just the winner."""

    X, y, _ = epochs

    model = build_model(name)
    model.fit(X[:TRAIN_TRIALS], y[:TRAIN_TRIALS])
    accuracy = model.score(X[TRAIN_TRIALS:], y[TRAIN_TRIALS:])

    assert accuracy > CHANCE + 0.10, f'{name} scored {accuracy:.3f}, barely above chance'


def test_online_decoder_agrees_with_offline(trained):
    """Streaming a trial sample by sample should give the same answer as the batch path.

    This is the check that catches an offline/online preprocessing mismatch, which
    is the classic way a decoder looks good in cross-validation and fails live.
    """

    from classification.realtime import OnlineDecoder

    detector, X_test, _ = trained
    decoder = OnlineDecoder.from_detector(detector, step_s=1.0, smoothing=1)

    offline = detector.detect(X_test[0]).top_label

    decoder.reset()
    decisions = decoder.push_chunk(X_test[0])

    assert decisions, 'a full-length trial should produce at least one decision'
    assert decisions[-1].label == offline
