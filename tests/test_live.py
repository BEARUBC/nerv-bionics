"""Checks on the live view's central promise: the model never sees the answer.

The live page shows you the true label while a detector guesses. That only means
anything if the label genuinely cannot reach the classifier, so the properties
that guarantee it are tested rather than left as a comment.
"""

import inspect

import numpy as np
import pytest

from classification.data import CLASSES, available_subjects
from classification.live import LiveSession

CHANCE = 0.25

pytestmark = pytest.mark.skipif(
    not available_subjects(),
    reason='No .mat recordings in acquisition/data to test against.',
)


@pytest.fixture(scope='module')
def session():
    """Build one live session for the whole module; it trains a detector."""

    return LiveSession(available_subjects()[0])


def test_streams_a_held_out_split(session):
    """Training and streaming sets must both be non-empty and add up."""

    summary = session.describe()

    assert summary['n_train'] > 0
    assert summary['n_trials'] > 0
    # The split is 70/30, so the streamed part should be the smaller one.
    assert summary['n_trials'] < summary['n_train']


def test_label_is_not_inside_any_frame(session):
    """The frames are what the page animates; none may carry the answer."""

    payload = session.trial(0)

    assert payload['truth'] in CLASSES, 'the viewer still needs the label'
    for frame in payload['frames']:
        assert 'truth' not in frame
        assert payload['truth'] not in [v for k, v in frame.items() if k != 'label'
                                        and isinstance(v, str)]


def test_frame_builder_cannot_receive_the_label(session):
    """The function that runs the model takes the signal and nothing else.

    A signature check rather than a behavioural one, because the guarantee is
    structural: if no label can be passed in, none can be used.
    """

    parameters = list(inspect.signature(session._frames_for).parameters)

    assert parameters == ['trial'], (
        f'_frames_for now takes {parameters}; the live view promises the label '
        'cannot reach the classifier, so it must take only the signal.'
    )


def test_probabilities_are_a_real_distribution(session):
    """Every frame should carry a usable probability per class."""

    frames = session.trial(0)['frames']
    assert frames, 'a full-length trial should produce at least one frame'

    for frame in frames:
        assert set(frame['probabilities']) == set(CLASSES)
        assert sum(frame['probabilities'].values()) == pytest.approx(1.0, abs=1e-3)
        assert frame['label'] == max(frame['probabilities'], key=frame['probabilities'].get)


def test_traces_are_drawable(session):
    """The page draws three channels of bounded, finite numbers."""

    frame = session.trial(0)['frames'][0]
    traces = frame['traces']

    assert len(traces) == len(session.channel_rows)
    for channel in traces:
        assert len(channel) > 20
        assert all(np.isfinite(v) for v in channel)
        # Scaled by a high percentile, so a little overshoot is expected but not
        # the raw microvolt range.
        assert max(abs(v) for v in channel) < 12


def test_beats_chance_on_trials_it_has_never_seen(session):
    """The honest number: first full window, on the held-out split."""

    correct = 0
    checked = min(30, session.n_trials)

    for index in range(checked):
        payload = session.trial(index)
        correct += payload['frames'][0]['label'] == payload['truth']

    accuracy = correct / checked
    assert accuracy > CHANCE + 0.15, (
        f'{accuracy:.2f} on unseen trials is barely above chance ({CHANCE})'
    )


def test_out_of_range_trial_is_rejected(session):
    """Asking for a trial that does not exist should say so."""

    with pytest.raises(IndexError):
        session.trial(session.n_trials)
