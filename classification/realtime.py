"""Online decoding: turn a rolling sample buffer into class predictions.

``benchmark.py`` answers "which model is best" offline. This module is what the
controller actually calls during a session. It holds a ring buffer of incoming
samples, and every ``step`` samples it preprocesses the most recent window and
asks the model for a prediction.

Two details matter for a real device:

- The window handed to the model must be the same length and go through the same
  preprocessing as the training epochs, or the covariances look nothing alike.
  ``OnlineDecoder`` stores the training window length and reuses
  ``preprocessing.prepare_trial`` to guarantee that.
- A single window's prediction is noisy. ``smoothing`` averages the last few
  probability vectors, and ``confidence_threshold`` withholds a command when the
  decoder is not sure, which is much safer than acting on a coin flip.

Example
-------
    from classification.data import load_subjects
    from classification.preprocessing import prepare
    from classification.realtime import OnlineDecoder

    X, y, subjects, fs = load_subjects(['A01T'])
    X, keep = prepare(X, fs)

    decoder = OnlineDecoder(model='riemann_ts_lr', fs=fs)
    decoder.fit(X[keep], y[keep])

    for sample in stream:              # sample: shape (22,)
        result = decoder.push(sample)
        if result is not None:
            print(result.label, result.confidence)
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from classification.models import build_model
from classification.preprocessing import DEFAULT_BAND, prepare_trial


@dataclass
class Decision:
    """One decoder output.

    Attributes
    ----------
    label : str or None
        Predicted class, or None when confidence was below the threshold.
    confidence : float
        Probability of the winning class after smoothing.
    probabilities : dict[str, float]
        Smoothed probability per class.
    latency_ms : float
        Time spent preprocessing and classifying this window.
    """

    label: object
    confidence: float
    probabilities: dict
    latency_ms: float


class OnlineDecoder:
    """Classify motor imagery from a live sample stream.

    Parameters
    ----------
    model : str or object, optional
        A registry key from :data:`classification.models.MODEL_REGISTRY`, or an
        already-built estimator.
    fs : float, optional
        Sampling rate in Hz.
    window_s : float, optional
        Window length in seconds. Ignored once :meth:`fit` has run, because the
        window then has to match the training epoch length.
    step_s : float, optional
        How often to emit a decision, in seconds.
    band : tuple[float, float] or None, optional
        Bandpass range applied to each window. Must match training.
    smoothing : int, optional
        Number of consecutive windows to average probabilities over. 1 disables
        smoothing.
    confidence_threshold : float, optional
        Minimum smoothed probability required to emit a label. Below it,
        ``Decision.label`` is None.
    """

    def __init__(self, model='riemann_ts_lr', fs=250.0, window_s=3.0, step_s=0.25,
                 band=DEFAULT_BAND, smoothing=3, confidence_threshold=0.0):
        self.model = build_model(model) if isinstance(model, str) else model
        self.model_name = model if isinstance(model, str) else type(model).__name__
        self.fs = float(fs)
        self.band = band
        self.smoothing = max(1, int(smoothing))
        self.confidence_threshold = float(confidence_threshold)

        self.window_samples = int(round(window_s * self.fs))
        self.step_samples = max(1, int(round(step_s * self.fs)))

        self._buffer = None
        self._samples_since_decision = 0
        self._probability_history = deque(maxlen=self.smoothing)
        self._fitted = False

    @classmethod
    def from_detector(cls, detector, step_s=0.25, smoothing=3, confidence_threshold=None):
        """Wrap an already-trained detector for streaming, without refitting.

        Parameters
        ----------
        detector : classification.detector.MotorImageryDetector
            A trained detector whose pipeline and settings should be reused.
        step_s : float, optional
            How often to emit a decision, in seconds.
        smoothing : int, optional
            Number of consecutive windows to average over.
        confidence_threshold : float, optional
            Minimum smoothed probability required to emit a label. Defaults to
            the detector's own ``min_confidence``.

        Returns
        -------
        OnlineDecoder
            A decoder ready for :meth:`push`, sharing the detector's model.
        """

        decoder = cls(
            model=detector.pipeline,
            fs=detector.fs,
            step_s=step_s,
            band=detector.band,
            smoothing=smoothing,
            confidence_threshold=(
                detector.min_confidence if confidence_threshold is None else confidence_threshold
            ),
        )

        decoder.attach(detector.pipeline, detector.n_channels, detector.window_samples,
                       detector.classes_)
        return decoder

    def attach(self, model, n_channels, window_samples, classes):
        """Use an already-trained model instead of calling :meth:`fit`.

        Parameters
        ----------
        model : object
            A fitted estimator with ``predict`` and ideally ``predict_proba``.
        n_channels : int
            Channels the model expects.
        window_samples : int
            Window length in samples the model was trained on.
        classes : array-like
            The model's class labels, in the order its probabilities come back.

        Returns
        -------
        OnlineDecoder
            self, ready to decode.
        """

        self.model = model
        self.n_channels = int(n_channels)
        self.window_samples = int(window_samples)
        self.classes_ = np.asarray(classes)
        self._buffer = deque(maxlen=self.window_samples)
        self._probability_history.clear()
        self._samples_since_decision = 0
        self._fitted = True

        return self

    def fit(self, X, y):
        """Calibrate the decoder on labelled epochs.

        Parameters
        ----------
        X : numpy.ndarray
            Preprocessed epochs of shape (n_trials, n_channels, n_samples).
        y : array-like
            Class labels.

        Returns
        -------
        OnlineDecoder
            self, calibrated.
        """

        X = np.asarray(X, dtype=float)

        # The online window has to match what the model was trained on.
        self.window_samples = X.shape[2]
        self.n_channels = X.shape[1]

        self.model.fit(X, y)
        self.classes_ = np.asarray(self.model.classes_)
        self._buffer = deque(maxlen=self.window_samples)
        self._fitted = True

        return self

    def reset(self):
        """Clear the sample buffer and smoothing history between trials.

        Returns
        -------
        None
        """

        if self._buffer is not None:
            self._buffer.clear()
        self._probability_history.clear()
        self._samples_since_decision = 0

    def push(self, sample):
        """Add one multi-channel sample and decide whether to emit a prediction.

        Parameters
        ----------
        sample : array-like
            One sample across channels, shape (n_channels,).

        Returns
        -------
        Decision or None
            A decision when the buffer is full and a step boundary was reached,
            otherwise None.
        """

        if not self._fitted:
            raise RuntimeError('Call fit() before push().')

        self._buffer.append(np.asarray(sample, dtype=float)[:self.n_channels])
        self._samples_since_decision += 1

        if len(self._buffer) < self.window_samples:
            return None
        if self._samples_since_decision < self.step_samples:
            return None

        self._samples_since_decision = 0
        return self.predict_window(np.stack(self._buffer, axis=1))

    def push_chunk(self, chunk):
        """Feed a block of samples, as BrainFlow's ``get_board_data`` returns.

        Parameters
        ----------
        chunk : numpy.ndarray
            Samples of shape (n_channels, n_samples).

        Returns
        -------
        list[Decision]
            Every decision triggered by this chunk, oldest first.
        """

        chunk = np.asarray(chunk, dtype=float)
        decisions = []

        for i in range(chunk.shape[1]):
            decision = self.push(chunk[:, i])
            if decision is not None:
                decisions.append(decision)

        return decisions

    def predict_window(self, window):
        """Classify one complete window.

        Parameters
        ----------
        window : numpy.ndarray
            Raw window of shape (n_channels, window_samples).

        Returns
        -------
        Decision
            The smoothed decision for this window.
        """

        start = time.perf_counter()

        batch = prepare_trial(window, self.fs, band=self.band)

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(batch)[0]
        else:
            # MDM and friends expose distances rather than probabilities; fall
            # back to a one-hot vector so the interface stays the same.
            predicted = self.model.predict(batch)[0]
            probabilities = (self.classes_ == predicted).astype(float)

        self._probability_history.append(probabilities)
        smoothed = np.mean(self._probability_history, axis=0)

        latency_ms = (time.perf_counter() - start) * 1000.0
        best = int(np.argmax(smoothed))
        confidence = float(smoothed[best])
        label = self.classes_[best] if confidence >= self.confidence_threshold else None

        return Decision(
            label=label,
            confidence=confidence,
            probabilities={str(c): float(p) for c, p in zip(self.classes_, smoothed)},
            latency_ms=latency_ms,
        )


def simulate(decoder, epochs, labels, verbose=True):
    """Replay labelled epochs through the decoder sample by sample.

    Useful as a sanity check that online and offline agree before touching real
    hardware: the accuracy here should land near the offline benchmark number.

    Parameters
    ----------
    decoder : OnlineDecoder
        A calibrated decoder.
    epochs : numpy.ndarray
        Raw epochs of shape (n_trials, n_channels, n_samples).
    labels : array-like
        True label per epoch.
    verbose : bool, optional
        Print the accuracy and latency summary.

    Returns
    -------
    dict
        ``{'accuracy', 'median_latency_ms', 'n_decisions'}``.
    """

    correct = 0
    scored = 0
    latencies = []

    for epoch, true_label in zip(epochs, labels):
        decoder.reset()
        decisions = decoder.push_chunk(epoch)
        if not decisions:
            continue

        final = decisions[-1]
        latencies.extend(d.latency_ms for d in decisions)

        if final.label is not None:
            scored += 1
            correct += int(final.label == true_label)

    accuracy = correct / scored if scored else float('nan')
    median_latency = float(np.median(latencies)) if latencies else float('nan')

    if verbose:
        print(f'Simulated {scored} decisions: accuracy {accuracy:.3f}, '
              f'median latency {median_latency:.2f} ms')

    return {'accuracy': accuracy, 'median_latency_ms': median_latency, 'n_decisions': scored}
