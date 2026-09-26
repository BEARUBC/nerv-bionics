"""The shipping detector: train once, save, load, and classify new signals.

`benchmark.py` answers "which model is best". This is what you actually use once
that question is settled. It wraps the winning pipeline (Riemannian tangent space
+ logistic regression) together with the preprocessing settings it was trained
under, so a saved detector carries everything needed to reproduce its own input.

That pairing is the point. A model saved on its own is close to useless here: feed
it a 2-second window when it was trained on 3-second windows, or skip the
bandpass, and it will still return confident, meaningless answers. Saving the
settings alongside the weights makes that mistake impossible.

Typical use
-----------
    from classification.detector import MotorImageryDetector

    detector = MotorImageryDetector()
    detector.fit_from_subjects(['A01T'])
    detector.save('models/a01t.joblib')

    # later, in another process
    detector = MotorImageryDetector.load('models/a01t.joblib')
    result = detector.detect(raw_trial)        # raw_trial: (22, 750)
    print(result.label, result.confidence)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from classification.data import CLASSES, load_subjects
from classification.models import build_model
from classification.preprocessing import DEFAULT_BAND, DEFAULT_REJECT_UV, prepare, prepare_trial

# The winner of the benchmark. Change this and you change what `fit` trains.
DEFAULT_MODEL = 'riemann_ts_lr'

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'


@dataclass
class Detection:
    """One classification result.

    Attributes
    ----------
    label : str or None
        The predicted class, or None when confidence fell below the detector's
        ``min_confidence`` and the detector chose to stay silent.
    confidence : float
        Probability of the top class, between 0 and 1.
    probabilities : dict[str, float]
        Probability for every class.
    top_label : str
        The best class regardless of the confidence threshold. Useful for logging
        what the detector was leaning towards when it declined to answer.
    """

    label: object
    confidence: float
    probabilities: dict = field(default_factory=dict)
    top_label: str = ''

    def __str__(self):
        if self.label is None:
            return f'(no call, leaning {self.top_label} at {self.confidence:.0%})'
        return f'{self.label} at {self.confidence:.0%}'


class MotorImageryDetector:
    """Detect imagined movement from EEG, using the benchmark-winning pipeline.

    Parameters
    ----------
    model : str, optional
        Registry key from :data:`classification.models.MODEL_REGISTRY`. Defaults
        to the benchmark winner.
    fs : float, optional
        Sampling rate in Hz of the signals this detector will see.
    tmin, tmax : float, optional
        Epoch window in seconds relative to cue onset, used when training from
        recordings and recorded so inference can match it.
    band : tuple[float, float] or None, optional
        Bandpass range applied before classification.
    min_confidence : float, optional
        Below this, :meth:`detect` returns ``label=None`` instead of guessing.
        0 means always answer.
    """

    def __init__(self, model=DEFAULT_MODEL, fs=250.0, tmin=0.5, tmax=3.5,
                 band=DEFAULT_BAND, min_confidence=0.0):
        self.model_name = model
        self.fs = float(fs)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.band = band
        self.min_confidence = float(min_confidence)

        self.pipeline = build_model(model)
        self.classes_ = None
        self.n_channels = None
        self.window_samples = None
        self.metadata = {}

    # ------------------------------------------------------------------ train

    def fit(self, X, y, preprocess=True):
        """Train on labelled epochs.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs of shape (n_trials, n_channels, n_samples). Raw by default;
            pass ``preprocess=False`` if they are already cleaned.
        y : array-like
            Class label per epoch.
        preprocess : bool, optional
            Run the standard chain and drop artifact epochs before fitting.

        Returns
        -------
        MotorImageryDetector
            self, trained.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if preprocess:
            X, keep = prepare(X, self.fs, band=self.band, reject_uv=DEFAULT_REJECT_UV)
            X, y = X[keep], y[keep]

        self.pipeline.fit(X, y)

        self.classes_ = np.asarray(self.pipeline.classes_)
        self.n_channels = X.shape[1]
        self.window_samples = X.shape[2]
        self.metadata.update({
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'n_training_trials': int(X.shape[0]),
            'classes': [str(c) for c in self.classes_],
        })

        return self

    def fit_from_subjects(self, subject_ids, data_dir=None):
        """Load recordings by name and train on all of them.

        Parameters
        ----------
        subject_ids : iterable of str
            Dataset stems such as ``['A01T']``.
        data_dir : str or pathlib.Path, optional
            Directory holding the ``.mat`` files.

        Returns
        -------
        MotorImageryDetector
            self, trained.
        """

        X, y, _, fs = load_subjects(
            subject_ids, data_dir=data_dir, tmin=self.tmin, tmax=self.tmax
        )

        if fs != self.fs:
            print(f'Sampling rate in the recordings is {fs:g} Hz; using that instead of {self.fs:g}.')
            self.fs = fs

        self.metadata['trained_on'] = list(subject_ids)
        return self.fit(X, y)

    # -------------------------------------------------------------- inference

    def detect(self, trial, preprocess=True):
        """Classify one trial or window.

        Parameters
        ----------
        trial : numpy.ndarray
            One epoch of shape (n_channels, n_samples). Must be the same length
            the detector was trained on.
        preprocess : bool, optional
            Apply the stored preprocessing chain first. Leave True unless the
            caller has already run it.

        Returns
        -------
        Detection
            Label, confidence and the full probability vector.
        """

        self._check_fitted()

        trial = np.asarray(trial, dtype=float)
        if trial.ndim != 2:
            raise ValueError(f'Expected one trial of shape (channels, samples), got {trial.shape}.')

        self._check_shape(trial.shape[0], trial.shape[1])

        if preprocess:
            batch = prepare_trial(trial, self.fs, band=self.band)
        else:
            batch = trial[np.newaxis, :, :]

        probabilities = self.pipeline.predict_proba(batch)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        top_label = str(self.classes_[best])

        return Detection(
            label=top_label if confidence >= self.min_confidence else None,
            confidence=confidence,
            probabilities={str(c): float(p) for c, p in zip(self.classes_, probabilities)},
            top_label=top_label,
        )

    def detect_many(self, trials, preprocess=True):
        """Classify a batch of trials in one pass.

        Much faster than looping over :meth:`detect` because the covariance and
        tangent-space steps vectorise over the batch.

        Parameters
        ----------
        trials : numpy.ndarray
            Epochs of shape (n_trials, n_channels, n_samples).
        preprocess : bool, optional
            Apply the stored preprocessing chain first.

        Returns
        -------
        list[Detection]
            One result per trial, in order.
        """

        self._check_fitted()

        trials = np.asarray(trials, dtype=float)
        if trials.ndim != 3:
            raise ValueError(f'Expected (n_trials, channels, samples), got {trials.shape}.')

        self._check_shape(trials.shape[1], trials.shape[2])

        if preprocess:
            # No artifact rejection here: at inference every trial gets an answer.
            batch, _ = prepare(trials, self.fs, band=self.band, reject_uv=None)
        else:
            batch = trials

        probabilities = self.pipeline.predict_proba(batch)
        results = []

        for row in probabilities:
            best = int(np.argmax(row))
            confidence = float(row[best])
            top_label = str(self.classes_[best])
            results.append(Detection(
                label=top_label if confidence >= self.min_confidence else None,
                confidence=confidence,
                probabilities={str(c): float(p) for c, p in zip(self.classes_, row)},
                top_label=top_label,
            ))

        return results

    def score(self, X, y, preprocess=True):
        """Report accuracy on labelled trials.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs of shape (n_trials, n_channels, n_samples).
        y : array-like
            True labels.
        preprocess : bool, optional
            Apply the stored preprocessing chain first.

        Returns
        -------
        float
            Fraction correct.
        """

        predictions = [d.top_label for d in self.detect_many(X, preprocess=preprocess)]
        return float(np.mean(np.asarray(predictions) == np.asarray(y)))

    # ------------------------------------------------------------- persistence

    def save(self, path):
        """Write the trained detector and its settings to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination ``.joblib`` file. Parent directories are created.

        Returns
        -------
        pathlib.Path
            The path written.
        """

        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'format_version': 1,
            'model_name': self.model_name,
            'pipeline': self.pipeline,
            'classes': self.classes_,
            'fs': self.fs,
            'tmin': self.tmin,
            'tmax': self.tmax,
            'band': self.band,
            'min_confidence': self.min_confidence,
            'n_channels': self.n_channels,
            'window_samples': self.window_samples,
            'metadata': self.metadata,
        }, path)

        return path

    @classmethod
    def load(cls, path):
        """Load a detector saved by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            The ``.joblib`` file to read.

        Returns
        -------
        MotorImageryDetector
            A ready-to-use detector.
        """

        blob = joblib.load(Path(path))

        detector = cls(
            model=blob['model_name'],
            fs=blob['fs'],
            tmin=blob['tmin'],
            tmax=blob['tmax'],
            band=blob['band'],
            min_confidence=blob['min_confidence'],
        )

        detector.pipeline = blob['pipeline']
        detector.classes_ = blob['classes']
        detector.n_channels = blob['n_channels']
        detector.window_samples = blob['window_samples']
        detector.metadata = blob.get('metadata', {})

        return detector

    # ------------------------------------------------------------------ guards

    def _check_fitted(self):
        """Raise a clear error if the detector has not been trained.

        Returns
        -------
        None
        """

        if self.classes_ is None:
            raise RuntimeError(
                'This detector is not trained yet. Call fit(), fit_from_subjects(), '
                'or load a saved one with MotorImageryDetector.load(path).'
            )

    def _check_shape(self, n_channels, n_samples):
        """Reject input that does not match the training geometry.

        A window of the wrong length still produces a confident answer, so this
        is checked rather than left to chance.

        Parameters
        ----------
        n_channels : int
            Channels in the incoming data.
        n_samples : int
            Samples per epoch in the incoming data.

        Returns
        -------
        None
        """

        if n_channels != self.n_channels:
            raise ValueError(
                f'This detector was trained on {self.n_channels} channels but got {n_channels}. '
                'Drop the EOG columns, or retrain on the montage you are streaming.'
            )

        if n_samples != self.window_samples:
            expected_s = self.window_samples / self.fs
            got_s = n_samples / self.fs
            raise ValueError(
                f'This detector was trained on {self.window_samples} samples '
                f'({expected_s:.2f} s) but got {n_samples} ({got_s:.2f} s). '
                'Window length has to match training exactly.'
            )

    def describe(self):
        """Return a one-block human-readable summary of the detector.

        Returns
        -------
        str
            Multi-line description, suitable for printing.
        """

        band = f'{self.band[0]:g}-{self.band[1]:g} Hz' if self.band else 'none'
        lines = [
            f'Model            : {self.model_name}',
            f'Classes          : {", ".join(str(c) for c in self.classes_) if self.classes_ is not None else "untrained"}',
            f'Input            : {self.n_channels} channels x {self.window_samples} samples '
            f'({(self.window_samples or 0) / self.fs:.2f} s @ {self.fs:g} Hz)',
            f'Cue window       : {self.tmin:g} to {self.tmax:g} s after cue onset',
            f'Bandpass         : {band}',
            f'Min confidence   : {self.min_confidence:.2f}',
        ]

        for key in ('trained_on', 'n_training_trials', 'trained_at'):
            if key in self.metadata:
                lines.append(f'{key:<17}: {self.metadata[key]}')

        return '\n'.join(lines)
