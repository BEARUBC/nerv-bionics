"""Backend for the live view: stream held-out trials past a trained detector.

The point of this module is a guarantee. When you watch a trial go past and see
what it really was, the model must not have had that information -- otherwise the
demo proves nothing.

Two things enforce that here:

1. **The model never trains on what you watch.** :class:`LiveSession` splits the
   recording, fits the detector on the training part, and streams only the
   held-out part. Every trial you see is one the model has never met.
2. **The label is not on the model's path.** The detector is handed a window of
   signal and nothing else; see :meth:`LiveSession._frames_for`. The true label is
   attached to the response afterwards, purely so the page can show it to you.

So the label reaches your screen and never the classifier.
"""

from classification._deps import require as _require
_require('classification.live')

import numpy as np

from classification.data import CLASSES, load_mat_epochs, DEFAULT_DATA_DIR
from classification.detector import MotorImageryDetector
from classification.preprocessing import prepare
from classification.realtime import OnlineDecoder

# Channels drawn on screen. Motor imagery shows up over the motor cortex, so
# these are the three worth watching.
DISPLAY_CHANNELS = ('C3', 'Cz', 'C4')

# Points per channel sent to the browser for one window. The window is 750
# samples; drawing every one is wasted bandwidth for a line a few hundred pixels
# wide.
DISPLAY_POINTS = 180

# Points in a picker thumbnail. Coarser than the live scope on purpose.
PREVIEW_POINTS = 60

# Fraction of the recording kept back for streaming.
HELD_OUT_FRACTION = 0.3

# How far past the cue to read, so the rolling window has room to slide.
EXTRA_SECONDS = 2.0


class LiveSession:
    """A trained detector plus the trials it has never seen.

    Parameters
    ----------
    subject : str
        Dataset stem, such as ``'A01T'``.
    model : str, optional
        Registry key of the pipeline to train.
    step_s : float, optional
        Seconds between decisions.
    smoothing : int, optional
        Number of consecutive windows averaged before deciding.
    seed : int, optional
        Seed for the train/stream split.
    """

    def __init__(self, subject, model='riemann_ts_lr', step_s=0.5, smoothing=3, seed=42):
        self.subject = subject
        self.model_name = model
        self.step_s = float(step_s)
        self.smoothing = int(smoothing)

        detector = MotorImageryDetector(model=model)

        # Training epochs: the standard window, cut and cleaned as usual.
        X_train, y_train, fs = load_mat_epochs(
            DEFAULT_DATA_DIR / f'{subject}.mat',
            tmin=detector.tmin, tmax=detector.tmax, verbose=False,
        )

        rng = np.random.RandomState(seed)
        order = rng.permutation(len(y_train))
        n_held = int(round(HELD_OUT_FRACTION * len(order)))
        held_idx = np.sort(order[:n_held])
        train_idx = np.sort(order[n_held:])

        X_clean, keep = prepare(X_train[train_idx], fs)
        detector.fs = fs
        detector.metadata['trained_on'] = [f'{subject} (training split only)']
        detector.fit(X_clean[keep], y_train[train_idx][keep], preprocess=False)

        self.detector = detector
        self.fs = fs

        # Streaming epochs: same cues, but read further past the cue so the
        # rolling buffer can slide. Only the held-out cues are kept.
        X_stream, y_stream, _ = load_mat_epochs(
            DEFAULT_DATA_DIR / f'{subject}.mat',
            tmin=detector.tmin, tmax=detector.tmax + EXTRA_SECONDS, verbose=False,
        )

        self.X = X_stream[held_idx]
        self.y = y_stream[held_idx]

        self.n_trials = int(len(self.y))
        self.n_train = int(len(train_idx))
        self.channel_rows = self._channel_rows()

        self.decoder = OnlineDecoder.from_detector(
            detector, step_s=self.step_s, smoothing=self.smoothing
        )

    @staticmethod
    def _channel_rows():
        """Map the display channel names to their row in the epoch array.

        Returns
        -------
        list[tuple[str, int]]
            Name and row index for each channel drawn.
        """

        from classification.data import CHANNEL_NAMES

        return [(name, CHANNEL_NAMES.index(name))
                for name in DISPLAY_CHANNELS if name in CHANNEL_NAMES]

    def describe(self):
        """Summarise the session for the page header.

        Returns
        -------
        dict
            Counts, class list and the settings in force.
        """

        return {
            'subject': self.subject,
            'model': self.model_name,
            'classes': list(CLASSES),
            'n_trials': self.n_trials,
            'n_train': self.n_train,
            'step_s': self.step_s,
            'smoothing': self.smoothing,
            'channels': [name for name, _ in self.channel_rows],
            'window_s': self.detector.window_samples / self.fs,
        }

    def catalog(self):
        """List every streamable trial with a thumbnail of its signal.

        The page shows this as a picker, so each entry carries enough to draw a
        small preview and to label it for the viewer. The previews come from the
        signal only; the model is not involved in building this list.

        Returns
        -------
        list[dict]
            One entry per held-out trial: ``index``, ``truth`` and ``preview``.
        """

        # A wider, coarser trace than the live scope: this is a thumbnail, and
        # one channel is enough to tell trials apart at a glance.
        row = self.channel_rows[len(self.channel_rows) // 2][1]
        entries = []

        for index in range(self.n_trials):
            signal = self.X[index][row]
            step = max(1, len(signal) // PREVIEW_POINTS)
            reduced = signal[::step][:PREVIEW_POINTS]
            scale = float(np.percentile(np.abs(reduced), 98)) or 1.0

            entries.append({
                'index': index,
                'truth': str(self.y[index]),
                'preview': [round(float(v / scale), 2) for v in reduced],
            })

        return entries

    def trial(self, index):
        """Run one held-out trial through the decoder and package it for display.

        Parameters
        ----------
        index : int
            Position in the held-out set.

        Returns
        -------
        dict
            Frames for the animation, plus the true label for the viewer.
        """

        if not 0 <= index < self.n_trials:
            raise IndexError(f'trial {index} out of range (0..{self.n_trials - 1})')

        trial = self.X[index]
        frames = self._frames_for(trial)

        return {
            'index': index,
            'n_trials': self.n_trials,
            'frames': frames,
            # Attached after the frames were computed. Nothing above this line
            # had access to it.
            'truth': str(self.y[index]),
        }

    def _frames_for(self, trial):
        """Slide the decoder across one trial and collect what to draw.

        The only argument is the signal. The label is deliberately not a
        parameter here, so it cannot reach the classifier even by accident.

        Parameters
        ----------
        trial : numpy.ndarray
            One raw epoch of shape (n_channels, n_samples).

        Returns
        -------
        list[dict]
            One entry per decision, each with the window drawn and the
            probabilities the model produced from it.
        """

        window = self.detector.window_samples
        self.decoder.reset()

        frames = []

        for sample_index in range(trial.shape[1]):
            decision = self.decoder.push(trial[:, sample_index])
            if decision is None:
                continue

            current = trial[:, sample_index - window + 1:sample_index + 1]

            # Decision.label is None when the decoder is below its confidence
            # floor; the page still wants to show what it was leaning towards.
            leading = max(decision.probabilities, key=decision.probabilities.get)

            frames.append({
                't': round((sample_index + 1) / self.fs, 3),
                'traces': self._downsample(current),
                'probabilities': {k: round(v, 4) for k, v in decision.probabilities.items()},
                'label': leading,
                'called': decision.label is not None,
                'confidence': round(decision.confidence, 4),
            })

        return frames

    def _downsample(self, window):
        """Reduce a window to a drawable number of points per channel.

        Parameters
        ----------
        window : numpy.ndarray
            Signal of shape (n_channels, n_samples).

        Returns
        -------
        list[list[float]]
            One list of points per display channel, scaled to roughly -1..1 so
            the page does not need to know about microvolts.
        """

        picks = [row for _, row in self.channel_rows]
        selected = window[picks]

        step = max(1, selected.shape[1] // DISPLAY_POINTS)
        reduced = selected[:, ::step][:, :DISPLAY_POINTS]

        # Scale by a high percentile rather than the max, so one spike does not
        # flatten the whole trace.
        scale = float(np.percentile(np.abs(reduced), 98)) or 1.0

        return [[round(float(v / scale), 3) for v in channel] for channel in reduced]
