"""Filter-bank tangent space, and the ensemble that pairs it with plain TS + LR.

Plain tangent space + logistic regression sees one covariance per trial, computed
over the whole 8-30 Hz band. That blends mu (8-12 Hz) and beta (13-30 Hz), which
desynchronise differently from person to person. The filter bank splits the band
first and builds a tangent-space vector per sub-band, so the classifier can
weight each rhythm on its own.

On its own the filter bank is high-variance: 5 bands x 253 features is a lot for
~130 training trials. Averaging its probabilities with plain TS + LR keeps most
of the gain and removes most of the variance. Measured over 20 random 50/25/25
splits of A01T and A04T (train on 50 %, score on the final 25 %):

    model                         A01T    A04T    mean    better than TS + LR
    TS + LR, 0.5-3.5 s            0.780   0.541   0.661   -
    ensemble, 0.5-4.5 s           0.815   0.573   0.694   14 of 20 splits

Both estimators take epochs that already went through
:func:`classification.preprocessing.prepare`, so the online decoder needs no
special case.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from classification.preprocessing import bandpass

# Sub-bands inside the shared 8-30 Hz preprocessing band: mu, then beta in 4 Hz
# steps. Bands outside 8-30 Hz would only see the main filter's roll-off.
FILTER_BANK = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 30.0))


class FilterBankTangentSpace(BaseEstimator, ClassifierMixin):
    """Tangent-space features per sub-band, concatenated, then logistic regression.

    Parameters
    ----------
    bands : tuple[tuple[float, float], ...], optional
        Sub-bands in Hz.
    fs : float, optional
        Sampling rate of the epochs.
    C : float, optional
        Inverse regularisation strength of the logistic regression.
    cov_estimator : str, optional
        pyriemann covariance estimator.
    """

    def __init__(self, bands=FILTER_BANK, fs=250.0, C=0.03, cov_estimator='oas'):
        self.bands = bands
        self.fs = fs
        self.C = C
        self.cov_estimator = cov_estimator

    def _features(self, X):
        """Stack the tangent-space vectors of every sub-band."""

        return np.hstack([
            projector.transform(bandpass(X, self.fs, band))
            for band, projector in zip(self.bands, self.projectors_)
        ])

    def fit(self, X, y):
        """Fit one tangent space per sub-band, then the classifier on all of them.

        Parameters
        ----------
        X : numpy.ndarray
            Preprocessed epochs of shape (n_trials, n_channels, n_samples).
        y : array-like
            Class label per epoch.

        Returns
        -------
        FilterBankTangentSpace
            self, fitted.
        """

        self.projectors_ = [
            Pipeline([('cov', Covariances(estimator=self.cov_estimator)),
                      ('tangent', TangentSpace(metric='riemann'))]).fit(bandpass(X, self.fs, band))
            for band in self.bands
        ]
        self.classifier_ = Pipeline([
            ('scale', StandardScaler()),
            ('lr', LogisticRegression(C=self.C, max_iter=2000)),
        ]).fit(self._features(X), y)
        self.classes_ = self.classifier_.classes_
        return self

    def predict_proba(self, X):
        """Class probabilities, columns in the order of ``classes_``."""

        return self.classifier_.predict_proba(self._features(X))

    def predict(self, X):
        """Most probable class per epoch."""

        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SoftVoteEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted average of the class probabilities of several fitted models.

    Parameters
    ----------
    estimators : list[tuple[object, float]]
        ``(unfitted_estimator, weight)`` pairs. Each must expose ``predict_proba``.
    """

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        """Fit every member on the same data.

        Parameters
        ----------
        X : numpy.ndarray
            Preprocessed epochs of shape (n_trials, n_channels, n_samples).
        y : array-like
            Class label per epoch.

        Returns
        -------
        SoftVoteEnsemble
            self, fitted.
        """

        self.fitted_ = [(estimator.fit(X, y), weight) for estimator, weight in self.estimators]
        self.classes_ = self.fitted_[0][0].classes_
        return self

    def predict_proba(self, X):
        """Weighted mean of the members' probabilities."""

        total = sum(weight for _, weight in self.fitted_)
        return sum(weight * model.predict_proba(X) for model, weight in self.fitted_) / total

    def predict(self, X):
        """Most probable class per epoch."""

        return self.classes_[self.predict_proba(X).argmax(axis=1)]
