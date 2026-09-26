"""The three selected classifiers, plus the ones we measured and set aside.

Selection came out of the 8-model benchmark in ``trydataset2a/main.ipynb`` and the
re-run in ``benchmark.py`` on a corrected 0.5-3.5 s epoch window. What we keep:

1. ``riemann_ts_lr``  - Riemannian tangent space + logistic regression. Best
   accuracy and best kappa of every pipeline tested.
2. ``riemann_mdm``    - minimum distance to Riemannian mean. Cheapest to fit,
   best cross-subject accuracy, and it shares its covariance front end with
   model 1.
3. ``csp_lda``        - the BCI-literature reference pipeline. Third on the
   composite score, and the number reviewers will expect to see.

``eegnet`` stays available but is not selected. On the two recordings in this
repository it came last: 0.543 accuracy against 0.664 for model 1, and 117 s to
calibrate against 0.21 s. Read that as a data-volume result rather than a verdict
on the architecture - a CNN fitted to roughly 214 trials per fold is starved.
Re-run the benchmark once the other seven BCI IV 2a subjects are in place before
writing it off.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# Covariance shrinkage. OAS is the safe default at 22 channels x a few hundred
# samples, where the sample covariance is poorly conditioned.
COV_ESTIMATOR = 'oas'

# Spatial filters per one-vs-rest CSP decomposition, matching the earlier notebook.
N_CSP_COMPONENTS = 6

# Inverse regularisation strength for the tangent-space classifier. The tangent
# space of a 22x22 covariance has 253 dimensions and a subject supplies a few
# hundred trials, so the default C=1.0 overfits. Sweeping C on both recordings
# showed a broad plateau between 0.003 and 0.02; 0.01 sits in the middle of it.
#
#   C     A01T   A04T   mean
#   1.0   0.824  0.504  0.664   <- scikit-learn default
#   0.01  0.817  0.565  0.691
#
# The gain is almost entirely on the harder subject, which is the behaviour you
# want: it costs the easy subject little and rescues the noisy one. Selected on
# two subjects only, so re-check it when the other seven are added.
TANGENT_SPACE_C = 0.01

RANDOM_STATE = 42


def build_riemann_ts_lr():
    """Rank 1: covariance -> tangent space -> multinomial logistic regression.

    Projecting each trial's 22x22 covariance into the tangent space at the
    geometric mean linearises the SPD manifold, so a linear classifier gets the
    full covariance instead of the handful of directions CSP keeps.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline taking epochs of shape (n_trials, n_channels, n_samples).
    """

    return Pipeline([
        ('cov', Covariances(estimator=COV_ESTIMATOR)),
        ('tangent', TangentSpace(metric='riemann')),
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=TANGENT_SPACE_C, random_state=RANDOM_STATE)),
    ])


def build_riemann_mdm():
    """Rank 2: nearest Riemannian class mean, no trained decision boundary.

    Fitting is just one geometric mean per class, which makes this the cheapest
    model to calibrate on a new user and the easiest to update incrementally.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline taking epochs of shape (n_trials, n_channels, n_samples).
    """

    return Pipeline([
        ('cov', Covariances(estimator=COV_ESTIMATOR)),
        ('mdm', MDM(metric='riemann')),
    ])


def build_csp_lda():
    """Rank 3: the standard CSP + LDA pipeline from the BCI literature.

    Kept both as the reference point every BCI paper reports and as a genuine
    third choice: it scores within two points of model 1, at the cost of a
    calibration step roughly six times slower and much weaker transfer to a
    subject it was not trained on.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline taking epochs of shape (n_trials, n_channels, n_samples).
    """

    from mne.decoding import CSP

    return Pipeline([
        ('csp', CSP(n_components=N_CSP_COMPONENTS, reg=COV_ESTIMATOR, log=True, norm_trace=False)),
        ('lda', LinearDiscriminantAnalysis()),
    ])


def build_csp_svm():
    """Not selected: CSP log-variance features with an RBF support vector machine.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline taking epochs of shape (n_trials, n_channels, n_samples).
    """

    from mne.decoding import CSP

    return Pipeline([
        ('csp', CSP(n_components=N_CSP_COMPONENTS, reg=COV_ESTIMATOR, log=True, norm_trace=False)),
        ('scale', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=RANDOM_STATE)),
    ])


def build_eegnet(**kwargs):
    """Not selected: EEGNet, imported lazily so the rest of the package needs no torch.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`classification.eegnet.EEGNetClassifier`.

    Returns
    -------
    classification.eegnet.EEGNetClassifier
        Unfitted scikit-learn compatible estimator.
    """

    from classification.eegnet import EEGNetClassifier

    return EEGNetClassifier(**kwargs)


# Selected models first, baselines after. `rank` is None for baselines.
MODEL_REGISTRY = {
    'riemann_ts_lr': {
        'label': 'Riemannian TS + LR',
        'builder': build_riemann_ts_lr,
        'rank': 1,
        'needs_torch': False,
    },
    'riemann_mdm': {
        'label': 'Riemannian MDM',
        'builder': build_riemann_mdm,
        'rank': 2,
        'needs_torch': False,
    },
    'csp_lda': {
        'label': 'CSP + LDA',
        'builder': build_csp_lda,
        'rank': 3,
        'needs_torch': False,
    },
    'csp_svm': {
        'label': 'CSP + SVM',
        'builder': build_csp_svm,
        'rank': None,
        'needs_torch': False,
    },
    'eegnet': {
        'label': 'EEGNet (CNN)',
        'builder': build_eegnet,
        'rank': None,
        'needs_torch': True,
    },
}

# Ranked best first. A model with rank None is available but was not selected.
SELECTED_MODELS = tuple(
    name for name, spec in sorted(
        MODEL_REGISTRY.items(), key=lambda item: (item[1]['rank'] is None, item[1]['rank'] or 0)
    )
    if spec['rank'] is not None
)


def build_model(name, **kwargs):
    """Instantiate a registered model by name.

    Parameters
    ----------
    name : str
        A key of :data:`MODEL_REGISTRY`.
    **kwargs
        Forwarded to the model's builder where it accepts arguments.

    Returns
    -------
    object
        An unfitted estimator exposing ``fit``, ``predict`` and usually
        ``predict_proba``.
    """

    if name not in MODEL_REGISTRY:
        raise KeyError(f'Unknown model {name!r}. Known models: {sorted(MODEL_REGISTRY)}')

    builder = MODEL_REGISTRY[name]['builder']
    return builder(**kwargs) if kwargs else builder()
