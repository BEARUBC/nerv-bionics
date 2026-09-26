"""Model layer for the NERV EEG motor-imagery pipeline.

This package holds the three classifiers selected after benchmarking, plus the
shared preprocessing they depend on. The acquisition package produces raw
samples; this package turns them into class labels.

Public entry points
-------------------
- ``classification.data.load_mat_epochs``   : .mat recording -> labelled epochs
- ``classification.detector``               : train, save, load, detect
- ``classification.models.build_model``     : name -> unfitted pipeline
- ``classification.benchmark``              : score every model on every axis
- ``classification.server`` / ``app``       : interactive test bench
- ``classification.realtime.OnlineDecoder`` : rolling-buffer inference

Nothing heavy is imported here on purpose. Importing the package used to pull in
scikit-learn immediately, which meant an interpreter without it failed with a
confusing traceback before :mod:`classification._deps` could explain the problem.
``build_model`` and ``MODEL_REGISTRY`` still work as attributes of the package;
they are resolved on first use.
"""

import os

# Must be set before any OpenMP-linked library loads. See the note on
# DEFAULT_THREADS in eegnet.py for the clash this guards against.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

__all__ = ['MODEL_REGISTRY', 'build_model']


def __getattr__(name):
    """Resolve the package's re-exports on first access.

    Parameters
    ----------
    name : str
        Attribute being looked up.

    Returns
    -------
    object
        The requested attribute.
    """

    if name in __all__:
        from classification import models
        return getattr(models, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
