"""EEGNet wrapped as a scikit-learn estimator.

EEGNet (Lawhern et al., 2018) is a compact CNN built for EEG: a temporal
convolution learns frequency filters, a depthwise convolution learns spatial
filters per frequency, and a separable convolution summarises the result. It is
about 2-3k parameters, which is why it can be trained on a few hundred trials
without immediately overfitting.

The ``EEGNetClassifier`` wrapper exists so this model drops into the same
benchmark loop, cross-validation and ``OnlineDecoder`` as the Riemannian
pipelines. torch is imported inside the class, so importing this module without
torch installed fails with a clear message instead of breaking the package.
"""

import os
import platform

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

TORCH_HINT = (
    'EEGNet needs PyTorch. Install it with:\n'
    '    pip install torch\n'
    'The Riemannian models in this package do not need it.'
)


def _require_torch():
    """Import torch or raise a message that says how to install it.

    Returns
    -------
    tuple
        ``(torch, torch.nn, torch.nn.functional)``.
    """

    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(TORCH_HINT) from exc

    return torch, nn, functional


# Set once per process by :func:`limit_threads`.
_THREADS_SET = False

# PyTorch's libomp and the OpenMP runtime behind numpy/scipy both end up loaded in
# this project's environment, and torch's multi-threaded convolutions abort the
# process when that happens. Running torch single-threaded sidesteps the clash and
# costs little, because EEGNet has only a few thousand parameters. Raise it via
# ``EEGNetClassifier(threads=...)`` on an environment where threading is safe.
DEFAULT_THREADS = 1


def limit_threads(torch, threads=DEFAULT_THREADS):
    """Cap torch's intra-op thread count, once per process.

    Parameters
    ----------
    torch : module
        The imported torch module.
    threads : int or None
        Thread count to set, or None to leave torch's default in place.

    Returns
    -------
    None
    """

    global _THREADS_SET

    if _THREADS_SET or threads is None:
        return

    torch.set_num_threads(int(threads))
    _THREADS_SET = True


def build_network(n_channels, n_samples, n_classes, F1=8, D=2, F2=16, dropout=0.5, kernel_length=64):
    """Construct the EEGNet module.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Samples per epoch.
    n_classes : int
        Number of output classes.
    F1 : int, optional
        Temporal filters in block 1.
    D : int, optional
        Depth multiplier for the spatial convolution.
    F2 : int, optional
        Pointwise filters in block 2.
    dropout : float, optional
        Dropout probability applied after each block.
    kernel_length : int, optional
        Temporal kernel width in samples. At 250 Hz, 64 covers about 256 ms.

    Returns
    -------
    torch.nn.Module
        The unfitted network.
    """

    torch, nn, _ = _require_torch()

    class EEGNet(nn.Module):
        """Compact convolutional network for raw EEG epochs."""

        def __init__(self):
            super().__init__()

            # Block 1: temporal convolution then depthwise spatial convolution.
            self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
            self.bn_temporal = nn.BatchNorm2d(F1)
            self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
            self.bn_spatial = nn.BatchNorm2d(F1 * D)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)

            # Block 2: separable convolution (depthwise then pointwise).
            self.conv_separable = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
            self.conv_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
            self.bn_separable = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)

            # Work out the flattened size once instead of hard-coding it.
            with torch.no_grad():
                probe = torch.zeros(1, 1, n_channels, n_samples)
                n_flat = self._features(probe).shape[1]

            self.classifier = nn.Linear(n_flat, n_classes)

        def _features(self, x):
            x = self.bn_temporal(self.conv_temporal(x))
            x = torch.nn.functional.elu(self.bn_spatial(self.conv_spatial(x)))
            x = self.drop1(self.pool1(x))
            x = torch.nn.functional.elu(self.bn_separable(self.conv_pointwise(self.conv_separable(x))))
            x = self.drop2(self.pool2(x))
            return x.flatten(start_dim=1)

        def forward(self, x):
            return self.classifier(self._features(x))

    return EEGNet()


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """Train and apply EEGNet through the scikit-learn estimator interface.

    Parameters
    ----------
    epochs : int, optional
        Maximum training epochs.
    batch_size : int, optional
        Mini-batch size.
    learning_rate : float, optional
        Adam learning rate.
    weight_decay : float, optional
        Adam L2 penalty.
    patience : int, optional
        Early-stopping patience in epochs, measured on the validation split.
    validation_fraction : float, optional
        Fraction of the training set held out for early stopping. Set to 0 to
        train for the full ``epochs`` with no holdout.
    dropout : float, optional
        Dropout probability.
    device : str, optional
        ``'cpu'``, ``'cuda'``, ``'mps'``, or None to pick the best available.
    threads : int or None, optional
        torch intra-op thread cap. Defaults to 1 for the reason documented on
        :data:`DEFAULT_THREADS`; pass None to leave torch's own default alone.
    random_state : int, optional
        Seed for torch and the validation split.
    verbose : bool, optional
        Print progress every 50 epochs.
    """

    def __init__(self, epochs=300, batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
                 patience=40, validation_fraction=0.1, dropout=0.5, device=None,
                 threads=DEFAULT_THREADS, random_state=42, verbose=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.dropout = dropout
        self.device = device
        self.threads = threads
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self, torch):
        """Pick the training device.

        Parameters
        ----------
        torch : module
            The imported torch module.

        Returns
        -------
        torch.device
            The device to train on.
        """

        if self.device is not None:
            return torch.device(self.device)

        if torch.cuda.is_available():
            return torch.device('cuda')

        # An x86_64 interpreter on an Apple Silicon Mac is running under Rosetta.
        # torch still reports MPS as available there, but using it segfaults, so
        # only trust MPS from a native arm64 interpreter.
        native_arm = platform.machine() == 'arm64'
        mps_backend = getattr(torch.backends, 'mps', None)
        if native_arm and mps_backend is not None and mps_backend.is_available():
            return torch.device('mps')

        return torch.device('cpu')

    def _normalise(self, X, fit=False):
        """Z-score epochs per channel using training-set statistics.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs of shape (n_trials, n_channels, n_samples).
        fit : bool, optional
            Compute and store the statistics instead of reusing stored ones.

        Returns
        -------
        numpy.ndarray
            Normalised epochs.
        """

        X = np.asarray(X, dtype=np.float32)

        if fit:
            self._mean = X.mean(axis=(0, 2), keepdims=True)
            self._std = X.std(axis=(0, 2), keepdims=True) + 1e-8

        return (X - self._mean) / self._std

    def fit(self, X, y):
        """Train the network with early stopping on a held-out split.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs of shape (n_trials, n_channels, n_samples).
        y : array-like
            Class labels.

        Returns
        -------
        EEGNetClassifier
            self, fitted.
        """

        torch, nn, _ = _require_torch()
        limit_threads(torch, self.threads)
        torch.manual_seed(self.random_state)

        device = self._resolve_device(torch)
        self._device = device

        self._encoder = LabelEncoder()
        y_encoded = self._encoder.fit_transform(y)
        self.classes_ = self._encoder.classes_

        X_norm = self._normalise(X, fit=True)
        n_trials, n_channels, n_samples = X_norm.shape

        self.network_ = build_network(
            n_channels, n_samples, len(self.classes_), dropout=self.dropout
        ).to(device)

        # Stratified-ish holdout: shuffle once, then split. Good enough for an
        # early-stopping signal and keeps the dependency surface small.
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n_trials)
        n_validation = int(round(self.validation_fraction * n_trials))
        validation_idx = order[:n_validation]
        train_idx = order[n_validation:]

        X_tensor = torch.tensor(X_norm).unsqueeze(1)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        X_train = X_tensor[train_idx].to(device)
        y_train = y_tensor[train_idx].to(device)
        has_validation = n_validation > 0
        if has_validation:
            X_validation = X_tensor[validation_idx].to(device)
            y_validation = y_tensor[validation_idx].to(device)

        optimiser = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.network_.train()
            batch_order = torch.randperm(X_train.shape[0], device=device)

            for start in range(0, X_train.shape[0], self.batch_size):
                batch = batch_order[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.network_(X_train[batch]), y_train[batch])
                loss.backward()
                optimiser.step()

            if not has_validation:
                continue

            self.network_.eval()
            with torch.no_grad():
                validation_loss = criterion(self.network_(X_validation), y_validation).item()

            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_state = {k: v.detach().clone() for k, v in self.network_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose and epoch % 50 == 0:
                print(f'  epoch {epoch:4d} | val loss {validation_loss:.4f}')

            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f'  early stop at epoch {epoch}')
                break

        if best_state is not None:
            self.network_.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        """Return class probabilities.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        numpy.ndarray
            Probabilities of shape (n_trials, n_classes).
        """

        torch, _, functional = _require_torch()
        limit_threads(torch, self.threads)

        self.network_.eval()
        X_tensor = torch.tensor(self._normalise(X)).unsqueeze(1).to(self._device)

        with torch.no_grad():
            logits = self.network_(X_tensor)

        return functional.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        """Return predicted class labels.

        Parameters
        ----------
        X : numpy.ndarray
            Epochs of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        numpy.ndarray
            Predicted labels in the original label space.
        """

        return self._encoder.inverse_transform(self.predict_proba(X).argmax(axis=1))

    def count_parameters(self):
        """Count trainable parameters in the fitted network.

        Returns
        -------
        int
            Number of trainable parameters.
        """

        return sum(p.numel() for p in self.network_.parameters() if p.requires_grad)

    def save(self, path):
        """Save network weights and the normalisation statistics.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination ``.pt`` file.

        Returns
        -------
        None
        """

        torch, _, _ = _require_torch()
        torch.save(
            {
                'state_dict': self.network_.state_dict(),
                'classes': self.classes_,
                'mean': self._mean,
                'std': self._std,
            },
            str(path),
        )
