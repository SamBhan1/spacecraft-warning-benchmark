"""Simple baselines that should precede deep sequence models."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


class PersistenceResidual:
    """Last-value forecast residual score with training-only scale."""

    def __init__(self) -> None:
        self.scale_: float | None = None
        self.center_: float | None = None

    def fit(self, X: np.ndarray) -> "PersistenceResidual":
        values = np.asarray(X, dtype=float)[:, : X.shape[1] // 2]
        last = values[:, -1]
        first_diff = values[:, 1:] - values[:, :-1]
        finite = first_diff[np.isfinite(first_diff)]
        self.center_ = float(np.nanmedian(last)) if np.isfinite(last).any() else 0.0
        scale = float(np.nanmedian(np.abs(finite - np.nanmedian(finite)))) if finite.size else 1.0
        self.scale_ = max(scale * 1.4826, 1e-8)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("fit must be called before score_samples")
        values = np.asarray(X, dtype=float)[:, : X.shape[1] // 2]
        last = np.where(np.isfinite(values[:, -1]), values[:, -1], self.center_)
        previous = values[:, -2]
        previous = np.where(np.isfinite(previous), previous, last)
        return np.abs(last - previous) / self.scale_


class LogisticWarning:
    """Regularized lag-feature baseline with training-only preprocessing."""

    def __init__(self, random_state: int = 7) -> None:
        self.model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=False)),
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        )
        self.calibrator: IsotonicRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticWarning":
        if len(np.unique(y)) < 2:
            raise ValueError("logistic warning baseline requires both classes in training data")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def calibrate(self, X_calibration: np.ndarray, y_calibration: np.ndarray) -> "LogisticWarning":
        """Fit a calibration map on the chronological calibration split only."""

        if len(np.unique(y_calibration)) < 2:
            raise ValueError("probability calibration requires both classes")
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            self.predict_proba(X_calibration), y_calibration
        )
        return self

    def predict_calibrated_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_proba(X)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw
