"""Conformal interval utilities with explicit fallback behavior."""

from __future__ import annotations

import numpy as np


def _finite_quantile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q, method="higher")) if values.size else float(fallback)


class ModeConditionalConformal:
    """Split-conformal absolute-residual intervals by observable mode.

    The method provides an empirical calibration layer. It does not claim
    conditional coverage under arbitrary distribution shift; sparse or unseen
    modes fall back to the global calibration radius and are reported by
    ``radius_for``.
    """

    def __init__(self, alpha: float = 0.1, min_group_size: int = 20) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.global_radius_: float | None = None
        self.group_radius_: dict[object, float] = {}

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, modes: np.ndarray | None = None) -> "ModeConditionalConformal":
        residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
        if residuals.size == 0:
            raise ValueError("at least one calibration residual is required")
        self.global_radius_ = _finite_quantile(residuals, 1 - self.alpha)
        self.group_radius_ = {}
        if modes is not None:
            mode_array = np.asarray(modes, dtype=object)
            for mode in np.unique(mode_array):
                group = residuals[mode_array == mode]
                if np.isfinite(group).sum() >= self.min_group_size:
                    self.group_radius_[mode] = _finite_quantile(group, 1 - self.alpha, self.global_radius_)
        return self

    def radius_for(self, modes: np.ndarray | None, n: int | None = None) -> np.ndarray:
        if self.global_radius_ is None:
            raise RuntimeError("fit must be called before radius_for")
        if modes is None:
            count = int(n or 1)
            return np.full(count, self.global_radius_, dtype=float)
        mode_array = np.asarray(modes, dtype=object)
        return np.asarray([self.group_radius_.get(mode, self.global_radius_) for mode in mode_array], dtype=float)

    def predict_interval(self, y_pred: np.ndarray, modes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(y_pred, dtype=float)
        radii = self.radius_for(modes, len(center))
        return center - radii, center + radii
