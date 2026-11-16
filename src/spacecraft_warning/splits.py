"""Chronological and event-safe splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    """Half-open index ranges for train, calibration, and confirmation data."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": self.train.tolist(),
            "calibration": self.calibration.tolist(),
            "test": self.test.tolist(),
        }


def chronological_split(
    n_samples: int,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
    purge_steps: int = 0,
) -> TemporalSplit:
    """Return chronological indices with no randomization.

    Fractions are measured from the complete sequence. The test set is always
    the latest segment, and the calibration set immediately precedes it.
    """

    if n_samples < 3:
        raise ValueError("at least three observations are required")
    if not 0 < calibration_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("fractions must be between 0 and 1")
    if purge_steps < 0:
        raise ValueError("purge_steps must be non-negative")
    if calibration_fraction + test_fraction >= 1:
        raise ValueError("calibration_fraction + test_fraction must be < 1")

    test_start = int(np.floor(n_samples * (1 - test_fraction)))
    calibration_start = int(np.floor(n_samples * (1 - test_fraction - calibration_fraction)))
    if calibration_start < 1 or test_start <= calibration_start or test_start >= n_samples:
        raise ValueError("fractions leave an unusable split")
    train_end = calibration_start - purge_steps
    calibration_end = test_start - purge_steps
    if train_end < 1 or calibration_end <= calibration_start:
        raise ValueError("purge_steps leaves an unusable split")
    return TemporalSplit(
        train=np.arange(0, train_end),
        calibration=np.arange(calibration_start, calibration_end),
        test=np.arange(test_start, n_samples),
    )


def event_safe_indices(
    origins: Iterable[int],
    event_starts: Iterable[int],
    lookback: int,
    horizon: int,
) -> np.ndarray:
    """Keep windows whose observed/predicted ranges do not straddle events.

    This conservative helper removes any origin whose lookback-to-horizon
    window touches an event start. It is useful for a nominal calibration set;
    warning evaluation should instead retain positive windows and group by
    event ID explicitly.
    """

    starts = np.asarray(list(event_starts), dtype=int)
    starts = starts[np.isfinite(starts)] if starts.size else starts
    kept: list[int] = []
    for origin in origins:
        left = int(origin) - lookback + 1
        right = int(origin) + horizon
        if not np.any((starts >= left) & (starts <= right)):
            kept.append(int(origin))
    return np.asarray(kept, dtype=int)
