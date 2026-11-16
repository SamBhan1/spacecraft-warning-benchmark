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


@dataclass(frozen=True)
class DomainSplit:
    """Indices for a domain-held-out evaluation.

    ``test`` contains every row from ``held_out_domain``. Train and calibration
    are chronological slices of the remaining domains, so the held-out domain
    cannot leak through overlapping rows or randomization.
    """

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    held_out_domain: object

    def as_dict(self) -> dict[str, object]:
        return {
            "train": self.train.tolist(),
            "calibration": self.calibration.tolist(),
            "test": self.test.tolist(),
            "held_out_domain": self.held_out_domain,
        }


def chronological_split(
    n_samples: int,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
    purge_steps: int = 0,
    timestamps: Iterable[object] | None = None,
    purge_seconds: float | None = None,
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
    if purge_seconds is not None and purge_seconds < 0:
        raise ValueError("purge_seconds must be non-negative")
    if purge_seconds is not None and timestamps is None:
        raise ValueError("timestamps are required when purge_seconds is set")
    if purge_seconds is not None and purge_steps:
        raise ValueError("use either purge_steps or purge_seconds, not both")
    if calibration_fraction + test_fraction >= 1:
        raise ValueError("calibration_fraction + test_fraction must be < 1")

    test_start = int(np.floor(n_samples * (1 - test_fraction)))
    calibration_start = int(np.floor(n_samples * (1 - test_fraction - calibration_fraction)))
    if calibration_start < 1 or test_start <= calibration_start or test_start >= n_samples:
        raise ValueError("fractions leave an unusable split")
    if timestamps is None:
        train_end = calibration_start - purge_steps
        calibration_end = test_start - purge_steps
        train = np.arange(0, train_end)
        calibration = np.arange(calibration_start, calibration_end)
    else:
        try:
            time_array = np.asarray(list(timestamps)).astype("datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamps must be datetime-like when used for physical purging") from exc
        if len(time_array) != n_samples or np.isnat(time_array).any():
            raise ValueError("timestamps must be complete and match n_samples")
        if np.any(time_array[1:] < time_array[:-1]):
            raise ValueError("timestamps must be monotonic for chronological splitting")
        if purge_seconds is None:
            train_end = calibration_start - purge_steps
            calibration_end = test_start - purge_steps
            train = np.arange(0, train_end)
            calibration = np.arange(calibration_start, calibration_end)
        else:
            embargo = np.timedelta64(int(round(purge_seconds * 1_000_000_000)), "ns")
            train = np.flatnonzero(
                (np.arange(n_samples) < calibration_start)
                & (time_array < time_array[calibration_start] - embargo)
            )
            calibration = np.flatnonzero(
                (np.arange(n_samples) >= calibration_start)
                & (np.arange(n_samples) < test_start)
                & (time_array < time_array[test_start] - embargo)
            )
    if len(train) < 1 or len(calibration) < 1:
        raise ValueError("purging leaves an unusable split")
    return TemporalSplit(
        train=train,
        calibration=calibration,
        test=np.arange(test_start, n_samples),
    )


def mission_holdout_split(
    missions: Iterable[object],
    held_out_mission: object,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
    purge_steps: int = 0,
    timestamps: Iterable[object] | None = None,
    purge_seconds: float | None = None,
) -> DomainSplit:
    """Create a chronological split with one complete mission held out.

    The held-out mission is never used for training, calibration, threshold
    selection, or preprocessing. The remaining missions are ordered as they
    appear in the input and split chronologically; callers should provide a
    deterministic mission ordering in their manifest.
    """

    mission_array = np.asarray(list(missions), dtype=object)
    if mission_array.size < 3:
        raise ValueError("at least three rows are required")
    test = np.flatnonzero(mission_array == held_out_mission)
    source = np.flatnonzero(mission_array != held_out_mission)
    if len(test) == 0 or len(source) < 3:
        raise ValueError("held-out mission must exist and leave at least three source rows")
    source_times = None
    if timestamps is not None:
        timestamp_array = np.asarray(list(timestamps), dtype=object)
        if len(timestamp_array) != len(mission_array):
            raise ValueError("timestamps must have equal length to missions")
        source_times = timestamp_array[source]
    local = chronological_split(
        len(source),
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        purge_steps=purge_steps,
        timestamps=source_times,
        purge_seconds=purge_seconds,
    )
    return DomainSplit(
        train=source[local.train],
        calibration=source[local.calibration],
        test=test,
        held_out_domain=held_out_mission,
    )


def assert_disjoint_groups(split: TemporalSplit | DomainSplit, groups: Iterable[object]) -> None:
    """Raise if a non-null event or episode group crosses split boundaries."""

    group_array = np.asarray(list(groups), dtype=object)
    partitions = [split.train, split.calibration, split.test]
    seen: dict[object, int] = {}
    for partition_number, indices in enumerate(partitions):
        for group in group_array[indices]:
            if group is None or (isinstance(group, float) and np.isnan(group)):
                continue
            previous = seen.setdefault(group, partition_number)
            if previous != partition_number:
                raise ValueError(f"group {group!r} crosses split boundaries")


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
