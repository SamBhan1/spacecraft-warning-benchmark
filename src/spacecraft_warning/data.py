"""Dataset validation and supervised-window construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "asset_id", "channel_id", "value"}


@dataclass(frozen=True)
class WindowedTelemetry:
    """A supervised warning table generated from one ordered telemetry stream."""

    X: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    modes: np.ndarray
    timestamps: np.ndarray
    targets: np.ndarray
    target_event_ids: np.ndarray


def validate_telemetry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and stably sort a long-form telemetry frame."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    clean = frame.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True)
    if clean["timestamp"].isna().any():
        raise ValueError("timestamp contains values that are not parseable")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    if clean[["asset_id", "channel_id"]].isna().any().any():
        raise ValueError("asset_id and channel_id cannot be missing")
    return clean.sort_values(["asset_id", "channel_id", "timestamp"], kind="stable").reset_index(drop=True)


def read_telemetry_csv(path: str | Path) -> pd.DataFrame:
    return validate_telemetry_frame(pd.read_csv(path))


def _event_target(event_starts: Iterable[int], origin: int, horizon: int) -> int:
    starts = np.asarray(list(event_starts), dtype=int)
    if starts.size == 0:
        return 0
    return int(np.any((starts > origin) & (starts <= origin + horizon)))


def _target_event_id(
    event_starts: np.ndarray,
    event_ids: np.ndarray,
    origin: int,
    horizon: int,
) -> object:
    """Return the first event identity in the forecast window, if available."""

    candidates = event_starts[(event_starts > origin) & (event_starts <= origin + horizon)]
    if candidates.size == 0 or event_ids.size == 0:
        return None
    candidate = event_ids[int(candidates[0])]
    return None if pd.isna(candidate) else candidate


def make_windows(
    values: Iterable[float],
    timestamps: Iterable[object],
    event_starts: Iterable[int] = (),
    event_ids: Iterable[object] | None = None,
    modes: Iterable[object] | None = None,
    lookback: int = 24,
    horizon: int = 6,
    max_gap_seconds: float | None = None,
) -> WindowedTelemetry:
    """Create lag windows without peeking beyond each forecast origin.

    Missing observations are retained as NaNs and accompanied by a missingness
    indicator in the final feature column for every lag. Imputation/scaling is
    intentionally left to the caller so it can be fitted inside the training
    split.
    """

    if lookback < 2 or horizon < 1:
        raise ValueError("lookback must be >= 2 and horizon must be >= 1")
    if max_gap_seconds is not None and max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    series = np.asarray(list(values), dtype=float)
    times = np.asarray(list(timestamps))
    if len(series) != len(times):
        raise ValueError("values and timestamps must have equal length")
    mode_array = np.asarray(list(modes), dtype=object) if modes is not None else np.full(len(series), "unknown", dtype=object)
    if len(mode_array) != len(series):
        raise ValueError("modes must have equal length to values")
    starts = np.asarray(list(event_starts), dtype=int)
    if event_ids is None:
        event_id_array = np.full(len(series), None, dtype=object)
    else:
        event_id_array = np.asarray(list(event_ids), dtype=object)
        if len(event_id_array) != len(series):
            raise ValueError("event_ids must have equal length to values")

    valid_origins = np.ones(len(series), dtype=bool)
    if max_gap_seconds is not None:
        parsed_times = pd.to_datetime(times, errors="coerce", utc=True)
        if parsed_times.isna().any():
            raise ValueError("timestamps must be parseable when max_gap_seconds is supplied")
        gap_seconds = parsed_times.to_series().diff().dt.total_seconds().to_numpy()
        for origin in range(lookback - 1, len(series) - horizon):
            window_gaps = gap_seconds[origin - lookback + 2 : origin + horizon + 1]
            valid_origins[origin] = not np.any(window_gaps > max_gap_seconds)

    X: list[np.ndarray] = []
    y: list[int] = []
    origins: list[int] = []
    mode_out: list[object] = []
    time_out: list[object] = []
    target_out: list[float] = []
    target_event_id_out: list[object] = []
    for origin in range(lookback - 1, len(series) - horizon):
        if not valid_origins[origin]:
            continue
        window = series[origin - lookback + 1 : origin + 1]
        missing = np.isnan(window).astype(float)
        # Preserve the missingness mechanism as explicit features. The raw
        # NaNs are imputed later using training-only statistics.
        X.append(np.concatenate([window, missing]))
        y.append(_event_target(starts, origin, horizon))
        origins.append(origin)
        mode_out.append(mode_array[origin])
        time_out.append(times[origin])
        target_out.append(series[origin + horizon])
        target_event_id_out.append(_target_event_id(starts, event_id_array, origin, horizon))
    return WindowedTelemetry(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=int),
        origins=np.asarray(origins, dtype=int),
        modes=np.asarray(mode_out, dtype=object),
        timestamps=np.asarray(time_out),
        targets=np.asarray(target_out, dtype=float),
        target_event_ids=np.asarray(target_event_id_out, dtype=object),
    )
