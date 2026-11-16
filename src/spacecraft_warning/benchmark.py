"""Run the conservative benchmark on a labeled long-form CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import LogisticWarning, PersistenceResidual
from .data import make_windows, read_telemetry_csv
from .metrics import calibration_metrics, event_warning_metrics
from .splits import chronological_split, event_safe_indices


def _median_cadence_seconds(group: pd.DataFrame) -> float | None:
    times = pd.to_datetime(group["timestamp"], utc=True)
    deltas = times.diff().dt.total_seconds().dropna()
    return float(deltas.median()) if len(deltas) else None


def _is_irregular_cadence(group: pd.DataFrame) -> bool:
    times = pd.to_datetime(group["timestamp"], utc=True)
    deltas = times.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    if len(deltas) < 2:
        return False
    median = float(np.median(deltas))
    return bool(np.any(np.abs(deltas - median) > max(1.0, abs(median) * 0.01)))


def _event_indices(group: pd.DataFrame, column: str) -> np.ndarray:
    values = group[column]
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        return np.flatnonzero(values.fillna(0).astype(bool).to_numpy())
    labels = pd.to_datetime(values, errors="coerce", utc=True)
    timestamps = pd.to_datetime(group["timestamp"], errors="coerce", utc=True)
    return np.flatnonzero(labels.notna().to_numpy() & timestamps.isin(labels).to_numpy())


def run_csv_benchmark(
    path: str | Path,
    lookback: int = 24,
    horizon: int = 6,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
    alpha: float = 0.05,
    cooldown_steps: int = 6,
) -> dict:
    """Evaluate every labeled asset/channel stream in a CSV.

    The input must contain `anomaly_start` as a boolean/numeric onset marker or
    as a timestamp matching the stream's `timestamp` column. Streams without
    that column are rejected because an unlabeled anomaly benchmark would be
    unable to distinguish warning from retrospective detection.
    """

    frame = read_telemetry_csv(path)
    if "anomaly_start" not in frame.columns:
        raise ValueError("CSV must contain anomaly_start labels for this benchmark")
    results: list[dict] = []
    group_columns = ["asset_id", "channel_id"]
    for (asset_id, channel_id), group in frame.groupby(group_columns, sort=False):
        group = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        events = _event_indices(group, "anomaly_start")
        windows = make_windows(
            group["value"].to_numpy(),
            group["timestamp"].to_numpy(),
            event_starts=events,
            modes=group["mode"].to_numpy() if "mode" in group else None,
            lookback=lookback,
            horizon=horizon,
        )
        if len(windows.X) < 20:
            continue
        split = chronological_split(
            len(windows.X),
            calibration_fraction=calibration_fraction,
            test_fraction=test_fraction,
            purge_steps=max(lookback, horizon),
        )
        calibration_origins = windows.origins[split.calibration]
        safe_origins = event_safe_indices(calibration_origins, events, lookback, horizon)
        calibration_rows = split.calibration[np.isin(calibration_origins, safe_origins)]
        if len(calibration_rows) == 0:
            raise ValueError(f"no event-free calibration windows for {asset_id}/{channel_id}")
        persistence = PersistenceResidual().fit(windows.X[split.train])
        calibration_scores = persistence.score_samples(windows.X[calibration_rows])
        test_scores = persistence.score_samples(windows.X[split.test])
        threshold = float(np.quantile(calibration_scores, 1 - alpha))
        warning = event_warning_metrics(
            test_scores,
            events,
            windows.origins[split.test],
            threshold=threshold,
            horizon=horizon,
            cooldown_steps=cooldown_steps,
            total_duration_steps=len(split.test),
        )
        result = {
            "asset_id": str(asset_id),
            "channel_id": str(channel_id),
            "events": int(len(events)),
            "threshold_from_calibration": threshold,
            "persistence_warning": warning,
            "median_cadence_seconds": _median_cadence_seconds(group),
            "irregular_cadence": _is_irregular_cadence(group),
        }
        if len(np.unique(windows.y[split.train])) > 1:
            logistic = LogisticWarning().fit(windows.X[split.train], windows.y[split.train])
            probability_calibrated = False
            if len(np.unique(windows.y[calibration_rows])) > 1:
                logistic.calibrate(windows.X[calibration_rows], windows.y[calibration_rows])
                probability_calibrated = True
            result["logistic_calibration"] = calibration_metrics(
                logistic.predict_calibrated_proba(windows.X[split.test]), windows.y[split.test]
            )
            result["logistic_probability_calibrated"] = probability_calibrated
        else:
            result["logistic_calibration"] = None
            result["logistic_probability_calibrated"] = False
        results.append(result)
    return {
        "dataset": str(Path(path)),
        "streams_evaluated": len(results),
        "results": results,
        "limitations": [
            "This runner evaluates anomaly-onset labels, not physical degradation or failure.",
            "Chronological isolation is per asset/channel; mission-held-out evaluation requires a dataset adapter.",
            "Thresholds and preprocessing are fit without test labels; no result is operational validation.",
        ],
    }
