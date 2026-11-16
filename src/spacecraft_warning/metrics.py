"""Event-level warning and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _datetime_ns(values: np.ndarray) -> np.ndarray:
    """Normalize naive and timezone-aware timestamps without numpy warnings."""

    return pd.to_datetime(list(values), utc=True, errors="raise").to_numpy(dtype="datetime64[ns]")


def monitored_exposure_seconds(
    timestamps: np.ndarray,
    max_gap_seconds: float | None = None,
) -> float:
    """Return observed wall-clock exposure, optionally censoring large gaps.

    A gap larger than ``max_gap_seconds`` is treated as unmonitored rather than
    silently becoming nominal exposure. This is the denominator needed for
    physical false-alarm rates on irregular telemetry.
    """

    values = _datetime_ns(np.asarray(timestamps))
    if values.size < 2 or np.isnat(values).any():
        return 0.0
    deltas = np.diff(values).astype("timedelta64[ns]").astype(np.int64) / 1_000_000_000
    if np.any(deltas < 0):
        raise ValueError("timestamps must be sorted for exposure accounting")
    if max_gap_seconds is not None:
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        deltas = deltas[deltas <= max_gap_seconds]
    return float(deltas.sum())


def _collapse_alerts(
    alerts: np.ndarray,
    cooldown_steps: int,
    timestamps: np.ndarray | None = None,
    cooldown_seconds: float | None = None,
) -> np.ndarray:
    if cooldown_steps < 0:
        raise ValueError("cooldown_steps must be non-negative")
    if cooldown_seconds is not None and cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must be non-negative")
    if cooldown_seconds is not None:
        if timestamps is None:
            raise ValueError("timestamps are required for a physical-time cooldown")
        time_array = _datetime_ns(np.asarray(timestamps))
        if len(time_array) != len(alerts) or np.isnat(time_array).any():
            raise ValueError("timestamps must be complete and match alerts")
    collapsed = np.zeros(len(alerts), dtype=bool)
    last = -10**9
    last_time: np.datetime64 | None = None
    for i, alert in enumerate(np.asarray(alerts, dtype=bool)):
        step_ok = i - last > cooldown_steps
        time_ok = (
            cooldown_seconds is None
            or last_time is None
            or (time_array[i] - last_time).astype("timedelta64[ns]").astype(np.int64) / 1_000_000_000
            > cooldown_seconds
        )
        if alert and step_ok and time_ok:
            collapsed[i] = True
            last = i
            if cooldown_seconds is not None:
                last_time = time_array[i]
    return collapsed


def event_warning_metrics(
    scores: np.ndarray,
    event_starts: np.ndarray,
    origins: np.ndarray,
    threshold: float,
    horizon: int,
    cooldown_steps: int = 1,
    total_duration_steps: int | None = None,
    timestamps: np.ndarray | None = None,
    cooldown_seconds: float | None = None,
    exposure_seconds: float | None = None,
    max_gap_seconds: float | None = None,
) -> dict[str, float | int | None]:
    """Measure warning alerts against event starts and physical exposure.

    The legacy step-based rate remains available for compatibility. When
    timestamps are supplied, ``false_alarms_per_day`` uses monitored wall-clock
    exposure and can censor communication gaps with ``max_gap_seconds``.
    """

    scores = np.asarray(scores, dtype=float)
    origins = np.asarray(origins, dtype=int)
    starts = np.asarray(event_starts, dtype=int)
    if len(scores) != len(origins):
        raise ValueError("scores and origins must have equal length")
    time_array = None
    if timestamps is not None:
        time_array = _datetime_ns(np.asarray(timestamps))
        if len(time_array) != len(scores) or np.isnat(time_array).any():
            raise ValueError("timestamps must be complete and match scores")
        if np.any(time_array[1:] < time_array[:-1]):
            raise ValueError("timestamps must be sorted")
    if len(origins):
        # Only events that could have been warned about by this score segment
        # belong in its denominator. Earlier train/calibration events are not
        # test misses, and later events are not yet observable.
        starts = starts[(starts > origins.min()) & (starts <= origins.max() + horizon)]
    raw = np.isfinite(scores) & (scores >= threshold)
    alert_mask = _collapse_alerts(raw, cooldown_steps, time_array, cooldown_seconds)
    alert_origins = origins[alert_mask]
    detected = 0
    lead_times: list[int] = []
    used_alerts: set[int] = set()
    for start in np.sort(starts):
        candidates = np.where((alert_origins < start) & (alert_origins >= start - horizon))[0]
        candidates = np.asarray([candidate for candidate in candidates if candidate not in used_alerts], dtype=int)
        if candidates.size:
            # Use the earliest alert so lead time measures genuine advance
            # warning. A single alert cannot claim two separate events.
            chosen = int(candidates[0])
            detected += 1
            lead_times.append(int(start - alert_origins[chosen]))
            used_alerts.add(chosen)
    false_alerts = max(0, len(alert_origins) - len(used_alerts))
    precision = float(len(used_alerts) / len(alert_origins)) if len(alert_origins) else 0.0
    recall = float(detected / len(starts)) if len(starts) else 0.0
    duration = int(total_duration_steps or (origins[-1] - origins[0] + 1 if len(origins) else 0))
    result: dict[str, float | int | None] = {
        "events": int(len(starts)),
        "detected_events": int(detected),
        "alerts": int(len(alert_origins)),
        "false_alerts": int(false_alerts),
        "precision": precision,
        "recall": recall,
        "false_alarms_per_1000_steps": float(false_alerts / max(duration, 1) * 1000),
        "mean_warning_lead_steps": float(np.mean(lead_times)) if lead_times else None,
    }
    if exposure_seconds is None and time_array is not None:
        exposure_seconds = monitored_exposure_seconds(time_array, max_gap_seconds=max_gap_seconds)
    if exposure_seconds is not None:
        if exposure_seconds < 0:
            raise ValueError("exposure_seconds must be non-negative")
        result["exposure_seconds"] = float(exposure_seconds)
        result["false_alarms_per_day"] = float(false_alerts / exposure_seconds * 86_400) if exposure_seconds else None
    return result


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Return Brier score and expected calibration error."""

    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=int)
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("probabilities and labels must be non-empty and equal length")
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p <= right if right == 1 else p < right)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {"brier": float(np.mean((p - y) ** 2)), "expected_calibration_error": ece}
