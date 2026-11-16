"""Event-level warning and calibration metrics."""

from __future__ import annotations

import numpy as np


def _collapse_alerts(alerts: np.ndarray, cooldown_steps: int) -> np.ndarray:
    collapsed = np.zeros(len(alerts), dtype=bool)
    last = -10**9
    for i, alert in enumerate(np.asarray(alerts, dtype=bool)):
        if alert and i - last > cooldown_steps:
            collapsed[i] = True
            last = i
    return collapsed


def event_warning_metrics(
    scores: np.ndarray,
    event_starts: np.ndarray,
    origins: np.ndarray,
    threshold: float,
    horizon: int,
    cooldown_steps: int = 1,
    total_duration_steps: int | None = None,
) -> dict[str, float | int | None]:
    """Measure warning alerts against event starts, not telemetry points."""

    scores = np.asarray(scores, dtype=float)
    origins = np.asarray(origins, dtype=int)
    starts = np.asarray(event_starts, dtype=int)
    if len(origins):
        # Only events that could have been warned about by this score segment
        # belong in its denominator. Earlier train/calibration events are not
        # test misses, and later events are not yet observable.
        starts = starts[(starts > origins.min()) & (starts <= origins.max() + horizon)]
    raw = np.isfinite(scores) & (scores >= threshold)
    alert_mask = _collapse_alerts(raw, cooldown_steps)
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
    return {
        "events": int(len(starts)),
        "detected_events": int(detected),
        "alerts": int(len(alert_origins)),
        "false_alerts": int(false_alerts),
        "precision": precision,
        "recall": recall,
        "false_alarms_per_1000_steps": float(false_alerts / max(duration, 1) * 1000),
        "mean_warning_lead_steps": float(np.mean(lead_times)) if lead_times else None,
    }


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
