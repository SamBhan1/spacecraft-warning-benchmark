"""Synthetic smoke test for the spacecraft warning benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import LogisticWarning, PersistenceResidual
from .conformal import ModeConditionalConformal
from .data import make_windows
from .metrics import calibration_metrics, event_warning_metrics
from .splits import chronological_split, event_safe_indices


def make_synthetic_stream(seed: int = 7, n: int = 2400) -> tuple[pd.DataFrame, np.ndarray]:
    """Create explicitly synthetic, mode-changing telemetry with anomalies."""

    rng = np.random.default_rng(seed)
    t = np.arange(n)
    mode = np.where((t // 300) % 2 == 0, "nominal", "maneuver")
    command = ((t % 300) == 0).astype(int)
    baseline = np.where(mode == "nominal", 0.2 * np.sin(t / 30), 1.0 + 0.35 * np.sin(t / 12))
    value = baseline + rng.normal(0, 0.08, n)
    event_starts = np.asarray([560, 1020, 1510, 2050], dtype=int)
    for start in event_starts:
        length = 18
        ramp = np.linspace(0.0, 2.0, length)
        value[start : start + length] += ramp + rng.normal(0, 0.05, length)
    missing = rng.random(n) < 0.03
    value[missing] = np.nan
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC"),
            "asset_id": "synthetic-spacecraft",
            "channel_id": "synthetic-thermal-channel",
            "value": value,
            "mode": mode,
            "command": command,
        }
    )
    frame["event_id"] = None
    for start in event_starts:
        frame.loc[start, "event_id"] = f"synthetic-event-{start}"
    return frame, event_starts


def run_demo(seed: int = 7) -> dict:
    frame, event_starts = make_synthetic_stream(seed=seed)
    windows = make_windows(
        frame["value"].to_numpy(),
        frame["timestamp"].to_numpy(),
        event_starts=event_starts,
        event_ids=frame["event_id"].to_numpy(),
        modes=frame["mode"].to_numpy(),
        lookback=24,
        horizon=12,
    )
    split = chronological_split(
        len(windows.X),
        calibration_fraction=0.2,
        test_fraction=0.2,
        timestamps=windows.timestamps,
        purge_seconds=24 * 60,
    )
    calibration_origins = windows.origins[split.calibration]
    safe_origins = event_safe_indices(calibration_origins, event_starts, lookback=24, horizon=12)
    calibration_rows = split.calibration[np.isin(calibration_origins, safe_origins)]
    if len(calibration_rows) == 0:
        raise RuntimeError("synthetic demo produced no event-free calibration windows")

    persistence = PersistenceResidual().fit(windows.X[split.train])
    cal_scores = persistence.score_samples(windows.X[calibration_rows])
    test_scores = persistence.score_samples(windows.X[split.test])
    # The threshold is selected from calibration only. It is intentionally
    # reported as a smoke-test policy, not as a tuned scientific result.
    threshold = float(np.quantile(cal_scores, 0.95))
    warning = event_warning_metrics(
        test_scores,
        event_starts,
        windows.origins[split.test],
        threshold=threshold,
        horizon=12,
        cooldown_steps=12,
        total_duration_steps=len(split.test),
        timestamps=windows.timestamps[split.test],
        cooldown_seconds=12 * 60,
        max_gap_seconds=5 * 60,
    )

    # Persistence point forecasts use the last observed finite value. The
    # conformal layer is calibrated only on the chronological calibration set.
    def last_value(X: np.ndarray) -> np.ndarray:
        values = X[:, : X.shape[1] // 2]
        last = values[:, -1]
        return np.where(np.isfinite(last), last, np.nanmedian(values, axis=1))

    cal_truth = windows.targets[calibration_rows]
    test_truth = windows.targets[split.test]
    cal_valid = np.isfinite(cal_truth)
    test_valid = np.isfinite(test_truth)
    conformal = ModeConditionalConformal(alpha=0.1, min_group_size=10).fit(
        cal_truth[cal_valid],
        last_value(windows.X[calibration_rows])[cal_valid],
        windows.modes[calibration_rows][cal_valid],
    )
    lower, upper = conformal.predict_interval(last_value(windows.X[split.test]), windows.modes[split.test])
    coverage = float(np.mean((test_truth[test_valid] >= lower[test_valid]) & (test_truth[test_valid] <= upper[test_valid])))

    logistic = LogisticWarning(random_state=seed).fit(windows.X[split.train], windows.y[split.train])
    probability_calibrated = len(np.unique(windows.y[split.calibration])) > 1
    if probability_calibrated:
        logistic.calibrate(windows.X[split.calibration], windows.y[split.calibration])
    probabilities = logistic.predict_calibrated_proba(windows.X[split.test])
    calibration = calibration_metrics(probabilities, windows.y[split.test])

    return {
        "synthetic_only": True,
        "n_observations": int(len(frame)),
        "n_windows": int(len(windows.X)),
        "event_starts": event_starts.tolist(),
        "split_sizes": {k: len(v) for k, v in split.as_dict().items()},
        "persistence_warning": warning,
        "logistic_calibration": calibration,
        "logistic_probability_calibrated": probability_calibrated,
        "conformal_test_coverage": coverage,
        "conformal_global_radius": conformal.global_radius_,
        "note": "Synthetic smoke test only; not a NASA or flight-performance result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run_demo(seed=args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
