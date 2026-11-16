import numpy as np
import pandas as pd
import pytest

from spacecraft_warning.conformal import ModeConditionalConformal
from spacecraft_warning.benchmark import run_csv_benchmark
from spacecraft_warning.data import make_windows, validate_telemetry_frame
from spacecraft_warning.metrics import calibration_metrics, event_warning_metrics
from spacecraft_warning.splits import chronological_split, event_safe_indices


def test_chronological_split_is_disjoint_and_ordered():
    split = chronological_split(100, calibration_fraction=0.2, test_fraction=0.2)
    assert set(split.train).isdisjoint(split.calibration)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.calibration).isdisjoint(split.test)
    assert split.train[-1] < split.calibration[0] < split.test[0]


def test_purged_split_leaves_embargo_rows():
    split = chronological_split(100, calibration_fraction=0.2, test_fraction=0.2, purge_steps=5)
    assert split.train[-1] + 5 < split.calibration[0]
    assert split.calibration[-1] + 5 < split.test[0]


def test_event_safe_indices_removes_touching_windows():
    kept = event_safe_indices([10, 20, 30], [20], lookback=5, horizon=3)
    assert kept.tolist() == [10, 30]


def test_windows_use_future_only_for_targets_and_preserve_missingness():
    values = np.arange(30, dtype=float)
    values[8] = np.nan
    windows = make_windows(values, np.arange(30), event_starts=[15], lookback=5, horizon=3)
    assert windows.X.shape[1] == 10
    assert windows.y.max() == 1
    # The missingness indicator is part of the feature vector.
    assert windows.X[4, 5 + 4] == 1.0


def test_conformal_falls_back_for_unseen_modes():
    method = ModeConditionalConformal(alpha=0.1, min_group_size=2).fit(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 2.0, 2.0]),
        np.array(["a", "a", "b", "b"], dtype=object),
    )
    radii = method.radius_for(np.array(["a", "unseen"], dtype=object))
    assert radii[0] != 0
    assert radii[1] == method.global_radius_


def test_event_metrics_count_events_not_points():
    result = event_warning_metrics(
        scores=np.array([0.0, 2.0, 2.0, 0.0, 0.0]),
        event_starts=np.array([3]),
        origins=np.arange(5),
        threshold=1.0,
        horizon=3,
        cooldown_steps=2,
        total_duration_steps=5,
    )
    assert result["detected_events"] == 1
    assert result["events"] == 1


def test_one_alert_cannot_claim_two_events():
    result = event_warning_metrics(
        scores=np.array([0.0, 2.0, 0.0, 0.0]),
        event_starts=np.array([3, 4]),
        origins=np.arange(4),
        threshold=1.0,
        horizon=3,
        cooldown_steps=1,
        total_duration_steps=4,
    )
    assert result["detected_events"] == 1


def test_event_metrics_excludes_events_outside_score_segment():
    result = event_warning_metrics(
        scores=np.array([0.0, 2.0, 0.0]),
        event_starts=np.array([1, 100]),
        origins=np.array([10, 11, 12]),
        threshold=1.0,
        horizon=3,
        cooldown_steps=1,
        total_duration_steps=3,
    )
    assert result["events"] == 0


def test_calibration_metrics_are_bounded():
    result = calibration_metrics(np.array([0.1, 0.9]), np.array([0, 1]))
    assert 0 <= result["brier"] <= 1
    assert 0 <= result["expected_calibration_error"] <= 1


def test_schema_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_telemetry_frame(pd.DataFrame({"value": [1.0]}))


def test_csv_runner_requires_and_uses_onset_labels(tmp_path):
    n = 160
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC"),
            "asset_id": "a",
            "channel_id": "c",
            "value": np.sin(np.arange(n) / 4),
            "anomaly_start": False,
        }
    )
    frame.loc[[40, 110], "anomaly_start"] = True
    path = tmp_path / "telemetry.csv"
    frame.to_csv(path, index=False)
    result = run_csv_benchmark(path, lookback=8, horizon=4)
    assert result["streams_evaluated"] == 1
    assert result["results"][0]["events"] == 2
