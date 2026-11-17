import numpy as np
import pandas as pd
import pytest

from spacecraft_warning.conformal import ModeConditionalConformal
from spacecraft_warning.benchmark import run_csv_benchmark
from spacecraft_warning.data import make_windows, validate_telemetry_frame
from spacecraft_warning.metrics import calibration_metrics, event_warning_metrics, monitored_exposure_seconds
from spacecraft_warning.splits import (
    assert_disjoint_groups,
    chronological_split,
    event_safe_indices,
    mission_holdout_split,
)


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


def test_physical_time_purge_handles_irregular_cadence():
    timestamps = np.arange(60, dtype="timedelta64[m]") + np.datetime64("2020-01-01T00:00:00")
    timestamps[30:] += np.timedelta64(30, "m")
    split = chronological_split(
        len(timestamps),
        calibration_fraction=0.2,
        test_fraction=0.2,
        timestamps=timestamps,
        purge_seconds=600,
    )
    assert timestamps[split.train[-1]] < timestamps[split.calibration[0]] - np.timedelta64(600, "s")
    assert timestamps[split.calibration[-1]] < timestamps[split.test[0]] - np.timedelta64(600, "s")


def test_mission_holdout_has_no_mission_overlap():
    missions = np.array(["m1"] * 4 + ["m2"] * 4 + ["m3"] * 4, dtype=object)
    split = mission_holdout_split(missions, "m3", calibration_fraction=0.25, test_fraction=0.25)
    assert set(missions[split.test]) == {"m3"}
    assert "m3" not in set(missions[split.train])
    assert "m3" not in set(missions[split.calibration])


def test_event_groups_cannot_cross_split_boundaries():
    split = chronological_split(12, calibration_fraction=0.25, test_fraction=0.25)
    groups = np.array(["a"] * 6 + ["b"] * 3 + ["c"] * 3, dtype=object)
    assert_disjoint_groups(split, groups)
    crossing = groups.copy()
    crossing[split.calibration[0]] = "a"
    with pytest.raises(ValueError, match="crosses split boundaries"):
        assert_disjoint_groups(split, crossing)


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


def test_windows_skip_context_that_crosses_large_time_gaps():
    timestamps = np.array(
        [
            "2020-01-01T00:00:00",
            "2020-01-01T00:00:01",
            "2020-01-01T00:00:02",
            "2020-01-01T00:10:00",
            "2020-01-01T00:10:01",
            "2020-01-01T00:10:02",
            "2020-01-01T00:10:03",
        ],
        dtype="datetime64[s]",
    )
    windows = make_windows(
        np.arange(len(timestamps), dtype=float),
        timestamps,
        lookback=3,
        horizon=1,
        max_gap_seconds=10,
    )
    assert windows.origins.tolist() == [5]


def test_windows_preserve_target_event_identity():
    values = np.arange(20, dtype=float)
    event_ids = np.full(20, None, dtype=object)
    event_ids[12] = "event-a"
    windows = make_windows(
        values,
        np.arange(20),
        event_starts=[12],
        event_ids=event_ids,
        lookback=4,
        horizon=3,
    )
    matching = windows.target_event_ids[windows.y == 1]
    assert matching.tolist() == ["event-a", "event-a", "event-a"]


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


def test_physical_exposure_censors_large_gaps():
    timestamps = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:01:00", "2020-01-01T01:01:00"],
        dtype="datetime64[ns]",
    )
    assert monitored_exposure_seconds(timestamps, max_gap_seconds=120) == 60.0
    result = event_warning_metrics(
        scores=np.array([2.0, 0.0, 0.0]),
        event_starts=np.array([2]),
        origins=np.array([0, 1, 2]),
        threshold=1.0,
        horizon=3,
        timestamps=timestamps,
        cooldown_seconds=30,
        max_gap_seconds=120,
    )
    assert result["exposure_seconds"] == 60.0
    assert result["false_alarms_per_day"] == 0.0


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
    assert result["results"][0]["logistic_warning"] is not None
    assert result["results"][0]["calibration_positive_windows"] == 0


def test_csv_runner_can_derive_horizon_from_physical_seconds(tmp_path):
    n = 200
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="5s", tz="UTC"),
            "asset_id": "a",
            "channel_id": "c",
            "value": np.sin(np.arange(n) / 4),
            "anomaly_start": False,
        }
    )
    frame.loc[120, "anomaly_start"] = True
    path = tmp_path / "physical_horizon.csv"
    frame.to_csv(path, index=False)
    result = run_csv_benchmark(path, lookback=8, horizon_seconds=30)
    stream = result["results"][0]
    assert stream["horizon_steps"] == 6
    assert stream["effective_horizon_seconds"] == 30.0


def test_csv_runner_can_hold_out_a_complete_mission(tmp_path):
    n = 240
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC"),
            "asset_id": "a",
            "channel_id": "c",
            "mission_id": ["m1"] * 120 + ["m2"] * 120,
            "value": np.sin(np.arange(n) / 4),
            "anomaly_start": False,
        }
    )
    frame.loc[[40, 180], "anomaly_start"] = True
    path = tmp_path / "multi_mission.csv"
    frame.to_csv(path, index=False)
    result = run_csv_benchmark(path, lookback=8, horizon=4, held_out_mission="m2")
    held_out = result["results"][0]
    assert held_out["held_out_mission"] == "m2"
    assert held_out["train_windows"] > 0
    assert held_out["calibration_windows"] > 0
    assert held_out["test_windows"] > 0
