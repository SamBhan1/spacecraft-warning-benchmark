import pandas as pd

from scripts.prepare_ops_sat_ad import prepare_ops_sat_ad


def test_ops_sat_adapter_marks_one_start_per_anomaly_segment(tmp_path):
    source = tmp_path / "segments.csv"
    output = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "channel": ["c", "c", "c", "c", "c"],
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="s"),
            "value": [0.0, 1.0, 1.1, 0.0, 2.0],
            "label": ["a2", "anomaly", "anomaly", "a3", "anomaly"],
            "segment": [1, 2, 2, 3, 4],
            "train": [1, 1, 1, 0, 0],
        }
    ).to_csv(source, index=False)

    summary = prepare_ops_sat_ad(source, output)
    prepared = pd.read_csv(output)

    assert summary["anomaly_starts"] == 2
    assert prepared["anomaly_start"].tolist() == [False, True, False, False, True]
    assert prepared.loc[1, "event_id"] == "ops-sat-segment-c-2"
    assert prepared.loc[2, "event_id"] == "ops-sat-segment-c-2"
    assert prepared.loc[3, "event_id"] != prepared.loc[1, "event_id"]


def test_ops_sat_adapter_rejects_invalid_values_and_duplicate_policy(tmp_path):
    source = tmp_path / "segments.csv"
    output = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "channel": ["c", "c"],
            "timestamp": ["2020-01-01T00:00:00Z"] * 2,
            "value": ["not-a-number", 1.0],
            "label": ["a2", "a2"],
            "segment": [1, 1],
            "train": [1, 1],
        }
    ).to_csv(source, index=False)

    try:
        prepare_ops_sat_ad(source, output)
    except ValueError as error:
        assert "non-numeric" in str(error)
    else:
        raise AssertionError("invalid telemetry values must be rejected")

    valid = pd.read_csv(source)
    valid["value"] = [1.0, 2.0]
    valid.to_csv(source, index=False)
    try:
        prepare_ops_sat_ad(source, output, duplicate_timestamp_policy="reject")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate timestamps must be rejectable")
