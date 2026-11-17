"""Convert the public OPS-SAT-AD segment file to the benchmark schema.

The source labels are expert-annotated telemetry segments. This adapter marks
the first sample of each contiguous annotated anomaly segment as an operational
anomaly start; it does not claim physical degradation or failure onset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def prepare_ops_sat_ad(
    source: str | Path,
    output: str | Path,
    duplicate_timestamp_policy: str = "retain",
) -> dict[str, object]:
    if duplicate_timestamp_policy not in {"retain", "reject"}:
        raise ValueError("duplicate_timestamp_policy must be 'retain' or 'reject'")
    source_path = Path(source)
    output_path = Path(output)
    frame = pd.read_csv(source_path, parse_dates=["timestamp"])
    required = {"channel", "timestamp", "value", "label", "segment", "train"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OPS-SAT-AD file is missing columns: {sorted(missing)}")
    nulls = frame[list(required)].isna().sum()
    if int(nulls.sum()) > 0:
        details = {column: int(count) for column, count in nulls.items() if count}
        raise ValueError(f"OPS-SAT-AD required columns contain missing values: {details}")
    parsed_timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if parsed_timestamps.isna().any():
        raise ValueError(f"OPS-SAT-AD timestamp contains {int(parsed_timestamps.isna().sum())} invalid rows")
    frame["timestamp"] = parsed_timestamps

    numeric_values = pd.to_numeric(frame["value"], errors="coerce")
    invalid_values = numeric_values.isna()
    if invalid_values.any():
        raise ValueError(f"OPS-SAT-AD value contains {int(invalid_values.sum())} non-numeric rows")
    frame["value"] = numeric_values

    duplicate_groups = frame.groupby(["channel", "timestamp"], sort=False).size()
    duplicate_timestamp_groups = int((duplicate_groups > 1).sum())
    duplicate_timestamp_rows = int(duplicate_groups[duplicate_groups > 1].sum())
    duplicate_groups_with_different_values = int(
        (
            frame.groupby(["channel", "timestamp"], sort=False)["value"].nunique(dropna=False)
            > 1
        ).sum()
    )
    if duplicate_timestamp_policy == "reject" and duplicate_timestamp_groups:
        raise ValueError(
            "OPS-SAT-AD contains duplicate channel/timestamp rows; "
            "use duplicate_timestamp_policy='retain' only for sequence-level evaluation"
        )

    frame = frame.sort_values(["channel", "timestamp", "segment"], kind="stable").reset_index(drop=True)
    label = frame["label"].astype(str).str.lower()
    anomaly = label.eq("anomaly") | frame["label"].astype(str).str.contains("anomal", case=False, na=False)
    previous_anomaly = anomaly.groupby(frame["channel"], sort=False).shift(fill_value=False)
    previous_segment = frame.groupby("channel", sort=False)["segment"].shift()
    anomaly_start = anomaly & (~previous_anomaly | frame["segment"].ne(previous_segment))
    event_id = pd.Series(pd.NA, index=frame.index, dtype="string")
    event_id.loc[anomaly] = (
        "ops-sat-segment-"
        + frame.loc[anomaly, "channel"].astype(str)
        + "-"
        + frame.loc[anomaly, "segment"].astype(str)
    )

    prepared = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "asset_id": "ops-sat",
            "channel_id": frame["channel"].astype(str),
            "mission_id": "ops-sat",
            "value": frame["value"],
            "event_id": event_id,
            "anomaly_start": anomaly_start.astype(bool),
            "source_label": frame["label"].astype(str),
            "source_segment": frame["segment"].astype(str),
            "source_train_split": frame["train"].astype(str),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "source_rows": int(len(frame)),
        "output_rows": int(len(prepared)),
        "channels": int(prepared["channel_id"].nunique()),
        "anomaly_starts": int(anomaly_start.sum()),
        "anomaly_rows": int(anomaly.sum()),
        "duplicate_timestamp_policy": duplicate_timestamp_policy,
        "duplicate_timestamp_groups": duplicate_timestamp_groups,
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "duplicate_groups_with_different_values": duplicate_groups_with_different_values,
        "label_semantics": "expert-annotated telemetry segments; not physical failure onset",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--duplicate-timestamp-policy",
        choices=("retain", "reject"),
        default="retain",
        help="Retain deterministic source order for sequence benchmarks, or reject ambiguity.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_ops_sat_ad(
                args.source,
                args.output,
                duplicate_timestamp_policy=args.duplicate_timestamp_policy,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
