"""Run the baseline benchmark on a labeled telemetry CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spacecraft_warning.benchmark import run_csv_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cooldown-steps", type=int, default=6)
    parser.add_argument("--purge-seconds", type=float, default=None)
    parser.add_argument("--cooldown-seconds", type=float, default=None)
    parser.add_argument("--max-gap-seconds", type=float, default=None)
    parser.add_argument("--held-out-mission", default=None)
    args = parser.parse_args()
    result = run_csv_benchmark(
        args.csv,
        lookback=args.lookback,
        horizon=args.horizon,
        alpha=args.alpha,
        cooldown_steps=args.cooldown_steps,
        purge_seconds=args.purge_seconds,
        cooldown_seconds=args.cooldown_seconds,
        max_gap_seconds=args.max_gap_seconds,
        held_out_mission=args.held_out_mission,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
