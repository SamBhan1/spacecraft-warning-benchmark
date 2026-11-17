# Spacecraft Warning Benchmark

This is the first implementation of the narrowed research project:

> Can a compact telemetry model produce a calibrated warning that a defined
> anomaly event will begin within a fixed horizon, while remaining reliable
> under unseen modes, missingness, spacecraft, and missions?

The package is deliberately evaluation-first. It provides chronological and
mission-held-out train/calibration/test splits, physical-time embargoes,
event-aware windows, persistence and logistic baselines, conformal forecast
intervals, mode-conditional calibration, and event-level warning metrics with
physical exposure accounting.

## Publication status and safety boundary

This is a research prototype and benchmark scaffold, approximately TRL 2–3.
It is not flight-qualified, safety-critical, or connected to spacecraft
command systems. The included demo is synthetic; it is not NASA flight data
and its metrics are not scientific performance claims. Do not infer physical
failure probability, remaining useful life, or mission reliability from an
anomaly interval alone.

Read [RESEARCH_CARD.md](RESEARCH_CARD.md) for intended use, claim boundaries,
data policy, and reproducibility requirements. Use
[PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md) before publishing
the repository or reporting a result.

The included demo data are synthetic and are never presented as NASA flight
data. To use real data, convert them to the schema in `data/SCHEMA.md` and keep
the source license and dataset card alongside the downloaded files.

## Quick start

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\run_demo.py --output .\artifacts\demo.json
pytest -q .\tests
```

If package installation is desired and the machine already has the build
dependencies cached, `python -m pip install --no-build-isolation -e .` is
supported. The `PYTHONPATH` route is the offline-safe
default.

The demo prints a JSON summary containing the chronological split, event counts,
warning metrics, and conformal interval coverage. Its generated anomaly labels
are synthetic; the result is a smoke test of the benchmark machinery, not a
scientific performance claim.

## Evaluation rules

1. Fit scaling, imputation, feature selection, model parameters, thresholds,
   and conformal quantiles using training/calibration data only.
2. Purge overlapping windows at split boundaries and keep windows from one event
   or contiguous episode in one split.
3. Freeze the confirmation set before selecting horizons, thresholds, or model
   features.
4. Report event-level precision/recall, missed events, false alarms per day,
   warning lead time, calibration, interval width, and worst-group behavior.
5. Treat command and mode context as ablations. A gain must survive command
   shuffling, command time shifts, availability masking, and held-out modes.
6. Never describe interval noncoverage or anomaly-onset warning as a physical
   failure probability without mission-specific labels and validation.

For irregular telemetry, pass `purge_seconds` and `cooldown_seconds` to the CSV
runner. Use `--horizon-seconds` to derive a channel-specific row horizon from
the median cadence; otherwise `--horizon` is a row count and has different
physical meaning across channels. The resulting `false_alarms_per_day` uses
observed wall-clock exposure; pass `max_gap_seconds` to censor communication
gaps instead of counting them as nominal monitoring time. Use
`mission_holdout_split` for a complete-domain confirmation test and
`assert_disjoint_groups` to fail fast when event or episode identities cross
split boundaries.

Example physical-time and mission-held-out run:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\run_csv_benchmark.py .\data\mission.csv `
  --horizon-seconds 1800 --purge-seconds 3600 --cooldown-seconds 1800 --max-gap-seconds 300 `
  --held-out-mission mission-2 --output .\artifacts\mission-2.json
```

The CSV runner reports false alarms per 1,000 rows and records median cadence
and irregular-cadence status. Its `logistic_calibration` field is a raw
uncalibrated diagnostic unless `logistic_probability_status` says otherwise;
event-free threshold calibration is intentionally reported separately. Convert
row-based rates to mission-day rates only after a dataset-specific cadence,
duplicate-timestamp, and gap policy has been reviewed.

## Data and licensing

This repository does not redistribute restricted spacecraft telemetry. Before
adding a dataset or adapter, verify its license and cite the original release.
Keep raw downloads and private research files outside the public repository.
Choose and add an explicit software license before public release.

## Layout

- `src/spacecraft_warning/` — reusable benchmark code.
- `data/SCHEMA.md` — input contract for real telemetry adapters, including
  mission and event provenance.
- `configs/default.json` — conservative default evaluation settings.
- `scripts/run_demo.py` — executable demo wrapper.
- `scripts/run_csv_benchmark.py` — adapter-free runner for labeled long-form CSVs.
- `tests/` — tests for split isolation, calibration, and metrics.
- `artifacts/demo.json` — checked-in synthetic smoke-test output; write local
  experiment results under `artifacts/local/`, which is ignored by Git.
