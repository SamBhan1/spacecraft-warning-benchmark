# Spacecraft Warning Benchmark

This is the first implementation of the narrowed research project:

> Can a compact telemetry model produce a calibrated warning that a defined
> anomaly event will begin within a fixed horizon, while remaining reliable
> under unseen modes, missingness, spacecraft, and missions?

The package is deliberately evaluation-first. It provides chronological
train/calibration/test splits, event-safe window construction, persistence and
logistic baselines, conformal forecast intervals, mode-conditional calibration,
and event-level warning metrics.

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
$env:PYTHONPATH = (Resolve-Path .\spacecraft_warning\src).Path
python .\spacecraft_warning\scripts\run_demo.py --output .\spacecraft_warning\artifacts\demo.json
pytest -q .\spacecraft_warning\tests
```

If package installation is desired and the machine already has the build
dependencies cached, `python -m pip install --no-build-isolation -e
.\spacecraft_warning` is supported. The `PYTHONPATH` route is the offline-safe
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

The CSV runner reports false alarms per 1,000 rows and records median cadence
and irregular-cadence status. Convert row-based rates to mission-day rates only
after a dataset-specific cadence and gap policy has been reviewed.

## Data and licensing

This repository does not redistribute restricted spacecraft telemetry. Before
adding a dataset or adapter, verify its license and cite the original release.
Keep raw downloads and private research files outside the public repository.
Choose and add an explicit software license before public release.

## Layout

- `src/spacecraft_warning/` — reusable benchmark code.
- `data/SCHEMA.md` — input contract for real telemetry adapters.
- `configs/default.json` — conservative default evaluation settings.
- `scripts/run_demo.py` — executable demo wrapper.
- `scripts/run_csv_benchmark.py` — adapter-free runner for labeled long-form CSVs.
- `tests/` — tests for split isolation, calibration, and metrics.
- `artifacts/` — local demo outputs; generated files are ignored by Git.
