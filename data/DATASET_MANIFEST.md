# Local dataset manifest

This file records the public data used for local adapter validation. The raw
files belong under the ignored `data/downloads/` directory and must not be
committed to this repository.

## OPS-SAT-AD

- Source record: <https://doi.org/10.5281/zenodo.12588359>
- Author code: <https://github.com/kplabs-pl/OPS-SAT-AD>
- Downloaded files: `dataset.csv` and `segments.csv`
- Verified MD5 for `dataset.csv`: `c12fb83863afa4463dec07303a20bb14`
- Verified MD5 for `segments.csv`: `1f4b6ffd0359efac96a6698ff557e0f9`
- Local adapter: `scripts/prepare_ops_sat_ad.py`
- Label semantics: expert-annotated telemetry segments and anomaly labels;
  these are not physical degradation or failure-onset labels.
- Evaluation status: retrospective anomaly benchmark only; the source dataset
  uses selected segments and is not a mission-day exposure denominator.
- Duplicate timestamp policy: the adapter retains source order for sequence-level
  reproduction and reports 22,114 duplicate `(channel, timestamp)` groups in
  the verified download; these rows have differing values, so physical-time
  alert rates require a separate packet/sequence resolution policy.
- Distribution status: the Zenodo API metadata did not expose an explicit
  license field at the time of local verification. Confirm the current record
  terms before redistributing the files or derived data.

### Local benchmark finding

Using the retained source order, a 30-second requested horizon, 60-second
purge/cooldown, and a 30-second maximum-gap censoring rule produced a diagnostic
rather than a successful warning result:

- 9 channels were evaluated from 303,493 rows and 434 annotated segment starts.
- The persistence baseline detected 6 of 59 test events (10.2% pooled recall),
  with 1.8% pooled alert precision.
- The supervised logistic warning baseline detected 4 of 51 test events in
  streams with both training classes (7.8% pooled recall, 3.3% pooled alert
  precision); its reported ECE was 0.368, and the
  probabilities were explicitly raw/uncalibrated because the event-free
  threshold-calibration set had no positive examples.
- The download contains 22,114 duplicate channel/timestamp groups with
  differing values. The benchmark therefore marks physical-time metrics as
  provisional and does not treat this as a clean mission-exposure study.

This does not support a 90–95% early-warning claim. It indicates that the
selected OPS-SAT-AD segments are suitable for adapter and leakage tests, but
not sufficient evidence for physical degradation prediction. Reaching a high
target requires a dataset with verified failure/precursor timing, a fixed
physical horizon, duplicate/packet resolution, domain-held-out confirmation,
and an independently frozen test set.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\prepare_ops_sat_ad.py `
  --source .\data\downloads\ops_sat_ad\segments.csv `
  --output .\data\downloads\ops_sat_ad\ops_sat_long.csv
python .\scripts\run_csv_benchmark.py `
  .\data\downloads\ops_sat_ad\ops_sat_long.csv `
  --horizon-seconds 30 --purge-seconds 60 --cooldown-seconds 60 --max-gap-seconds 30 `
  --output .\data\downloads\ops_sat_ad\benchmark.json
```

The resulting score is a reproducibility and adapter check. It must not be
reported as a physical-failure warning result or as mission-level precision.
