# Research card

## Status

This repository is a research prototype and benchmark scaffold, approximately
TRL 2–3. It is not flight-qualified, safety-critical, clinical, or an
operational spacecraft-control system.

## Research question

Can a compact telemetry model produce a calibrated warning for a defined anomaly
event within a fixed horizon while remaining reliable under unseen modes,
missingness, spacecraft, and missions?

## Intended use

- Reproducible experiments on public or authorized spacecraft telemetry.
- Comparison of simple baselines, calibrated warning rules, and uncertainty
  methods.
- Testing split, label, timing, missingness, and alert-budget assumptions.

## Out of scope

- Commanding, controlling, or connecting to a spacecraft.
- Safety, mission assurance, maintenance, or autonomous operations decisions.
- Claims that an anomaly is a physical failure or that a failure will occur.
- A universal 90–95% performance guarantee.

## Data policy

The included demo is synthetic and must not be described as NASA or flight
data. Do not commit downloaded telemetry, credentials, restricted files, or
raw source archives. For real-data experiments, publish only data that the
license permits, plus the dataset citation, access instructions, version or
release identifier, checksum when allowed, preprocessing description, and
known restrictions.

## Claim policy

Every result must identify its dataset, label definition, forecast horizon,
prediction-time information boundary, split, exposure denominator, calibration
procedure, uncertainty interval, and limitations. Report negative and failed
experiments. Treat the 90–95% target as a preregistered hypothesis, never as an
achieved result unless a frozen held-out evaluation supports that exact claim.

## Reproducibility contract

The release should include the exact environment, commands, random seeds,
dataset adapter, split manifest, preprocessing rules, baseline settings,
metrics, confidence intervals, and generated artifacts. Confirmation data must
not be used to select models, thresholds, features, horizons, or displayed
operating points.

## AI-assisted development disclosure

If AI tools materially contributed to code, analysis, or documentation, record
the tool and contribution in the release notes and independently review the
result. Do not treat generated text or citations as evidence without checking
the underlying source.
