# Public release checklist

Use this checklist before making the repository public or attaching a result
to a paper, proposal, or mission claim.

## Release blockers

- [ ] Choose and add an explicit software license.
- [ ] Confirm every included file is owned by the project or redistributable.
- [ ] Confirm no telemetry, raw mission files, credentials, tokens, private
      paths, or operational endpoints are committed.
- [ ] Confirm the README and research card say that the system is not
      flight-qualified and that the demo is synthetic.
- [ ] Confirm every reported number has a source, dataset version, split,
      label definition, and reproducible command.
- [ ] Confirm restricted datasets are referenced by instructions and citations,
      not copied into the repository.

## Scientific integrity

- [ ] Freeze the primary metric, horizon, alert budget, and split before
      confirmation evaluation.
- [ ] Keep training, calibration, and confirmation data separate.
- [ ] Use event-separated and chronological splits where the claim requires
      warning or transfer.
- [ ] Report uncertainty, independent support, missingness strata, false-alert
      exposure, and negative results.
- [ ] Do not convert selected-fragment benchmark scores into mission-day
      reliability claims.
- [ ] Review all citations and verify that they support the exact sentence.

## Reproduction

- [ ] Run the documented smoke test from a clean environment.
- [ ] Run the complete test suite.
- [ ] Record dependency versions and platform assumptions.
- [ ] Include the data schema, adapter instructions, preprocessing, seed, and
      output format.
- [ ] Check that generated artifacts are either reproducible or clearly labeled
      as local outputs.

## Security and privacy

- [ ] Run a secret scan and inspect the full diff, including hidden files.
- [ ] Keep `.env`, credentials, caches, raw downloads, and local research
      vaults outside the public repository.
- [ ] Verify the code has no network call, command channel, or spacecraft
      integration unless separately reviewed and explicitly intended.
- [ ] Add a private vulnerability-reporting path before advertising the project
      as production software.

## What this repository may claim now

It may claim to provide an evaluation-first spacecraft telemetry warning
benchmark scaffold, synthetic smoke tests, and a documented plan for public
data experiments. It may not claim a validated 90–95% warning system, flight
readiness, safety assurance, or generalization to unseen missions.
