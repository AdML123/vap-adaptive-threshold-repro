# Adaptive Per-Bin Thresholds for VAP

This repository contains the disclosure-safe code, figures, aggregate tables, and
audit manifests for the Switchboard voice activity projection label-calibration
study. The main result is a per-bin label-loss audit followed by frozen-backbone
output-head retraining with uniform, fixed per-bin, and activity-conditioned
thresholds.

## What is included

- experiments/: label generation, calibration, training, evaluation, aggregation,
  bootstrap, and figure scripts.
- results/tables/: the valid three-seed head-only Table II aggregate, formal
  integrity audit, paired forward/reverse bootstrap, and QA manifest.
- results/figures/: publication-ready PDF and SVG exports.
- tests/: regression tests for label definitions, full-data training guards,
  and evaluation semantics.
- environments/requirements.txt: Python dependencies used by the project.

## Data boundary

The Switchboard audio, word alignments, DAMSL annotations, and pretrained
checkpoint are not redistributed. They are external inputs that must be obtained
under the applicable provider terms. The configuration in
analysis/config_formal_recovery.json uses relative placeholders for those
resources; replace them with local paths before running an analysis.

The reported VAD is an alignment-derived proxy VAD. The original private VAP VAD
pipeline was unavailable, so results should not be presented as a byte-for-byte
reproduction of the original VAP training labels.

## Verification

Run the tests from this directory:

    python -m pytest tests/test_vap_adaptive.py -q

The formal integrity manifest records 18/18 valid full-data head-only runs and
the test evaluation covers 232 conversation-disjoint test conversations. The
paired direction result uses 2,000 conversation-cluster bootstrap replicates.

The extended floor re-selection and 20-epoch full-model fine-tuning runs were
not completed and are intentionally not represented as completed results.

## Rebuilding figures

The included aggregate source artifacts are sufficient to regenerate the
disclosure-safe figures without raw audio:

    python experiments/vap_adaptive/make_figures.py --repo . --output results/figures

The included figure files are derived outputs from the validated run and do not
require the restricted source data for inspection.

## Citation and release

Please cite the repository using CITATION.cff. The current public release is
tagged v0.1.1. A Zenodo DOI will be added to
this file and to the README only after the GitHub release has been archived by
Zenodo. No DOI is claimed before that archive step.

## License

Code is released under the license in LICENSE. Redistribution of the
Switchboard source data remains governed by its provider and is outside this
repository license.
