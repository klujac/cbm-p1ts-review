# CBM-P1TS — Code and Results for Peer Review

**Paper:** Auditable distillation of tabular foundation models into
percentile-based fuzzy rules under a concept bottleneck
**Submitted to:** Information Sciences (Elsevier)
**Status:** revised version (v26), September 2026

This repository is provided for peer review. It contains the complete
source code and the pre-computed fold-level results needed to verify
every quantitative claim in the paper, for both the original submission
and the revision.

## Verify every number in the paper (no GPU, no training, seconds)

    pip install pandas numpy scipy
    python3 src/v26/verify_paper_numbers.py

The script recomputes the main comparison table, the ablation table, the
large-scale table, the collapse threshold of Proposition 10 and the
statistical tests from the released fold-level results, and prints each
computed value next to the value printed in the paper. All 44 checks
should pass.

## What the revision changed

The benchmark driver `src/benchmark_driver_v25_3.py` is **unchanged**
from the original submission. Every revision experiment is a thin runner
in `src/v26/` that imports that driver and changes exactly one factor,
so the comparisons are controlled by construction. Each run records the
driver's SHA-256 in its manifest, which lets a reader verify that no
hidden modification was made.

    sha256sum src/benchmark_driver_v25_3.py
    # compare with driver_sha256 in results/v26/manifests/*.json

New in the revision: concept dropout removed (841 folds re-run), six
one-factor ablations (6600 tasks), six baselines tuned with the same
budget as our model (3600 tasks), TabICLv2 (600 tasks), a large-scale
track up to 130,064 rows (90 tasks) and a subsampling-stability
experiment (270 tasks). In total 21,242 fold-level tasks, none of which
failed.

## Files

    src/benchmark_driver_v25_3.py    Benchmark driver (unchanged since submission)
    src/cacp_compat.py               Reporting module (Wilcoxon, Holm, CSV)
    src/rule_gen_v25_1.py            Diabetes IF-THEN rule generator
    src/xgboost_spawn_smoke_test.py  Pre-run sanity check

    src/v26/run_v26.py               Main run with concept dropout removed
    src/v26/run_v26_fh.py            One-factor ablations (frozen hyper-parameters)
    src/v26/tuned_baselines.py       Tuning wrapper for the six baselines
    src/v26/run_v26_cpu.py           Runner for the tuned baselines
    src/v26/run_tabicl.py            TabICLv2 on the same folds
    src/v26/run_v26_large.py         Large-scale track
    src/v26/run_stability.py         Subsampling-stability experiment
    src/v26/replay_v25.py            Reconstructs archived models; concept
                                     fidelity, purity and margin diagnostics
    src/v26/test_eq10.py             Unit test of Equation (10)
    src/v26/priors_check.py          Class priors after subsampling; imbalance ratios
    src/v26/hparams_times_v25.py     Selected hyper-parameters and task times
    src/v26/fetch_large_datasets.py  Downloads the six large datasets from OpenML
    src/v26/verify_paper_numbers.py  Recomputes every published number
    src/v26/*.sh                     Launchers (sharded, resumable)
    src/v26/tasks/*.csv              Task lists defining each run exactly

    results/comparison.csv           Original fold-level results (9600 rows)
    results/comparison_result.csv    Original dataset-level means
    results/wilcoxon_holm/           Original Holm-adjusted p-values
    results/v26/*.csv                Fold-level results of every revision run
    results/v26/manifests/*.json     Run manifests with protocol and code hashes

    env/environment.json             Software versions + TabPFN SHA-256
    env/requirements_frozen_v25_1.txt Pinned dependencies
    REPRODUCE.md                     Original reproduction guide
    REPRODUCE_v26.md                 Reproduction guide for the revision
    CHANGELOG.md                     Version history

## Protocol integrity

Original run:

    9e2fd6f6c51d2e9150021e0e2ef5e8c539775437f6c81a0950760b7b4987819e

Each revision run has its own protocol hash, derived from the variant
name and the driver's SHA-256 and recorded in
`results/v26/manifests/`. No protocol parameter was modified after a
run began.

## What is not redistributed

The TabPFN-3 checkpoint, trained student weights and the per-task
restart checkpoints (about 2.3 GB) are not included. The checkpoint is
obtained from Hugging Face and verified by SHA-256 as described in
REPRODUCE.md; the restart checkpoints are only needed by
`replay_v25.py`, and every number it produces is included in
`results/v26/`.

## Hardware and software

    CPU: AMD EPYC 9354 (32 cores), RAM: 251 GiB
    GPU: 2 x NVIDIA H200 NVL (143 GiB VRAM each)
    OS:  Ubuntu 24.04 LTS, Python 3.12.3
    PyTorch 2.10.0+cu128, TabPFN 8.0.8
    TabPFN-3 checkpoint SHA-256: d0d865d5...ea18f3988
    TabICLv2 requires a separate environment (see REPRODUCE_v26.md)

## Licence

Code: MIT. TabPFN-3: TABPFN-3 License v1.0 (Prior Labs, 2026).
Academic use permitted; commercial use requires a Prior Labs licence.
