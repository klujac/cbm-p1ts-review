# CBM-P1TS — Code and Results for Peer Review

**Paper:** Auditable distillation of tabular foundation models into
percentile-based fuzzy rules under a concept bottleneck
**Submitted to:** Information Sciences (Elsevier)
**Protocol hash:** 9e2fd6f6c51d2e9150021e0e2ef5e8c539775437f6c81a0950760b7b4987819e

This repository is provided for peer review.
It contains the complete source code and pre-computed results
sufficient to verify every quantitative claim in the paper.

## Verify key results instantly (no GPU needed)

    python3 -c "
    import pandas as pd
    df = pd.read_csv('results/comparison.csv')
    ds = df.groupby(['Algorithm','Dataset'])['AUC_ROC'].mean()
    print(ds.groupby('Algorithm').mean().sort_values(ascending=False))
    "
    # CBM-KE should appear at position 5, value ~0.935

## Files

    src/benchmark_driver_v25_3.py   - Main benchmark driver (Tables 5-8)
    src/cacp_compat.py              - Reporting module (Wilcoxon, Holm, CSV)
    src/rule_gen_v25_1.py           - Diabetes IF-THEN rule generator
    src/xgboost_spawn_smoke_test.py - Pre-run sanity check
    results/comparison.csv          - Raw fold-level results (9600 rows)
    results/comparison_result.csv   - Aggregated dataset-level means
    results/wilcoxon_holm/          - Holm-adjusted p-values (Tables 6-7)
    env/environment.json            - Software versions + TabPFN SHA-256
    env/requirements_frozen_v25_1.txt - Pinned dependencies
    REPRODUCE.md                    - Full reproduction instructions

## Protocol integrity

The immutable protocol hash recorded before any results were computed:
9e2fd6f6c51d2e9150021e0e2ef5e8c539775437f6c81a0950760b7b4987819e

This hash covers datasets, classifiers, seeds, folds, HPO parameters
and metrics. It was not modified after the run began.

## Hardware and software

    CPU: AMD EPYC 9354 (32 cores), RAM: 251 GiB
    GPU: 2 x NVIDIA H200 NVL (143 GiB VRAM each)
    OS:  Ubuntu 24.04 LTS, Python 3.12.3
    PyTorch 2.10.0+cu128, TabPFN 8.0.8
    TabPFN-3 checkpoint SHA-256: d0d865d5...ea18f3988

Full version list: env/requirements_frozen_v25_1.txt

## Licence

Code: MIT. TabPFN-3: TABPFN-3 License v1.0 (Prior Labs, 2026).
Academic use permitted; commercial use requires Prior Labs licence.
