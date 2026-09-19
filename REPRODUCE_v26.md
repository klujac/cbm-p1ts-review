# Reproduction Guide for the Revised Results (v26)

This guide reproduces every number in the revised manuscript. The
benchmark driver `src/benchmark_driver_v25_3.py` is **byte-for-byte the
one used for the original submission**; each revision experiment is a
thin runner in `src/v26/` that imports it and changes exactly one
factor. The driver's SHA-256 is recorded in every run manifest, so a
reader can verify that no hidden modification was made.

    sha256sum src/benchmark_driver_v25_3.py
    # c2cf... (see results/v26/manifests/*.json, field driver_sha256)

## 0. What can be checked without a GPU

All fold-level outputs are included, so every table of the paper can be
recomputed from `results/v26/*.csv` in seconds:

    python3 src/v26/verify_paper_numbers.py

This script recomputes the main comparison table, the ablation table,
the large-scale table and the statistical tests, and prints each value
next to the one printed in the paper.

## 1. Environment

Identical to the original submission (Section "Hardware and software"
of the paper):

    python3 -m venv .venv && source .venv/bin/activate
    export PYTHONNOUSERSITE=1
    pip install --no-cache-dir -r env/requirements_frozen_v25_1.txt
    export LDRV25_TABPFN_MODEL_PATH=$(pwd)/tabpfn_models/tabpfn-v3-classifier-v3_default.ckpt
    export LDRV25_REQUIRE_TABPFN_TOKEN=0
    export DRIVER_DIR=$(pwd)/src

TabICLv2 requires a separate environment, because it needs a newer
PyTorch than the pinned one:

    python3 -m venv .venv_tabicl && .venv_tabicl/bin/pip install tabicl torch pandas scikit-learn

The three large datasets of the stability experiment and the six
datasets of the large-scale track are downloaded from OpenML by
`src/v26/fetch_large_datasets.py` into `DATASETS_LARGE/`.

## 2. Main run: concept dropout removed (Section 3.4, Table 4)

Search space restricted to `concept_dropout_p = 0`; all other protocol
elements unchanged. Only the 841 folds in which the original tuner had
selected `p_c > 0` need re-running; the remaining 359 folds already
satisfied the constraint and are reused from `results/comparison.csv`.

    cd src/v26 && bash run_v26.sh main_pc0 tasks/tasks_main_pc0.csv 8

    Cost: 204 GPU task-hours, about 35 h wall-clock on 2x H200 with
    4 processes per GPU. Resumable: re-running the same command skips
    finished tasks.
    Output: results_v26/main_pc0/*.json, one per task, each recording
    the number of Optuna trials and the set of p_c values tried.

## 3. Ablations (Section 4.3, Table 7)

Each variant freezes the hyperparameters, restarts and seeds of the
corresponding full model and changes exactly one factor. The finished
Optuna study is rebuilt in memory from the stored parameters, so only
the final training runs.

    bash run_v26_fh.sh fh_noresid  tasks/tasks_all1200.csv 8   # S = 0
    bash run_v26_fh.sh fh_norules  tasks/tasks_all1200.csv 8   # W = 0
    bash run_v26_fh.sh fh_ls0      tasks/tasks_all1200.csv 8   # eps_ls = 0
    bash run_v26_fh.sh fh_nokl     tasks/tasks_ke600.csv    8  # KL term removed
    bash run_v26_fh.sh fh_fixedppl tasks/tasks_all1200.csv 8   # c = PPL targets
    bash run_v26_fh.sh fh_clamp    tasks/tasks_main_pc0.csv 8  # clamped dropout
    bash run_v26_fh.sh fh_ref_v25  tasks/tasks_main_pc0.csv 8  # determinism check

    Cost: about 4 h wall-clock per variant on 2x H200.

`fh_ref_v25` is the control: it retrains with the archived v25
hyperparameters unchanged and must reproduce the archived metrics
exactly. On diabetes, seed 42, fold 0 it returns accuracy 0.7272727 and
AUC-ROC 0.780, the values in `results/comparison.csv`.

## 4. Tuned baselines (Section 4.1, Tables 3 and 4)

Six baselines receive the same 40-trial Optuna budget, inner split and
selection criterion as CBM-P1TS, with their own default configuration
evaluated as an additional candidate.

    bash run_v26_cpu.sh tasks/tasks_tuned6.csv 16

    Cost: about 470 CPU-hours, roughly 30 h wall-clock on 16 processes.
    CPU only; can run concurrently with the GPU experiments.

## 5. TabICLv2 (Section 4.1, Table 4)

Evaluated in context, without tuning, on exactly the same folds. The
fold split depends only on the labels and the seed, so
`StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)`
reproduces the driver's folds.

    .venv_tabicl/bin/python src/v26/run_tabicl.py src/v26/tasks/tasks_tabicl.csv 0 1

    Cost: about 30 min on one GPU for all 600 tasks.

## 6. Large-scale track (Section 4.4, Table 8)

    python3 src/v26/fetch_large_datasets.py
    cd src/v26
    bash run_v26_large.sh CBM     tasks/tasks_large.csv  8
    bash run_v26_large.sh CBM_KE  tasks/tasks_large5.csv 8
    bash run_v26_large.sh TabPFN  tasks/tasks_large5.csv 8

    Documented deviation: 15 tuning trials instead of 40 and five folds
    instead of ten. Cost: about 30 h wall-clock for CBM (MiniBooNE alone
    takes 6 h per fold), 10 h for CBM-KE, under 1 h for TabPFN-3.

## 7. Subsampling stability (Section 4.5)

    cd src/v26
    bash run_stability.sh CBM_KE   tasks/tasks_stab.csv 4
    bash run_stability.sh TabPFN   tasks/tasks_stab.csv 4
    bash run_stability.sh CatBoost tasks/tasks_stab.csv 4

    Cost: about 17 h wall-clock for CBM-KE on one GPU, minutes for the
    other two.

## 8. Auxiliary checks

    python3 src/v26/test_eq10.py
    # Equation (10) against the PyTorch implementation; agreement to 3e-16

    python3 src/v26/priors_check.py
    # Class proportions after subsampling vs the full OpenML releases,
    # and the imbalance ratios used in Proposition 10

    python3 src/v26/hparams_times_v25.py <path to v25 run>
    # Distribution of the selected hyperparameters, including the share
    # of folds with p_c > 0 (69.2% CBM, 71.0% CBM-KE)

    python3 src/v26/replay_v25.py <path to v25 run> 0 1
    # Reconstructs archived models without training and computes the
    # concept-fidelity, purity and margin-attribution diagnostics of
    # Sections 4.3 and 4.6; requires the archived restart checkpoints,
    # which are not redistributed (see README)

## 9. Total cost

    GPU:  about 440 task-hours (roughly 4 days on 2x H200)
    CPU:  about 470 task-hours (roughly 30 h on 16 processes)
    Tasks: 21,242 fold-level tasks, none of which failed

Every run writes a manifest with the protocol hash, the driver SHA-256,
the runner SHA-256 and the library versions; copies are in
`results/v26/manifests/`.
