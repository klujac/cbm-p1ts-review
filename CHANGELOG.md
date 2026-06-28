# Changelog

All notable changes to the CBM-P1TS benchmark code. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/). The frozen results
reported in the manuscript were produced by **v25.3**; the immutable protocol
hash is recorded in `protocol.json` and the pinned dependency manifest in
`environment.json`.

## [v25.3]
### Fixed
- Every SQLite connection is explicitly closed; atomic-result recovery reuses
  a single connection per scan.
- Added a file-descriptor regression self-test and a master FD counter.

The scientific protocol is unchanged from v25.

## [v25]
### Added
- Crash-safe, resumable execution: a shared SQLite task queue for GPU (H200)
  and CPU workers, with atomic task results and lease/heartbeat recovery after
  worker death.
- Automatic GPU-memory waiting with exponential-backoff retry.
- Persistent Optuna studies per `(fold, model, seed)`; restart and per-epoch
  checkpoints for final CBM training.
- Supervisor that restarts workers and balances GPU tasks; resume is the
  default, so completed work is never recomputed.
### Changed
- Strict inner-split preprocessing and class weights.
- Deterministic (serialised) GPU RNG by default for publication runs.
- Explicit TabPFN `model_path` + SHA-256 instead of cache-file discovery.
### Fixed
- Metrics and HPO no longer use a numerical zero as an error sentinel.
- Failed checkpoint rows are retried rather than resumed as complete.
- The concept count is lower-bounded so that `2^M >= K` (fail-fast otherwise).

## [v23]
### Changed
- The Optuna objective is evaluated **after** rule pruning, so the rule budget
  is an effective hyperparameter.
- PPL percentiles and mRMR concept selection are fitted on the inner-train
  split only and applied to the inner-validation split.
- Restart/trial seeds are set before model construction, so they govern weight
  initialisation.
### Fixed
- AUC-ROC returns `NaN` when class probabilities are unavailable.
- Failed folds are recorded as `NaN` with `Status=FAILED` (not `0.0`).
### Added
- Concept-count lower bound `M >= ceil(log2 K)`, guaranteeing `2^M >= K` for
  per-class rule coverage.
- TabPFN-3 version and checkpoint SHA-256 recorded in `environment.json`.

CBM and CBM-KE are recomputed with this version; the external baselines are
unaffected and may be reused from the previous run.

## [v22]
### Changed
- Main protocol lightened for tractable full runs (`trials=40`, `restarts=2`,
  `epochs-final=250`); `M` remains Optuna-tuned in `[2, min(12, d)]`.
- The `group2` preset uses `trials=40`, `epochs_hpo=80`, `patience=40`,
  `epochs_final=250`, `restarts=2`.
### Removed
- Ablation removed from the default path (still available via `--ablation`).

## [v21]
### Added
- Six baselines (gated imports): XGBoost, LightGBM, CatBoost, EBM, RuleFit, and
  FIGS; binary-only rule learners are wrapped in One-vs-Rest for multiclass.
- Ablation variants: `_noHedge`, `_noResid`, `_noLS`, `_noCalib`.
- Multi-seed runs (`--seeds 42,43,44`) with per-seed result directories and a
  `Seed` column.
- Holm-corrected pairwise Wilcoxon reports (vs CBM-KE and vs CBM) from
  dataset-level means.
- Environment manifest written to `environment.json`.
### Changed
- `--strict-validation` / `--seeds` / `--ablation` propagated to spawned GPU
  workers on the interactive path.
- SafeTabPFN default limits aligned with the protocol (`K<=10`, `n<=50k`;
  overridable via environment variables).

## [v20]
### Changed
- The inner-validation split is no longer used for final CBM weight updates.
- In KE mode, the TabPFN teacher is fitted only on the inner-fit subset.
- Corrected knowledge-distillation temperature softening.

## [v02]
### Changed
- Vectorised batching of GPU-resident tensors (replaces per-sample DataLoader
  indexing), the main source of the earlier CBM training slowdown.
- CPU classifiers executed in processes (joblib/loky) with automatic thread
  fallback.
- TF32 enabled on Hopper GPUs (disable via environment variable).
### Added
- Optional Optuna HPO wall-clock budget per fold.
- Checkpointing per `(dataset, classifier, fold)`; resume by default.
- Resource monitor (CPU/RAM/GPU utilisation, progress, ETA, stall warning).
- Global socket timeout on OpenML and TabPFN-checkpoint downloads.
- TabPFN pre-warm wrapped in a watchdog thread.
### Removed
- Dead post-`as_completed` timeout (replaced by stall detection).

**Reproducibility note.** Vectorised batching changes the random-batch
shuffling order (CUDA RNG instead of the DataLoader CPU RNG), so for a given
seed the numerical results may differ minimally from the pre-v02 code.
