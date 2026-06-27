"""
benchmark_driver_v25_3.py  —  spawn-safe resilient publication protocol
================================================================
V25.3 execution-engine repair (scientific protocol unchanged):
  * every SQLite connection is explicitly closed;
  * atomic-result recovery reuses one connection per scan;
  * an FD regression self-test and master FD counter are included.
V25 adds crash-safe, monotonic execution on top of V24:
  * shared SQLite task queue for both H200 workers and CPU workers;
  * atomic task results and lease/heartbeat recovery after worker death;
  * automatic GPU-memory waiting and retry with exponential backoff;
  * persistent Optuna studies per fold/model/seed;
  * restart and epoch checkpoints for final CBM training;
  * supervisor restarts workers and dynamically balances GPU tasks;
  * resume is the default: completed work is never recomputed.
  * strict inner-split preprocessing and class weights;
  * no numerical zero as an error sentinel in metrics or HPO;
  * failed checkpoint rows are retried, not silently resumed;
  * mathematically enforced 2^M >= K (fail-fast when infeasible);
  * serialized GPU RNG by default for deterministic publication runs;
  * explicit TabPFN model_path + SHA-256 (no cache-file guessing).
================================================================
Historical V23 fixes SIX review findings on top of V22 (all marked [V24-FIXn]):
  FIX1  Optuna objective now evaluates the model AFTER rule pruning, so
        the rule budget rb is a real hyperparameter (was a silent no-op).
  FIX2  PPL percentiles + mRMR concept selection are fitted on the inner
        TRAIN split only and APPLIED to the inner-val split; the inner-val
        labels no longer participate in concept selection.
  FIX3  _auc_roc returns NaN when probabilities are unavailable instead of
        silently substituting balanced accuracy under the AUC_ROC name.
  FIX4  failed folds are recorded as NaN + Status='FAILED'+Error, never as
        a spurious 0.0 that would drag dataset-level means down.
  FIX5  #concepts lower bound M >= ceil(log2 K) guarantees 2^M >= K, the
        combinatorial condition for per-class rule coverage.
  FIX6  restart/trial seeds are set BEFORE model construction (so they
        govern weight init); TabPFN-3 version + checkpoint SHA-256 are
        recorded in environment.json for an exactly pinned teacher.
NOTE: CBM and CBM-KE must be recomputed with this driver; baselines are
unaffected by FIX1/FIX2/FIX5/FIX6 and may be reused from the v22 run.
================================================================
Previous header (V22):
benchmark_driver_v22.py  —  cost-reduced protocol (paper-aligned)
================================================================
V22 changes vs V21 (all marked [V22]):
  * Main protocol lightened for tractable full runs WITHOUT touching M
    (M stays Optuna-tuned in [2, min(12,d)] — quality preserved):
        --trials   default 40   (was 50/120)
        --restarts default 2
        --epochs-final default 250
  * Ablation REMOVED from the default path (--ablation still available
    on demand for a reviewer rebuttal, but OFF by default).
  * 'group 2' preset (full 20-dataset run) now uses trials=40,
    epochs_hpo=80, patience=40, epochs_final=250, restarts=2.
Everything else identical to V21 (baselines T1, multi-seed T3,
Holm T4, environment manifest A5, strict validation).
Base: benchmark_driver_v21.py revision of
================================================================
V21 adds, on top of V20 (strict validation, corrected KD softening):
  [V21-T1] Six new baselines (gated imports): XGBoost, LightGBM,
           CatBoost, EBM (interpret), RuleFit and FIGS (imodels);
           binary-only rule learners are wrapped in One-vs-Rest for
           multiclass tasks.
  [V21-T2] --ablation adds four CBM_KE ablation variants:
           _noHedge (hedge search off), _noResid (residual concept-class
           connection zeroed+frozen), _noLS (label smoothing 0),
           _noCalib (temperature scaling off).
  [V21-T3] --seeds "42,43,44": full multi-seed runs; per-seed result
           directories + concatenated comparison.csv with a Seed column.
  [V21-T4] Holm-corrected pairwise Wilcoxon reports
           (wilcoxon_holm_vs_CBM_KE.csv / _vs_CBM.csv) computed from
           dataset-level means over folds and seeds.
  [V21-A5] _record_versions(): environment manifest written to
           environment.json (makes the manuscript's reproducibility
           sentence true).
  [V21-P4] --strict-validation / --seeds / --ablation propagated to
           spawned GPU workers also on the interactive-menu path.
  [V21-P5] SafeTabPFN default limits aligned with the paper protocol
           (K<=10, n<=50k); override via LDRV20_TABPFN_*_MAX.
Base: benchmark_driver_v20.py (which is a revision of
=======================================================================
Changes with respect to v01 (full list and justification: diagnoza.tex):
  [V02-1] _FastTensorLoader: vectorised batching of GPU-resident tensors
          (replaces DataLoader+TensorDataset, which indexed the data
          ONE SAMPLE AT A TIME in Python -> hundreds of micro-kernels
          per batch). The main source of the ~10-50x CBM training slowdown.
  [V02-2] CPU classifiers executed in PROCESSES (joblib/loky) instead of
          threads (GIL) — automatic fallback to threads.
  [V02-3] TF32 on H200 (LDRV2_TF32=0 to disable).
  [V02-4] Optuna HPO wall-clock budget per fold: --hpo-timeout S
          (0 = disabled).
  [V02-5] Checkpointing per (dataset, classifier, fold) + --no-resume.
          After a restart, already-computed tasks are skipped.
  [V02-6] Resource monitor: every --monitor-interval s prints to the
          console the CPU load, RAM, util/VRAM of every GPU, progress %,
          ETA, and a [STALL] warning when no task has finished
          for >30 min.
  [V02-7] Removed the dead timeout fut.result(timeout=...) after
          as_completed (it could never fire) — replaced by STALL detection.
  [V02-8] Global socket timeout (LDRV2_NET_TIMEOUT, default 180 s):
          network calls without a timeout (OpenML download, TabPFN
          checkpoint download) raise instead of hanging forever — the
          root cause of the observed 10-day startup hang in v01.
  [V02-9] [STALL] warning fires also when zero tasks have completed
          (startup-phase hang), not only mid-experiment.
  [V02-10] TabPFN pre-warm wrapped in a watchdog thread
          (LDRV2_PREWARM_TIMEOUT, default 600 s).
NOTE: [V02-1] changes the order of random batch shuffling (CUDA RNG instead
of the DataLoader CPU RNG), so numerical results may differ minimally from v01
for the same SEED. V20 further changes the default CBM/CBM_KE validation
protocol: the inner validation split is no longer used for final CBM weight
updates, and the TabPFN teacher in KE mode is fitted only on the inner-fit
subset. Use --no-strict-validation only for historical v16 reproduction.
"""
import os
import sys
import gc
import warnings
import random
import subprocess
import time
import datetime
import signal
import copy
import math
import hashlib
import inspect
import sqlite3
import json
import uuid
import traceback
import re
from pathlib import Path
from contextlib import contextmanager
from itertools import product as iproduct

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
import joblib

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import (
    LabelEncoder, RobustScaler,
    OneHotEncoder, StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from torch.utils.data import DataLoader, TensorDataset

# [V25-STANDALONE] Local reporting/statistics implementation.
# No external ``cacp`` package or ``cacp/`` directory is required.
from cacp_compat import (
    process_comparison_results, process_comparison_results_plots,
    process_comparison_result_winners, process_times, process_wilcoxon,
    dataset_info, classifier_info,
)


try:
    from xgboost import XGBClassifier
    _XGBOOST = True
except ImportError:
    _XGBOOST = False
    print("[WARN] xgboost unavailable. Installation: pip install xgboost")

# ESWA v20 reported experiments use TabPFN-only distillation.
# The disabled optional tree model development branch is disabled in this reviewer package
# regardless of whether the optional `disabled_optional_tree_model` library is installed.
_DISABLED_OPTIONAL_TREE_CLASSIFIER = None
_DISABLED_OPTIONAL_TREE_MODEL = False
print("[INFO] ESWA v20 benchmark protocol uses the reported classifier set only.")

try:
    from catboost import CatBoostClassifier
    _CATBOOST = True
except ImportError:
    _CATBOOST = False
    print("[WARN] catboost unavailable. Installation: pip install catboost")

try:
    from imodels import FIGSClassifier, RuleFitClassifier
    _IMODELS = True
except ImportError:
    _IMODELS = False
    print("[WARN] imodels unavailable. Installation: pip install imodels")

try:
    import corels
    from corels import CorelsClassifier
    _CORELS = True
except ImportError:
    _CORELS = False
    print("[WARN] corels unavailable. Installation: pip install corels")

try:
    from aix360.algorithms.rbm import BoostingEnsemble, BooleanRuleCG
    _BRCG = True
except ImportError:
    _BRCG = False
    print("[WARN] aix360 unavailable. Installation: pip install aix360")

try:
    from imli import IMLI
    _IMLI = True
except ImportError:
    _IMLI = False
    print("[WARN] imli unavailable. Installation: pip install imli")

try:
    from aix360.algorithms.rbm import BayesianRuleSet
    _BRS = True
except ImportError:
    _BRS = False
    print("[WARN] BRS unavailable (requires aix360).")

try:
    import pysbrl
    from pysbrl import RuleListClassifier
    _BRL = True
except ImportError:
    _BRL = False
    print("[WARN] pysbrl unavailable. Installation: pip install pysbrl")

try:
    from pyids import IDS, IDSClassifier
    _IDS = True
except ImportError:
    _IDS = False
    print("[WARN] pyids unavailable. Installation: pip install pyids")

try:
    from dl85 import DL85Classifier
    _DL85 = True
except ImportError:
    _DL85 = False
    print("[WARN] dl8.5 unavailable. Installation: pip install dl8.5")

try:
    from mdlp.discretization import MDLP
    _MDLP = True
except ImportError:
    _MDLP = False
    print("[WARN] mdlp unavailable. Installation: pip install mdlp-discretization")

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _EBM = True
except ImportError:
    _EBM = False
    print("[WARN] interpret unavailable. Installation: pip install interpret")

try:
    from wittgenstein import RIPPER
    _RIPPER = True
except ImportError:
    _RIPPER = False
    print("[WARN] wittgenstein unavailable. Installation: pip install wittgenstein")

try:
    from tabpfn import TabPFNClassifier
    _TABPFN = True
except ImportError:
    _TABPFN = False
    print("[WARN] tabpfn unavailable. Installation: pip install tabpfn")

# ---------------------------------------------------------------------
# [V21-T1] Optional baseline imports (all gated; the benchmark runs with
# whatever is installed and prints what is missing).
# ---------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False
    print("[WARN] xgboost unavailable. Installation: pip install xgboost")
try:
    from lightgbm import LGBMClassifier
    _LGBM = True
except ImportError:
    _LGBM = False
    print("[WARN] lightgbm unavailable. Installation: pip install lightgbm")
try:
    from catboost import CatBoostClassifier
    _CATB = True
except ImportError:
    _CATB = False
    print("[WARN] catboost unavailable. Installation: pip install catboost")
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _EBM = True
except ImportError:
    _EBM = False
    print("[WARN] interpret unavailable (EBM). Installation: pip install interpret")
try:
    from imodels import RuleFitClassifier, FIGSClassifier
    _IMODELS = True
except ImportError:
    _IMODELS = False
    print("[WARN] imodels unavailable (RuleFit/FIGS). Installation: pip install imodels")


_GPU_WORKER_ID = os.environ.get('LDR_GPU_ID', None)

print(f">>> INITIALISATION LDR "
      f"{'[WORKER GPU-' + _GPU_WORKER_ID + ']' if _GPU_WORKER_ID else '[MASTER]'} ...",
      flush=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# [V02-3] TF32 for nn.Linear matrix multiplications on Hopper GPUs
# (H100/H200): 2-8x speed-up of Linear layers at a relative error of ~1e-3
# (10-bit mantissa). To disable: export LDRV2_TF32=0
if torch.cuda.is_available() and os.environ.get('LDRV2_TF32', '1') == '1':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(">>> [V02] TF32 ENABLED (LDRV2_TF32=0 to disable)", flush=True)

# [V02-8] Global socket timeout: every network call issued without an
# explicit timeout (OpenML dataset download in fetch_openml, TabPFN model
# checkpoint download during pre-warm/first fit) RAISES an exception after
# LDRV2_NET_TIMEOUT seconds instead of blocking forever. Root cause of the
# observed 10-day startup hang: zero tasks ever completed, GPU util 0%,
# bare CUDA context (~612 MiB) — the worker was stuck inside a network
# call that has no timeout in v01. Exceptions raised here are handled by
# the existing try/except blocks (dataset skipped / pre-warm skipped).
import socket as _socket
_socket.setdefaulttimeout(float(os.environ.get('LDRV2_NET_TIMEOUT', '180')))
print(f">>> [V02] Global network timeout: "
      f"{_socket.getdefaulttimeout():.0f}s (LDRV2_NET_TIMEOUT)", flush=True)

os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import argparse as _argparse

def _parse_args():
    p = _argparse.ArgumentParser(
        description='LDR2H200 — CBM benchmark',
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
        add_help=True)
    p.add_argument('--trials',       type=int, default=40,   # [V22]
                   help='Number of Optuna HPO trials')
    p.add_argument('--epochs-hpo',   type=int, default=60,
                   help='Epochs per HPO trial')
    p.add_argument('--epochs-final', type=int, default=250,  # [V22]
                   help='Final training epochs')
    p.add_argument('--patience',     type=int, default=25,
                   help='Early stopping patience')
    p.add_argument('--restarts',     type=int, default=2,    # [V22]
                   help='Number of final restarts')
    p.add_argument('--gpu-workers',  type=int, default=None,
                   help='Parallel GPU workers (default: auto = 3 per GPU)')
    p.add_argument('--cpu-workers',  type=int, default=None,
                   help='Parallel CPU workers (default: auto = 75%% of CPU cores)')
    p.add_argument('--full', action='store_true',
                   help='Full mode (slow, highest quality)')
    p.add_argument('--superfast', action='store_true',
                   help='SUPER-FAST mode: 4 datasets 2bin+2multi, trials=8 — max ~15 min')
    p.add_argument('--fast', action='store_true',
                   help='Ultra-fast mode (~4h)')
    p.add_argument('--datasets-group', choices=['0','1','2'], default=None,
                   help='Dataset subset: 0=super(4), 1=small(10), 2=all(20)')
    p.add_argument('--no-menu', action='store_true',
                   help='Skip the interactive menu (script mode)')
    # ---------------- [V02] new options ----------------
    p.add_argument('--hpo-timeout', type=float, default=0.0,
                   help='[V02-4] Wall-clock budget [s] for Optuna HPO per '
                        'CBM fit (0 = no limit, protocol identical to v01)')
    p.add_argument('--monitor-interval', type=int, default=60,
                   help='[V02-6] Resource monitor print interval [s]')
    p.add_argument('--no-resume', action='store_true',
                   help='[V02-5] Ignore checkpoint_rows.csv and recompute all')
    p.add_argument('--cpu-backend', choices=['loky', 'threads'],
                   default='loky',
                   help='[V02-2] Backend for CPU classifiers '
                        '(loky = processes, no GIL; threads = v01 behaviour)')
    p.add_argument('--seeds', type=str, default='42',
                   help='[V21-T3] Comma-separated random seeds, e.g. "42,43,44". '
                        'Each seed gets its own CV split and result subfolder.')
    p.add_argument('--ablation', action='store_true',
                   help='[V21-T2] Additionally evaluate CBM_KE ablation variants '
                        '(noHedge, noResid, noLS, noCalib).')
    p.add_argument('--strict-validation', action=_argparse.BooleanOptionalAction,
                   default=True,
                   help='[V20] Keep the inner validation split completely out of final CBM training; '
                        'also fit the TabPFN teacher only on the inner-fit split for distillation. '
                        'Use --no-strict-validation only to reproduce the historical v16 protocol.')
    # ---------------- [V25] resilient execution ----------------
    p.add_argument('--resilient', action='store_true',
                   help='Use the crash-safe SQLite queue and supervised workers.')
    p.add_argument('--resilient-self-test', action='store_true',
                   help=_argparse.SUPPRESS)
    p.add_argument('--resilient-fd-self-test', action='store_true',
                   help=_argparse.SUPPRESS)
    p.add_argument('--run-id', type=str, default='v25_publication',
                   help='Stable run identifier; reuse it to resume the same experiment.')
    p.add_argument('--state-db', type=str, default=None,
                   help='SQLite state database (default: results_LDR/<run-id>/run_state.sqlite3).')
    p.add_argument('--worker-role', choices=['master','gpu','cpu'], default='master',
                   help=_argparse.SUPPRESS)
    p.add_argument('--worker-id', type=str, default='', help=_argparse.SUPPRESS)
    p.add_argument('--gpu-id', type=int, default=None, help=_argparse.SUPPRESS)
    p.add_argument('--n-folds', type=int, default=10, choices=[2,3,5,10],
                   help='Number of outer CV folds in resilient mode.')
    p.add_argument('--cpu-processes', type=int, default=None,
                   help='Total CPU task processes in resilient mode.')
    p.add_argument('--min-free-vram-gib', type=float, default=24.0,
                   help='Do not claim a GPU task below this free-VRAM threshold.')
    p.add_argument('--lease-seconds', type=int, default=300,
                   help='Task lease duration; heartbeat renews it while a task runs.')
    p.add_argument('--heartbeat-seconds', type=int, default=60,
                   help='Task/worker heartbeat interval.')
    p.add_argument('--retry-base-seconds', type=int, default=30,
                   help='Initial retry delay for transient failures.')
    p.add_argument('--retry-max-seconds', type=int, default=900,
                   help='Maximum retry delay for transient failures.')
    p.add_argument('--max-worker-restarts', type=int, default=20,
                   help='Maximum automatic restarts of a crashed worker process.')
    p.add_argument('--idle-sleep-seconds', type=int, default=20,
                   help='Worker sleep interval when no eligible task is available.')
    p.add_argument('--epoch-checkpoint-interval', type=int, default=10,
                   help='Save final-training state every N epochs (0 disables).')
    p.add_argument('--max-nonresource-attempts', type=int, default=2,
                   help='Attempts before a deterministic non-resource failure becomes permanent.')
    p.add_argument('--reset-run', action='store_true',
                   help='Delete only this run-id state before creating the queue.')
    args, _ = p.parse_known_args()
    if args.superfast:
        args.trials, args.epochs_hpo, args.patience = 8, 20, 8
        args.epochs_final, args.restarts = 60, 1
        if args.datasets_group is None:
            args.datasets_group = '0'
    elif args.full:
        args.trials, args.epochs_hpo, args.patience = 120, 100, 50
        args.epochs_final, args.restarts = 450, 3
    elif args.fast:
        args.trials, args.epochs_hpo, args.patience = 30, 40, 15
        args.epochs_final, args.restarts = 150, 1
    if not hasattr(args, 'epochs_hpo'):
        args.epochs_hpo = args.epochs_hpo if hasattr(args, 'epochs_hpo') else 60
    return args

_ARGS = _parse_args()

CBM_N_TRIALS      = _ARGS.trials
CBM_EPOCHS_HPO    = _ARGS.epochs_hpo
CBM_EPOCHS_FINAL  = _ARGS.epochs_final
CBM_PATIENCE_ES   = _ARGS.patience
CBM_N_RESTARTS    = _ARGS.restarts

# [V02] new global configuration
_CBM_HPO_TIMEOUT  = float(getattr(_ARGS, 'hpo_timeout', 0.0) or 0.0)
_MON_INTERVAL     = int(getattr(_ARGS, 'monitor_interval', 60))
_RESUME           = not getattr(_ARGS, 'no_resume', False)
_CPU_BACKEND      = getattr(_ARGS, 'cpu_backend', 'loky')
_SEEDS = [int(s) for s in str(getattr(_ARGS, 'seeds', '42')).split(',') if s.strip()]
if not _SEEDS:
    _SEEDS = [42]
_ABLATION = bool(getattr(_ARGS, 'ablation', False))
print(f">>> [V21] seeds={_SEEDS}  ablation={_ABLATION}", flush=True)

_STRICT_VALIDATION = bool(getattr(_ARGS, 'strict_validation', True))
print(f">>> [V20] strict_validation={_STRICT_VALIDATION} "
      f"(inner validation excluded from final CBM training when True)", flush=True)

# ---------------------------------------------------------------------
# AUTOMATIC HARDWARE DETECTION (used if --gpu-workers / --cpu-workers
# were not specified explicitly on the command line)
# ---------------------------------------------------------------------
import multiprocessing as _mp
_n_gpus_detected = torch.cuda.device_count()
_n_cores_detected = _mp.cpu_count()

if _ARGS.gpu_workers is None:
    # Heuristic: 3 workers per GPU; TabPFN model is ~1.5 GB so even
    # an H200 (141 GB) easily fits many parallel TabPFN contexts.
    # Fallback to 1 if no GPU detected (everything will run on CPU).
    _N_GPU_WORKERS = max(1, 3 * _n_gpus_detected) if _n_gpus_detected > 0 else 1
else:
    _N_GPU_WORKERS = _ARGS.gpu_workers

if _ARGS.cpu_workers is None:
    # Heuristic: 75% of logical CPU cores, capped at 32, min 4.
    _N_CPU_WORKERS = max(4, min(32, int(0.75 * _n_cores_detected)))
else:
    _N_CPU_WORKERS = _ARGS.cpu_workers

print(f"", flush=True)
print(f">>> HARDWARE DETECTION:", flush=True)
print(f"    CPU cores (logical): {_n_cores_detected}", flush=True)
print(f"    CUDA GPUs detected:  {_n_gpus_detected}", flush=True)
if _n_gpus_detected > 0:
    for _i in range(_n_gpus_detected):
        _nm  = torch.cuda.get_device_name(_i)
        _mem = torch.cuda.get_device_properties(_i).total_memory / 1e9
        print(f"      GPU {_i}: {_nm} ({_mem:.1f} GB)", flush=True)
print(f"    Chosen: --gpu-workers={_N_GPU_WORKERS}, "
      f"--cpu-workers={_N_CPU_WORKERS}"
      f"{' (auto)' if _ARGS.gpu_workers is None or _ARGS.cpu_workers is None else ' (manual)'}",
      flush=True)

print(f"", flush=True)
print(f">>> CBM CONFIGURATION:", flush=True)
print(f"    trials={CBM_N_TRIALS}  epochs_hpo={CBM_EPOCHS_HPO}  "
      f"patience={CBM_PATIENCE_ES}  epochs_final={CBM_EPOCHS_FINAL}  "
      f"restarts={CBM_N_RESTARTS}", flush=True)
print(f"    gpu_workers={_N_GPU_WORKERS}  cpu_workers={_N_CPU_WORKERS}", flush=True)
_scale = CBM_N_TRIALS / 120 * CBM_EPOCHS_HPO / 100 * CBM_PATIENCE_ES / 50
_t_est_h = 5 * 24 * _scale
print(f"    Estimated time (relative to full): {_scale*100:.0f}% → ~{_t_est_h:.0f}h", flush=True)
print(f"", flush=True)

N_MIN = 500
N_MAX = 10_000

DATA_PATH = "./DATASETS_LDR"
os.makedirs(DATA_PATH, exist_ok=True)

_phys_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')
_cuda_ok  = torch.cuda.is_available()

if _cuda_ok:
    device    = 'cuda'
    _n_gpu    = torch.cuda.device_count()
    _gpu_name = torch.cuda.get_device_name(0)
    _gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f">>> DEVICE: cuda  (CUDA_VISIBLE_DEVICES={_phys_gpu})", flush=True)
    print(f"    GPU: {_gpu_name}  |  VRAM: {_gpu_mem:.1f} GB  |  n_gpu={_n_gpu}",
          flush=True)

    if not (getattr(_ARGS, 'resilient', False)
            and getattr(_ARGS, 'worker_role', 'master') == 'master'):
        _dummy = torch.zeros(1, device='cuda'); del _dummy
        print(f"    CUDA context: OK", flush=True)
    else:
        print(f"    CUDA context deferred in resilient master", flush=True)
else:
    device = 'cpu'
    print(f">>> [WARN] CUDA UNAVAILABLE → device=cpu", flush=True)
    print(f"    CUDA_VISIBLE_DEVICES={_phys_gpu}", flush=True)
    print(f"    torch.version.cuda={torch.version.cuda}  "
          f"torch={torch.__version__}", flush=True)
    if _phys_gpu not in ('not_set', '', '-1'):
        try:
            _t = torch.zeros(1).cuda(); del _t
            device = 'cuda'
            print(f"    [RECOVER] .cuda() works → device=cuda", flush=True)
        except Exception as _e:
            print(f"    [RECOVER FAILED] {_e}", flush=True)
            print(f"    >>> CBM will train on CPU -- very slowly!", flush=True)
print(f">>> device={device}", flush=True)

import threading
try:
    from scipy.optimize import minimize_scalar
except ImportError:
    minimize_scalar = None

# [V02-2] Lock removed: threading.Lock is not picklable (cloudpickle/
# loky), and dict[tid]=v / dict.get(tid) operations are atomic in CPython
# (GIL). The key = thread id, hence no races between tasks.
_PROBA_CACHE = {}


class _FastTensorLoader:
    """[V02-1] Vectorised loader for GPU-resident tensors.

    Replaces DataLoader(TensorDataset(...)), which for every batch performs
    bs * len(tensors) individual GPU indexing operations in Python
    (TensorDataset.__getitem__) + torch.stack, i.e. O(bs) micro-kernels.
    Here: 1 randperm per epoch + 1 index_select per tensor per batch,
    i.e. O(1) kernels per batch. Compatible interface: iteration yields
    tuples (x, y, c[, soft]) already on the proper device.
    """

    def __init__(self, tensors, batch_size, shuffle=False):
        assert len(tensors) >= 1
        self.tensors    = tuple(tensors)
        self.batch_size = int(batch_size)
        self.shuffle    = bool(shuffle)
        self.n          = self.tensors[0].shape[0]

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.tensors[0].device)
            for s in range(0, self.n, self.batch_size):
                sel = idx[s:s + self.batch_size]
                yield tuple(t.index_select(0, sel) for t in self.tensors)
        else:
            for s in range(0, self.n, self.batch_size):
                yield tuple(t[s:s + self.batch_size] for t in self.tensors)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


class ProbaCachingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = (self.clf.classes_
                         if hasattr(self.clf, 'classes_')
                         else np.unique(y))
        return self

    def predict(self, X):
        tid = threading.get_ident()
        try:
            if hasattr(self.clf, 'predict_proba'):
                proba = self.clf.predict_proba(X)
            elif hasattr(self.clf, 'decision_function'):
                df    = self.clf.decision_function(X)
                df    = df - df.max(axis=1, keepdims=True)
                exp   = np.exp(df)
                proba = exp / exp.sum(axis=1, keepdims=True)
            else:
                proba = None

            _PROBA_CACHE[tid] = proba
            self._last_proba  = proba   # [V02-2] instance copy (loky-safe)
        except Exception:
            _PROBA_CACHE[tid] = None
            self._last_proba  = None

        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_params(self, deep=True):
        return {'clf': self.clf}

    def set_params(self, **params):
        if 'clf' in params:
            self.clf = params['clf']
        return self

def _auc_roc(y_true, y_pred, labels=None):
    tid = threading.get_ident()
    proba = _PROBA_CACHE.get(tid, None)

    if proba is not None:
        try:
            K = proba.shape[1] if proba.ndim == 2 else 2
            if K == 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            else:
                return float(roc_auc_score(
                    y_true, proba,
                    multi_class='ovr', average='macro'))
        except Exception:
            pass

    # [V24-FIX3] No silent fallback to a DIFFERENT functional. If class
    # probabilities are unavailable or ROC-AUC cannot be computed, AUC is
    # undefined for this task; return NaN rather than balanced accuracy
    # under the AUC_ROC name (the two functionals are not interchangeable).
    return float('nan')


def _f1_macro(y_true, y_pred, labels=None):
    try:
        return float(f1_score(y_true, y_pred,
                              average='macro', zero_division=0))
    except Exception:
        return float('nan')


def _mcc(y_true, y_pred, labels=None):
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return float('nan')


def _accuracy(y_true, y_pred, labels=None):
    try:
        return float(accuracy_score(y_true, y_pred))
    except Exception:
        return float('nan')


def _precision_macro(y_true, y_pred, labels=None):
    from sklearn.metrics import precision_score
    try:
        return float(precision_score(y_true, y_pred,
                                     average='macro', zero_division=0))
    except Exception:
        return float('nan')


def _recall_macro(y_true, y_pred, labels=None):
    from sklearn.metrics import recall_score
    try:
        return float(recall_score(y_true, y_pred,
                                  average='macro', zero_division=0))
    except Exception:
        return float('nan')


def _csv_has_data(csv_path, min_rows: int = 1) -> bool:
    p = str(csv_path)
    if not os.path.exists(p):
        return False
    try:
        df_check = pd.read_csv(p, nrows=min_rows)
        return len(df_check) >= min_rows
    except Exception:
        return False


def _safe_cacp_call(fn, *args, label: str = "", min_rows: int = 1,
                    csv_dir=None, **kwargs):
    if csv_dir is not None:
        _check_dir = csv_dir
    else:
        _check_dir = args[0] if args else None

    if _check_dir is not None:
        _csv_p = os.path.join(str(_check_dir), 'comparison.csv')
        if not _csv_has_data(_csv_p, min_rows=min_rows):
            print(f"  [CACP-SKIP] {label}: comparison.csv empty/missing "
                  f"(min_rows={min_rows}). Skipping.", flush=True)
            return False
    try:
        fn(*args, **kwargs)
        print(f"  [CACP] ✓ {label}", flush=True)
        return True
    except pd.errors.EmptyDataError as e:
        print(f"  [CACP] ✗ {label}: EmptyDataError – {e}", flush=True)
        return False
    except Exception as e:
        print(f"  [CACP] ✗ {label}: {type(e).__name__} – {e}", flush=True)
        return False


CACP_METRICS = (
    ('AUC_ROC',   _auc_roc),
    ('Accuracy',  _accuracy),
    ('Precision', _precision_macro),
    ('Recall',    _recall_macro),
    ('F1_macro',  _f1_macro),
    ('MCC',       _mcc),
)

class Fold:
    def __init__(self, x_train_raw, y_train,
                 x_test_raw, y_test, index,
                 cat_idx, num_idx, col_names=None):
        self.y_train = y_train
        self.y_test  = y_test
        self.index   = index
        self.labels  = np.unique(np.concatenate([y_train, y_test]))
        # [V24] Preserve raw outer-fold data and schema.  CBM/CBM_KE use
        # these arrays so their inner validation preprocessing can be fitted
        # strictly on the inner-training subset rather than on all outer-train
        # rows.  Other baselines continue to use x_train/x_test below.
        self.x_train_raw = np.asarray(x_train_raw)
        self.x_test_raw  = np.asarray(x_test_raw)
        self.cat_idx = list(cat_idx)
        self.num_idx = list(num_idx)
        self.raw_feature_names = (list(col_names) if col_names is not None
                                  else [f"f{i}" for i in range(self.x_train_raw.shape[1])])

        transformers = []
        if cat_idx:
            cat_pipe = Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore',
                                      sparse_output=False)),
            ])
            transformers.append(('cat', cat_pipe, cat_idx))
        if num_idx:
            num_pipe = Pipeline([
                ('imp',   SimpleImputer(strategy='median')),
                ('scale', RobustScaler()),
            ])
            transformers.append(('num', num_pipe, num_idx))

        if transformers:
            ct = ColumnTransformer(transformers, remainder='drop')
            self.x_train = ct.fit_transform(x_train_raw).astype(np.float32)
            self.x_test  = ct.transform(x_test_raw).astype(np.float32)
        else:
            x_tr = x_train_raw.astype(np.float32) if x_train_raw.dtype != np.float32 \
                   else x_train_raw
            x_te = x_test_raw.astype(np.float32) if x_test_raw.dtype != np.float32 \
                   else x_test_raw
            imp = SimpleImputer(strategy='median')
            x_tr = imp.fit_transform(x_tr)
            x_te = imp.transform(x_te)
            scaler = RobustScaler()
            self.x_train = scaler.fit_transform(x_tr).astype(np.float32)
            self.x_test  = scaler.transform(x_te).astype(np.float32)


class LocalDataset:
    def __init__(self, name, X_raw, y_raw):
        self.name = name

        if isinstance(X_raw, pd.DataFrame):
            self._cat_idx = [i for i, c in enumerate(X_raw.columns)
                             if not pd.api.types.is_numeric_dtype(X_raw[c])]
            self._num_idx = [i for i, c in enumerate(X_raw.columns)
                             if i not in self._cat_idx]
            self._col_names = list(X_raw.columns)
            self.X_raw = X_raw.values
        else:
            self.X_raw = np.array(X_raw)
            self._cat_idx = []
            self._num_idx = list(range(self.X_raw.shape[1]))
            self._col_names = [f"f{i}" for i in range(self.X_raw.shape[1])]

        if hasattr(y_raw, "to_numpy"):  y_raw = y_raw.to_numpy()
        elif hasattr(y_raw, "values"):  y_raw = y_raw.values
        y_str = y_raw.astype(str) if not np.issubdtype(
            y_raw.dtype, np.number) else y_raw
        self.y = LabelEncoder().fit_transform(y_str)

        if isinstance(X_raw, pd.DataFrame):
            X_num = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            self.X = X_num.values.astype(np.float32)
        else:
            X_f = np.zeros(self.X_raw.shape, dtype=np.float32)
            for j in range(self.X_raw.shape[1]):
                try:
                    X_f[:, j] = self.X_raw[:, j].astype(np.float32)
                except (ValueError, TypeError):
                    pass
            self.X = X_f

    def folds(self, n_folds=10, shuffle=True, random_state=SEED, **kwargs):
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        for i, (tr, te) in enumerate(skf.split(self.X_raw, self.y)):
            yield Fold(
                self.X_raw[tr], self.y[tr],
                self.X_raw[te], self.y[te],
                i, self._cat_idx, self._num_idx,
                col_names=self._col_names)

    def __iter__(self):
        for i in range(len(self.X_raw)):
            yield self.X_raw[i], self.y[i]

    def __len__(self):    return len(self.X_raw)

    def get_data(self):
        return self.X, self.y, None, None

DATASETS_BINARY = {
    'diabetes':    {'name': 'diabetes',                         'version': 1},
    'credit-g':    {'name': 'credit-g',                         'version': 1},
    'blood':       {'name': 'blood-transfusion-service-center', 'version': 1},
    'wdbc':        {'name': 'wdbc',                             'version': 1},
    'tic-tac-toe': {'name': 'tic-tac-toe',                      'version': 1},
    'spambase':    {'name': 'spambase',                         'version': 1},
    'magic':       {'name': 'MagicTelescope',                   'version': 1},
    'bank':        {'name': 'bank-marketing',                   'version': 1},
    'phoneme':     {'name': 'phoneme',                          'version': 1},
    'kr-vs-kp':    {'name': 'kr-vs-kp',                         'version': 1},
}

DATASETS_MULTI = {
    'balance-scale': {'name': 'balance-scale',    'version': 1},
    'vehicle':       {'name': 'vehicle',          'version': 1},
    'car':           {'name': 'car',              'version': 3},
    'segment':       {'name': 'segment',          'version': 1},
    'satimage':      {'name': 'satimage',         'version': 1},
    'pendigits':     {'name': 'pendigits',        'version': 1},
    'mfeat-factors': {'name': 'mfeat-factors',    'version': 1},
    'optdigits':     {'name': 'optdigits',        'version': 1},
    'page-blocks':   {'name': 'page-blocks',      'version': 1},
    'wine-quality':  {'name': 'wine-quality-white','version': 1},
}

def _load_group(datasets_info, group_name):
    loaded = []
    print(f"\n>>> LOADING: {group_name} ({len(datasets_info)} datasets)",
          flush=True)
    for key, info in datasets_info.items():
        sub_csv = os.path.join(DATA_PATH, f"{key}_ldr.csv")
        _csv_ok = False
        if os.path.exists(sub_csv):
            try:
                df = pd.read_csv(sub_csv)
                if len(df) > 0:
                    _csv_ok = True
                    print(f"  [OK] {key:20s} n={len(df)}", flush=True)
                else:
                    print(f"  [WARN] {key}: cache file empty (0 rows) → "
                          f"removing and downloading again.", flush=True)
                    os.remove(sub_csv)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as _e:
                print(f"  [WARN] {key}: corrupted cache file ({_e}) → "
                      f"removing and downloading again.", flush=True)
                try:
                    os.remove(sub_csv)
                except OSError:
                    pass
        if _csv_ok:
            pass
        else:
            print(f"  [↓]  Downloading {key}...", end=" ", flush=True)
            try:
                raw = fetch_openml(
                    name=info['name'], version=info['version'],
                    as_frame=True, parser='liac-arff')
                n_sample = min(N_MAX, len(raw.frame))
                df = raw.frame.sample(
                    n=n_sample, random_state=SEED).reset_index(drop=True)
                df.to_csv(sub_csv, index=False)
                print(f"OK (n={n_sample}).", flush=True)
            except Exception as exc:
                print(f"ERROR: {exc}", flush=True)
                continue

        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        n_classes = y.nunique()
        if len(df) < N_MIN:
            print(f"  [SKIP] {key}: n={len(df)} < N_MIN={N_MIN}. "
                  f"Skipping dataset.", flush=True)
            continue
        if n_classes > 10:
            print(f"  [WARN] {key}: K={n_classes} > 10 (TabPFN limit). "
                  f"TabPFN will be skipped for this dataset.", flush=True)
        if X.shape[1] > 500:
            print(f"  [WARN] {key}: d={X.shape[1]} > 500 (TabPFN limit).",
                  flush=True)

        try:
            ds = LocalDataset(key, X, y)
            ds._n_classes = n_classes
            loaded.append(ds)
        except Exception as exc:
            print(f"  [WARN] Skipping {key}: {exc}", flush=True)

    print(f"  Loaded: {len(loaded)}/{len(datasets_info)}", flush=True)
    return loaded

_SUPERFAST_BIN   = ['diabetes', 'blood']
_SUPERFAST_MULTI = ['balance-scale', 'vehicle']

_GROUP1_BIN   = ['diabetes', 'blood','kr-vs-kp','credit-g','tic-tac-toe','wdbc']
_GROUP1_MULTI = ['balance-scale','vehicle','car','segment']

_GROUP2_BIN   = list(DATASETS_BINARY.keys())
_GROUP2_MULTI = list(DATASETS_MULTI.keys())


def get_datasets_by_group(group):
    if group == '0':
        return _SUPERFAST_BIN, _SUPERFAST_MULTI
    elif group == '1':
        return _GROUP1_BIN, _GROUP1_MULTI
    else:
        return _GROUP2_BIN, _GROUP2_MULTI

def get_datasets_for_group(group):
    bin_keys, multi_keys = get_datasets_by_group(group)
    bin_info   = {k: DATASETS_BINARY[k] for k in bin_keys   if k in DATASETS_BINARY}
    multi_info = {k: DATASETS_MULTI[k]  for k in multi_keys if k in DATASETS_MULTI}
    bin_ds   = _load_group(bin_info,   "BINARY")
    multi_ds = _load_group(multi_info, "MULTICLASS")
    return bin_ds + multi_ds

def get_all_datasets():
    bin_ds   = _load_group(DATASETS_BINARY, "BINARY (K=2)")
    multi_ds = _load_group(DATASETS_MULTI,  "MULTICLASS (K>2)")
    return bin_ds + multi_ds

MAX_CONCEPTS = 12
P_LOW_PCT    = 20.0
P_HIGH_PCT   = 80.0


def _sat01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _apply_residual_ablation(model):
    """[V21-T2] Zero and freeze the residual concept->class connection."""
    with torch.no_grad():
        model.concept_to_class.weight.zero_()
        if model.concept_to_class.bias is not None:
            model.concept_to_class.bias.zero_()
    for _p in model.concept_to_class.parameters():
        _p.requires_grad_(False)
    return model


def _soften_teacher_probabilities(tp: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return temperature-softened teacher probabilities.

    TabPFN exposes probabilities, not logits. For distillation at T>1 we
    therefore use log-probabilities as logits up to an additive constant:
    q_T = softmax(log(q) / T). This is the probability-space analogue of
    Hinton-style temperature scaling and avoids the v16 bug where only the
    student distribution was temperature-scaled.
    """
    T = float(max(temperature, 1e-6))
    tp = tp.clamp(min=1e-12)
    return F.softmax(torch.log(tp) / T, dim=1)

def _temperature_scaling(model,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          T_range=(0.1, 10.0)) -> float:
    if minimize_scalar is None:
        print("  [TempScale] scipy unavailable → T=1.0. "
              "Installation: pip install scipy", flush=True)
        return 1.0

    model.eval()
    with torch.inference_mode():
        Xt    = torch.tensor(X_val, dtype=torch.float32, device=device)
        logits = model(Xt).cpu().numpy()

    y_int = y_val.astype(np.int64)

    def _nll(T):
        T     = float(T)
        if T < 1e-4:
            return 1e9
        logits_scaled = logits / T
        lsm   = logits_scaled - np.log(
                    np.sum(np.exp(logits_scaled -
                                  logits_scaled.max(1, keepdims=True)),
                           axis=1, keepdims=True)
                ) - logits_scaled.max(1, keepdims=True)
        nll   = -np.mean(lsm[np.arange(len(y_int)), y_int])
        return float(nll)

    result = minimize_scalar(_nll, bounds=T_range, method='bounded',
                             options={'xatol': 1e-4})
    T_star = float(result.x)
    nll_base  = _nll(1.0)
    nll_calib = _nll(T_star)
    print(f"  [TempScale] T*={T_star:.4f}  NLL: {nll_base:.4f} → {nll_calib:.4f}",
          flush=True)
    return T_star


FOLD_TIMEOUT_S = 180


class _TimeoutException(Exception):
    pass


class TimeoutWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, timeout_s=FOLD_TIMEOUT_S):
        self.clf       = clf
        self.timeout_s = timeout_s
        self._timed_out = False
        self._majority  = None

    def fit(self, X, y):
        import threading, ctypes

        self._timed_out = False
        self._majority  = None
        self.classes_   = np.unique(y)

        vals, counts    = np.unique(y, return_counts=True)
        self._majority  = vals[np.argmax(counts)]

        fit_done    = threading.Event()
        fit_result  = [None]
        thread_id   = [None]

        def _fit_worker():
            thread_id[0] = threading.current_thread().ident
            try:
                self.clf.fit(X, y)
                if hasattr(self.clf, 'classes_'):
                    self.classes_ = self.clf.classes_
            except Exception as e:
                fit_result[0] = e
            finally:
                fit_done.set()

        t = threading.Thread(target=_fit_worker, daemon=True)
        t.start()
        finished = fit_done.wait(timeout=self.timeout_s)

        if not finished:
            self._timed_out = True
            if thread_id[0] is not None:
                try:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(thread_id[0]),
                        ctypes.py_object(SystemExit))
                except Exception:
                    pass
            print(f"  [TIMEOUT] {type(self.clf).__name__} "
                  f"> {self.timeout_s}s → majority fallback", flush=True)
        elif fit_result[0] is not None:
            print(f"  [TIMEOUT/ERR] {type(self.clf).__name__}: "
                  f"{fit_result[0]} → majority fallback", flush=True)
            self._timed_out = True

        return self

    def predict(self, X):
        if self._timed_out or self._majority is None:
            return np.full(len(X), self._majority if self._majority is not None else 0)
        try:
            return self.clf.predict(X)
        except Exception:
            return np.full(len(X), self._majority)

    def predict_proba(self, X):
        if self._timed_out:
            K = len(self.classes_)
            return np.full((len(X), K), 1.0 / K)
        try:
            return self.clf.predict_proba(X)
        except Exception:
            K = len(self.classes_)
            return np.full((len(X), K), 1.0 / K)

    def get_params(self, deep=True):
        return {'clf': self.clf, 'timeout_s': self.timeout_s}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class BinarizingWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, max_bins=5, max_binary_cols=200,
                 timeout_s=FOLD_TIMEOUT_S):
        self.clf             = clf
        self.max_bins        = max_bins
        self.max_binary_cols = max_binary_cols
        self.timeout_s       = timeout_s
        self._disc           = None
        self._ohe            = None
        self._var_thresh     = None
        self._inner_clf      = None

    def _binarize_fit(self, X, y):
        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.feature_selection import VarianceThreshold

        if _MDLP:
            try:
                disc = MDLP(random_state=SEED)
                disc.fit(X, y)
                X_disc = disc.transform(X).astype(np.int32)
                self._disc = disc
            except Exception as e:
                print(f"  [BinarizingWrapper] MDLP error: {e} → KBins fallback",
                      flush=True)
                disc = KBinsDiscretizer(
                    n_bins=self.max_bins, encode='ordinal',
                    strategy='quantile')
                disc.fit(X)
                X_disc = disc.transform(X).astype(np.int32)
                self._disc = disc
        else:
            disc = KBinsDiscretizer(
                n_bins=self.max_bins, encode='ordinal',
                strategy='quantile')
            disc.fit(X)
            X_disc = disc.transform(X).astype(np.int32)
            self._disc = disc

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_bin = ohe.fit_transform(X_disc)
        self._ohe = ohe

        if X_bin.shape[1] > self.max_binary_cols:
            vt = VarianceThreshold()
            vt.fit(X_bin)
            X_bin = vt.transform(X_bin)
            if X_bin.shape[1] > self.max_binary_cols:
                rng  = np.random.RandomState(SEED)
                cols = rng.choice(X_bin.shape[1],
                                  self.max_binary_cols, replace=False)
                cols.sort()
                X_bin = X_bin[:, cols]
                self._var_thresh = (vt, cols)
            else:
                self._var_thresh = (vt, None)

        return X_bin.astype(np.float32)

    def _binarize_transform(self, X):
        from sklearn.preprocessing import KBinsDiscretizer
        X_disc = self._disc.transform(X).astype(np.int32)
        X_bin  = self._ohe.transform(X_disc)
        if self._var_thresh is not None:
            vt, cols = self._var_thresh
            X_bin = vt.transform(X_bin)
            if cols is not None:
                X_bin = X_bin[:, cols]
        return X_bin.astype(np.float32)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_bin = self._binarize_fit(X, y)

        inner = TimeoutWrapper(self.clf, timeout_s=self.timeout_s)
        inner.fit(X_bin, y)
        self._inner_clf = inner
        return self

    def predict(self, X):
        X_bin = self._binarize_transform(X)
        return self._inner_clf.predict(X_bin)

    def predict_proba(self, X):
        X_bin = self._binarize_transform(X)
        return self._inner_clf.predict_proba(X_bin)

    def get_params(self, deep=True):
        return {
            'clf': self.clf,
            'max_bins': self.max_bins,
            'max_binary_cols': self.max_binary_cols,
            'timeout_s': self.timeout_s,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def _mrmr_select_fast(C_all: np.ndarray, y: np.ndarray,
                      max_k: int) -> np.ndarray:
    d = C_all.shape[1]
    if max_k >= d:
        return np.arange(d)

    mi_rel = mutual_info_classif(C_all, y, random_state=SEED)
    mi_rel = np.nan_to_num(mi_rel, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        corr_mat = np.corrcoef(C_all.T)
        corr_mat = np.nan_to_num(np.abs(corr_mat), nan=0.0)
    except Exception:
        return np.argsort(mi_rel)[::-1][:max_k]

    selected = []
    remaining = list(range(d))

    first = int(np.argmax(mi_rel))
    selected.append(first)
    remaining.remove(first)

    for _ in range(max_k - 1):
        if not remaining:
            break
        scores = []
        for j in remaining:
            rel = mi_rel[j]
            red = float(np.mean([corr_mat[j, s] for s in selected]))
            scores.append(rel - 0.5 * red)
        best_j = remaining[int(np.argmax(scores))]
        selected.append(best_j)
        remaining.remove(best_j)

    return np.array(selected[:max_k])


def _build_fuzzy_concepts_train(X_train: np.ndarray,
                                y_train: np.ndarray,
                                max_k: int = MAX_CONCEPTS):
    n, d = X_train.shape
    cands     = []
    ramp_par  = []

    for j in range(d):
        col = X_train[:, j].astype(np.float64)
        fin = col[np.isfinite(col)]
        uq  = np.unique(fin)

        if len(uq) <= 2:
            cands.append(_sat01(col).astype(np.float32))
            ramp_par.append(('binary', 0.0, 1.0))
        else:
            a = float(np.percentile(fin, P_LOW_PCT))  if len(fin) else 0.0
            b = float(np.percentile(fin, P_HIGH_PCT)) if len(fin) else 1.0
            if abs(b - a) < 1e-10:
                cands.append(np.full(n, 0.5, dtype=np.float32))
            else:
                cands.append(_sat01((col - a) / (b - a)).astype(np.float32))
            ramp_par.append(('cont', a, b))

    C_all   = np.stack(cands, axis=1)
    scores  = mutual_info_classif(C_all, y_train, random_state=SEED)
    scores  = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    K_eff   = int(min(max_k, d))
    idx_top = _mrmr_select_fast(C_all, y_train, K_eff)
    C_train = C_all[:, idx_top].astype(np.float32)

    return C_train, ramp_par, idx_top, K_eff


def _apply_fuzzy_concepts_test(X_test: np.ndarray,
                               ramp_par: list,
                               idx_top: np.ndarray) -> np.ndarray:
    n    = X_test.shape[0]
    cands = []
    for j, (typ, a, b) in enumerate(ramp_par):
        col = X_test[:, j].astype(np.float64)
        if typ == 'binary':
            cands.append(_sat01(col).astype(np.float32))
        else:
            if abs(b - a) < 1e-10:
                cands.append(np.full(n, 0.5, dtype=np.float32))
            else:
                cands.append(_sat01((col - a) / (b - a)).astype(np.float32))
    C_all = np.stack(cands, axis=1)
    return C_all[:, idx_top].astype(np.float32)

class CBMP1TSModelV4(nn.Module):
    def __init__(self, input_dim: int, num_concepts: int,
                 num_classes: int, hidden_dim: int = 256, dropout: float = 0.2,
                 concept_dropout_p: float = 0.0):
        super().__init__()
        self.num_concepts    = num_concepts
        self.num_classes     = num_classes
        self.actual_num_rules = 2 ** num_concepts

        indices = list(iproduct([0, 1], repeat=num_concepts))
        self.register_buffer('rule_matrix',
                             torch.tensor(indices, dtype=torch.float32))

        self.register_buffer('rule_mask',
                             torch.ones(self.actual_num_rules, dtype=torch.bool))
        self.register_buffer('_rule_mask_f',
                             torch.ones(self.actual_num_rules, dtype=torch.float32))

        self.register_buffer('hedge_exponents',
                             torch.ones(self.actual_num_rules, num_concepts))

        h = hidden_dim
        self.concept_predictor = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(h // 2, num_concepts),
            nn.Sigmoid()
        )

        self.rule_weights = nn.Parameter(
            torch.randn(self.actual_num_rules, num_classes) * 0.02)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

        self.concept_to_class = nn.Linear(num_concepts, num_classes)
        nn.init.xavier_uniform_(self.concept_to_class.weight)
        nn.init.constant_(self.concept_to_class.bias, 0.0)

        self.concept_dropout_p = concept_dropout_p
        self._concept_drop = nn.Dropout(p=max(0.0, concept_dropout_p))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_details: bool = False):
        c      = self.concept_predictor(x)
        if self.training and self.concept_dropout_p > 0:
            c  = self._concept_drop(c)
        c_exp  = c.unsqueeze(1)
        r_exp  = self.rule_matrix.unsqueeze(0)
        mem    = (c_exp * r_exp + (1.0 - c_exp) * (1.0 - r_exp)
                  ).clamp(min=1e-7)
        mem    = mem ** self.hedge_exponents.unsqueeze(0)
        gen    = mem.prod(dim=2)
        masked = gen * self._rule_mask_f.unsqueeze(0)
        logits = (torch.matmul(masked, self.rule_weights)
                  + self.class_bias
                  + self.concept_to_class(c))
        if return_details:
            return logits, c, masked
        return logits

    def keep_balanced_rules(self, k_total: int = None) -> int:
        min_r   = self.num_classes
        if k_total is None:
            k_total = min_r
        else:
            k_total = max(int(k_total), min_r)
        k_total = min(k_total, self.actual_num_rules)

        with torch.no_grad():
            W = self.rule_weights.detach().cpu().numpy()

        W_exp = np.exp(W - W.max(axis=1, keepdims=True))
        P     = W_exp / (W_exp.sum(axis=1, keepdims=True) + 1e-12)
        H     = -np.sum(P * np.log(P + 1e-12), axis=1)
        K_log = np.log(max(self.num_classes, 2))
        importance = np.max(np.abs(W), axis=1) * (1.0 - H / K_log)

        selected, used = set(), set()
        for c in range(self.num_classes):
            for ridx in np.argsort(W[:, c])[::-1]:
                if ridx not in used:
                    selected.add(int(ridx)); used.add(int(ridx)); break

        for ridx in np.argsort(importance)[::-1]:
            if len(selected) >= k_total:
                break
            if ridx not in selected:
                selected.add(int(ridx))

        new_mask = torch.zeros(self.actual_num_rules, dtype=torch.bool,
                               device=self.rule_weights.device)
        for ridx in selected:
            new_mask[ridx] = True
        self.rule_mask = new_mask
        self._rule_mask_f = new_mask.float()
        with torch.no_grad():
            self.rule_weights.data[~new_mask] = 0.0
            self.hedge_exponents[~new_mask] = 1.0
        return len(selected)


def _atomic_torch_save(payload, path):
    """Crash-safe torch.save: write+fsync to a temporary file, then replace."""
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        torch.save(payload, tmp)
        with open(tmp, 'rb') as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _state_dict_cpu(state):
    if state is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _train_cbm_v4(model: CBMP1TSModelV4,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  epochs: int,
                  lr: float,
                  lambda_concept: float,
                  lambda_sparse: float,
                  cw_tensor: torch.Tensor,
                  weight_decay: float = 1e-5,
                  kd_temperature: float = 2.0,
                  kd_alpha_hard: float = 0.5,
                  label_smoothing: float = 0.05,
                  trial=None,
                  silent: bool = True,
                  checkpoint_path=None,
                  checkpoint_interval: int = 0,
                  checkpoint_metadata=None):
    """Train CBM, optionally resuming and checkpointing every N epochs.

    The checkpoint includes model, optimizer, scheduler, early-stopping and RNG
    states.  It is used for final restarts; Optuna trials remain atomic units.
    """
    ce_crit  = nn.CrossEntropyLoss(weight=cw_tensor,
                                   label_smoothing=float(label_smoothing))
    mse_crit = nn.MSELoss()
    opt      = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
    sched    = optim.lr_scheduler.ReduceLROnPlateau(
                   opt, mode='max', factor=0.5, patience=10)

    warmup_ep   = 5
    patience_es = CBM_PATIENCE_ES
    best_acc    = 0.0
    best_state  = None
    no_imp      = 0
    start_epoch = 0

    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
            expected = checkpoint_metadata or {}
            saved_meta = ck.get('metadata', {})
            if expected and saved_meta != expected:
                raise ValueError('checkpoint metadata does not match current protocol')
            model.load_state_dict(ck['model_state'])
            opt.load_state_dict(ck['optimizer_state'])
            sched.load_state_dict(ck['scheduler_state'])
            start_epoch = int(ck['epoch']) + 1
            best_acc = float(ck.get('best_acc', 0.0))
            best_state = ck.get('best_state')
            if best_state is not None:
                best_state = {k: v.to(device) for k, v in best_state.items()}
            no_imp = int(ck.get('no_imp', 0))
            if ck.get('python_rng') is not None:
                random.setstate(ck['python_rng'])
            if ck.get('numpy_rng') is not None:
                np.random.set_state(ck['numpy_rng'])
            if ck.get('torch_rng') is not None:
                torch.set_rng_state(ck['torch_rng'])
            if torch.cuda.is_available() and ck.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(ck['cuda_rng'])
            if not silent:
                print(f"   [V25-RESUME-EPOCH] {checkpoint_path}: epoch {start_epoch}/{epochs}",
                      flush=True)
        except Exception as exc:
            print(f"   [V25-CKPT] Cannot resume epoch checkpoint ({exc}); restarting it.",
                  flush=True)
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass
            start_epoch = 0
            best_acc = 0.0
            best_state = None
            no_imp = 0

    def _save_epoch(epoch):
        if not checkpoint_path or checkpoint_interval <= 0:
            return
        payload = {
            'epoch': int(epoch),
            'model_state': _state_dict_cpu(model.state_dict()),
            'optimizer_state': opt.state_dict(),
            'scheduler_state': sched.state_dict(),
            'best_acc': float(best_acc),
            'best_state': _state_dict_cpu(best_state),
            'no_imp': int(no_imp),
            'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            'metadata': checkpoint_metadata or {},
        }
        _atomic_torch_save(payload, checkpoint_path)

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        if epoch < warmup_ep:
            for pg in opt.param_groups:
                pg['lr'] = lr * (epoch + 1) / warmup_ep

        model.train()
        for batch in train_loader:
            x_b = batch[0].to(device)
            y_b = batch[1].to(device)
            c_b = batch[2].to(device)
            tp  = batch[3].to(device).float() if len(batch) >= 4 else None

            opt.zero_grad(set_to_none=True)
            logits, c_pred, _ = model(x_b, return_details=True)

            if tp is not None:
                T = float(kd_temperature)
                alpha_hard = float(np.clip(kd_alpha_hard, 0.0, 1.0))
                tp_T = _soften_teacher_probabilities(tp, T)
                kd = F.kl_div(F.log_softmax(logits / T, dim=1),
                              tp_T, reduction='batchmean') * (T * T)
                cls_l = alpha_hard * ce_crit(logits, y_b) + (1.0 - alpha_hard) * kd
            else:
                cls_l = ce_crit(logits, y_b)

            loss = (cls_l
                    + lambda_concept * mse_crit(c_pred, c_b)
                    + lambda_sparse  * torch.mean(torch.abs(model.rule_weights)))

            loss.backward()
            _gnorm = 0.5 if epoch < 10 else (1.0 if epoch < 100 else 2.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_gnorm)
            opt.step()

        model.eval()
        corr = tot = 0
        with torch.inference_mode():
            for vb in val_loader:
                xv, yv = vb[0].to(device), vb[1].to(device)
                corr  += (torch.argmax(model(xv), dim=1) == yv).sum().item()
                tot   += yv.size(0)
        val_acc = corr / max(1, tot)

        if epoch >= warmup_ep:
            sched.step(val_acc)

        if val_acc > best_acc + 1e-5:
            best_acc   = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if checkpoint_path and checkpoint_interval > 0 and (
                (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs):
            _save_epoch(epoch)

        if no_imp >= patience_es and epoch > warmup_ep + 10:
            if not silent:
                print(f"   ES@{epoch+1} best={best_acc:.4f}", flush=True)
            _save_epoch(epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc

def _evaluate_acc_nll(model: CBMP1TSModelV4,
                      loader: DataLoader,
                      n_cls: int):
    model.eval()
    corr = tot = 0
    nll_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device)
            logits = model(xb)
            probs  = F.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
            corr  += (preds == yb).sum().item()
            tot   += yb.size(0)
            p_true = probs[torch.arange(yb.size(0), device=device), yb]
            nll_sum += (-torch.log(p_true.clamp(min=1e-12))).sum().item()
    acc = corr / max(1, tot)
    nll = nll_sum / max(1, tot)
    return acc, nll


def _evaluate_acc_nll_gpu(model: CBMP1TSModelV4,
                          X_val: torch.Tensor,
                          y_val: torch.Tensor,
                          n_cls: int):
    model.eval()
    with torch.inference_mode():
        logits = model(X_val)
        probs  = F.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        corr   = (preds == y_val).sum().item()
        n      = y_val.size(0)
        p_true = probs[torch.arange(n, device=y_val.device), y_val]
        nll    = (-torch.log(p_true.clamp(min=1e-12))).mean().item()
    return corr / max(1, n), nll

_HEDGE_CANDIDATES = [0.5, 1.0, 2.0]
_HEDGE_LABELS     = {0.5: 'ml', 1.0: '', 2.0: 'v'}


def optimize_linguistic_hedges(model: CBMP1TSModelV4,
                               val_loader: DataLoader,
                               n_cls: int,
                               max_iter: int = 20,
                               silent: bool = False):
    model.eval()
    M       = model.num_concepts
    active  = torch.where(model.rule_mask)[0].cpu().tolist()

    if not active:
        if not silent:
            print("  [HEDGE] No active rules — skipped.", flush=True)
        return 0

    X_val_parts, y_val_parts = [], []
    for batch in val_loader:
        X_val_parts.append(batch[0].to(device))
        y_val_parts.append(batch[1].to(device))
    X_val_gpu = torch.cat(X_val_parts, dim=0)
    y_val_gpu = torch.cat(y_val_parts, dim=0)
    del X_val_parts, y_val_parts

    with torch.no_grad():
        model.hedge_exponents.fill_(1.0)

    base_acc, base_nll = _evaluate_acc_nll_gpu(model, X_val_gpu, y_val_gpu, n_cls)
    best_acc, best_nll = base_acc, base_nll

    if not silent:
        print(f"  [HEDGE] Start: ACC={base_acc:.4f} NLL={base_nll:.6f} "
              f"| rules={len(active)} concepts={M}", flush=True)

    def _better(a, n, ra, rn, eps_a=1e-8, eps_n=1e-9):
        if a > ra + eps_a:
            return True
        if abs(a - ra) <= eps_a and n < rn - eps_n:
            return True
        return False

    n_changed_total = 0
    for it in range(max_iter):
        improved = False
        for r_idx in active:
            for j in range(M):
                cur = float(model.hedge_exponents[r_idx, j].item())
                best_val = cur
                ref_a, ref_n = best_acc, best_nll

                for alpha in _HEDGE_CANDIDATES:
                    if abs(alpha - cur) < 1e-6:
                        continue
                    with torch.no_grad():
                        model.hedge_exponents[r_idx, j] = alpha
                    acc_t, nll_t = _evaluate_acc_nll_gpu(
                        model, X_val_gpu, y_val_gpu, n_cls)
                    if _better(acc_t, nll_t, ref_a, ref_n):
                        ref_a, ref_n = acc_t, nll_t
                        best_val = alpha
                        improved = True

                with torch.no_grad():
                    model.hedge_exponents[r_idx, j] = best_val
                best_acc, best_nll = ref_a, ref_n

        if not silent:
            print(f"  [HEDGE] Iter {it+1}: ACC={best_acc:.4f} NLL={best_nll:.6f}",
                  flush=True)
        if not improved:
            break

    n_changed = int(
        (model.hedge_exponents[active, :] - 1.0).abs().gt(1e-6).sum().item()
    )
    if not silent:
        delta = best_acc - base_acc
        print(f"  [HEDGE] End: ΔACC={delta:+.4f} | "
              f"changed positions: {n_changed}/{len(active)*M}", flush=True)
        rmat = list(iproduct([0, 1], repeat=M))
        for r_idx in active:
            for j in range(M):
                alpha = float(model.hedge_exponents[r_idx, j].item())
                if abs(alpha - 1.0) > 1e-6:
                    antecedent = "HIGH" if rmat[r_idx][j] == 1 else "LOW"
                    prefix     = _HEDGE_LABELS.get(alpha, f'^{alpha:.2g}')
                    print(f"    Rule {r_idx:>4}, concept {j}: "
                          f"{antecedent} → {prefix}{antecedent}  (α={alpha})",
                          flush=True)
    return n_changed

class CBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trials: int = None, mode: str = 'standalone',
                 ablate: str = None):
        # [V21-T2] ablate in {None,'hedges','residual','smoothing','calibration'}
        self.n_trials     = n_trials if n_trials is not None else CBM_N_TRIALS
        self.mode         = mode
        self.ablate       = ablate
        self.model_       = None
        self._temperature = 1.0
        self._ramp_par = None
        self._idx_top  = None
        self._input_preprocessor = None
        self._raw_input = False
        self._raw_cat_idx = []
        self._raw_num_idx = []
        self._raw_feature_names = None
        self._transformed_feature_names = None
        # [V25] Set by the resilient task runner.  It identifies a
        # unique seed/dataset/algorithm/fold state directory.
        self._resilience_context = None

    def _prepare_inputs(self, X, idx_tr, idx_val):
        """Fit preprocessing strictly on inner-train for CBM validation."""
        if not getattr(self, '_raw_input', False):
            X_all = np.asarray(X, dtype=np.float32)
            self._input_preprocessor = None
            self._transformed_feature_names = [f"f{i}" for i in range(X_all.shape[1])]
            return X_all, X_all[idx_tr], X_all[idx_val]

        X_raw = np.asarray(X)
        fit_idx = idx_tr if _STRICT_VALIDATION else np.arange(len(X_raw))
        cat_idx = list(getattr(self, '_raw_cat_idx', []))
        num_idx = list(getattr(self, '_raw_num_idx', []))
        raw_names = list(getattr(self, '_raw_feature_names', None) or
                         [f"f{i}" for i in range(X_raw.shape[1])])
        transformers = []
        if cat_idx:
            transformers.append(('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), cat_idx))
        if num_idx:
            transformers.append(('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scale', RobustScaler()),
            ]), num_idx))
        prep = (ColumnTransformer(transformers, remainder='drop')
                if transformers else Pipeline([
                    ('imp', SimpleImputer(strategy='median')),
                    ('scale', RobustScaler()),
                ]))
        prep.fit(X_raw[fit_idx])
        X_all = np.asarray(prep.transform(X_raw), dtype=np.float32)
        self._input_preprocessor = prep
        try:
            names = [str(v).split('__', 1)[-1]
                     for v in prep.get_feature_names_out(raw_names)]
        except Exception:
            names = [f"f{i}" for i in range(X_all.shape[1])]
        if len(names) != X_all.shape[1]:
            names = [f"f{i}" for i in range(X_all.shape[1])]
        self._transformed_feature_names = names
        return X_all, X_all[idx_tr], X_all[idx_val]

    def _transform_inputs(self, X):
        if self._input_preprocessor is None:
            return np.asarray(X, dtype=np.float32)
        return np.asarray(self._input_preprocessor.transform(np.asarray(X)),
                          dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        y = np.asarray(y)
        n_cls = int(len(np.unique(y)))
        n = len(y)
        idx_all = np.arange(n)
        idx_tr, idx_val = train_test_split(
            idx_all, test_size=0.20, stratify=y, random_state=SEED)

        # [V24-FIX2] Split first; fit preprocessing, class weights, PPL
        # percentiles and mRMR strictly on inner-train.
        X_all, X_tr, X_val = self._prepare_inputs(X, idx_tr, idx_val)
        d = X_all.shape[1]
        y_tr, y_val = y[idx_tr], y[idx_val]

        cw_source = y_tr if _STRICT_VALIDATION else y
        all_classes = np.unique(y)
        source_classes = np.unique(cw_source)
        source_weights = compute_class_weight(
            'balanced', classes=source_classes, y=cw_source)
        cw_map = dict(zip(source_classes, source_weights))
        cw = np.asarray([cw_map.get(c, 1.0) for c in all_classes], dtype=np.float32)
        cw_tensor = torch.tensor(cw, dtype=torch.float32, device=device)

        C_tr, ramp_par, idx_top, k_avail = _build_fuzzy_concepts_train(
            X_tr, y_tr, max_k=MAX_CONCEPTS)
        C_val = _apply_fuzzy_concepts_test(X_val, ramp_par, idx_top)
        C_full = _apply_fuzzy_concepts_test(X_all, ramp_par, idx_top)
        self._ramp_par = ramp_par
        self._idx_top  = idx_top

        soft_full = None
        soft_tr   = None
        teacher_  = None
        if self.mode == 'ke':
            soft_parts = []
            teacher_   = None
            # V20: strict mode prevents the teacher from seeing the inner
            # validation labels used for HPO, hedge selection and calibration.
            # This is more conservative than v16 and may change reported numbers.
            teacher_fit_X = X_tr if _STRICT_VALIDATION else X_all
            teacher_fit_y = y_tr if _STRICT_VALIDATION else y
            teacher_pred_X = X_tr if _STRICT_VALIDATION else X_all
            if _TABPFN:
                try:
                    teacher_ = _make_tabpfn_v25(device)
                    teacher_.fit(teacher_fit_X, teacher_fit_y)
                    sp = teacher_.predict_proba(teacher_pred_X).astype(np.float32)
                    soft_parts.append(sp)
                    print(f"  [CBM_KE] TabPFN teacher OK "
                          f"(fit_n={len(teacher_fit_y)}, pred_n={len(teacher_pred_X)}, K={n_cls})",
                          flush=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"TabPFN teacher failed in CBM_KE: {exc}") from exc
            # ESWA v20: no additional teacher is used; CBM_KE uses TabPFN only.
            if soft_parts:
                if _STRICT_VALIDATION:
                    soft_tr = np.mean(soft_parts, axis=0).astype(np.float32)
                    soft_full = None
                else:
                    soft_full = np.mean(soft_parts, axis=0).astype(np.float32)
                    soft_tr   = soft_full[idx_tr]
                print(f"  [CBM_KE] Soft labels: {len(soft_parts)} teacher(s); "
                      f"strict_validation={_STRICT_VALIDATION}", flush=True)
            else:
                raise RuntimeError(
                    "CBM_KE requested, but no TabPFN soft labels were produced.")

        def make_train_loader(X_a, y_a, C_a, bs, soft_a=None):
            # [V02-1] FastTensorLoader instead of DataLoader(TensorDataset)
            ts = [torch.tensor(X_a, dtype=torch.float32, device=device),
                  torch.tensor(y_a, dtype=torch.long,    device=device),
                  torch.tensor(C_a, dtype=torch.float32, device=device)]
            if soft_a is not None:
                ts.append(torch.tensor(soft_a, dtype=torch.float32, device=device))
            return _FastTensorLoader(ts, batch_size=bs, shuffle=True)

        def make_val_loader(X_a, y_a, C_a, bs):
            return _FastTensorLoader(
                [torch.tensor(X_a, dtype=torch.float32, device=device),
                 torch.tensor(y_a, dtype=torch.long,    device=device),
                 torch.tensor(C_a, dtype=torch.float32, device=device)],
                batch_size=bs, shuffle=False)

        k_max_hpo = int(min(MAX_CONCEPTS, k_avail))
        required_m = max(2, int(math.ceil(math.log2(max(2, n_cls)))))
        if k_max_hpo < required_m:
            raise ValueError(
                f"Cannot guarantee per-class rule coverage: K={n_cls}, "
                f"requires M>={required_m}, but only {k_max_hpo} concepts "
                f"are available after preprocessing/selection.")
        m_min = required_m
        _t_hpo_start = time.time()
        print(f"  [CBM] HPO start: n_trials={self.n_trials}, "
              f"M_range=[{m_min},{k_max_hpo}], K={n_cls}, n={n}, d={d}",
              flush=True)

        def objective(trial):
            k_c = trial.suggest_int('num_concepts', m_min, k_max_hpo)

            rb_lo = int(min(n_cls,       2 ** k_c))
            rb_hi = int(min(n_cls + 16,  2 ** k_c))
            if rb_lo > rb_hi:
                rb_lo = rb_hi
            rb = trial.suggest_int('rule_budget', rb_lo, rb_hi)

            bs = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            hd = trial.suggest_categorical('hidden_dim', [64, 128, 256, 384, 512, 768, 1024])
            dr = trial.suggest_float('dropout',        0.0,  0.5)
            lr = trial.suggest_float('lr',             1e-4, 1e-2,  log=True)
            wd = trial.suggest_float('weight_decay',   1e-6, 1e-3,  log=True)
            lc = trial.suggest_float('lambda_concept', 0.05, 5.0)
            ls   = trial.suggest_float('lambda_sparse',  1e-6, 1e-2,  log=True)
            T_kd  = trial.suggest_float('kd_temperature', 1.0, 5.0) if soft_tr is not None else 2.0
            a_kd  = trial.suggest_float('kd_alpha_hard', 0.2, 0.9) if soft_tr is not None else 1.0
            cdrop = trial.suggest_categorical('concept_dropout_p',
                                              [0.0, 0.05, 0.1, 0.15])

            # [V24-FIX6] deterministic per-trial init: seed before build
            _seed_t = SEED + 1009 * (trial.number + 1)
            torch.manual_seed(_seed_t)
            np.random.seed(_seed_t % (2**31))
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed_t)
            m   = CBMP1TSModelV4(d, k_c, n_cls, hd, dr,
                                  concept_dropout_p=cdrop).to(device)
            if getattr(self, 'ablate', None) == 'residual':   # [V21-T2]
                _apply_residual_ablation(m)
            trl = make_train_loader(X_tr, y_tr, C_tr[:, :k_c], bs,
                                    soft_a=soft_tr if soft_tr is not None else None)
            vl  = make_val_loader(X_val, y_val, C_val[:, :k_c], bs)

            try:
                m, val_acc = _train_cbm_v4(
                    m, trl, vl,
                    epochs=CBM_EPOCHS_HPO, lr=lr, weight_decay=wd,
                    lambda_concept=lc, lambda_sparse=ls,
                    kd_temperature=T_kd,
                    kd_alpha_hard=a_kd,
                    cw_tensor=cw_tensor,
                    label_smoothing=(0.0 if getattr(self, 'ablate', None)
                                     == 'smoothing' else 0.05),
                    trial=trial, silent=True)
                # [V24-FIX1] Evaluate the objective AFTER pruning so that the
                # rule budget rb actually influences Optuna. Previously the
                # pre-pruning val_acc was returned, making rb a no-op in HPO.
                m.keep_balanced_rules(k_total=rb)
                val_acc_pruned, _ = _evaluate_acc_nll(m, vl, n_cls)
                ret = float(val_acc_pruned)
                if not np.isfinite(ret):
                    raise ValueError("non-finite pruned validation accuracy")
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as _e:
                trial.set_user_attr('error', repr(_e))
                if trial.number == 0:
                    print(f"  [CBM-HPO] Error trial-0: {_e}", flush=True)
                # [V25] A transient CUDA/driver/resource error must abort the
                # task and be retried later; it must not silently consume one
                # of the requested scientific HPO trials.
                if _is_retryable_gpu_error_text(repr(_e)):
                    raise
                raise optuna.exceptions.TrialPruned(str(_e))
            finally:
                del m; gc.collect()
            return ret

        import importlib.util as _iutil
        _cmaes_available = _iutil.find_spec('cmaes') is not None
        if not _cmaes_available:
            print("  [HPO] package 'cmaes' unavailable → TPE "
                  "(pip install cmaes to enable CMA-ES)", flush=True)

        _sampler = None
        if _cmaes_available:
            try:
                _sampler = optuna.samplers.CmaEsSampler(
                    seed=SEED,
                    n_startup_trials=25,
                    warn_independent_sampling=False,
                )
            except Exception as _ce:
                print(f"  [HPO] CmaEsSampler error: {_ce} → TPE", flush=True)
                _sampler = None

        if _sampler is None:
            _sampler = optuna.samplers.TPESampler(
                seed=SEED, n_startup_trials=25)

        _ctx = getattr(self, '_resilience_context', None) or {}
        _task_dir = _ctx.get('task_dir')
        if _task_dir:
            os.makedirs(_task_dir, exist_ok=True)
            _study_db = os.path.join(_task_dir, 'optuna.sqlite3')
            _study_name = 'cbm_' + hashlib.sha256(
                json.dumps({
                    'protocol_hash': _ctx.get('protocol_hash'),
                    'seed': _ctx.get('seed'), 'dataset': _ctx.get('dataset'),
                    'algorithm': _ctx.get('algorithm'), 'fold': _ctx.get('fold'),
                    'mode': self.mode, 'ablate': self.ablate,
                }, sort_keys=True).encode('utf-8')).hexdigest()[:24]
            try:
                _storage = optuna.storages.RDBStorage(
                    url=f"sqlite:///{_study_db}",
                    engine_kwargs={'connect_args': {'timeout': 120}},
                    heartbeat_interval=60,
                    grace_period=300)
            except TypeError:
                _storage = f"sqlite:///{_study_db}"
            study = optuna.create_study(
                study_name=_study_name,
                storage=_storage,
                load_if_exists=True,
                direction='maximize',
                sampler=_sampler,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=15, n_warmup_steps=20))
            try:
                optuna.storages.fail_stale_trials(study)
            except Exception:
                pass
        else:
            study = optuna.create_study(
                direction='maximize', sampler=_sampler,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=15, n_warmup_steps=20))

        # Only COMPLETE and scientifically PRUNED trials consume the target
        # budget.  Transient resource FAIL trials are repeated after resume.
        _finished_scientific = sum(
            t.state in (optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED)
            for t in study.trials)
        _remaining_trials = max(0, int(self.n_trials) - int(_finished_scientific))
        if _task_dir and _finished_scientific:
            print(f"  [V25-HPO-RESUME] {_finished_scientific}/{self.n_trials} "
                  f"trials already durable; remaining={_remaining_trials}", flush=True)

        _hpo_to = _CBM_HPO_TIMEOUT if _CBM_HPO_TIMEOUT > 0 else None
        if _hpo_to is not None:
            print(f"  [CBM] HPO wall-clock budget: {_hpo_to:.0f}s", flush=True)
        if _remaining_trials > 0:
            study.optimize(objective, n_trials=_remaining_trials,
                           timeout=_hpo_to, show_progress_bar=False)

        _t_hpo_end = time.time()
        complete_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None and np.isfinite(float(t.value))
        ]
        if not complete_trials:
            errors = [t.user_attrs.get('error') for t in study.trials
                      if t.user_attrs.get('error')]
            raise RuntimeError(
                "CBM HPO produced no finite completed trial. "
                f"Example errors: {errors[:3]}")
        best_trial = max(complete_trials, key=lambda t: float(t.value))
        bp      = best_trial.params
        best_k  = int(bp['num_concepts'])
        best_rb = int(bp['rule_budget'])
        best_bs = int(bp['batch_size'])
        best_hd = int(bp['hidden_dim'])
        best_dr = float(bp['dropout'])
        best_lr = float(bp['lr'])
        best_lc = float(bp['lambda_concept'])
        best_ls   = float(bp['lambda_sparse'])
        best_T_kd = float(bp.get('kd_temperature', 2.0))
        best_a_kd  = float(bp.get('kd_alpha_hard', 1.0 if soft_tr is None else 0.5))
        best_cdrop = float(bp.get('concept_dropout_p', 0.0))

        best_wd_disp = float(bp.get('weight_decay', 1e-5))
        print(f"  [CBM] HPO done: {_t_hpo_end-_t_hpo_start:.1f}s | "
              f"M={best_k}, rules={best_rb}, hd={best_hd}, "
              f"lr={best_lr:.5f}, wd={best_wd_disp:.2e}, "
              f"T_kd={best_T_kd:.3f}, alpha_hard={best_a_kd:.3f}, "
              f"best_val_acc={float(best_trial.value):.4f}",
              flush=True)

        _t_fin_start = time.time()
        print(f"  [CBM] Final training: {CBM_EPOCHS_FINAL} epochs × {CBM_N_RESTARTS} restarts, M={best_k}, "
              f"rb={best_rb}, bs={best_bs}", flush=True)
        if _STRICT_VALIDATION:
            X_fin, y_fin = X_tr, y_tr
            C_fin = C_tr[:, :best_k]
            soft_use = soft_tr
            print("  [CBM] V20 strict final fit: inner validation is held out "
                  "for restart selection, hedge search and calibration.", flush=True)
        else:
            X_fin, y_fin = X_all, y
            C_fin = C_full[:, :best_k]
            soft_use = soft_full if soft_full is not None else None
            print("  [CBM] Historical final fit: uses all outer-training data.", flush=True)

        trl_fin = make_train_loader(X_fin, y_fin, C_fin, best_bs, soft_a=soft_use)
        vl_fin  = make_val_loader(X_val, y_val, C_val[:, :best_k], best_bs)

        best_wd        = float(bp.get('weight_decay', 1e-5))
        _best_val_fin  = -1.0
        self.model_    = None

        _ctx = getattr(self, '_resilience_context', None) or {}
        _task_dir = _ctx.get('task_dir')
        _epoch_interval = int(_ctx.get(
            'epoch_checkpoint_interval',
            getattr(_ARGS, 'epoch_checkpoint_interval', 10)))
        _restart_meta_base = {
            'protocol_hash': _ctx.get('protocol_hash'),
            'seed': int(_ctx.get('seed', SEED)),
            'dataset': _ctx.get('dataset'),
            'algorithm': _ctx.get('algorithm'),
            'fold': int(_ctx.get('fold', -1)),
            'best_params': bp,
            'epochs': int(CBM_EPOCHS_FINAL),
        }

        for _restart in range(CBM_N_RESTARTS):
            _seed_r = SEED + _restart * 37
            _restart_complete = (os.path.join(_task_dir, f'restart_{_restart}.pt')
                                 if _task_dir else None)
            _restart_epoch = (os.path.join(_task_dir, f'restart_{_restart}_epoch.pt')
                              if _task_dir else None)
            _meta = dict(_restart_meta_base, restart=int(_restart), restart_seed=int(_seed_r))

            _loaded_complete = False
            _m_try = None
            _val_fin = float('-inf')
            if _restart_complete and os.path.isfile(_restart_complete):
                try:
                    _saved = torch.load(_restart_complete, map_location=device,
                                        weights_only=False)
                    if _saved.get('metadata') != _meta:
                        raise ValueError('completed restart metadata mismatch')
                    _m_try = CBMP1TSModelV4(
                        d, best_k, n_cls, best_hd, best_dr,
                        concept_dropout_p=best_cdrop).to(device)
                    if getattr(self, 'ablate', None) == 'residual':
                        _apply_residual_ablation(_m_try)
                    _m_try.load_state_dict(_saved['model_state'])
                    _val_fin = float(_saved['val_acc'])
                    _loaded_complete = True
                    print(f"  [V25-RESTART-RESUME] restart {_restart+1}/"
                          f"{CBM_N_RESTARTS}: val_acc={_val_fin:.4f}", flush=True)
                except Exception as exc:
                    print(f"  [V25-RESTART] Invalid {_restart_complete}: {exc}; recomputing.",
                          flush=True)
                    try: os.remove(_restart_complete)
                    except OSError: pass

            if not _loaded_complete:
                torch.manual_seed(_seed_r)
                np.random.seed(_seed_r % (2**31))
                random.seed(_seed_r)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed_r)
                _m_try = CBMP1TSModelV4(
                    d, best_k, n_cls, best_hd, best_dr,
                    concept_dropout_p=best_cdrop).to(device)
                if getattr(self, 'ablate', None) == 'residual':
                    _apply_residual_ablation(_m_try)
                _m_try, _val_fin = _train_cbm_v4(
                    _m_try, trl_fin, vl_fin,
                    epochs=CBM_EPOCHS_FINAL, lr=best_lr,
                    lambda_concept=best_lc, lambda_sparse=best_ls,
                    kd_temperature=best_T_kd,
                    kd_alpha_hard=best_a_kd,
                    weight_decay=best_wd,
                    label_smoothing=(0.0 if getattr(self, 'ablate', None)
                                     == 'smoothing' else 0.05),
                    cw_tensor=cw_tensor, silent=(_restart > 0),
                    checkpoint_path=_restart_epoch,
                    checkpoint_interval=_epoch_interval,
                    checkpoint_metadata=_meta)
                if _restart_complete:
                    _atomic_torch_save({
                        'metadata': _meta,
                        'val_acc': float(_val_fin),
                        'model_state': _state_dict_cpu(_m_try.state_dict()),
                    }, _restart_complete)
                    try:
                        if _restart_epoch and os.path.isfile(_restart_epoch):
                            os.remove(_restart_epoch)
                    except OSError:
                        pass

            print(f"  [CBM] Restart {_restart+1}/{CBM_N_RESTARTS}: "
                  f"val_acc={_val_fin:.4f}", flush=True)
            if _val_fin > _best_val_fin:
                _best_val_fin = _val_fin
                self.model_   = copy.deepcopy(_m_try)
            del _m_try
            gc.collect()
        torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        print(f"  [CBM] Best restart val_acc={_best_val_fin:.4f}", flush=True)

        _t_fin_end = time.time()
        print(f"  [CBM] Final training: {_t_fin_end-_t_fin_start:.1f}s",
              flush=True)
        kept = self.model_.keep_balanced_rules(k_total=best_rb)
        print(f"  [CBM] Active rules after pruning: {kept} "
              f"(target={best_rb}, K={n_cls})", flush=True)

        if getattr(self, 'ablate', None) == 'hedges':        # [V21-T2]
            print("  [ABLATE] Hedge optimisation skipped "
                  "(all exponents stay at 1).", flush=True)
        else:
            n_hedge = optimize_linguistic_hedges(
                self.model_, vl_fin, n_cls,
                max_iter=20, silent=False)
            print(f"  [HEDGE] Modified exponents: {n_hedge}", flush=True)

        if getattr(self, 'ablate', None) == 'calibration':   # [V21-T2]
            self._temperature = 1.0
            print("  [ABLATE] Temperature scaling skipped (T=1).", flush=True)
        else:
            try:
                self._temperature = _temperature_scaling(
                    self.model_, X_val, y_val)
            except Exception as e:
                print(f"  [TempScale] ERROR: {e} → T=1.0", flush=True)
                self._temperature = 1.0

        self.model_.eval()
        self.classes_ = np.unique(y)

        if teacher_ is not None:
            del teacher_
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xp = self._transform_inputs(X)
        Xt = torch.tensor(Xp, dtype=torch.float32, device=device)
        with torch.inference_mode():
            return torch.argmax(self.model_(Xt), dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._transform_inputs(X)
        Xt = torch.tensor(Xp, dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits = self.model_(Xt)
            T = getattr(self, '_temperature', 1.0)
            return F.softmax(logits / T, dim=1).cpu().numpy()

class RIPPERBase(BaseEstimator, ClassifierMixin):
    def __init__(self, k=2, random_state=SEED, pos_class=None):
        self.k            = k
        self.random_state = random_state
        self.pos_class    = pos_class
        self._clf         = None

    def fit(self, X, y):
        if not _RIPPER:
            raise ImportError("pip install wittgenstein")
        y_str = y.astype(str)
        pc = str(self.pos_class) if self.pos_class is not None else str(1)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df['__target__'] = y_str
        self._clf = RIPPER(k=self.k, random_state=self.random_state)
        self._clf.fit(df, class_feat='__target__', pos_class=pc)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        return np.array([int(p) for p in self._clf.predict(df)])


class RIPPERMulticlassWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, k=2, random_state=SEED):
        self.k            = k
        self.random_state = random_state

    def fit(self, X, y):
        if not _RIPPER:
            raise ImportError("pip install wittgenstein")
        self.classes_ = np.unique(y)
        self._clfs    = []
        if len(self.classes_) == 2:
            clf = RIPPERBase(k=self.k, random_state=self.random_state,
                             pos_class=self.classes_[1])
            clf.fit(X, y)
            self._clfs.append(clf)
        else:
            for cls in self.classes_:
                y_bin = (y == cls).astype(int)
                clf   = RIPPERBase(k=self.k, random_state=self.random_state,
                                   pos_class=1)
                clf.fit(X, y_bin)
                self._clfs.append(clf)
        return self

    def predict(self, X):
        if len(self.classes_) == 2:
            preds = self._clfs[0].predict(X)
            return np.where(preds == 1,
                            self.classes_[1], self.classes_[0])
        else:
            votes = np.zeros((len(X), len(self.classes_)), dtype=int)
            for j, clf in enumerate(self._clfs):
                votes[:, j] = clf.predict(X)
            winners = np.argmax(votes, axis=1)
            return self.classes_[winners]


os.environ.setdefault('TABPFN_ACCEPT_MODEL_LICENSE', 'true')

# ---------------------------------------------------------------------
# [V20] TabPFN-3 teacher / baseline configuration
# ---------------------------------------------------------------------
# Current PriorLabs tabpfn releases use TabPFN-3 by default.  The v20
# protocol deliberately relies on that default instead of forcing the old
# ModelVersion.V2_5 checkpoint.  For non-interactive tmux/SSH runs, set
# TABPFN_TOKEN in the environment (for example by sourcing .env).  Without
# it, recent TabPFN versions may block on an interactive licence/API-key
# prompt; by default v20 fails fast with a clear diagnostic instead.
_TABPFN_MODE = 'tabpfn-3-explicit-checkpoint'
_TABPFN_PACKAGE_VERSION = 'unknown'
if _TABPFN:
    try:
        import tabpfn as _tabpfn_pkg
        _TABPFN_PACKAGE_VERSION = getattr(_tabpfn_pkg, '__version__', 'unknown')
    except Exception:
        pass

# [V24-FIX6] Publication runs pin the exact checkpoint explicitly.  The
# official TabPFN API supports TabPFNClassifier(model_path=...).  Guessing the
# largest file in a cache is forbidden because it may not be the file used.
_TABPFN_MODEL_PATH = os.environ.get(
    'LDRV25_TABPFN_MODEL_PATH',
    os.environ.get('LDRV24_TABPFN_MODEL_PATH',
                   os.environ.get('TABPFN_MODEL_PATH', ''))
).strip()
if _TABPFN_MODEL_PATH:
    _TABPFN_MODEL_PATH = os.path.abspath(os.path.expanduser(_TABPFN_MODEL_PATH))
_TABPFN_REQUIRE_PINNED_MODEL = (
    os.environ.get('LDRV25_REQUIRE_PINNED_TABPFN_MODEL',
                   os.environ.get('LDRV24_REQUIRE_PINNED_TABPFN_MODEL', '1')) != '0'
)
_TABPFN_REQUIRE_TOKEN = (
    os.environ.get(
        'LDRV25_REQUIRE_TABPFN_TOKEN',
        os.environ.get(
            'LDRV24_REQUIRE_TABPFN_TOKEN',
            os.environ.get('LDRV20_REQUIRE_TABPFN_TOKEN', '1'))
    ) != '0'
)
_TABPFN_FINGERPRINT = None

if _TABPFN:
    print(f">>> TabPFN teacher/baseline: explicit checkpoint mode "
          f"(tabpfn package={_TABPFN_PACKAGE_VERSION}, "
          f"model_path={_TABPFN_MODEL_PATH or 'UNSET'})", flush=True)


def _ensure_tabpfn_auth_ready():
    if not _TABPFN or not _TABPFN_REQUIRE_TOKEN:
        return
    if os.environ.get('TABPFN_TOKEN', '').strip():
        return
    raise RuntimeError(
        "TABPFN_TOKEN is not set. Put TABPFN_TOKEN=... in a local .env "
        "file and load it before running. For an intentional interactive "
        "login only, set LDRV25_REQUIRE_TABPFN_TOKEN=0."
    )


def _record_tabpfn_fingerprint():
    global _TABPFN_FINGERPRINT
    if _TABPFN_FINGERPRINT is not None:
        return _TABPFN_FINGERPRINT
    info = {
        'tabpfn_version': _TABPFN_PACKAGE_VERSION,
        'model_path': _TABPFN_MODEL_PATH or None,
        'sha256': None,
        'pinned': False,
    }
    if not _TABPFN_MODEL_PATH:
        if _TABPFN_REQUIRE_PINNED_MODEL:
            raise RuntimeError(
                "Exact TabPFN checkpoint is not pinned. Set "
                "LDRV25_TABPFN_MODEL_PATH=/absolute/path/to/model.ckpt."
            )
        info['warning'] = 'unpinned default checkpoint; exploratory run only'
        _TABPFN_FINGERPRINT = info
        return info
    if not os.path.isfile(_TABPFN_MODEL_PATH):
        raise FileNotFoundError(
            f"Pinned TabPFN checkpoint does not exist: {_TABPFN_MODEL_PATH}")
    h = hashlib.sha256()
    with open(_TABPFN_MODEL_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    info['sha256'] = h.hexdigest()
    info['size_bytes'] = os.path.getsize(_TABPFN_MODEL_PATH)
    info['pinned'] = True
    _TABPFN_FINGERPRINT = info
    return info


def _make_tabpfn_v25(device_=device):
    if not _TABPFN:
        raise ImportError("tabpfn is not installed")
    _ensure_tabpfn_auth_ready()
    fp = _record_tabpfn_fingerprint()
    kwargs = {'device': device_}
    if fp.get('model_path'):
        # Official TabPFN API: TabPFNClassifier(model_path=...).
        kwargs['model_path'] = fp['model_path']
    try:
        return TabPFNClassifier(**kwargs)
    except TypeError as exc:
        sig = inspect.signature(TabPFNClassifier)
        raise RuntimeError(
            f"Installed TabPFNClassifier signature {sig} does not accept the "
            "pinned model_path required by the v24 protocol. Install the "
            "recorded compatible tabpfn version."
        ) from exc

# [V21-P5] defaults follow the paper protocol (n<=10k after subsampling,
# K<=10); raise explicitly via env for out-of-protocol experiments.
_TABPFN_N_MAX = int(os.environ.get('LDRV25_TABPFN_N_MAX', os.environ.get('LDRV24_TABPFN_N_MAX', os.environ.get('LDRV20_TABPFN_N_MAX', '50000'))))
_TABPFN_D_MAX = int(os.environ.get('LDRV25_TABPFN_D_MAX', os.environ.get('LDRV24_TABPFN_D_MAX', os.environ.get('LDRV20_TABPFN_D_MAX', '500'))))
_TABPFN_K_MAX = int(os.environ.get('LDRV25_TABPFN_K_MAX', os.environ.get('LDRV24_TABPFN_K_MAX', os.environ.get('LDRV20_TABPFN_K_MAX', '10'))))
print(f">>> SafeTabPFN v20 limits: n≤{_TABPFN_N_MAX}, d≤{_TABPFN_D_MAX}, "
      f"K≤{_TABPFN_K_MAX} (override with LDRV25_TABPFN_*_MAX)", flush=True)


class SafeTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(self, device_=device):
        self.device_ = device_
        self._clf    = None
        self._mode   = _TABPFN_MODE

    def _make_tabpfn(self):
        clf = _make_tabpfn_v25(self.device_)
        return clf, _TABPFN_MODE

    def fit(self, X, y):
        K = len(np.unique(y))
        n, d = X.shape
        if n > _TABPFN_N_MAX or d > _TABPFN_D_MAX or K > _TABPFN_K_MAX:
            print(f"  [SafeTabPFN] Limits exceeded "
                  f"(n={n},d={d},K={K}) → fallback LogisticRegression", flush=True)
            self._mode = 'fallback'
            self._clf = LogisticRegression(
                max_iter=1000, class_weight='balanced',
                random_state=SEED)
        else:
            self._clf, self._mode = self._make_tabpfn()
            print(f"  [SafeTabPFN] Using {self._mode} "
                  f"(n={n}, d={d}, K={K})", flush=True)
        self._clf.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

class _CORElSWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, max_card=2, c=0.01, n_iter=10000, random_state=SEED):
        self.max_card     = max_card
        self.c            = c
        self.n_iter       = n_iter
        self.random_state = random_state
        self._clf         = None

    def fit(self, X, y):
        if not _CORELS:
            raise ImportError("pip install corels")
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)

        if n_cls == 2:
            self._clf = CorelsClassifier(
                max_card=self.max_card,
                c=self.c,
                n_iter=self.n_iter,
                verbosity=[])
            feat_names = [f"f{i}" for i in range(X.shape[1])]
            label_names = [str(c) for c in self.classes_]
            self._clf.fit(X.astype(np.uint8), y.astype(np.int32),
                         features=feat_names, prediction_name=label_names[-1])
        else:
            self._clf = []
            for cls in self.classes_:
                y_bin = (y == cls).astype(np.int32)
                clf_k = CorelsClassifier(
                    max_card=self.max_card, c=self.c,
                    n_iter=self.n_iter, verbosity=[])
                feat_names = [f"f{i}" for i in range(X.shape[1])]
                clf_k.fit(X.astype(np.uint8), y_bin,
                         features=feat_names, prediction_name=str(cls))
                self._clf.append(clf_k)
        return self

    def predict(self, X):
        if isinstance(self._clf, list):
            votes = np.stack([c.predict(X.astype(np.uint8))
                              for c in self._clf], axis=1)
            return self.classes_[np.argmax(votes, axis=1)]
        preds = self._clf.predict(X.astype(np.uint8))
        return np.array([self.classes_[1] if p else self.classes_[0]
                         for p in preds])

    def predict_proba(self, X):
        preds = self.predict(X)
        K = len(self.classes_)
        proba = np.zeros((len(X), K))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx):
                proba[i, idx[0]] = 1.0
        return proba


class _BRCGWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda0=0.001, lambda1=0.001, random_state=SEED):
        self.lambda0      = lambda0
        self.lambda1      = lambda1
        self.random_state = random_state

    def fit(self, X, y):
        if not _BRCG:
            raise ImportError("pip install aix360")
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        if n_cls == 2:
            self._clf = BooleanRuleCG(lambda0=self.lambda0,
                                      lambda1=self.lambda1,
                                      CNF=False)
            import pandas as pd
            Xdf = pd.DataFrame(X.astype(np.uint8),
                               columns=[f"f{i}" for i in range(X.shape[1])])
            ydf = pd.Series((y == self.classes_[1]).astype(int))
            self._clf.fit(Xdf, ydf)
        else:
            self._clf = OneVsRestClassifier(
                BooleanRuleCG(lambda0=self.lambda0,
                              lambda1=self.lambda1, CNF=False))
            self._clf.fit(X.astype(np.uint8), y)
        return self

    def predict(self, X):
        import pandas as pd
        if hasattr(self._clf, 'predict'):
            if isinstance(self._clf, BooleanRuleCG):
                Xdf = pd.DataFrame(X.astype(np.uint8),
                                   columns=[f"f{i}" for i in range(X.shape[1])])
                p = self._clf.predict(Xdf).values.flatten().astype(int)
                return np.where(p == 1, self.classes_[1], self.classes_[0])
            return self._clf.predict(X.astype(np.uint8))
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        preds = self.predict(X)
        K = len(self.classes_)
        proba = np.zeros((len(X), K))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx):
                proba[i, idx[0]] = 1.0
        return proba


class _IMLIWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_clauses=4, rule_length=3, random_state=SEED):
        self.num_clauses  = num_clauses
        self.rule_length  = rule_length
        self.random_state = random_state

    def fit(self, X, y):
        if not _IMLI:
            raise ImportError("pip install imli")
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        if n_cls == 2:
            self._clf = IMLI(num_clause=self.num_clauses,
                             len_clause=self.rule_length)
            self._clf.fit(X.astype(np.float64), y.astype(int))
        else:
            self._clf = OneVsRestClassifier(
                IMLI(num_clause=self.num_clauses,
                     len_clause=self.rule_length))
            self._clf.fit(X.astype(np.float64), y.astype(int))
        return self

    def predict(self, X):
        return self._clf.predict(X.astype(np.float64))

    def predict_proba(self, X):
        preds = self.predict(X)
        K = len(self.classes_)
        proba = np.zeros((len(X), K))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx): proba[i, idx[0]] = 1.0
        return proba


class _BRSWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=2000, n_chains=2, random_state=SEED):
        self.n_iter       = n_iter
        self.n_chains     = n_chains
        self.random_state = random_state

    def fit(self, X, y):
        if not _BRS:
            raise ImportError("pip install aix360")
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        if n_cls == 2:
            self._clf = BayesianRuleSet()
            import pandas as pd
            Xdf = pd.DataFrame(X.astype(np.uint8),
                               columns=[f"f{i}" for i in range(X.shape[1])])
            ydf = pd.Series((y == self.classes_[1]).astype(int))
            self._clf.fit(Xdf, ydf)
        else:
            self._clf = None
            self._ovr = OneVsRestClassifier(
                BayesianRuleSet())
            self._ovr.fit(X.astype(np.uint8), y)
        return self

    def predict(self, X):
        if hasattr(self, '_ovr') and self._ovr is not None:
            return self._ovr.predict(X.astype(np.uint8))
        import pandas as pd
        Xdf = pd.DataFrame(X.astype(np.uint8),
                           columns=[f"f{i}" for i in range(X.shape[1])])
        p = self._clf.predict(Xdf).astype(int)
        return np.where(p == 1, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        preds = self.predict(X)
        K = len(self.classes_)
        proba = np.zeros((len(X), K))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx): proba[i, idx[0]] = 1.0
        return proba


class _BRLWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_chains=3, n_iter=30000, random_state=SEED):
        self.n_chains     = n_chains
        self.n_iter       = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        if not _BRL:
            raise ImportError("pip install pysbrl")
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        if n_cls == 2:
            self._clf = RuleListClassifier(
                n_chains=self.n_chains,
                max_iter=self.n_iter,
                random_state=self.random_state)
            self._clf.fit(X.astype(np.uint8), y.astype(int))
        else:
            self._clf = OneVsRestClassifier(
                RuleListClassifier(n_chains=self.n_chains,
                                   max_iter=self.n_iter,
                                   random_state=self.random_state))
            self._clf.fit(X.astype(np.uint8), y.astype(int))
        return self

    def predict(self, X):
        return self._clf.predict(X.astype(np.uint8))

    def predict_proba(self, X):
        try:
            return self._clf.predict_proba(X.astype(np.uint8))
        except Exception:
            preds = self.predict(X)
            K = len(self.classes_)
            proba = np.zeros((len(X), K))
            for i, p in enumerate(preds):
                idx = np.where(self.classes_ == p)[0]
                if len(idx): proba[i, idx[0]] = 1.0
            return proba


class _IDSWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_array=None, random_state=SEED):
        self.lambda_array = lambda_array or [1, 1, 1, 1, 1, 1, 1]
        self.random_state = random_state

    def fit(self, X, y):
        if not _IDS:
            raise ImportError("pip install pyids")
        import pandas as pd
        self.classes_ = np.unique(y)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df['__target__'] = y
        try:
            ids_clf = IDS()
            ids_clf.fit(df, class_col='__target__',
                        lambda_array=self.lambda_array)
            self._clf = ids_clf
        except Exception as e:
            print(f"  [IDS] fit error: {e} → fallback DT", flush=True)
            self._clf = DecisionTreeClassifier(
                max_depth=5, random_state=self.random_state)
            self._clf.fit(X, y)
        return self

    def predict(self, X):
        import pandas as pd
        if isinstance(self._clf, DecisionTreeClassifier):
            return self._clf.predict(X)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        try:
            return np.array(self._clf.predict(df))
        except Exception:
            return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        preds = self.predict(X)
        K = len(self.classes_)
        proba = np.zeros((len(X), K))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx): proba[i, idx[0]] = 1.0
        return proba


class _DL85Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=4, time_limit=120, random_state=SEED):
        self.max_depth  = max_depth
        self.time_limit = time_limit
        self.random_state = random_state

    def fit(self, X, y):
        if not _DL85:
            raise ImportError("pip install dl8.5")
        self.classes_ = np.unique(y)
        self._clf = DL85Classifier(max_depth=self.max_depth,
                                   time_limit=self.time_limit)
        self._clf.fit(X.astype(np.uint8), y.astype(int))
        return self

    def predict(self, X):
        return self._clf.predict(X.astype(np.uint8))

    def predict_proba(self, X):
        try:
            return self._clf.predict_proba(X.astype(np.uint8))
        except Exception:
            preds = self.predict(X)
            K = len(self.classes_)
            proba = np.zeros((len(X), K))
            for i, p in enumerate(preds):
                idx = np.where(self.classes_ == p)[0]
                if len(idx): proba[i, idx[0]] = 1.0
            return proba

def build_classifiers():
    """Build exactly the classifiers reported in the ESWA v20 manuscript.

    The reported experimental comparison contains:
      - proposed models: CBM and CBM_KE;
      - interpretable baselines: DT_depth5, DT_depth3, LogReg, KNN_k5;
      - non-interpretable baselines: TabPFN, RandomForest, MLP, SVM_RBF.

    Development-only optional baselines such as disabled optional tree model, XGBoost, CatBoost,
    EBM, FIGS, RuleFit, CORELS, BRCG, IMLI, BRS, BRL, IDS, DL8.5 and RIPPER
    are intentionally not appended here, because they are not reported in the
    v20 manuscript tables.
    """
    def wrap(factory):
        return lambda n_i, n_c: ProbaCachingClassifier(factory(n_i, n_c))

    clfs = []

    clfs.append(('CBM',
        wrap(lambda n_i, n_c: CBMClassifier(mode='standalone', n_trials=CBM_N_TRIALS))))
    clfs.append(('CBM_KE',
        wrap(lambda n_i, n_c: CBMClassifier(mode='ke',         n_trials=CBM_N_TRIALS))))

    clfs.append(('DT_depth5',
        wrap(lambda n_i, n_c: DecisionTreeClassifier(
            max_depth=5, random_state=SEED, class_weight='balanced'))))
    clfs.append(('DT_depth3',
        wrap(lambda n_i, n_c: DecisionTreeClassifier(
            max_depth=3, random_state=SEED, class_weight='balanced'))))

    clfs.append(('LogReg',
        wrap(lambda n_i, n_c: LogisticRegression(
            max_iter=1000, random_state=SEED,
            class_weight='balanced', solver='lbfgs'))))

    clfs.append(('KNN_k5',
        wrap(lambda n_i, n_c: KNeighborsClassifier(n_neighbors=5, n_jobs=1))))

    if _TABPFN:
        clfs.append(('TabPFN',
            wrap(lambda n_i, n_c: SafeTabPFN(device_=device))))
    else:
        print("[INFO] TabPFN skipped (missing 'tabpfn').")

    clfs.append(('RandomForest',
        wrap(lambda n_i, n_c: RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight='balanced', random_state=SEED, n_jobs=1))))

    clfs.append(('MLP',
        wrap(lambda n_i, n_c: MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=500,
            random_state=SEED, early_stopping=True,
            validation_fraction=0.1))))

    clfs.append(('SVM_RBF',
        wrap(lambda n_i, n_c: SVC(
            kernel='rbf', probability=True,
            class_weight='balanced', random_state=SEED, C=1.0))))

    # ----------------------------------------------------------------
    # [V21-T1] Gradient-boosted trees + modern interpretable baselines.
    # All CPU-side, n_jobs/threads = 1 for fair per-task timing and clean
    # process-level parallelism. Binary-only rule learners are wrapped in
    # One-vs-Rest for multiclass tasks.
    # ----------------------------------------------------------------
    if _XGB:
        clfs.append(('XGBoost',
            wrap(lambda n_i, n_c: XGBClassifier(
                n_estimators=300, tree_method='hist',
                eval_metric='logloss', random_state=SEED,
                n_jobs=1, verbosity=0))))
    else:
        print("[INFO] XGBoost skipped (missing 'xgboost').")
    if _LGBM:
        clfs.append(('LightGBM',
            wrap(lambda n_i, n_c: LGBMClassifier(
                n_estimators=300, random_state=SEED,
                n_jobs=1, verbose=-1))))
    else:
        print("[INFO] LightGBM skipped (missing 'lightgbm').")
    if _CATB:
        clfs.append(('CatBoost',
            wrap(lambda n_i, n_c: CatBoostClassifier(
                iterations=300, random_seed=SEED, verbose=0,
                thread_count=1, allow_writing_files=False))))
    else:
        print("[INFO] CatBoost skipped (missing 'catboost').")
    if _EBM:
        clfs.append(('EBM',
            wrap(lambda n_i, n_c: ExplainableBoostingClassifier(
                random_state=SEED, n_jobs=1))))
    else:
        print("[INFO] EBM skipped (missing 'interpret').")
    if _IMODELS:
        def _mk_rulefit(n_c):
            try:
                base = RuleFitClassifier(random_state=SEED)
            except TypeError:
                base = RuleFitClassifier()
            return (OneVsRestClassifier(base, n_jobs=1)
                    if n_c > 2 else base)
        def _mk_figs(n_c):
            try:
                base = FIGSClassifier(random_state=SEED)
            except TypeError:
                base = FIGSClassifier()
            return (OneVsRestClassifier(base, n_jobs=1)
                    if n_c > 2 else base)
        clfs.append(('RuleFit', wrap(lambda n_i, n_c: _mk_rulefit(n_c))))
        clfs.append(('FIGS',    wrap(lambda n_i, n_c: _mk_figs(n_c))))
    else:
        print("[INFO] RuleFit/FIGS skipped (missing 'imodels').")

    # [V21-T2] Ablation variants of CBM_KE (GPU tasks; the prefix
    # 'CBM_KE_' routes them to the GPU executor via _is_gpu_clf).
    if _ABLATION:
        clfs.append(('CBM_KE_noHedge',
            wrap(lambda n_i, n_c: CBMClassifier(
                mode='ke', n_trials=CBM_N_TRIALS, ablate='hedges'))))
        clfs.append(('CBM_KE_noResid',
            wrap(lambda n_i, n_c: CBMClassifier(
                mode='ke', n_trials=CBM_N_TRIALS, ablate='residual'))))
        clfs.append(('CBM_KE_noLS',
            wrap(lambda n_i, n_c: CBMClassifier(
                mode='ke', n_trials=CBM_N_TRIALS, ablate='smoothing'))))
        clfs.append(('CBM_KE_noCalib',
            wrap(lambda n_i, n_c: CBMClassifier(
                mode='ke', n_trials=CBM_N_TRIALS, ablate='calibration'))))

    print(f">>> Loaded classifiers for ESWA v21 protocol: {len(clfs)}"
          f" (ablation={'on' if _ABLATION else 'off'})", flush=True)
    for name, _ in clfs:
        print(f"    - {name}", flush=True)
    return clfs

CBM_NAMES = {'CBM', 'CBM_KE'}

SOTA_INTERPRETABLE_NAMES = {
    'EBM', 'FIGS', 'RuleFit', 'RIPPER', 'LogReg',
    'CORELS', 'BRCG', 'IMLI', 'BRS', 'BRL', 'IDS', 'DL8.5',
    'DT_depth5', 'DT_depth3',
}
SOTA_INTERPRETABLE_TOP5_COUNT = 5
SOTA_NONINTERPRETABLE_NAMES = {'TabPFN', 'RandomForest'}

def _load_balanced_split(datasets, n_splits=2):
    costs = [(ds, ds.X_raw.shape[0] * ds.X_raw.shape[1])
             for ds in datasets]
    costs.sort(key=lambda x: x[1], reverse=True)

    partitions = [[] for _ in range(n_splits)]
    part_costs  = [0] * n_splits

    for ds, cost in costs:
        idx = int(np.argmin(part_costs))
        partitions[idx].append(ds)
        part_costs[idx] += cost

    print(f"  [LOADBALANCE] Split {len(datasets)} datasets into {n_splits} GPU:",
          flush=True)
    for i, (part, cost) in enumerate(zip(partitions, part_costs)):
        names = [ds.name for ds in part]
        print(f"    GPU-{i}: {names}  (cost n×d: {cost:,})", flush=True)

    return partitions

def print_dataset_summary(datasets):
    print("\n" + "="*65)
    print(f"{'DATASET':20s} {'n':>6} {'d':>6} {'K':>4} {'Type':>10}")
    print("="*65)
    for ds in datasets:
        K    = len(np.unique(ds.y))
        n, d = ds.X_raw.shape
        typ  = "binary" if K == 2 else f"K={K}"
        print(f"  {ds.name:20s} {n:6d} {d:6d} {K:4d} {typ:>10}")
    print("="*65 + "\n")



def _select_top_n_by_friedman_rank(df_all: pd.DataFrame,
                                   candidate_names: set,
                                   metrics,
                                   n_top: int = 5) -> set:
    if not candidate_names:
        return set()

    metric_names = [m[0] for m in metrics]
    alg_ranks    = {a: [] for a in candidate_names}

    for ds_name in df_all['Dataset'].unique():
        df_ds = df_all[df_all['Dataset'] == ds_name]
        for mn in metric_names:
            if mn not in df_ds.columns:
                continue
            means = {}
            for alg in candidate_names:
                rows = df_ds[df_ds['Algorithm'] == alg][mn].dropna()
                if len(rows):
                    means[alg] = float(rows.mean())
            if not means:
                continue
            sorted_algs = sorted(means.keys(), key=lambda a: means[a],
                                 reverse=True)
            for rank, alg in enumerate(sorted_algs, start=1):
                alg_ranks[alg].append(rank)

    avg_ranks = {a: np.mean(v) if v else 999.0
                 for a, v in alg_ranks.items()}


    sorted_by_rank = sorted(avg_ranks.items(), key=lambda x: x[1])
    print("  [FriedmanRank] Ranking of interpretable SOTA models:", flush=True)
    for alg, rk in sorted_by_rank:
        print(f"    {alg:20s}  avg_rank={rk:.2f}", flush=True)

    top_names = {a for a, _ in sorted_by_rank[:n_top]}
    return top_names

def post_process_separate_comparisons(result_dir, classifiers, metrics,
                                      cbm_names=None,
                                      sota_interp_names=None,
                                      sota_noninterp_names=None,
                                      top5_count=5):
    from pathlib import Path
    result_dir = Path(result_dir)
    csv_path   = result_dir / 'comparison.csv'
    if not csv_path.exists():
        print(f"  [SEPARATE] comparison.csv not found in {result_dir}",
              flush=True)
        return

    if cbm_names is None:
        cbm_names = CBM_NAMES
    if sota_interp_names is None:
        sota_interp_names = SOTA_INTERPRETABLE_NAMES
    if sota_noninterp_names is None:
        sota_noninterp_names = SOTA_NONINTERPRETABLE_NAMES

    try:
        df_all = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"  [SEPARATE] comparison.csv empty ({csv_path}) → skipping.",
              flush=True)
        return
    except Exception as e:
        print(f"  [SEPARATE] Error reading comparison.csv: {e} → skipping.",
              flush=True)
        return
    available_algs = set(df_all['Algorithm'].unique())
    print(f"  [SEPARATE] Available algorithms: {sorted(available_algs)}",
          flush=True)

    def _run_subset(subset_names, subdir_name):
        wanted = (cbm_names | subset_names) & available_algs
        missing = (cbm_names | subset_names) - available_algs
        if missing:
            print(f"  [SEPARATE/{subdir_name}] Missing algorithms: {missing}",
                  flush=True)
        if len(wanted) < 2:
            print(f"  [SEPARATE/{subdir_name}] Too few algorithms ({wanted}). Skipping.",
                  flush=True)
            return

        out_dir = result_dir / subdir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        df_sub  = df_all[df_all['Algorithm'].isin(wanted)].copy()
        out_csv = out_dir / 'comparison.csv'
        df_sub.to_csv(out_csv, index=False)
        print(f"  [SEPARATE/{subdir_name}] comparison.csv: "
              f"{len(df_sub)} rows, algorithms={sorted(wanted)}", flush=True)

        clf_sub = [(n, f) for n, f in classifiers if n in wanted]

        pfx = f"[SEPARATE/{subdir_name}]"
        _safe_cacp_call(process_comparison_results, out_dir, metrics,
                        label=f"{pfx} comparison_result.csv + .tex")
        _safe_cacp_call(process_comparison_results_plots, out_dir, metrics,
                        label=f"{pfx} plot/")
        _safe_cacp_call(process_comparison_result_winners, out_dir, metrics,
                        label=f"{pfx} winner/")
        _safe_cacp_call(process_times, out_dir,
                        label=f"{pfx} time/")
        _safe_cacp_call(process_wilcoxon, clf_sub, out_dir, metrics,
                        label=f"{pfx} wilcoxon/",
                        csv_dir=out_dir)

        print(f"  [SEPARATE] {subdir_name}/ → {out_dir}", flush=True)

    _banner("CBM+CBM_KE vs ALL SOTA interpretable")
    _run_subset(sota_interp_names, 'interpretable_full')

    _banner("CBM+CBM_KE vs TOP-5 SOTA interpretable (Friedman rank)")
    top5_interp = _select_top_n_by_friedman_rank(
        df_all, sota_interp_names & available_algs,
        metrics, n_top=getattr(sys.modules[__name__],
                               'SOTA_INTERPRETABLE_TOP5_COUNT', 5))
    print(f"  TOP-5 interpr. (Friedman): {sorted(top5_interp)}",
          flush=True)
    _run_subset(top5_interp, 'interpretable')

    _banner("CBM+CBM_KE vs SOTA non-interpretable")
    _run_subset(sota_noninterp_names, 'noninterpretable')

def merge_results(dir0, dir1, merged_dir):
    import glob
    os.makedirs(merged_dir, exist_ok=True)

    for fname in ['comparison.csv']:
        f0 = os.path.join(dir0, fname)
        f1_ = os.path.join(dir1, fname)
        if os.path.exists(f0) and os.path.exists(f1_):
            try:
                df0 = pd.read_csv(f0)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                print(f"  [MERGE] WARN: {f0} empty/corrupted → skipping.", flush=True)
                df0 = pd.DataFrame()
            try:
                df1 = pd.read_csv(f1_)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                print(f"  [MERGE] WARN: {f1_} empty/corrupted → skipping.", flush=True)
                df1 = pd.DataFrame()
            if df0.empty and df1.empty:
                print(f"  [MERGE] WARN: both files empty → skipping merge.", flush=True)
                continue
            merged = pd.concat([df0, df1], ignore_index=True)
            out_path = os.path.join(merged_dir, fname)
            merged.to_csv(out_path, index=False)
            print(f"  [MERGE] {fname}: {len(df0)}+{len(df1)} → {len(merged)} rows",
                  flush=True)
        elif os.path.exists(f0):
            import shutil
            shutil.copy(f0, os.path.join(merged_dir, fname))
            print(f"  [MERGE] {fname}: gpu0 only", flush=True)
        elif os.path.exists(f1_):
            import shutil
            shutil.copy(f1_, os.path.join(merged_dir, fname))
            print(f"  [MERGE] {fname}: gpu1 only", flush=True)

    for sub in ['wilcoxon', 'plot', 'winner', 'time', 'info']:
        for src_dir in [dir0, dir1]:
            src = os.path.join(src_dir, sub)
            if os.path.isdir(src):
                dst = os.path.join(merged_dir, sub)
                os.makedirs(dst, exist_ok=True)
                import shutil
                for f in glob.glob(os.path.join(src, '**', '*'), recursive=True):
                    if os.path.isfile(f):
                        rel = os.path.relpath(f, src)
                        dst_f = os.path.join(dst, rel)
                        os.makedirs(os.path.dirname(dst_f), exist_ok=True)
                        shutil.copy2(f, dst_f)

    from pathlib import Path
    merged_path = Path(merged_dir)
    _merged_csv = os.path.join(merged_dir, 'comparison.csv')
    if _csv_has_data(_merged_csv):
        print(f"  [MERGE] Running CACP post-processing on merged results...",
              flush=True)
        _safe_cacp_call(process_comparison_results, merged_path, CACP_METRICS,
                        label="comparison_result.csv + .tex")
        _safe_cacp_call(process_comparison_results_plots, merged_path, CACP_METRICS,
                        label="plot/ (merged)")
        _safe_cacp_call(process_comparison_result_winners, merged_path, CACP_METRICS,
                        label="winner/ (merged)")
        _safe_cacp_call(process_times, merged_path,
                        label="time/ (merged)")
    else:
        print(f"  [MERGE] comparison.csv empty or missing → skipping CACP.",
              flush=True)

    try:
        _clfs_all = build_classifiers()
        post_process_separate_comparisons(
            merged_dir, _clfs_all, CACP_METRICS)
        print(f"  [MERGE] ✓ interpretable/ + noninterpretable/", flush=True)
    except Exception as e:
        print(f"  [MERGE] ✗ separate comparisons: {e}", flush=True)

    print(f"  [MERGE] Merged results → {merged_dir}", flush=True)

from concurrent.futures import ThreadPoolExecutor, as_completed

# [V24] PyTorch/NumPy RNGs are process-global.  Publication-integrity mode
# serializes complete GPU tasks so parallel threads cannot overwrite one
# another's seeds during model construction, shuffling or training.
_SERIALIZE_GPU_TASKS = os.environ.get('LDRV25_SERIALIZE_GPU_TASKS', os.environ.get('LDRV24_SERIALIZE_GPU_TASKS', '1')) != '0'
_GPU_TASK_LOCK = threading.RLock()

_GPU_CLF_NAMES = {'CBM', 'CBM_KE', 'TabPFN'}


# =====================================================================
# [V21-T3] multi-seed orchestration, [V21-T4] Holm-corrected Wilcoxon,
# [V21-A5] environment manifest
# =====================================================================
def _set_run_seed(s):
    """[V21-T3] Re-seed every RNG and switch the global SEED."""
    global SEED
    SEED = int(s)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _run_experiment_all_seeds(datasets, classifiers, results_directory,
                              metrics, n_gpu_workers, n_cpu_workers,
                              n_folds=10):
    """[V21-T3] Run the full experiment once per seed.

    Single seed: behaves exactly like v20 (results in results_directory).
    Multiple seeds: per-seed subdirectories seed<NN>/ plus a concatenated
    comparison.csv (with a Seed column) at the top level, so the existing
    merge/post-processing keeps working.
    NOTE: dataset subsampling to N_MAX is performed once at load time with
    the base seed 42 (as disclosed in the manuscript); seeds vary the CV
    splits and all training randomness.
    """
    if len(_SEEDS) == 1:
        _set_run_seed(_SEEDS[0])
        parallel_run_experiment(datasets, classifiers,
                                results_directory=results_directory,
                                metrics=metrics,
                                n_gpu_workers=n_gpu_workers,
                                n_cpu_workers=n_cpu_workers,
                                n_folds=n_folds)
        return
    for _s in _SEEDS:
        _set_run_seed(_s)
        _dir_s = os.path.join(results_directory, f'seed{_s}')
        os.makedirs(_dir_s, exist_ok=True)
        print(f"\n{'='*68}\n>>> [V21-T3] SEED {_s} "
              f"({_SEEDS.index(_s)+1}/{len(_SEEDS)})\n{'='*68}", flush=True)
        parallel_run_experiment(datasets, classifiers,
                                results_directory=_dir_s,
                                metrics=metrics,
                                n_gpu_workers=n_gpu_workers,
                                n_cpu_workers=n_cpu_workers,
                                n_folds=n_folds)
    frames = []
    for _s in _SEEDS:
        _p = os.path.join(results_directory, f'seed{_s}', 'comparison.csv')
        try:
            _df = pd.read_csv(_p)
            _df['Seed'] = _s
            frames.append(_df)
        except Exception as _e:
            print(f"  [V21-T3] WARN: cannot read {_p}: {_e}", flush=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(results_directory, 'comparison.csv'), index=False)
        print(f"  [V21-T3] Concatenated {len(frames)} seed runs -> "
              f"{results_directory}/comparison.csv", flush=True)


_META_COLS = {'Dataset', 'Algorithm', 'CV index', 'Seed',
              'Number of classes', 'Train size', 'Test size',
              'Train time [s]', 'Prediction time [s]', 'Status', 'Error',
              'Protocol hash', 'Attempts'}


def _holm_adjust(pvals):
    """[V21-T4] Holm step-down adjustment. pvals: list of floats (NaN ok).
    Returns list of adjusted p in the original order."""
    import math
    idx = [i for i, p in enumerate(pvals) if p == p]      # non-NaN
    m = len(idx)
    adj = [float('nan')] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, min(1.0, val))
        adj[i] = running
    return adj


def _holm_wilcoxon_reports(comparison_csv, out_dir,
                           refs=('CBM_KE', 'CBM')):
    """[V21-T4] Pairwise Wilcoxon vs each reference with Holm correction
    across the full (algorithms x metrics) family. Operates on
    dataset-level means over folds (and seeds, if present)."""
    try:
        from scipy.stats import wilcoxon as _wilcoxon
    except ImportError:
        print("  [V21-T4] scipy unavailable -> Holm report skipped.",
              flush=True)
        return
    try:
        df = pd.read_csv(comparison_csv)
    except Exception as _e:
        print(f"  [V21-T4] cannot read {comparison_csv}: {_e}", flush=True)
        return
    metric_cols = [c for c in df.columns
                   if c not in _META_COLS
                   and pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        print("  [V21-T4] no metric columns found.", flush=True)
        return
    agg = (df.groupby(['Dataset', 'Algorithm'])[metric_cols]
             .mean().reset_index())
    algs = sorted(agg['Algorithm'].unique())
    os.makedirs(out_dir, exist_ok=True)
    for ref in refs:
        if ref not in algs:
            continue
        rows, pv = [], []
        ref_tab = agg[agg['Algorithm'] == ref].set_index('Dataset')
        for alg in algs:
            if alg == ref:
                continue
            alg_tab = agg[agg['Algorithm'] == alg].set_index('Dataset')
            common = ref_tab.index.intersection(alg_tab.index)
            for met in metric_cols:
                x = ref_tab.loc[common, met].to_numpy()
                y = alg_tab.loc[common, met].to_numpy()
                try:
                    p = float(_wilcoxon(x, y, zero_method='pratt',
                                        alternative='two-sided').pvalue)
                except Exception:
                    p = float('nan')
                rows.append([alg, met, len(common), p])
                pv.append(p)
        adj = _holm_adjust(pv)
        out = pd.DataFrame(rows, columns=['Algorithm', 'Metric',
                                          'N_datasets', 'p_raw'])
        out['p_holm'] = adj
        out['significant_at_0.05'] = out['p_holm'] < 0.05
        _path = os.path.join(out_dir, f'wilcoxon_holm_vs_{ref}.csv')
        out.to_csv(_path, index=False)
        print(f"  [V21-T4] Holm-corrected Wilcoxon vs {ref}: {_path} "
              f"(family size m={int(out['p_raw'].notna().sum())})",
              flush=True)


def _record_versions(out_dir):
    """Write a v24 environment and exact-teacher manifest."""
    import json as _json
    import platform as _platform
    info = {
        'driver': 'benchmark_driver_v25.py',
        'protocol_version': 25,
        'timestamp': datetime.datetime.now().isoformat(),
        'python': sys.version,
        'platform': _platform.platform(),
        'machine': _platform.machine(),
        'torch': torch.__version__,
        'cuda': getattr(torch.version, 'cuda', None),
        'cudnn': (torch.backends.cudnn.version()
                  if torch.cuda.is_available() else None),
        'gpus': ([torch.cuda.get_device_name(i)
                  for i in range(torch.cuda.device_count())]
                 if torch.cuda.is_available() else []),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'strict_validation': _STRICT_VALIDATION,
        'seed': SEED,
        'seeds': _SEEDS,
        'ablation': _ABLATION,
        'tf32': os.environ.get('LDRV2_TF32', '1'),
        'serialize_gpu_tasks': _SERIALIZE_GPU_TASKS,
        'tabpfn': _TABPFN_PACKAGE_VERSION,
        'tabpfn_fingerprint': (_record_tabpfn_fingerprint()
                               if _TABPFN else None),
        'v25_integrity_and_resilience': [
            'objective-after-pruning',
            'strict-inner-preprocessing-and-concept-selection',
            'inner-train-class-weights',
            'nan-not-zero-on-failure',
            'retry-failed-checkpoints',
            'M>=ceil(log2 K)-fail-fast',
            'seed-before-model-init',
            'serialized-gpu-rng',
            'explicit-tabpfn-model-path-and-sha256',
            'local-cacp-compatible-reporting-no-external-cacp',
        ],
    }
    for _mod in ('sklearn', 'optuna', 'cacp_compat', 'xgboost', 'lightgbm',
                 'catboost', 'interpret', 'imodels', 'joblib'):
        try:
            _m = __import__(_mod)
            info[_mod] = getattr(_m, '__version__', 'unknown')
        except Exception:
            info[_mod] = None
    try:
        os.makedirs(out_dir, exist_ok=True)
        _p = os.path.join(out_dir, 'environment.json')
        with open(_p, 'w', encoding='utf-8') as _f:
            _json.dump(info, _f, indent=2)
        print(f"  [V25] Environment manifest -> {_p}", flush=True)
    except Exception as _e:
        print(f"  [V25] manifest write error: {_e}", flush=True)

def _is_gpu_clf(name):
    return any(name == g or name.startswith(g + '_') for g in _GPU_CLF_NAMES)


def _run_one_task(ds_name, fold, clf_name, clf_factory, metrics_list,
                  fold_seed=None, task_context=None):
    if fold_seed is not None:
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed % (2**31))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

    n_classes  = len(fold.labels)
    train_size = len(fold.x_train)
    test_size  = len(fold.x_test)
    n_features = fold.x_train.shape[1]

    result = {
        'Dataset':             ds_name,
        'Algorithm':           clf_name,
        'Number of classes':   n_classes,
        'Train size':          train_size,
        'Test size':           test_size,
        'CV index':            fold.index,
        'Train time [s]':      float('nan'),
        'Prediction time [s]': float('nan'),
        'Status':              'FAILED',
        'Error':               '',
    }

    _gpu_lock_acquired = False
    try:
        if _SERIALIZE_GPU_TASKS and _is_gpu_clf(clf_name):
            _GPU_TASK_LOCK.acquire()
            _gpu_lock_acquired = True

        clf = clf_factory(n_features, n_classes)
        X_train_input, X_test_input = fold.x_train, fold.x_test

        try:
            _inner = clf.clf if hasattr(clf, 'clf') else clf
            if isinstance(_inner, CBMClassifier):
                _inner._resilience_context = task_context
                _inner._raw_input = True
                _inner._raw_cat_idx = getattr(fold, 'cat_idx', [])
                _inner._raw_num_idx = getattr(fold, 'num_idx', [])
                _inner._raw_feature_names = getattr(fold, 'raw_feature_names', None)
                X_train_input = fold.x_train_raw
                X_test_input = fold.x_test_raw
        except Exception:
            pass

        _t_train = time.time()
        clf.fit(X_train_input, fold.y_train)

        result['Train time [s]'] = time.time() - _t_train

        _t_pred = time.time()
        y_pred = clf.predict(X_test_input)
        result['Prediction time [s]'] = time.time() - _t_pred

        _PROBA_CACHE[threading.get_ident()] = getattr(
            clf, '_last_proba', _PROBA_CACHE.get(threading.get_ident()))

        result['Status'] = 'OK'
        for metric_name, metric_fn in metrics_list:
            try:
                value = metric_fn(fold.y_test, y_pred, labels=fold.labels)
            except TypeError:
                value = metric_fn(fold.y_test, y_pred)
            except Exception:
                value = float('nan')
            try:
                result[metric_name] = float(value)
            except Exception:
                result[metric_name] = float('nan')

        undefined = [name for name, _ in metrics_list
                     if not np.isfinite(result.get(name, float('nan')))]
        if undefined:
            result['Status'] = 'PARTIAL'
            result['Error'] = 'Undefined metrics: ' + ', '.join(undefined)

    except Exception as exc:
        print(f"  [PARALLEL] ERROR {clf_name} fold={fold.index} "
              f"ds={ds_name}: {exc}", flush=True)
        result['Status'] = 'FAILED'
        result['Error'] = repr(exc)
        for m_name, _ in metrics_list:
            result[m_name] = float('nan')
    finally:
        if _gpu_lock_acquired:
            _GPU_TASK_LOCK.release()

    return result


def _progress_bar(pct, width=24):
    filled = int(width * pct / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"


def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if d > 0:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

_DS_TIME_HISTORY = []


def _print_progress(ds_idx, n_ds, n_done_results, n_total_results,
                    t_start, ds_name, ds_time, ds_n=0, ds_d=0, ds_K=0):
    global _DS_TIME_HISTORY

    elapsed = time.time() - t_start

    if ds_n > 0 and ds_d > 0:
        _DS_TIME_HISTORY.append((ds_name, ds_n, ds_d, ds_K, ds_time))

    pct_ds  = 100.0 * (ds_idx + 1) / n_ds
    pct_res = 100.0 * n_done_results / max(1, n_total_results)
    pct     = (pct_ds + pct_res) / 2.0

    if pct > 1.0:
        eta_linear = elapsed * (100.0 - pct) / pct
    else:
        eta_linear = 0.0

    eta_ndk = None
    if len(_DS_TIME_HISTORY) >= 2:
        try:
            import numpy as _np2
            X_h = _np2.array([h[1]*h[2]*h[3] for h in _DS_TIME_HISTORY],
                             dtype=float)
            y_h = _np2.array([h[4] for h in _DS_TIME_HISTORY], dtype=float)
            a = _np2.cov(X_h, y_h)[0,1] / max(_np2.var(X_h), 1e-12)
            b = _np2.mean(y_h) - a * _np2.mean(X_h)
            _all_ds_meta = {**DATASETS_BINARY, **DATASETS_MULTI}
            done_names = {h[0] for h in _DS_TIME_HISTORY}
            remaining_cost = 0
            for _dsname, _dsinfo in _all_ds_meta.items():
                if _dsname not in done_names:
                    remaining_cost += 1000 * 20 * 5
            eta_ndk = max(0, a * remaining_cost + b * len(_all_ds_meta))
        except Exception:
            eta_ndk = None

    if eta_ndk is not None and eta_ndk > 0:
        eta_s = eta_ndk
        eta_method = "n×d×K"
    else:
        eta_s = eta_linear
        eta_method = "linear"

    bar       = _progress_bar(pct)
    t_elapsed = _fmt_time(elapsed)
    t_eta     = _fmt_time(eta_s)
    t_finish  = datetime.datetime.now() + datetime.timedelta(seconds=eta_s)
    t_finish_str = t_finish.strftime('%Y-%m-%d %H:%M')

    speed = (n_done_results / elapsed * 60) if elapsed > 1 else 0.0

    print(f"\n{'═'*65}", flush=True)
    print(f"  PROGRESS {bar} {pct:5.1f}%  |  dataset {ds_idx+1}/{n_ds}  "
          f"[{ds_name}]", flush=True)
    print(f"  Results:   {n_done_results}/{n_total_results} CSV rows "
          f"|  Speed: {speed:.1f} results/min", flush=True)
    print(f"  Elapsed:  {t_elapsed}", flush=True)
    print(f"  ETA:      ~{t_eta}  (method: {eta_method})", flush=True)
    print(f"  Total est.:   ~{t_finish_str}", flush=True)
    print(f"  Last dataset:  {ds_name} {ds_n}×{ds_d} K={ds_K} → {ds_time:.0f}s "
          f"({ds_time/3600:.2f}h)", flush=True)

    print(f"  CBM cfg:  trials={CBM_N_TRIALS} epochs_hpo={CBM_EPOCHS_HPO} "
          f"patience={CBM_PATIENCE_ES} final={CBM_EPOCHS_FINAL} "
          f"restarts={CBM_N_RESTARTS}", flush=True)
    print(f"{'═'*65}\n", flush=True)

import threading as _pt_threading

class _ProgressTracker:
    def __init__(self):
        self._lock        = _pt_threading.Lock()
        self.ds_total     = 0
        self.ds_done      = 0
        self.ds_name      = ""
        self.tasks_total  = 0
        self.tasks_done   = 0
        self.t_start      = time.time()
        self.t_ds_start   = time.time()
        self.t_last_task  = time.time()   # [V02-6] for STALL detection
        self._last_print  = 0.0
        self._print_every = 10.0

    def reset(self, ds_total, tasks_total):
        with self._lock:
            self.ds_total    = ds_total
            self.tasks_total = tasks_total
            self.ds_done     = 0
            self.tasks_done  = 0
            self.t_start     = time.time()
            self.t_ds_start  = time.time()
            self.t_last_task = time.time()

    def snapshot(self):
        """[V02-6] Atomic state snapshot for the resource monitor."""
        with self._lock:
            return dict(pct=self._pct(), eta=self._eta(),
                        ds_done=self.ds_done, ds_total=self.ds_total,
                        tasks_done=self.tasks_done,
                        tasks_total=self.tasks_total,
                        ds_name=self.ds_name,
                        t_last_task=self.t_last_task,
                        elapsed=time.time() - self.t_start)

    def start_dataset(self, ds_name):
        with self._lock:
            self.ds_name    = ds_name
            self.t_ds_start = time.time()
        self._print_stage(force=True)

    def task_done(self, clf_name, fold_idx, t_task):
        with self._lock:
            self.tasks_done += 1
            self.t_last_task = time.time()
        now = time.time()
        with self._lock:
            last = self._last_print
        if now - last >= self._print_every:
            with self._lock:
                self._last_print = now
            self._print_inline(clf_name, fold_idx, t_task)

    def dataset_done(self, ds_name, n_results, t_ds):
        with self._lock:
            self.ds_done += 1
        self._print_stage(force=True, ds_done=True, n_res=n_results, t_ds=t_ds)

    def _pct(self):
        p_tasks = self.tasks_done / max(1, self.tasks_total)
        p_ds    = self.ds_done    / max(1, self.ds_total)
        return 100.0 * (0.7 * p_tasks + 0.3 * p_ds)

    def _bar(self, pct, w=28):
        f = int(w * pct / 100)
        return "█"*f + "░"*(w-f)

    def _fmt(self, s):
        s = int(max(0, s))
        d, s = divmod(s, 86400)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"

    def _eta(self):
        elapsed = time.time() - self.t_start
        pct     = self._pct()
        if pct > 1.0:
            return elapsed * (100.0 - pct) / pct
        return 0.0

    def _print_inline(self, clf_name, fold_idx, t_task):
        with self._lock:
            pct  = self._pct()
            eta  = self._eta()
            ds_d = self.ds_done
            ds_t = self.ds_total
            ds_n = self.ds_name
            td   = self.tasks_done
            tt   = self.tasks_total
        t_finish = datetime.datetime.now() + datetime.timedelta(seconds=eta)
        bar      = self._bar(pct)
        print(f"  [{bar}] {pct:5.1f}%"
              f"  ds {ds_d}/{ds_t}"
              f"  task {td}/{tt}"
              f"  |  {ds_n}/{clf_name} f{fold_idx}"
              f"  |  {t_task:.0f}s"
              f"  |  ETA {self._fmt(eta)}"
              f"  ({t_finish.strftime('%H:%M')})",
              flush=True)

    def _print_stage(self, force=False, ds_done=False, n_res=0, t_ds=0):
        now = time.time()
        with self._lock:
            last = self._last_print
            if not force and (now - last < self._print_every):
                return
            self._last_print = now
            pct  = self._pct()
            eta  = self._eta()
            elapsed = now - self.t_start
            ds_d = self.ds_done
            ds_t = self.ds_total
            ds_n = self.ds_name
            td   = self.tasks_done
            tt   = self.tasks_total
        bar      = self._bar(pct, w=32)
        t_finish = datetime.datetime.now() + datetime.timedelta(seconds=eta)

        print(f"", flush=True)
        print(f"  ╔══════════════════════════════════════════════════════════════╗",
              flush=True)
        if ds_done:
            print(f"  ║  ✓ DATASET COMPLETED: {ds_n:25s}  {n_res} results  {t_ds:.0f}s",
                  flush=True)
        print(f"  ║  PROGRESS: [{bar}] {pct:5.1f}%", flush=True)
        print(f"  ║  Datasets:  {ds_d:2d}/{ds_t:2d} ({100*ds_d/max(1,ds_t):.0f}%)"
              f"  |  Tasks: {td:5d}/{tt:5d} ({100*td/max(1,tt):.0f}%)",
              flush=True)
        print(f"  ║  Elapsed: {self._fmt(elapsed)}"
              f"  |  ETA: ~{self._fmt(eta)}"
              f"  |  Finish: ~{t_finish.strftime('%Y-%m-%d %H:%M')}",
              flush=True)
        print(f"  ║  CBM: trials={CBM_N_TRIALS} hpo={CBM_EPOCHS_HPO}"
              f" pat={CBM_PATIENCE_ES} final={CBM_EPOCHS_FINAL}"
              f" rest={CBM_N_RESTARTS}",
              flush=True)
        if not ds_done:
            print(f"  ║  Current dataset: {ds_n}", flush=True)
        print(f"  ╚══════════════════════════════════════════════════════════════╝",
              flush=True)
        print(f"", flush=True)


_PROGRESS = _ProgressTracker()


# =====================================================================
# [V02-6] RESOURCE MONITOR: CPU / RAM / GPU / progress / ETA / STALL
# =====================================================================
_STALL_THRESHOLD_S = 1800   # 30 min without a finished task => warning


def _read_gpu_stats():
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                rows.append(parts)
        return rows
    except Exception:
        return []


def _read_ram_gib():
    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                info[k] = int(v.split()[0])          # kB
        tot   = info.get('MemTotal', 0)     / 1048576.0
        avail = info.get('MemAvailable', 0) / 1048576.0
        return tot, tot - avail
    except Exception:
        return 0.0, 0.0


def _resource_monitor_worker(interval: int = 60):
    n_cores = _mp.cpu_count()
    while True:
        time.sleep(max(5, interval))
        try:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            try:
                la1, la5, _ = os.getloadavg()
            except OSError:
                la1 = la5 = 0.0
            ram_tot, ram_used = _read_ram_gib()
            gpu_rows = _read_gpu_stats()
            gpu_str = "  ".join(
                f"GPU{r[0]}: {r[1]}% util, {r[2]}/{r[3]} MiB"
                for r in gpu_rows) or "GPU: n/a"
            snap = _PROGRESS.snapshot()
            eta_str = _fmt_time(snap['eta'])
            stall_s = time.time() - snap['t_last_task']
            line = (f"  [MON {ts}] CPU load1m={la1:.1f}/{n_cores}"
                    f" | RAM {ram_used:.1f}/{ram_tot:.0f} GiB"
                    f" | {gpu_str}"
                    f" | tasks {snap['tasks_done']}/{snap['tasks_total']}"
                    f" ({snap['pct']:.1f}%)"
                    f" | ETA ~{eta_str}")
            print(line, flush=True)
            # [V02-9] STALL must fire ALSO when tasks_done == 0: the
            # observed v01 failure stalled BEFORE the first task completed
            # (startup network call), so a tasks_done>0 guard would have
            # hidden it for 10 days exactly like the heartbeat did.
            if stall_s > _STALL_THRESHOLD_S:
                _where = ("startup/dataset-loading/pre-warm phase"
                          if snap['tasks_done'] == 0
                          else f"dataset='{snap['ds_name']}'")
                print(f"  [MON][STALL] No task finished for "
                      f"{stall_s/60:.0f} min ({_where}). "
                      f"Check: py-spy dump --pid {os.getpid()}", flush=True)
        except Exception:
            pass


def _start_resource_monitor(interval: int = 60):
    t = threading.Thread(target=_resource_monitor_worker,
                         args=(interval,), daemon=True)
    t.start()
    print(f"  [V02] Resource monitor started (every {interval}s)", flush=True)

def parallel_run_experiment(datasets, classifiers, results_directory, metrics,
                            n_gpu_workers=5, n_cpu_workers=14, n_folds=10):
    os.makedirs(results_directory, exist_ok=True)

    gpu_clfs = [(n, f) for n, f in classifiers if _is_gpu_clf(n)]
    cpu_clfs = [(n, f) for n, f in classifiers if not _is_gpu_clf(n)]

    print(f"\n  [PARALLEL] GPU classifiers ({len(gpu_clfs)}): "
          f"{[n for n,_ in gpu_clfs]}", flush=True)
    print(f"  [PARALLEL] CPU classifiers ({len(cpu_clfs)}): "
          f"{[n for n,_ in cpu_clfs]}", flush=True)
    print(f"  [PARALLEL] GPU workers={n_gpu_workers}, "
          f"CPU workers={n_cpu_workers}", flush=True)

    all_results = []
    _t_total_start = time.time()

    _n_folds       = n_folds
    _n_clf         = len(classifiers)
    _n_ds          = len(datasets)
    _n_total_res   = _n_ds * _n_folds * _n_clf
    _n_done_res    = 0
    print(f"  [PROGRESS] Estimated number of results: "
          f"{_n_ds} datasets × {_n_folds} folds × {_n_clf} clf "
          f"= {_n_total_res} CSV rows", flush=True)
    _PROGRESS.reset(ds_total=_n_ds, tasks_total=_n_total_res)

    # -----------------------------------------------------------------
    # [V02-5] CHECKPOINTING: every completed (Dataset, Algorithm, fold)
    # is immediately appended to checkpoint_rows.csv; after a restart
    # these tasks are skipped (unless --no-resume).
    # -----------------------------------------------------------------
    _ckpt_path = os.path.join(results_directory, 'checkpoint_rows.csv')
    _ckpt_lock = _pt_threading.Lock()
    _done_keys = set()
    if _RESUME and os.path.exists(_ckpt_path):
        try:
            _df_ck = pd.read_csv(_ckpt_path)
            if 'Status' not in _df_ck.columns:
                _df_ck['Status'] = 'OK'
            _status = _df_ck['Status'].fillna('OK').astype(str).str.upper()
            _df_ok = _df_ck[_status.isin(['OK', 'PARTIAL'])].copy()
            _df_ok = _df_ok.drop_duplicates(
                subset=['Dataset', 'Algorithm', 'CV index'], keep='last')
            for _, _r in _df_ok.iterrows():
                _done_keys.add((str(_r['Dataset']), str(_r['Algorithm']),
                                int(_r['CV index'])))
            all_results.extend(_df_ok.to_dict('records'))
            _n_done_res += len(_df_ok)
            with _PROGRESS._lock:
                _PROGRESS.tasks_done = len(_df_ok)
            print(f"  [V24-RESUME] checkpoint: {len(_done_keys)} successful/partial tasks "
                  f"will be skipped; FAILED tasks will be retried", flush=True)
        except Exception as _ce:
            print(f"  [V02-RESUME] checkpoint unreadable ({_ce}) → "
                  f"starting from scratch", flush=True)
            _done_keys = set()

    def _ckpt_append(res: dict):
        try:
            with _ckpt_lock:
                _hdr = not os.path.exists(_ckpt_path)
                pd.DataFrame([res]).to_csv(
                    _ckpt_path, mode='a', header=_hdr, index=False)
        except Exception as _we:
            print(f"  [V02-CKPT] write error: {_we}", flush=True)

    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device=device)
            torch.cuda.synchronize()
            print(f"  [PRE-WARM] CUDA context initialized on {device}",
                  flush=True)
        except Exception as _e:
            print(f"  [PRE-WARM] CUDA init warning: {_e}", flush=True)

    if _TABPFN and any(_is_gpu_clf(n) for n, _ in classifiers):
        print(f"  [PRE-WARM] TabPFN model-weight pre-loading...", flush=True)
        # [V02-10] Pre-warm runs in a watchdog thread: even if a network
        # layer ignores the global socket timeout, the main flow continues
        # after LDRV2_PREWARM_TIMEOUT seconds (default 600).
        def _prewarm_tabpfn():
            try:
                import tempfile
                import numpy as np
                _X_dummy = np.random.rand(10, 4).astype(np.float32)
                _y_dummy = np.array([0]*5 + [1]*5)
                _tabpfn_pw = _make_tabpfn_v25(device)
                _tabpfn_pw.fit(_X_dummy, _y_dummy)
                del _tabpfn_pw, _X_dummy, _y_dummy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  [PRE-WARM] TabPFN OK (weights in cache, GPU warm)",
                      flush=True)
            except Exception as _e:
                print(f"  [PRE-WARM] TabPFN warning: {_e} (continuing)",
                      flush=True)
        _pw_t = _pt_threading.Thread(target=_prewarm_tabpfn, daemon=True)
        _pw_t.start()
        _pw_to = float(os.environ.get('LDRV2_PREWARM_TIMEOUT', '600'))
        _pw_t.join(timeout=_pw_to)
        if _pw_t.is_alive():
            print(f"  [PRE-WARM][WARN] TabPFN pre-warm exceeded "
                  f"{_pw_to:.0f}s -> continuing without it "
                  f"(weights will load lazily on first TabPFN fit).",
                  flush=True)

    for ds_idx, ds in enumerate(datasets):
        _t_ds_start = time.time()
        n_cls = len(np.unique(ds.y))
        n, d  = ds.X_raw.shape

        print(f"\n{'─'*60}", flush=True)
        print(f"  [{ds_idx+1}/{len(datasets)}] {ds.name:20s}  "
              f"n={n}, d={d}, K={n_cls}", flush=True)
        _PROGRESS.start_dataset(ds.name)
        print(f"{'─'*60}", flush=True)

        _t_folds = time.time()
        # [V21-T3] CV split depends on the CURRENT global seed
        folds = list(ds.folds(n_folds=_n_folds, random_state=SEED))
        print(f"  Folds: {len(folds)} (pre-computed in "
              f"{time.time()-_t_folds:.1f}s)", flush=True)

        # ----------------------- CPU TASKS [V02-2] ---------------------
        cpu_task_args = []
        for fold in folds:
            for clf_name, clf_factory in cpu_clfs:
                if (ds.name, clf_name, fold.index) in _done_keys:
                    continue
                seed_f = SEED + fold.index
                cpu_task_args.append(
                    (ds.name, fold, clf_name, clf_factory, metrics, seed_f))

        n_cpu_tasks = len(cpu_task_args)
        n_cpu_skip  = len(folds) * len(cpu_clfs) - n_cpu_tasks
        print(f"  CPU tasks: {n_cpu_tasks} "
              f"({len(cpu_clfs)} clf × {len(folds)} folds"
              f"{', skipped(ckpt)=' + str(n_cpu_skip) if n_cpu_skip else ''}) "
              f"| backend={_CPU_BACKEND}", flush=True)

        ds_results   = []
        _cpu_results = []
        _cpu_lock    = _pt_threading.Lock()

        def _on_cpu_result(res):
            with _cpu_lock:
                _cpu_results.append(res)
            _ckpt_append(res)
            _PROGRESS.task_done(res['Algorithm'], res['CV index'],
                                res.get('Train time [s]', 0))

        def _cpu_threads_run(task_args):
            ex = ThreadPoolExecutor(max_workers=n_cpu_workers,
                                    thread_name_prefix=f'cpu_{ds.name}')
            futs = {ex.submit(_run_one_task, a0, a1, a2, a3, a4,
                              fold_seed=a5): (a2, a1.index)
                    for (a0, a1, a2, a3, a4, a5) in task_args}
            for fut in as_completed(futs):
                clf_n2, fold_i2 = futs[fut]
                try:
                    res = fut.result()
                except Exception as _fe2:
                    print(f"  [CPU-ERROR] {clf_n2} fold={fold_i2}: "
                          f"{type(_fe2).__name__}: {_fe2}", flush=True)
                    continue
                _on_cpu_result(res)
            ex.shutdown(wait=True)

        def _cpu_collector():
            """Runs CPU classifiers in PROCESSES (joblib/loky), streaming
            results as they finish; falls back to v01 thread pool."""
            if not cpu_task_args:
                return
            if _CPU_BACKEND != 'loky':
                _cpu_threads_run(cpu_task_args)
                return
            try:
                from joblib import Parallel, delayed
                try:
                    _par = Parallel(n_jobs=n_cpu_workers, backend='loky',
                                    return_as='generator_unordered')
                except (TypeError, ValueError):
                    _par = Parallel(n_jobs=n_cpu_workers, backend='loky',
                                    return_as='generator')
                _gen = _par(delayed(_run_one_task)(a0, a1, a2, a3, a4,
                                                   fold_seed=a5)
                            for (a0, a1, a2, a3, a4, a5) in cpu_task_args)
                _n_cpu_done = 0
                for res in _gen:
                    _on_cpu_result(res)
                    _n_cpu_done += 1
                    if _n_cpu_done % 20 == 0 or _n_cpu_done == n_cpu_tasks:
                        _pct2 = 100.0 * _n_cpu_done / max(1, n_cpu_tasks)
                        print(f"    CPU [{_pct2:5.1f}%] "
                              f"{_n_cpu_done}/{n_cpu_tasks}  "
                              f"{res['Algorithm']} f{res['CV index']}",
                              flush=True)
            except Exception as _le:
                print(f"  [V02-CPU] loky failed "
                      f"({type(_le).__name__}: {_le}) → thread fallback",
                      flush=True)
                with _cpu_lock:
                    _done_now = {(r['Dataset'], r['Algorithm'],
                                  int(r['CV index'])) for r in _cpu_results}
                _remaining = [a for a in cpu_task_args
                              if (a[0], a[2], a[1].index) not in _done_now]
                _cpu_threads_run(_remaining)

        _cpu_thread = _pt_threading.Thread(target=_cpu_collector, daemon=True)
        _cpu_thread.start()

        # ----------------------- GPU TASKS -----------------------------
        gpu_futures = {}
        gpu_executor = ThreadPoolExecutor(
            max_workers=n_gpu_workers,
            thread_name_prefix=f'gpu_{ds.name}')

        n_gpu_skip = 0
        for fold in folds:
            for clf_name, clf_factory in gpu_clfs:
                if (ds.name, clf_name, fold.index) in _done_keys:
                    n_gpu_skip += 1
                    continue
                seed_f = SEED + fold.index
                fut = gpu_executor.submit(
                    _run_one_task, ds.name, fold, clf_name,
                    clf_factory, metrics, fold_seed=seed_f)
                gpu_futures[fut] = (clf_name, fold.index)

        n_gpu_tasks = len(gpu_futures)
        print(f"  GPU tasks: {n_gpu_tasks} "
              f"({len(gpu_clfs)} clf × {len(folds)} folds"
              f"{', skipped(ckpt)=' + str(n_gpu_skip) if n_gpu_skip else ''})",
              flush=True)

        n_done = 0
        # [V02-7] NOTE: in v01, fut.result(timeout=3600) after as_completed
        # could never raise TimeoutError (as_completed yields only ALREADY
        # completed futures). A hung task is now detected by the [STALL]
        # monitor instead of the dead timeout.
        for fut in as_completed(gpu_futures):
            clf_n, fold_i = gpu_futures[fut]
            try:
                res = fut.result()
            except Exception as _fe:
                print(f"  [GPU-ERROR] {clf_n} fold={fold_i}: "
                      f"{type(_fe).__name__}: {_fe}", flush=True)
                continue
            ds_results.append(res)
            _ckpt_append(res)
            n_done += 1
            if n_done % 3 == 0 or n_done == n_gpu_tasks:
                _pct = 100.0 * n_done / max(1, n_gpu_tasks)
                print(f"    GPU [{_pct:5.1f}%] {n_done}/{n_gpu_tasks}  "
                      f"{clf_n} f{fold_i} "
                      f"{res['Train time [s]']:.0f}s", flush=True)
            _PROGRESS.task_done(clf_n, fold_i, res.get('Train time [s]', 0))

        gpu_executor.shutdown(wait=True)
        _t_gpu_done = time.time()
        print(f"  GPU done: {_t_gpu_done - _t_ds_start:.1f}s", flush=True)

        _cpu_thread.join()
        with _cpu_lock:
            ds_results.extend(_cpu_results)

        all_results.extend(ds_results)
        _n_done_res += len(ds_results)
        _dt_ds = time.time() - _t_ds_start
        _PROGRESS.dataset_done(ds.name, len(ds_results), _dt_ds)
        print(f"  ✓ {ds.name}: {len(ds_results)} results in {_dt_ds:.1f}s "
              f"({_dt_ds/len(folds):.1f}s/fold)", flush=True)

        _print_progress(
            ds_idx      = ds_idx,
            n_ds        = _n_ds,
            n_done_results  = _n_done_res,
            n_total_results = _n_total_res,
            t_start     = _t_total_start,
            ds_name     = ds.name,
            ds_time     = _dt_ds,
            ds_n        = n,
            ds_d        = d,
            ds_K        = n_cls,
        )

        del folds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from pathlib import Path
    result_dir = Path(results_directory)

    df = pd.DataFrame(all_results)
    csv_path = result_dir / 'comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"  [PARALLEL] comparison.csv: {len(df)} rows (fold-level)", flush=True)

    if len(df) == 0:
        print(f"  [CACP-SKIP] No results (len(df)==0) → skipping post-processing.",
              flush=True)
    else:
        _safe_cacp_call(process_comparison_results, result_dir, metrics,
                        label="comparison_result.csv + .tex")
        _safe_cacp_call(process_comparison_results_plots, result_dir, metrics,
                        label="plot/ (distribution plots)")
        _safe_cacp_call(process_comparison_result_winners, result_dir, metrics,
                        label="winner/ (rankings)")
        _safe_cacp_call(process_times, result_dir,
                        label="time/ (runtime)")
        _safe_cacp_call(process_wilcoxon, classifiers, result_dir, metrics,
                        label="wilcoxon/ (statistical tests)",
                        csv_dir=result_dir)
        try:
            dataset_info(datasets, result_dir)
            classifier_info(classifiers, result_dir)
            print(f"  [CACP] ✓ info/ (datasets + classifiers metadata)",
                  flush=True)
        except Exception as e:
            print(f"  [CACP] ✗ info: {type(e).__name__} – {e}", flush=True)

        try:
            post_process_separate_comparisons(
                result_dir, classifiers, metrics)
            print(f"  [CACP] ✓ interpretable/ + noninterpretable/", flush=True)
        except Exception as e:
            print(f"  [CACP] ✗ separate comparisons: {type(e).__name__} – {e}",
                  flush=True)

    _dt_total = time.time() - _t_total_start
    print(f"\n{'='*60}", flush=True)
    print(f"  PARALLEL RUNNER FINISHED", flush=True)
    print(f"  Datasets: {len(datasets)}, Classifiers: {len(classifiers)}", flush=True)
    print(f"  Total results: {len(all_results)}", flush=True)
    print(f"  Time: {str(datetime.timedelta(seconds=int(_dt_total)))}", flush=True)
    print(f"  CSV: {csv_path}", flush=True)
    print(f"{'='*60}", flush=True)

    return df

_T0_GLOBAL = time.time()


def _elapsed() -> str:
    s = int(time.time() - _T0_GLOBAL)
    return str(datetime.timedelta(seconds=s))


def _banner(msg: str, width: int = 70) -> None:
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*width}", flush=True)
    print(f"  [{ts}]  +{_elapsed()}  {msg}", flush=True)
    print(f"{'='*width}", flush=True)


def _heartbeat_worker(interval: int = 120) -> None:
    while True:
        time.sleep(interval)
        ts  = datetime.datetime.now().strftime('%H:%M:%S')
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
        print(f"  [♥ HEARTBEAT gpu={gpu}  elapsed={_elapsed()}  {ts}]",
              flush=True)


def _start_heartbeat(interval: int = 120) -> None:
    t = __import__('threading').Thread(
        target=_heartbeat_worker, args=(interval,), daemon=True)
    t.start()


def _sighandler(sig, frame):
    print(f"\n[SIGNAL {sig}] Received interrupt signal. ",
          f"Elapsed={_elapsed()}", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT,  _sighandler)
signal.signal(signal.SIGTERM, _sighandler)



# ============================================================================
# V25 RESILIENT EXECUTION ENGINE
# ============================================================================
_V25_TERMINAL = {'OK', 'PARTIAL', 'PERMANENT_FAILURE'}
_V25_SUCCESS = {'OK', 'PARTIAL'}


def _is_retryable_gpu_error_text(text):
    s = str(text).lower()
    needles = (
        'cuda out of memory', 'outofmemoryerror', 'cublas_status_alloc_failed',
        'cudnn_status_alloc_failed', 'cuda error', 'driver shutting down',
        'device is busy or unavailable', 'illegal memory access', 'nccl',
        'misaligned address', 'unspecified launch failure', 'context is destroyed',
        'connection reset', 'timed out', 'timeout', 'temporarily unavailable',
    )
    return any(n in s for n in needles)


def _v25_now():
    return time.time()


def _v25_safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def _v25_file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _v25_run_paths():
    run_id = _v25_safe_name(getattr(_ARGS, 'run_id', 'v25_publication'))
    root = os.path.abspath(os.path.join('results_LDR', run_id))
    db = (os.path.abspath(_ARGS.state_db) if getattr(_ARGS, 'state_db', None)
          else os.path.join(root, 'run_state.sqlite3'))
    return run_id, root, db


def _v25_protocol_payload():
    ckpt = globals().get('_TABPFN_MODEL_PATH', '')
    return {
        'driver_sha256': _v25_file_sha256(__file__),
        'checkpoint_sha256': (_v25_file_sha256(ckpt)
                              if ckpt and os.path.isfile(ckpt) else None),
        'datasets_group': _ARGS.datasets_group or '2',
        'n_folds': int(_ARGS.n_folds),
        'seeds': list(_SEEDS),
        'trials': int(CBM_N_TRIALS),
        'epochs_hpo': int(CBM_EPOCHS_HPO),
        'epochs_final': int(CBM_EPOCHS_FINAL),
        'patience': int(CBM_PATIENCE_ES),
        'restarts': int(CBM_N_RESTARTS),
        'strict_validation': bool(_STRICT_VALIDATION),
        'ablation': bool(_ABLATION),
    }


def _v25_protocol_hash():
    # [V25.2] A controlled resume may keep the original protocol hash after
    # an execution-engine-only repair.  The resume script reads the hash from
    # the existing SQLite run and exports it to every worker.
    override = os.environ.get('LDRV25_PROTOCOL_HASH_OVERRIDE', '').strip().lower()
    if override:
        if not re.fullmatch(r'[0-9a-f]{64}', override):
            raise ValueError('LDRV25_PROTOCOL_HASH_OVERRIDE must be 64 hex characters.')
        return override
    blob = json.dumps(_v25_protocol_payload(), sort_keys=True,
                      separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def _v25_connect(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path, timeout=120, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=FULL')
    con.execute('PRAGMA busy_timeout=120000')
    con.execute('PRAGMA foreign_keys=ON')
    return con


@contextmanager
def _v25_db(db_path):
    """Open and always close one SQLite connection.

    sqlite3.Connection.__exit__ commits or rolls back but does not close the
    connection. V25.2 used ``with _v25_db(...)`` and therefore leaked
    file descriptors during repeated heartbeat and recovery operations.
    """
    con = _v25_connect(db_path)
    try:
        yield con
    finally:
        con.close()


def _v25_open_fd_count():
    try:
        return len(os.listdir('/proc/self/fd'))
    except Exception:
        return -1


def _v25_init_schema(db_path):
    with _v25_db(db_path) as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS runs (
            protocol_hash TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            protocol_hash TEXT NOT NULL,
            seed INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            fold INTEGER NOT NULL,
            kind TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 100,
            status TEXT NOT NULL DEFAULT 'PENDING',
            attempts INTEGER NOT NULL DEFAULT 0,
            worker_id TEXT,
            gpu_id INTEGER,
            lease_until REAL,
            next_retry_at REAL NOT NULL DEFAULT 0,
            last_error TEXT,
            result_json TEXT,
            result_path TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY(protocol_hash, seed, dataset, algorithm, fold)
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_claim
          ON tasks(protocol_hash, kind, status, next_retry_at, priority);
        CREATE TABLE IF NOT EXISTS workers (
            protocol_hash TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            role TEXT NOT NULL,
            pid INTEGER,
            gpu_id INTEGER,
            status TEXT,
            current_task TEXT,
            heartbeat REAL NOT NULL,
            message TEXT,
            PRIMARY KEY(protocol_hash, worker_id)
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol_hash TEXT NOT NULL,
            ts REAL NOT NULL,
            worker_id TEXT,
            level TEXT,
            event TEXT,
            details TEXT
        );
        ''')


def _v25_event(db, ph, worker, level, event, details=''):
    try:
        with _v25_db(db) as con:
            con.execute('INSERT INTO events(protocol_hash,ts,worker_id,level,event,details) '
                        'VALUES(?,?,?,?,?,?)',
                        (ph, _v25_now(), worker, level, event, str(details)[:10000]))
    except Exception:
        pass


def _v25_worker_heartbeat(db, ph, worker_id, role, status,
                          current_task=None, gpu_id=None, message=''):
    now = _v25_now()
    with _v25_db(db) as con:
        con.execute('''INSERT INTO workers
            (protocol_hash,worker_id,role,pid,gpu_id,status,current_task,heartbeat,message)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(protocol_hash,worker_id) DO UPDATE SET
              role=excluded.role,pid=excluded.pid,gpu_id=excluded.gpu_id,
              status=excluded.status,current_task=excluded.current_task,
              heartbeat=excluded.heartbeat,message=excluded.message''',
            (ph, worker_id, role, os.getpid(), gpu_id, status,
             current_task, now, str(message)[:2000]))


def _v25_reset_expired(con, ph):
    now = _v25_now()
    con.execute('''UPDATE tasks SET status='RETRYABLE', worker_id=NULL,
        gpu_id=NULL, lease_until=NULL, next_retry_at=?,
        last_error=COALESCE(last_error,'') || '\n[V25] expired worker lease',
        updated_at=?
        WHERE protocol_hash=? AND status='RUNNING'
          AND lease_until IS NOT NULL AND lease_until < ?''',
        (now, now, ph, now))


def _v25_requeue_worker_tasks(db, ph, worker_id, reason):
    # [V25.2] A dead worker cannot renew its lease. Requeue all tasks still
    # owned by that worker immediately instead of waiting lease_seconds.
    now = _v25_now()
    with _v25_db(db) as con:
        changed = con.execute('''UPDATE tasks
            SET status='RETRYABLE', worker_id=NULL, gpu_id=NULL,
                lease_until=NULL, next_retry_at=?,
                last_error=COALESCE(last_error,'') || ?,
                updated_at=?
            WHERE protocol_hash=? AND status='RUNNING' AND worker_id=?''',
            (now, '\n[V25.2] worker terminated: ' + str(reason)[:1000],
             now, ph, worker_id)).rowcount
        con.execute('''UPDATE workers SET status='CRASHED',
            current_task=NULL, heartbeat=?, message=?
            WHERE protocol_hash=? AND worker_id=?''',
            (now, str(reason)[:2000], ph, worker_id))
    if changed:
        print(f"[V25.2-REQUEUE] {worker_id}: {changed} task(s) -> RETRYABLE; "
              f"reason={reason}", flush=True)
    return int(changed)


def _v25_claim_task(db, ph, kind, worker_id, gpu_id, lease_seconds):
    now = _v25_now()
    con = _v25_connect(db)
    try:
        con.execute('BEGIN IMMEDIATE')
        _v25_reset_expired(con, ph)
        row = con.execute('''SELECT * FROM tasks
            WHERE protocol_hash=? AND kind=?
              AND status IN ('PENDING','RETRYABLE')
              AND next_retry_at <= ?
            ORDER BY priority, seed, dataset, fold, algorithm
            LIMIT 1''', (ph, kind, now)).fetchone()
        if row is None:
            con.execute('COMMIT')
            return None
        key = (row['protocol_hash'], row['seed'], row['dataset'],
               row['algorithm'], row['fold'])
        updated = con.execute('''UPDATE tasks SET status='RUNNING',
            attempts=attempts+1, worker_id=?, gpu_id=?, lease_until=?,
            updated_at=?
            WHERE protocol_hash=? AND seed=? AND dataset=? AND algorithm=?
              AND fold=? AND status IN ('PENDING','RETRYABLE')''',
            (worker_id, gpu_id, now + lease_seconds, now, *key)).rowcount
        if updated != 1:
            con.execute('ROLLBACK')
            return None
        claimed = con.execute('''SELECT * FROM tasks WHERE protocol_hash=?
            AND seed=? AND dataset=? AND algorithm=? AND fold=?''', key).fetchone()
        con.execute('COMMIT')
        return dict(claimed)
    except Exception:
        try: con.execute('ROLLBACK')
        except Exception: pass
        raise
    finally:
        con.close()


def _v25_renew_lease(db, task, worker_id, lease_seconds):
    now = _v25_now()
    with _v25_db(db) as con:
        con.execute('''UPDATE tasks SET lease_until=?, updated_at=?
            WHERE protocol_hash=? AND seed=? AND dataset=? AND algorithm=?
              AND fold=? AND status='RUNNING' AND worker_id=?''',
            (now + lease_seconds, now, task['protocol_hash'], task['seed'],
             task['dataset'], task['algorithm'], task['fold'], worker_id))


def _v25_task_id(task):
    return (f"s{task['seed']}:{task['dataset']}:{task['algorithm']}:"
            f"f{task['fold']}")


def _v25_task_dir(root, task):
    return os.path.join(root, 'state', task['protocol_hash'][:16],
                        f"seed{task['seed']}",
                        _v25_safe_name(task['dataset']),
                        _v25_safe_name(task['algorithm']),
                        f"fold{task['fold']}")


def _v25_atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, allow_nan=True, indent=2)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)



def _v25_recover_atomic_results(root, db, ph):
    # Reconcile JSON files written just before a master/worker crash.
    # [V25.3] One explicitly closed SQLite connection is reused for the full
    # scan; V25.2 opened one connection per JSON file and leaked descriptors.
    directory = os.path.join(root, 'task_results', ph[:16])
    if not os.path.isdir(directory):
        return 0
    recovered = 0
    sql = ("UPDATE tasks SET status=?, result_json=?, result_path=?, "
           "last_error=?, lease_until=NULL, next_retry_at=0, "
           "worker_id=NULL, gpu_id=NULL, updated_at=? "
           "WHERE protocol_hash=? AND seed=? AND dataset=? "
           "AND algorithm=? AND fold=? "
           "AND status NOT IN ('OK','PARTIAL')")
    with _v25_db(db) as con:
        for path in Path(directory).glob('*.json'):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    result = json.load(fh)
                if result.get('Protocol hash') != ph:
                    continue
                status = str(result.get('Status', '')).upper()
                if status not in _V25_SUCCESS:
                    continue
                key = (ph, int(result['Seed']), str(result['Dataset']),
                       str(result['Algorithm']), int(result['CV index']))
                payload = json.dumps(result, ensure_ascii=False, allow_nan=True)
                now = _v25_now()
                changed = con.execute(
                    sql, (status, payload, str(path), result.get('Error',''),
                          now, *key)).rowcount
                recovered += int(changed)
            except Exception as exc:
                print(f"[V25-RECOVER] ignored invalid result {path}: {exc}",
                      flush=True)
    if recovered:
        print(f"[V25-RECOVER] restored {recovered} atomic results into SQLite.",
              flush=True)
    return recovered

def _v25_finish_task(db, root, task, result, worker_id):
    status = str(result.get('Status', 'FAILED')).upper()
    if status not in _V25_SUCCESS:
        raise ValueError(f'_v25_finish_task called for non-success status={status}')
    result = dict(result)
    result['Seed'] = int(task['seed'])
    result['Protocol hash'] = task['protocol_hash']
    result['Attempts'] = int(task['attempts'])
    out = os.path.join(root, 'task_results', task['protocol_hash'][:16],
                       _v25_safe_name(_v25_task_id(task)) + '.json')
    _v25_atomic_json(out, result)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=True)
    now = _v25_now()
    with _v25_db(db) as con:
        con.execute('BEGIN IMMEDIATE')
        changed = con.execute('''UPDATE tasks SET status=?, result_json=?, result_path=?,
            last_error=?, lease_until=NULL, next_retry_at=0, updated_at=?
            WHERE protocol_hash=? AND seed=? AND dataset=? AND algorithm=?
              AND fold=? AND worker_id=? AND status='RUNNING' ''',
            (status, payload, out, result.get('Error',''), now,
             task['protocol_hash'], task['seed'], task['dataset'],
             task['algorithm'], task['fold'], worker_id)).rowcount
        if changed != 1:
            con.execute('ROLLBACK')
            raise RuntimeError('Task lease was lost before atomic result commit.')
        con.execute('COMMIT')


def _v25_fail_task(db, task, worker_id, error, retryable):
    attempts = int(task.get('attempts', 1))
    now = _v25_now()
    if retryable:
        delay = min(int(_ARGS.retry_max_seconds),
                    int(_ARGS.retry_base_seconds) * (2 ** min(attempts - 1, 20)))
        status = 'RETRYABLE'
        next_retry = now + delay
    elif attempts < int(_ARGS.max_nonresource_attempts):
        status = 'RETRYABLE'
        next_retry = now + min(60, int(_ARGS.retry_max_seconds))
    else:
        status = 'PERMANENT_FAILURE'
        next_retry = 0
    with _v25_db(db) as con:
        con.execute('''UPDATE tasks SET status=?, next_retry_at=?,
            last_error=?, lease_until=NULL, worker_id=NULL, gpu_id=NULL,
            updated_at=?
            WHERE protocol_hash=? AND seed=? AND dataset=? AND algorithm=?
              AND fold=?''',
            (status, next_retry, str(error)[:20000], now,
             task['protocol_hash'], task['seed'], task['dataset'],
             task['algorithm'], task['fold']))
    return status, max(0, next_retry - now)


def _v25_kind_pending(db, ph, kind):
    with _v25_db(db) as con:
        _v25_reset_expired(con, ph)
        row = con.execute('''SELECT COUNT(*) AS n FROM tasks
            WHERE protocol_hash=? AND kind=?
              AND status NOT IN ('OK','PARTIAL','PERMANENT_FAILURE')''',
            (ph, kind)).fetchone()
        return int(row['n'])


def _v25_summary(db, ph):
    with _v25_db(db) as con:
        _v25_reset_expired(con, ph)
        rows = con.execute('''SELECT status, COUNT(*) n FROM tasks
            WHERE protocol_hash=? GROUP BY status''', (ph,)).fetchall()
    return {r['status']: int(r['n']) for r in rows}


def _v25_free_vram_gib():
    if not torch.cuda.is_available():
        return 0.0
    free_b, _ = torch.cuda.mem_get_info()
    return free_b / (1024 ** 3)


def _v25_cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache()
        except Exception: pass
        try: torch.cuda.ipc_collect()
        except Exception: pass


def _v25_build_cache():
    group = _ARGS.datasets_group or '2'
    datasets = get_datasets_for_group(group)
    ds_map = {d.name: d for d in datasets}
    classifiers = build_classifiers()
    clf_map = {n: f for n, f in classifiers}
    return ds_map, clf_map


def _v25_execute_task(task, root, ds_map, clf_map, fold_cache):
    seed = int(task['seed'])
    _set_run_seed(seed)
    ds = ds_map[task['dataset']]
    fkey = (seed, ds.name)
    if fkey not in fold_cache:
        fold_cache[fkey] = list(ds.folds(
            n_folds=int(_ARGS.n_folds), random_state=seed))
    fold = fold_cache[fkey][int(task['fold'])]
    factory = clf_map[task['algorithm']]
    task_dir = _v25_task_dir(root, task)
    os.makedirs(task_dir, exist_ok=True)
    context = {
        'protocol_hash': task['protocol_hash'], 'seed': seed,
        'dataset': task['dataset'], 'algorithm': task['algorithm'],
        'fold': int(task['fold']), 'task_dir': task_dir,
        'epoch_checkpoint_interval': int(_ARGS.epoch_checkpoint_interval),
    }
    return _run_one_task(task['dataset'], fold, task['algorithm'], factory,
                         CACP_METRICS, fold_seed=seed + int(task['fold']),
                         task_context=context)


def _v25_task_heartbeat_loop(stop, db, ph, task, worker_id, role, gpu_id):
    while not stop.wait(max(5, int(_ARGS.heartbeat_seconds))):
        try:
            _v25_renew_lease(db, task, worker_id, int(_ARGS.lease_seconds))
            _v25_worker_heartbeat(db, ph, worker_id, role, 'RUNNING',
                                  _v25_task_id(task), gpu_id)
        except Exception as exc:
            print(f"[V25-HEARTBEAT] {worker_id}: {exc}", flush=True)


def _v25_worker_loop(kind, worker_id, gpu_id=None, preloaded=None):
    run_id, root, db = _v25_run_paths()
    ph = _v25_protocol_hash()
    ds_map, clf_map = preloaded if preloaded is not None else _v25_build_cache()
    fold_cache = {}
    role = 'gpu' if kind == 'GPU' else 'cpu'
    _v25_worker_heartbeat(db, ph, worker_id, role, 'STARTING', gpu_id=gpu_id)
    print(f"[V25-WORKER] {worker_id} role={role} pid={os.getpid()} root={root}",
          flush=True)

    while True:
        if kind == 'GPU':
            free_gib = _v25_free_vram_gib()
            if free_gib < float(_ARGS.min_free_vram_gib):
                _v25_worker_heartbeat(
                    db, ph, worker_id, role, 'WAITING_FOR_GPU', gpu_id=gpu_id,
                    message=f'free={free_gib:.1f} GiB < {_ARGS.min_free_vram_gib:.1f} GiB')
                print(f"[V25-WAIT] {worker_id}: free VRAM {free_gib:.1f} GiB; "
                      f"requires {_ARGS.min_free_vram_gib:.1f} GiB", flush=True)
                time.sleep(max(5, int(_ARGS.idle_sleep_seconds)))
                continue

        task = _v25_claim_task(db, ph, kind, worker_id, gpu_id,
                               int(_ARGS.lease_seconds))
        if task is None:
            remaining = _v25_kind_pending(db, ph, kind)
            if remaining == 0:
                _v25_worker_heartbeat(db, ph, worker_id, role, 'FINISHED', gpu_id=gpu_id)
                print(f"[V25-WORKER] {worker_id}: no remaining {kind} tasks.", flush=True)
                return 0
            _v25_worker_heartbeat(db, ph, worker_id, role, 'IDLE', gpu_id=gpu_id,
                                  message=f'{remaining} tasks pending/retryable/running')
            time.sleep(max(2, int(_ARGS.idle_sleep_seconds)))
            continue

        tid = _v25_task_id(task)
        print(f"[V25-CLAIM] {worker_id}: {tid}, attempt={task['attempts']}", flush=True)
        _v25_worker_heartbeat(db, ph, worker_id, role, 'RUNNING', tid, gpu_id)
        stop = threading.Event()
        hb = threading.Thread(target=_v25_task_heartbeat_loop,
                              args=(stop, db, ph, task, worker_id, role, gpu_id),
                              daemon=True)
        hb.start()
        try:
            result = _v25_execute_task(task, root, ds_map, clf_map, fold_cache)
            if str(result.get('Status','FAILED')).upper() in _V25_SUCCESS:
                _v25_finish_task(db, root, task, result, worker_id)
                print(f"[V25-COMMIT] {worker_id}: {tid} -> {result['Status']}", flush=True)
            else:
                err = result.get('Error', 'task returned FAILED')
                retryable = (kind == 'GPU' and _is_retryable_gpu_error_text(err))
                state, delay = _v25_fail_task(db, task, worker_id, err, retryable)
                print(f"[V25-{state}] {worker_id}: {tid}; retry in {delay:.0f}s; {err}",
                      flush=True)
        except BaseException as exc:
            err = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            retryable = (kind == 'GPU' and _is_retryable_gpu_error_text(err))
            state, delay = _v25_fail_task(db, task, worker_id, err, retryable)
            print(f"[V25-{state}] {worker_id}: {tid}; retry in {delay:.0f}s; {exc}",
                  flush=True)
        finally:
            stop.set(); hb.join(timeout=5)
            _v25_cleanup_cuda()
            _v25_worker_heartbeat(db, ph, worker_id, role, 'IDLE', gpu_id=gpu_id)


def _v25_cpu_child(index):
    # [V25.2] Under spawn every process imports and initialises native
    # libraries independently.  Nothing containing XGBoost/OpenMP state is
    # inherited from the supervisor.
    return _v25_worker_loop('CPU', f'cpu-{index:02d}', None, preloaded=None)


def _v25_cpu_supervisor():
    n = int(_ARGS.cpu_processes or max(1, min(24, _mp.cpu_count() - 4)))
    # [V25.2] Never fork after importing PyTorch/XGBoost/OpenMP.
    ctx = _mp.get_context('spawn')
    children = {}
    restarts = {i: 0 for i in range(n)}
    _, _, db = _v25_run_paths()
    ph = _v25_protocol_hash()

    def spawn_child(i):
        p = ctx.Process(target=_v25_cpu_child, args=(i,),
                        name=f'v25-cpu-{i:02d}')
        p.start()
        children[i] = p
        print(f"[V25-CPU-SUP] spawned cpu-{i:02d} pid={p.pid} method=spawn",
              flush=True)

    for i in range(n):
        spawn_child(i)

    while children:
        for i, p in list(children.items()):
            if p.is_alive():
                continue
            p.join(timeout=1)
            wid = f'cpu-{i:02d}'
            rc = p.exitcode
            _v25_requeue_worker_tasks(db, ph, wid, f'child exit rc={rc}')
            if _v25_kind_pending(db, ph, 'CPU') == 0:
                children.pop(i, None)
                continue
            if restarts[i] >= int(_ARGS.max_worker_restarts):
                raise RuntimeError(f'CPU child {i} exceeded restart limit')
            restarts[i] += 1
            print(f"[V25-CPU-SUP] {wid} rc={rc}; restart {restarts[i]}",
                  flush=True)
            spawn_child(i)
        time.sleep(5)
    return 0


def _v25_initialize_tasks():
    run_id, root, db = _v25_run_paths()
    if getattr(_ARGS, 'reset_run', False) and os.path.isdir(root):
        import shutil
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    _v25_init_schema(db)
    ph = _v25_protocol_hash()
    config = _v25_protocol_payload()
    now = _v25_now()
    datasets = get_datasets_for_group(_ARGS.datasets_group or '2')
    classifiers = build_classifiers()
    rows = []
    for seed in _SEEDS:
        for ds in datasets:
            for alg, _ in classifiers:
                kind = 'GPU' if _is_gpu_clf(alg) else 'CPU'
                priority = 10 if kind == 'GPU' else 100
                for fold in range(int(_ARGS.n_folds)):
                    rows.append((ph, int(seed), ds.name, alg, fold, kind,
                                 priority, now, now))
    with _v25_db(db) as con:
        con.execute('BEGIN IMMEDIATE')
        con.execute('''INSERT INTO runs(protocol_hash,run_id,config_json,created_at,updated_at)
            VALUES(?,?,?,?,?) ON CONFLICT(protocol_hash) DO UPDATE SET
              updated_at=excluded.updated_at''',
            (ph, run_id, json.dumps(config, sort_keys=True), now, now))
        con.executemany('''INSERT OR IGNORE INTO tasks
            (protocol_hash,seed,dataset,algorithm,fold,kind,priority,
             status,attempts,next_retry_at,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,'PENDING',0,0,?,?)''', rows)
        _v25_reset_expired(con, ph)
        con.execute('COMMIT')
    _v25_recover_atomic_results(root, db, ph)
    _record_versions(root)
    if os.environ.get('LDRV25_PROTOCOL_HASH_OVERRIDE', '').strip():
        # Keep the original protocol.json as the publication record.  Record
        # the execution-engine repair separately.
        resume_record = {
            'protocol_hash': ph,
            'resume_driver_sha256': _v25_file_sha256(__file__),
            'repair': 'v25.3 closes every SQLite connection and reuses one connection during atomic-result recovery; scientific protocol unchanged',
            'timestamp': datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(root, 'protocol_resume_v25_3.json'),
                  'w', encoding='utf-8') as fh:
            json.dump(resume_record, fh, indent=2)
    else:
        with open(os.path.join(root, 'protocol.json'), 'w', encoding='utf-8') as fh:
            json.dump({'protocol_hash': ph, **config}, fh, indent=2)
    print(f"[V25-QUEUE] run_id={run_id} protocol={ph[:16]} tasks={len(rows)} db={db}",
          flush=True)
    return ph, root, db


def _v25_materialize_results(root, db, ph):
    with _v25_db(db) as con:
        rows = con.execute('''SELECT result_json FROM tasks
            WHERE protocol_hash=? AND status IN ('OK','PARTIAL')
            ORDER BY seed,dataset,algorithm,fold''', (ph,)).fetchall()
    records = [json.loads(r['result_json']) for r in rows if r['result_json']]
    df = pd.DataFrame(records)
    out = os.path.join(root, 'comparison.csv')
    tmp = out + f'.tmp.{os.getpid()}'
    df.to_csv(tmp, index=False); os.replace(tmp, out)
    print(f"[V25-RESULTS] {len(df)} durable rows -> {out}", flush=True)
    if len(df):
        result_dir = Path(root)
        _safe_cacp_call(process_comparison_results, result_dir, CACP_METRICS,
                        label='comparison_result.csv + .tex')
        _safe_cacp_call(process_comparison_results_plots, result_dir, CACP_METRICS,
                        label='plots')
        _safe_cacp_call(process_comparison_result_winners, result_dir, CACP_METRICS,
                        label='winners')
        _safe_cacp_call(process_times, result_dir, label='times')
        _holm_wilcoxon_reports(out, os.path.join(root, 'wilcoxon_holm'))
    return df


def _v25_worker_command(role, worker_id, gpu_id=None):
    cmd = [sys.executable, os.path.abspath(__file__),
           '--resilient', '--worker-role', role,
           '--worker-id', worker_id,
           '--run-id', _ARGS.run_id,
           '--datasets-group', _ARGS.datasets_group or '2',
           '--n-folds', str(_ARGS.n_folds),
           '--trials', str(CBM_N_TRIALS),
           '--epochs-hpo', str(CBM_EPOCHS_HPO),
           '--epochs-final', str(CBM_EPOCHS_FINAL),
           '--patience', str(CBM_PATIENCE_ES),
           '--restarts', str(CBM_N_RESTARTS),
           '--seeds', ','.join(map(str, _SEEDS)),
           '--cpu-processes', str(_ARGS.cpu_processes or max(1, min(24, _mp.cpu_count()-4))),
           '--min-free-vram-gib', str(_ARGS.min_free_vram_gib),
           '--lease-seconds', str(_ARGS.lease_seconds),
           '--heartbeat-seconds', str(_ARGS.heartbeat_seconds),
           '--retry-base-seconds', str(_ARGS.retry_base_seconds),
           '--retry-max-seconds', str(_ARGS.retry_max_seconds),
           '--max-worker-restarts', str(_ARGS.max_worker_restarts),
           '--idle-sleep-seconds', str(_ARGS.idle_sleep_seconds),
           '--epoch-checkpoint-interval', str(_ARGS.epoch_checkpoint_interval),
           '--max-nonresource-attempts', str(_ARGS.max_nonresource_attempts),
           '--no-menu']
    if _ARGS.state_db: cmd += ['--state-db', _ARGS.state_db]
    if _ABLATION: cmd += ['--ablation']
    if not _STRICT_VALIDATION: cmd += ['--no-strict-validation']
    if gpu_id is not None: cmd += ['--gpu-id', str(gpu_id)]
    return cmd


def _v25_master():
    ph, root, db = _v25_initialize_tasks()
    n_gpus = min(2, torch.cuda.device_count())
    if n_gpus < 1:
        raise RuntimeError('Resilient publication run requires at least one CUDA GPU.')
    logs = os.path.join(root, 'logs'); os.makedirs(logs, exist_ok=True)
    specs = [('gpu', f'gpu-{i}', i) for i in range(n_gpus)] + [('cpu','cpu-supervisor',None)]
    procs = {}; handles = {}; restarts = {wid: 0 for _, wid, _ in specs}

    def spawn(role, wid, gid):
        env = os.environ.copy()
        if role == 'gpu':
            env['CUDA_VISIBLE_DEVICES'] = str(gid)
            env['LDR_GPU_ID'] = str(gid)
        else:
            env['CUDA_VISIBLE_DEVICES'] = ''
            env['LDR_GPU_ID'] = ''
        path = os.path.join(logs, wid + '.log')
        fh = open(path, 'a', buffering=1)
        p = subprocess.Popen(_v25_worker_command(role, wid, gid), env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
        procs[wid] = (p, role, gid); handles[wid] = fh
        print(f"[V25-MASTER] spawned {wid} pid={p.pid}; log={path}", flush=True)

    for spec in specs: spawn(*spec)
    try:
        while True:
            _v25_recover_atomic_results(root, db, ph)
            summary = _v25_summary(db, ph)
            terminal = sum(summary.get(s,0) for s in _V25_TERMINAL)
            total = sum(summary.values())
            print(f"[V25-MASTER {datetime.datetime.now():%H:%M:%S}] "
                  f"{summary} terminal={terminal}/{total} "
                  f"master_fds={_v25_open_fd_count()}", flush=True)
            if total and terminal == total:
                break
            for wid, (p, role, gid) in list(procs.items()):
                rc = p.poll()
                if rc is None: continue
                handles[wid].close()
                kind = 'GPU' if role == 'gpu' else 'CPU'
                _v25_requeue_worker_tasks(db, ph, wid,
                                          f'top-level worker exit rc={rc}')
                if _v25_kind_pending(db, ph, kind) == 0:
                    procs.pop(wid, None); handles.pop(wid, None); continue
                if restarts[wid] >= int(_ARGS.max_worker_restarts):
                    raise RuntimeError(f'{wid} exceeded max worker restarts')
                restarts[wid] += 1
                delay = min(int(_ARGS.retry_max_seconds),
                            int(_ARGS.retry_base_seconds) * 2 ** min(restarts[wid]-1, 10))
                print(f"[V25-MASTER] {wid} rc={rc}; restart {restarts[wid]} in {delay}s",
                      flush=True)
                time.sleep(delay); spawn(role, wid, gid)
            time.sleep(30)
    finally:
        for wid, (p, _, _) in procs.items():
            if p.poll() is None: p.terminate()
        for wid, (p, _, _) in procs.items():
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: p.kill()
        for fh in handles.values():
            try: fh.close()
            except Exception: pass
    df = _v25_materialize_results(root, db, ph)
    summary = _v25_summary(db, ph)
    failures = int(summary.get('PERMANENT_FAILURE', 0))
    print(f"[V25-DONE] rows={len(df)} summary={summary}", flush=True)
    return 2 if failures else 0


def _v25_entry():
    role = getattr(_ARGS, 'worker_role', 'master')
    if role == 'master':
        return _v25_master()
    if role == 'gpu':
        return _v25_worker_loop('GPU', _ARGS.worker_id or f'gpu-{_ARGS.gpu_id}',
                                _ARGS.gpu_id)
    if role == 'cpu':
        return _v25_cpu_supervisor()
    raise ValueError(role)

if __name__ == '__main__':

    # [V25.1-FIX1] Dispatch resilient mode before entering the legacy
    # two-worker execution path.  The previous v25 file defined the
    # resilient queue but never called _v25_entry().
    if getattr(_ARGS, 'resilient_fd_self_test', False):
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory(prefix='v25_3_fd_test_') as _td:
            _db = os.path.join(_td, 'fd_test.sqlite3')
            _v25_init_schema(_db)
            _before = _v25_open_fd_count()
            for _ in range(2500):
                with _v25_db(_db) as _con:
                    _con.execute('SELECT 1').fetchone()
            gc.collect()
            _after = _v25_open_fd_count()
            _delta = _after - _before if _before >= 0 and _after >= 0 else 0
            if _delta > 3:
                print(f'V25_3_SQLITE_FD_TEST_FAILED before={_before} after={_after}',
                      flush=True)
                raise SystemExit(3)
            print(f'V25_3_SQLITE_FD_TEST_OK before={_before} after={_after}',
                  flush=True)
        raise SystemExit(0)
    if getattr(_ARGS, 'resilient_self_test', False):
        print('V25_1_RESILIENT_DISPATCH_OK', flush=True)
        raise SystemExit(0)
    if getattr(_ARGS, 'resilient', False):
        raise SystemExit(_v25_entry())

    _role = f"WORKER-GPU{_GPU_WORKER_ID}" if _GPU_WORKER_ID else "MASTER"

    if _GPU_WORKER_ID is None:
        print(f">>> ACTIVE PARAMETERS:", flush=True)
        print(f"    --trials={CBM_N_TRIALS}  --epochs-hpo={CBM_EPOCHS_HPO}  "
              f"--patience={CBM_PATIENCE_ES}", flush=True)
        print(f"    --epochs-final={CBM_EPOCHS_FINAL}  --restarts={CBM_N_RESTARTS}", flush=True)
        print(f"    --gpu-workers={_N_GPU_WORKERS}  --cpu-workers={_N_CPU_WORKERS}", flush=True)
        _scale = CBM_N_TRIALS/120 * CBM_EPOCHS_HPO/100 * CBM_PATIENCE_ES/50
        print(f"    Estimated time vs. -full: {_scale*100:.0f}%  (~{5*24*_scale:.0f}h "
              f"vs 5 days full)", flush=True)
        print(f"    To change: ./okno0.sh → restart with different parameters", flush=True)
        print(f"", flush=True)
    print(f"\n{'#'*70}", flush=True)
    print(f"#  LDR2H200.py   |  role={_role}  |  PID={os.getpid()}",
          flush=True)
    print(f"#  Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)
    print(f"{'#'*70}\n", flush=True)


    def _show_menu():
        print(f"\n{'═'*68}", flush=True)
        print(f"  LDR2H200 — Interactive configuration menu", flush=True)
        print(f"  (skip: python3 LDR2H200.py --no-menu [options])", flush=True)
        print(f"{'═'*68}", flush=True)

        print(f"\n  STEP 1/3 — Choose the dataset subset:")
        print(f"")
        print(f"    0 — SUPER-FAST  (4 datasets: 2 bin + 2 multi)   ~15 min  ← TEST")
        print(f"                    diabetes, blood  |  balance-scale(K=3), vehicle(K=4)")
        print(f"    1 — SMALL        (10 datasets)                   ~1-2 days (fast)")
        print(f"                    6 cheap binary + 4 cheap multiclass")
        print(f"    2 — ALL   (20 datasets)                   ~4h–5 days")
        print(f"                    full set")
        print(f"")
        while True:
            try:
                c = input("  Your choice [0/1/2]: ").strip()
                if c in ("0","1","2"):
                    break
                print("  Enter 0, 1 or 2.")
            except (EOFError, KeyboardInterrupt):
                print("\n  Aborting.")
                sys.exit(0)
        ds_group = c


        if ds_group == "0":
            print(f"\n  ℹ  Selected SUPER-FAST — recommended CBM mode: 0 (SUPER-FAST)")

        print(f"\n  STEP 2/3 — Choose CBM HPO mode:")
        print(f"")
        print(f"    0 — SUPER-FAST  trials=8,   hpo=20  epochs, patience=8,")
        print(f"                    final=60   epochs, restarts=1  ← recommended for testing")
        print(f"    1 — FAST        trials=30,  hpo=40  epochs, patience=15,")
        print(f"                    final=150  epochs, restarts=1  (~4h total)")
        print(f"    2 — NORMAL      trials=50,  hpo=60  epochs, patience=25,")
        print(f"                    final=250  epochs, restarts=2  (~15h total)")
        print(f"    3 — FULL        trials=120, hpo=100 epochs, patience=50,")
        print(f"                    final=450  epochs, restarts=3  (~5 days)")
        print(f"    4 — CUSTOM      (enter parameters manually)")
        print(f"")
        while True:
            try:
                c = input("  CBM mode [0/1/2/3/4]: ").strip()
                if c in ("0","1","2","3","4"):
                    break
                print("  Enter 0, 1, 2, 3 or 4.")
            except (EOFError, KeyboardInterrupt):
                print("\n  Aborting.")
                sys.exit(0)

        _PRESETS = {
            "0": dict(trials=8,   epochs_hpo=20,  patience=8,  epochs_final=60,  restarts=1),
            "1": dict(trials=30,  epochs_hpo=40,  patience=15, epochs_final=150, restarts=1),
            "2": dict(trials=40,  epochs_hpo=80,  patience=40, epochs_final=250, restarts=2),  # [V22] full-run protocol
            "3": dict(trials=120, epochs_hpo=100, patience=50, epochs_final=450, restarts=3),
        }
        if c in _PRESETS:
            pr = _PRESETS[c]
            n_trials, epochs_hpo, patience = pr["trials"], pr["epochs_hpo"], pr["patience"]
            epochs_final, restarts = pr["epochs_final"], pr["restarts"]
        else:
            def _ask(prompt, default):
                try:
                    v = input(f"  {prompt} [default {default}]: ").strip()
                    return int(v) if v else default
                except (ValueError, EOFError):
                    return default
            n_trials     = _ask("trials (number of HPO trials)",     50)
            epochs_hpo   = _ask("epochs-hpo  (epochs HPO)",      60)
            patience     = _ask("patience    (early stopping)",  25)
            epochs_final = _ask("epochs-final (final training)",  250)
            restarts     = _ask("restarts    (final restarts)",  2)

        print(f"\n  STEP 3/3 — Number of CV folds:")
        print(f"    3  — super-fast  5  — faster  10 — standard (more accurate)")
        print(f"")
        while True:
            try:
                c = input("  Number of folds [3/5/10, Enter=10]: ").strip()
                if c == "":
                    n_folds = 10; break
                elif c in ("3","5","10"):
                    n_folds = int(c); break
                print("  Type 3, 5, 10 or Enter.")
            except (EOFError, KeyboardInterrupt):
                n_folds = 10; break

        _GN = {"0":"SUPER-FAST(4)","1":"SMALL(10)","2":"ALL(20)"}
        print(f"\n{'═'*68}", flush=True)
        print(f"  ACTIVE CONFIGURATION:", flush=True)
        print(f"  Datasets:   {_GN[ds_group]}  |  Folds: {n_folds}", flush=True)
        print(f"  CBM:      trials={n_trials}  hpo={epochs_hpo}  "
              f"patience={patience}  final={epochs_final}  restarts={restarts}", flush=True)
        _scale = n_trials/120 * epochs_hpo/100 * patience/50
        print(f"  ETA est.: ~{_scale*100:.1f}% of full time  "
              f"(~{5*24*_scale:.1f}h for 20 datasets)", flush=True)
        print(f"{'═'*68}", flush=True)
        try:
            ok = input("  Run? [Y/n]: ").strip()
        except (EOFError, KeyboardInterrupt):
            ok = "T"
        if ok.lower() in ("n","nie","no"):
            print("  Cancelled."); sys.exit(0)

        return ds_group, n_trials, epochs_hpo, patience, epochs_final, restarts, n_folds

    _run_menu = (
        _GPU_WORKER_ID is None and
        not getattr(_ARGS, 'no_menu', False) and
        _ARGS.datasets_group is None
    )
    if _run_menu:
        (_menu_ds_group, _menu_trials, _menu_hpo, _menu_patience,
         _menu_final, _menu_restarts, _menu_folds) = _show_menu()
        CBM_N_TRIALS    = _menu_trials
        CBM_EPOCHS_HPO  = _menu_hpo
        CBM_PATIENCE_ES = _menu_patience
        CBM_EPOCHS_FINAL= _menu_final
        CBM_N_RESTARTS  = _menu_restarts
        _N_FOLDS_MENU   = _menu_folds
        _DS_GROUP_MENU  = _menu_ds_group
        _menu_cli_args = [
            '--trials', str(_menu_trials),
            '--epochs-hpo', str(_menu_hpo),
            '--patience', str(_menu_patience),
            '--epochs-final', str(_menu_final),
            '--restarts', str(_menu_restarts),
            '--datasets-group', _menu_ds_group,
            '--no-menu',
            # [V02] propagate new options to spawned workers
            '--hpo-timeout', str(_CBM_HPO_TIMEOUT),
            '--monitor-interval', str(_MON_INTERVAL),
            '--cpu-backend', _CPU_BACKEND,
            # [V21-P4] propagate V20/V21 protocol flags to workers
            '--seeds', ','.join(str(s) for s in _SEEDS),
        ] + (['--no-resume'] if not _RESUME else []) \
          + (['--ablation'] if _ABLATION else []) \
          + ([] if _STRICT_VALIDATION else ['--no-strict-validation'])
    else:
        _N_FOLDS_MENU  = 10
        _DS_GROUP_MENU = _ARGS.datasets_group
        _menu_cli_args = sys.argv[1:]

    n_gpus = torch.cuda.device_count()


    if _GPU_WORKER_ID is not None:
        gpu_id   = int(_GPU_WORKER_ID)
        res_dir  = f'./results_LDR/gpu{gpu_id}'
        os.makedirs(res_dir, exist_ok=True)

        _start_heartbeat(interval=120)
        _start_resource_monitor(interval=_MON_INTERVAL)   # [V02-6]

        if _DS_GROUP_MENU is not None:
            all_datasets = get_datasets_for_group(_DS_GROUP_MENU)
        else:
            all_datasets = get_all_datasets()
        classifiers  = build_classifiers()

        _partitions = _load_balanced_split(all_datasets, n_splits=2)
        datasets = _partitions[gpu_id] if gpu_id < len(_partitions) else []
        label    = f"GPU-{gpu_id} (load-balanced, {len(datasets)} datasets)"

        n_folds   = _N_FOLDS_MENU
        n_ds      = len(datasets)
        n_clf     = len(classifiers)
        total_exp = n_ds * n_folds

        print_dataset_summary(datasets)
        _banner(f"START WORKER GPU-{gpu_id} ({label})")
        print(f"    Datasets:        {n_ds}", flush=True)
        print(f"    Classifiers: {n_clf}", flush=True)
        print(f"    Folds/dataset:   {n_folds}", flush=True)
        print(f"    Total folds: {total_exp}", flush=True)
        print(f"    Metrics:        {[m[0] for m in CACP_METRICS]}", flush=True)
        print(f"    Results ->        {res_dir}", flush=True)
        print(f"", flush=True)
        print(f"  Monitor progress:", flush=True)
        print(f"    tail -f {res_dir}/../worker{gpu_id}.log", flush=True)
        print(f"    or (tmux): tail -f {res_dir}/../worker{gpu_id}.log | grep -E 'fold|CBM|HEDGE|WORKER|♥'",
              flush=True)
        print(f"", flush=True)

        t_exp_start = time.time()

        _record_versions(res_dir)                       # [V21-A5]
        _run_experiment_all_seeds(                      # [V21-T3]
            datasets,
            classifiers,
            results_directory=res_dir,
            metrics=CACP_METRICS,
            n_gpu_workers=_N_GPU_WORKERS,
            n_cpu_workers=_N_CPU_WORKERS,
            n_folds=_N_FOLDS_MENU,
        )

        t_exp_end = time.time()
        t_total   = t_exp_end - t_exp_start
        _banner(f"WORKER GPU-{gpu_id} FINISHED")
        print(f"    Results:       {res_dir}", flush=True)
        print(f"    Total time:  {str(datetime.timedelta(seconds=int(t_total)))}",
              flush=True)
        print(f"    Time/fold:    {t_total/max(1,total_exp):.1f} s", flush=True)

    elif n_gpus >= 2:
        print(f">>> MULTI-GPU MODE: {n_gpus} GPU detected. "
              f"Launching 2 workers.", flush=True)

        res_dir     = './results_LDR'
        log_gpu0    = os.path.join(res_dir, 'worker0.log')
        log_gpu1    = os.path.join(res_dir, 'worker1.log')
        os.makedirs(res_dir, exist_ok=True)

        def _make_env(gpu_id):
            env = os.environ.copy()
            env['CUDA_DEVICE_ORDER']  = 'PCI_BUS_ID'
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['LDR_GPU_ID']           = str(gpu_id)
            env['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
            return env

        print(f"  GPU-0 -> load-balanced datasets | log: {log_gpu0}", flush=True)
        print(f"  GPU-1 -> load-balanced datasets | log: {log_gpu1}", flush=True)

        with open(log_gpu0, 'w') as f0, open(log_gpu1, 'w') as f1:
            _cli_args = _menu_cli_args
            p0 = subprocess.Popen(
                [sys.executable, __file__] + _cli_args,
                env=_make_env(0), stdout=f0, stderr=subprocess.STDOUT)
            p1 = subprocess.Popen(
                [sys.executable, __file__] + _cli_args,
                env=_make_env(1), stdout=f1, stderr=subprocess.STDOUT)

            print(f"  PID worker-0: {p0.pid}", flush=True)
            print(f"  PID worker-1: {p1.pid}", flush=True)
            print(f"", flush=True)
            print(f"  ── Progress monitoring (tmux / SSH) ──────────────────────────",
                  flush=True)
            print(f"  # Worker 0 (binary):", flush=True)
            print(f"  tail -f {log_gpu0}", flush=True)
            print(f"  # Worker 1 (multiclass):", flush=True)
            print(f"  tail -f {log_gpu1}", flush=True)
            print(f"  # GPU:", flush=True)
            print(f"  watch -n 10 nvidia-smi", flush=True)
            print(f"  # Kill everything:", flush=True)
            print(f"  kill {p0.pid} {p1.pid}", flush=True)
            print(f"  ─────────────────────────────────────────────────────────────",
                  flush=True)
            print(f"", flush=True)

            _mon_interval = 30
            while True:
                rc0_now = p0.poll()
                rc1_now = p1.poll()
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                w0_status = f"RC={rc0_now}" if rc0_now is not None else "RUNNING"
                w1_status = f"RC={rc1_now}" if rc1_now is not None else "RUNNING"
                print(f"  [MASTER {ts}  +{_elapsed()}] "
                      f"worker0={w0_status}  worker1={w1_status}",
                      flush=True)
                if rc0_now is not None and rc1_now is not None:
                    break
                time.sleep(_mon_interval)

        rc0, rc1 = p0.returncode, p1.returncode
        print(f"\n>>> WORKERS FINISHED. "
              f"Codes: GPU-0={rc0}, GPU-1={rc1}", flush=True)

        if rc0 == 0 and rc1 == 0:
            print(">>> MERGING RESULTS...", flush=True)
            merge_results(
                f'{res_dir}/gpu0',
                f'{res_dir}/gpu1',
                f'{res_dir}/merged')
            _record_versions(os.path.join(res_dir, 'merged'))   # [V21-A5]
            _holm_wilcoxon_reports(                              # [V21-T4]
                os.path.join(res_dir, 'merged', 'comparison.csv'),
                os.path.join(res_dir, 'merged', 'wilcoxon_holm'))
            print(f"\n>>> DONE. Final results: {res_dir}/merged/",
                  flush=True)
        else:
            print(f"[ERROR] Worker ended with an error. "
                  f"Check: {log_gpu0}, {log_gpu1}", flush=True)

    else:
        print(f">>> SINGLE-GPU/CPU MODE (detected {n_gpus} GPU).", flush=True)
        res_dir = './results_LDR/single'
        os.makedirs(res_dir, exist_ok=True)
        _start_resource_monitor(interval=_MON_INTERVAL)   # [V02-6]

        if _DS_GROUP_MENU is not None:
            datasets = get_datasets_for_group(_DS_GROUP_MENU)
        else:
            datasets = get_all_datasets()
        classifiers = build_classifiers()
        print_dataset_summary(datasets)

        print(f"\n>>> START EXPERIMENT LDR", flush=True)
        print(f"    Datasets:       {len(datasets)}", flush=True)
        print(f"    Classifiers:{len(classifiers)}", flush=True)
        print(f"    Metrics:       {[m[0] for m in CACP_METRICS]}", flush=True)
        print(f"    Results ->       {res_dir}\n", flush=True)

        _record_versions(res_dir)                       # [V21-A5]
        _run_experiment_all_seeds(                      # [V21-T3]
            datasets,
            classifiers,
            results_directory=res_dir,
            metrics=CACP_METRICS,
            n_gpu_workers=_N_GPU_WORKERS,
            n_cpu_workers=_N_CPU_WORKERS,
            n_folds=_N_FOLDS_MENU
        )
        _holm_wilcoxon_reports(                         # [V21-T4]
            os.path.join(res_dir, 'comparison.csv'),
            os.path.join(res_dir, 'wilcoxon_holm'))
        print(f"\n>>> EXPERIMENT FINISHED. Results: {res_dir}", flush=True)
