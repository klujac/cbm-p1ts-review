"""
rule_gen.py - companion of benchmark_driver.py: generates the
natural-language P1-TS rule explanations for the trained CBM-P1TS model
(used for the illustrative diabetes example), with TabPFN-3 as the
distillation teacher.

Shares the resilient execution engine of benchmark_driver.py:
vectorised batching of GPU-resident tensors, process-isolated CPU
classifiers with thread fallback, an optional Optuna HPO wall-clock
budget, per-(dataset, classifier, fold) checkpointing with
resume-by-default, a periodic resource monitor, and global socket
timeouts on OpenML and TabPFN-checkpoint downloads.

Runtime configuration is read from environment variables; the
dependency manifest is written to environment.json.
Changelog: see CHANGELOG.md
"""
# reviewer package: rule_gen.py
# Standalone interactive version with fast / medium / normal modes.
# This script reproduces the illustrative diabetes rule-generation
# example from Section 3.12 of the manuscript.
# Key features:
# TabPFN-3 is used for knowledge-enhanced rule generation.
# Hardware auto-detection: number of GPUs and CPU cores is detected
# at runtime and the number of GPU/CPU workers is chosen accordingly.
# Single-process mode for small benchmarks (<=3 datasets) to avoid
# subprocess overhead and VRAM contention.
# Per-fold progress reporting with ETA, so the user always sees
# how far the experiment has progressed.

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
import json
import uuid
from pathlib import Path
import inspect
from itertools import product as iproduct
from collections import defaultdict
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

# Local reporting/statistics implementation.
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

# rule generation uses TabPFN-3 as the teacher.
_DISABLED_TREE_TEACHER_CLASSIFIER = None
_DISABLED_TREE_TEACHER = False
print("[INFO] Rule-generation protocol: TabPFN-3 teacher, CBM_KE student.")

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

_GPU_WORKER_ID = os.environ.get('LDR_GPU_ID', None)

print(f">>> INITIALIZATION LDR "
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

os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# TF32 for nn.Linear matmuls on Hopper (H100/H200): 2-8x faster
# Linear layers at ~1e-3 relative error. Disable: export LDRV2_TF32=0
if torch.cuda.is_available() and os.environ.get('LDRV2_TF32', '1') == '1':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(">>> [V02] TF32 ENABLED (LDRV2_TF32=0 to disable)", flush=True)

# Global socket timeout: network calls without an explicit timeout
# (OpenML download, TabPFN checkpoint download) raise after
# LDRV2_NET_TIMEOUT seconds instead of blocking forever. The exceptions
# are handled by the existing try/except blocks.
import socket as _socket
_socket.setdefaulttimeout(float(os.environ.get('LDRV2_NET_TIMEOUT', '180')))
print(f">>> [V02] Global network timeout: "
      f"{_socket.getdefaulttimeout():.0f}s (LDRV2_NET_TIMEOUT)", flush=True)

# runtime configuration via environment variables
_CBM_HPO_TIMEOUT = float(os.environ.get('LDRV2_HPO_TIMEOUT', '0') or 0)
_MON_INTERVAL    = int(os.environ.get('LDRV2_MON_INTERVAL', '60'))
_RESUME          = os.environ.get('LDRV2_NO_RESUME', '0') != '1'
_CPU_BACKEND     = os.environ.get('LDRV2_CPU_BACKEND', 'loky')
_STRICT_VALIDATION = os.environ.get('LDRV25_STRICT_VALIDATION', os.environ.get('LDRV20_STRICT_VALIDATION', '1')) != '0'
print(f">>> [V20] strict_validation={_STRICT_VALIDATION} "
      f"(set LDRV25_STRICT_VALIDATION=0 only for historical reproduction)", flush=True)

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
    _dummy = torch.zeros(1, device='cuda'); del _dummy
    print(f"    CUDA context: OK", flush=True)
else:
    device = 'cpu'
    print(f">>> [WARN] CUDA UNAVAILABLE -> device=cpu", flush=True)
    print(f"    CUDA_VISIBLE_DEVICES={_phys_gpu}", flush=True)
    print(f"    torch.version.cuda={torch.version.cuda}  "
          f"torch={torch.__version__}", flush=True)
    if _phys_gpu not in ('not_set', '', '-1'):
        try:
            _t = torch.zeros(1).cuda(); del _t
            device = 'cuda'
            print(f"    [RECOVER] .cuda() works -> device=cuda", flush=True)
        except Exception as _e:
            print(f"    [RECOVER FAILED] {_e}", flush=True)
            print(f"    >>> CBM will train on CPU -- very slowly!", flush=True)
print(f">>> device={device}", flush=True)

import threading
try:
    from scipy.optimize import minimize_scalar
except ImportError:
    minimize_scalar = None

# Lock removed: threading.Lock is not picklable (cloudpickle/
# loky), and dict[tid]=v / dict.get(tid) are atomic in CPython (GIL).
_PROBA_CACHE = {}


class _FastTensorLoader:
    """[V02-1] Vectorised loader for GPU-resident tensors.

    Replaces DataLoader(TensorDataset(...)), which performs bs*len(tensors)
    individual GPU indexing operations in Python per batch (+ torch.stack),
    i.e. O(bs) micro-kernels. Here: 1 randperm per epoch + 1 index_select
    per tensor per batch, i.e. O(1) kernels per batch. Compatible
    interface: iteration yields tuples already on the proper device.
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
            self._last_proba  = proba  # instance copy (loky-safe)
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

# [FIX] Early-stopping patience used in _train_cbm_v4
CBM_PATIENCE_ES = 40

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

    # AUC is undefined without usable scores; never
    # substitute a different metric under the AUC_ROC name.
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
                 cat_idx, num_idx, col_names=None, class_names = None):
        self.y_train = y_train
        self.y_test  = y_test
        self.index   = index
        self.labels  = np.unique(np.concatenate([y_train, y_test]))
        # Preserve raw outer-fold data and schema. CBM/CBM_KE use
        # these arrays so their inner validation preprocessing can be fitted
        # strictly on the inner-training subset rather than on all outer-train
        # rows. Other baselines continue to use x_train/x_test below.
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

            self.rs_centers = {}
            self.rs_scales = {}
            if num_idx:
                _rs = ct.named_transformers_['num'].named_steps['scale']
                for j, orig_j in enumerate(num_idx):
                    col_name = col_names[orig_j] if col_names else f"f{orig_j}"
                    self.rs_centers[col_name] = float(_rs.center_[j])
                    self.rs_scales[col_name] = float(_rs.scale_[j])

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
            self.rs_centers = {}
            self.rs_scales = {}

        num_names = [col_names[i] for i in num_idx] if col_names else \
            [f"f{i}" for i in num_idx]
        cat_names = [col_names[i] for i in cat_idx] if col_names else \
            [f"f{i}" for i in cat_idx]
        self.feature_names = num_names + cat_names
        _labels_int = np.unique(np.concatenate([y_train, y_test]))
        if class_names is not None:
            self.class_names_orig = [class_names[i] for i in _labels_int]
        else:
            self.class_names_orig = [str(i) for i in _labels_int]
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
        self._le = LabelEncoder()
        self.y = self._le.fit_transform(y_str)
        self._class_names = [str(c) for c in self._le.classes_]

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
                col_names=self._col_names,
                class_names = self._class_names)
    def __iter__(self):
        for i in range(len(self.X_raw)):
            yield self.X_raw[i], self.y[i]

    def __len__(self):    return len(self.X_raw)

    def get_data(self):
        return self.X, self.y, None, None

DATASETS_BINARY = {
    'diabetes':    {'name': 'diabetes',                         'version': 1},
    # 'credit-g': {'name': 'credit-g', 'version': 1},
    # 'blood': {'name': 'blood-transfusion-service-center', 'version': 1},
    # 'wdbc': {'name': 'wdbc', 'version': 1},
    # 'tic-tac-toe': {'name': 'tic-tac-toe', 'version': 1},
    # 'spambase': {'name': 'spambase', 'version': 1},
    # 'magic': {'name': 'MagicTelescope', 'version': 1},
    # 'bank': {'name': 'bank-marketing', 'version': 1},
    # 'phoneme': {'name': 'phoneme', 'version': 1},
    # 'kr-vs-kp': {'name': 'kr-vs-kp', 'version': 1},
}

DATASETS_MULTI = {
    # 'balance-scale': {'name': 'balance-scale', 'version': 1},
    # 'vehicle': {'name': 'vehicle', 'version': 1},
    # 'car': {'name': 'car', 'version': 3},
    # 'segment': {'name': 'segment', 'version': 1},
    # 'satimage': {'name': 'satimage', 'version': 1},
    # 'pendigits': {'name': 'pendigits', 'version': 1},
    # 'mfeat-factors': {'name': 'mfeat-factors', 'version': 1},
    # 'optdigits': {'name': 'optdigits', 'version': 1},
    # 'page-blocks': {'name': 'page-blocks', 'version': 1},
    # 'wine-quality': {'name': 'wine-quality-white','version': 1},
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
                    print(f"  [WARN] {key}: cache file empty (0 rows) -> "
                          f"removing and downloading again.", flush=True)
                    os.remove(sub_csv)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as _e:
                print(f"  [WARN] {key}: corrupted cache file ({_e}) -> "
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


def get_all_datasets():
    bin_ds   = _load_group(DATASETS_BINARY, "BINARY (K=2)")
    multi_ds = _load_group(DATASETS_MULTI,  "MULTICLASS (K>2)")
    return bin_ds + multi_ds

MAX_CONCEPTS_DEFAULT = 12
MAX_CONCEPTS = 12
P_LOW_PCT    = 20.0
P_HIGH_PCT   = 80.0

# --------------------------------------------------------------------------
# Standalone interactive execution modes
# --------------------------------------------------------------------------
# normal: closest to the original full rule-generation protocol.
# medium: approximately one order of magnitude faster than normal.
# fast: quick rule-inspection mode.
# All modes remain data-derived. The fast and medium modes only reduce
# optimisation/training budgets; they do not use hard-coded rules.
# --------------------------------------------------------------------------

MODE_CONFIGS = {
    "fast": {
        "description": "quick rule inspection; lowest computation budget",
        "n_trials": 3,
        "max_concepts": 2,
        "hpo_epochs": 10,
        "final_epochs": 30,
        "final_restarts": 1,
        "hedge_iter": 2,
        "n_folds": 2,
        "standalone_tabpfn": False,
        "n_gpu_workers": 1,
        "n_cpu_workers": 1,
        "hidden_dim_grid": [64, 128, 256],
        "batch_size_grid": [32, 64, 128],
        "force_min_rule_budget": True,
    },
    "medium": {
        "description": "intermediate budget; faster than normal, richer than fast",
        "n_trials": 12,
        "max_concepts": 6,
        "hpo_epochs": 50,
        "final_epochs": 150,
        "final_restarts": 2,
        "hedge_iter": 5,
        "n_folds": 10,
        "standalone_tabpfn": True,
        "n_gpu_workers": None,
        "n_cpu_workers": None,
        "hidden_dim_grid": [64, 128, 256, 384],
        "batch_size_grid": [32, 64, 128],
        "force_min_rule_budget": False,
    },
    "normal": {
        "description": "original/full rule-generation protocol; slowest",
        "n_trials": 120,
        "max_concepts": 12,
        "hpo_epochs": 100,
        "final_epochs": 450,
        "final_restarts": 3,
        "hedge_iter": 20,
        "n_folds": 10,
        "standalone_tabpfn": True,
        "n_gpu_workers": None,
        "n_cpu_workers": None,
        "hidden_dim_grid": [64, 128, 256, 384, 512, 768, 1024],
        "batch_size_grid": [16, 32, 64, 128],
        "force_min_rule_budget": False,
    },
}

RUN_MODE = "normal"
MODE_CFG = MODE_CONFIGS[RUN_MODE]


def _select_run_mode_from_cli_or_prompt() -> str:
    """Return one of: fast, medium, normal.

    Supported forms:
        python3 rule_gen.py
        python3 rule_gen.py --mode fast
        python3 rule_gen.py --mode medium
        python3 rule_gen.py --mode normal
        python3 rule_gen.py fast
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="ESWA v25 diabetes rule-generation: fast / medium / normal"
    )
    parser.add_argument(
        "positional_mode",
        nargs="?",
        choices=["fast", "medium", "normal"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "medium", "normal"],
        default=None,
        help="Execution mode.",
    )
    parser.add_argument("--fast", action="store_true", help="Shortcut for --mode fast.")
    parser.add_argument("--medium", action="store_true", help="Shortcut for --mode medium.")
    parser.add_argument("--normal", action="store_true", help="Shortcut for --mode normal.")
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Do not ask interactively; use --mode or default normal.",
    )

    args, unknown = parser.parse_known_args()

    # Remove launcher-only arguments so the legacy part of the script does not
    # see them later if it inspects sys.argv.
    sys.argv = [sys.argv[0]] + unknown

    selected = args.mode or args.positional_mode
    if args.fast:
        selected = "fast"
    if args.medium:
        selected = "medium"
    if args.normal:
        selected = "normal"

    if selected is not None:
        return selected

    if args.no_menu:
        return "normal"

    print("=" * 78)
    print("ESWA v25 diabetes rule-generation")
    print("=" * 78)
    print("Choose execution mode:")
    print()
    print("  1) fast")
    print("     Quick inspection of generated rules.")
    print("     Very reduced budget; not manuscript-table reproduction.")
    print()
    print("  2) medium")
    print("     Intermediate budget.")
    print("     Much faster than normal, less reduced than fast.")
    print()
    print("  3) normal")
    print("     Original/full rule-generation protocol.")
    print("     Slowest; closest to manuscript protocol.")
    print()
    print("  q) quit")
    print("-" * 78)

    aliases = {
        "1": "fast", "f": "fast", "fast": "fast",
        "2": "medium", "m": "medium", "medium": "medium",
        "3": "normal", "n": "normal", "normal": "normal",
        "q": "quit", "quit": "quit", "exit": "quit",
    }

    while True:
        choice = input("Your choice [1/2/3/q]: ").strip().lower()
        if choice in aliases:
            value = aliases[choice]
            if value == "quit":
                print("No mode selected. Exiting.")
                raise SystemExit(0)
            return value
        print("Please choose 1, 2, 3, or q.")


def _apply_run_mode(mode: str) -> None:
    global RUN_MODE, MODE_CFG, MAX_CONCEPTS
    RUN_MODE = mode
    MODE_CFG = MODE_CONFIGS[mode]
    MAX_CONCEPTS = int(MODE_CFG["max_concepts"])

    print("=" * 78)
    print(f"SELECTED MODE: {RUN_MODE.upper()}")
    print("=" * 78)
    print(f"Description:       {MODE_CFG['description']}")
    print("Dataset:           diabetes only")
    print(f"CBM_KE HPO trials: {MODE_CFG['n_trials']}")
    print(f"Max concepts:      {MODE_CFG['max_concepts']}")
    print(f"HPO epochs:        {MODE_CFG['hpo_epochs']}")
    print(f"Final training:    {MODE_CFG['final_epochs']} epochs × {MODE_CFG['final_restarts']} restart(s)")
    print(f"Hedge iterations:  {MODE_CFG['hedge_iter']}")
    print(f"CV folds:          {MODE_CFG['n_folds']}")
    print(f"Standalone TabPFN: {MODE_CFG['standalone_tabpfn']}")
    print("Teacher:           TabPFN-3 default")
    print("Teacher protocol:  TabPFN-3 -> CBM_KE")
    print("-" * 78)


_apply_run_mode(_select_run_mode_from_cli_or_prompt())



def _sat01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


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
        print("  [TempScale] scipy unavailable -> T=1.0. "
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
    print(f"  [TempScale] T*={T_star:.4f}  NLL: {nll_base:.4f} -> {nll_calib:.4f}",
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
                  f"> {self.timeout_s}s -> majority fallback", flush=True)
        elif fit_result[0] is not None:
            print(f"  [TIMEOUT/ERR] {type(self.clf).__name__}: "
                  f"{fit_result[0]} -> majority fallback", flush=True)
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
                print(f"  [BinarizingWrapper] MDLP error: {e} -> KBins fallback",
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
                 concept_dropout_p: float = 0.0,
                 concept_names: list = None):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.actual_num_rules = 2 ** num_concepts
        self.concept_names = concept_names or [f"f{i}" for i in range(num_concepts)]
        self.rule_indices_list = list(iproduct([0, 1], repeat=num_concepts))

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
                          f"{antecedent} -> {prefix}{antecedent}  (α={alpha})",
                          flush=True)
    return n_changed

_HEDGE_LABELS_REPORT = {0.5: 'ml', 1.0: '', 2.0: 'v'}

def get_full_description(rule_idx, rule_indices_list, concept_names, hedge_exponents=None):
    indices = rule_indices_list[rule_idx]
    conditions_high = []
    conditions_low = []
    for k, idx in enumerate(indices):
        exp = 1.0
        if hedge_exponents is not None:
            try:
                exp = float(hedge_exponents[rule_idx, k])
            except Exception:
                exp = 1.0
        prefix = _HEDGE_LABELS_REPORT.get(exp, f'^{exp:.2g}')
        if idx == 1:
            conditions_high.append(f"{concept_names[k]}={prefix}HIGH")
        else:
            conditions_low.append(f"{concept_names[k]}={prefix}LOW")
    all_conditions = conditions_high + conditions_low
    if not all_conditions:
        return "ALL CONCEPTS INDEFINITE"
    return " AND ".join(all_conditions)


def extract_metarules(model, class_names, concept_names):
    weights = model.rule_weights.detach().cpu().numpy()
    mask = model.rule_mask.detach().cpu().numpy()
    hedge_exp = model.hedge_exponents.detach().cpu() \
                if hasattr(model, 'hedge_exponents') else None
    rules_data = []
    for idx in np.where(mask)[0]:
        class_idx = int(np.argmax(weights[idx]))
        rules_data.append({
            'rule_idx': int(idx),
            'desc': get_full_description(
                int(idx), model.rule_indices_list,
                concept_names, hedge_exp),
            'class_name': str(class_names[class_idx]),
            'class_id': int(class_idx),
            'weight': float(weights[idx, class_idx]),
        })
    rules_data.sort(key=lambda x: (x['class_id'], -x['weight']))
    return rules_data


def format_aggregated_rules_report(rules_data, concept_meta=None, concept_names=None):
    lines = [
        "=" * 80,
        "GENERATED RULES (HIGH/LOW + LINGUISTIC HEDGES)",
        "  Notation: mlHIGH = more or less HIGH (μ^{1/2}), vHIGH = very HIGH (μ²)",
        "           mlLOW  = more or less LOW,              vLOW  = very LOW",
        "=" * 80,
    ]
    if concept_meta and concept_names:
        lines.append("--- CONCEPTS (JK: HIGH=mu(z), LOW=1-mu(z)) ---")
        for i, nm in enumerate(concept_names):
            meta = concept_meta[i] if i < len(concept_meta) else None
            if isinstance(meta, dict) and meta.get("type") == "numeric":
                a = meta.get("a", None)
                b = meta.get("b", None)
                lines.append(
                    f"{nm}: mu_HIGH(z)=sat((z-a)/(b-a)), "
                    f"a={a:.6g}, b={b:.6g}, LOW=1-HIGH")
            elif isinstance(meta, dict):
                lines.append(f"{nm}: binary/categorical, HIGH=1, LOW=0")
        lines.append("")

    rules_by_class = defaultdict(list)
    for r in rules_data:
        rules_by_class[r['class_name']].append(r)

    lines.append(f"Number of atomic rules: {len(rules_data)}")
    lines.append("-" * 80)

    for class_name, rules in rules_by_class.items():
        lines.append(f"\n>>> DECISION CLASS: {class_name} <<<")
        rules.sort(key=lambda x: x['weight'], reverse=True)
        if len(rules) == 1:
            lines.append(
                f"   IF  ( {rules[0]['desc']} )\n"
                f"   THEN -> {class_name}  [Weight: {rules[0]['weight']:.4f}]")
        else:
            lines.append("   IF")
            for i, r in enumerate(rules):
                suffix = 'OR' if i < len(rules) - 1 else ''
                lines.append(
                    f"      ( {r['desc']} )  "
                    f"[Weight: {r['weight']:.4f}] {suffix}")
            lines.append(f"   THEN -> {class_name}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)

def _is_retryable_gpu_error_text(text):
    s = str(text).lower()
    return any(x in s for x in (
        'cuda out of memory', 'outofmemoryerror', 'cublas_status_alloc_failed',
        'cudnn_status_alloc_failed', 'cuda error', 'driver shutting down',
        'device is busy or unavailable', 'illegal memory access', 'nccl',
        'unspecified launch failure', 'context is destroyed', 'timed out',
        'timeout', 'temporarily unavailable'))


def _rule_v25_task_dir(dataset='diabetes', algorithm='CBM_KE', fold=0):
    base = os.environ.get('LDRV25_RULE_STATE_DIR', './results_rule_gen_v25/state')
    return os.path.abspath(os.path.join(base, RUN_MODE, f'seed{SEED}',
                                        str(dataset), str(algorithm), f'fold{fold}'))


class CBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trials: int = None, mode: str = 'standalone', feature_names: list = None):
        self.n_trials     = int(MODE_CFG['n_trials'] if n_trials is None else n_trials)
        self.mode         = mode
        self.feature_names = feature_names
        self.model_       = None
        self._temperature = 1.0
        self._ramp_par = None
        self._idx_top  = None
        self._concept_names = None
        self._ramp_meta = None
        self.rs_centers = {}
        self.rs_scales = {}
        self._input_preprocessor = None
        self._raw_input = False
        self._raw_cat_idx = []
        self._raw_num_idx = []
        self._raw_feature_names = None
        self._transformed_feature_names = None
        self._resilience_context = None
        self._inverse_transform_params = {}

    def _prepare_inputs(self, X, idx_tr, idx_val):
        """Fit preprocessing without inner-validation leakage.

        For raw mixed-type inputs, the transformer is fitted on inner-train
        only in strict mode, stored for outer-test inference, and then applied
        unchanged to inner-val/all rows.  Already-numeric legacy inputs pass
        through unchanged.
        """
        if not getattr(self, '_raw_input', False):
            X_all = np.asarray(X, dtype=np.float32)
            self._input_preprocessor = None
            self._transformed_feature_names = [f"f{i}" for i in range(X_all.shape[1])]
            self._inverse_transform_params = {}
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

        if transformers:
            prep = ColumnTransformer(transformers, remainder='drop')
        else:
            prep = Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scale', RobustScaler()),
            ])
        prep.fit(X_raw[fit_idx])
        X_all = np.asarray(prep.transform(X_raw), dtype=np.float32)
        self._input_preprocessor = prep

        # Names and inverse RobustScaler parameters aligned with transformed
        # columns; this is used by the standalone rule report.
        try:
            names = [str(v).split('__', 1)[-1]
                     for v in prep.get_feature_names_out(raw_names)]
        except Exception:
            names = [f"f{i}" for i in range(X_all.shape[1])]
        if len(names) != X_all.shape[1]:
            names = [f"f{i}" for i in range(X_all.shape[1])]
        self._transformed_feature_names = names
        self._inverse_transform_params = {}
        try:
            if isinstance(prep, ColumnTransformer) and num_idx:
                cat_width = 0
                if cat_idx:
                    ohe = prep.named_transformers_['cat'].named_steps['ohe']
                    cat_width = int(sum(len(c) for c in ohe.categories_))
                rs = prep.named_transformers_['num'].named_steps['scale']
                for j in range(len(num_idx)):
                    self._inverse_transform_params[cat_width + j] = (
                        float(rs.center_[j]), float(rs.scale_[j]))
            elif isinstance(prep, Pipeline):
                rs = prep.named_steps['scale']
                for j in range(X_all.shape[1]):
                    self._inverse_transform_params[j] = (
                        float(rs.center_[j]), float(rs.scale_[j]))
        except Exception:
            self._inverse_transform_params = {}
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

        # Split first; fit preprocessing, PPL percentiles and mRMR
        # only on inner-train in strict mode.
        X_all, X_tr, X_val = self._prepare_inputs(X, idx_tr, idx_val)
        d = X_all.shape[1]
        y_tr, y_val = y[idx_tr], y[idx_val]

        cw_source = y_tr if _STRICT_VALIDATION else y
        cw_classes = np.unique(y)
        cw_map = dict(zip(np.unique(cw_source), compute_class_weight(
            'balanced', classes=np.unique(cw_source), y=cw_source)))
        cw = np.asarray([cw_map.get(c, 1.0) for c in cw_classes], dtype=np.float32)
        cw_tensor = torch.tensor(cw, dtype=torch.float32, device=device)

        C_tr, ramp_par, idx_top, k_avail = _build_fuzzy_concepts_train(
            X_tr, y_tr, max_k=MAX_CONCEPTS)
        C_val = _apply_fuzzy_concepts_test(X_val, ramp_par, idx_top)
        C_full = _apply_fuzzy_concepts_test(X_all, ramp_par, idx_top)
        self._ramp_par = ramp_par
        self._idx_top  = idx_top

        all_names = (self._transformed_feature_names or
                     [f"f{i}" for i in range(d)])
        self._concept_names_all = [all_names[i] for i in idx_top]

        soft_full = None
        soft_tr   = None
        teacher_  = None
        if self.mode == 'ke':
            soft_parts = []
            teacher_   = None
            # V20: strict mode prevents the teacher from seeing the inner
            # validation labels used for HPO, hedge selection and calibration.
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
            # The optional tree-teacher development branch remains disabled in
            # the reviewer package. It is deliberately not revived in v20.
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
            # FastTensorLoader instead of DataLoader(TensorDataset)
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
            rb_hi = rb_lo if MODE_CFG.get('force_min_rule_budget', False) else int(min(n_cls + 16,  2 ** k_c))
            if rb_lo > rb_hi:
                rb_lo = rb_hi
            rb = trial.suggest_int('rule_budget', rb_lo, rb_hi)

            bs = trial.suggest_categorical('batch_size', list(MODE_CFG['batch_size_grid']))
            hd = trial.suggest_categorical('hidden_dim', list(MODE_CFG['hidden_dim_grid']))
            dr = trial.suggest_float('dropout',        0.0,  0.5)
            lr = trial.suggest_float('lr',             1e-4, 1e-2,  log=True)
            wd = trial.suggest_float('weight_decay',   1e-6, 1e-3,  log=True)
            lc = trial.suggest_float('lambda_concept', 0.05, 5.0)
            ls   = trial.suggest_float('lambda_sparse',  1e-6, 1e-2,  log=True)
            T_kd  = trial.suggest_float('kd_temperature', 1.0, 5.0) if soft_tr is not None else 2.0
            a_kd  = trial.suggest_float('kd_alpha_hard', 0.2, 0.9) if soft_tr is not None else 1.0
            cdrop = trial.suggest_categorical('concept_dropout_p',
                                              [0.0, 0.05, 0.1, 0.15])

            _seed_t = SEED + 1009 * (trial.number + 1)
            torch.manual_seed(_seed_t)
            np.random.seed(_seed_t % (2**31))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_seed_t)
            m   = CBMP1TSModelV4(d, k_c, n_cls, hd, dr,
                                  concept_dropout_p=cdrop,
                                  concept_names=self._concept_names_all[:k_c]).to(device)
            trl = make_train_loader(X_tr, y_tr, C_tr[:, :k_c], bs,
                                    soft_a=soft_tr if soft_tr is not None else None)
            vl  = make_val_loader(X_val, y_val, C_val[:, :k_c], bs)

            try:
                m, val_acc = _train_cbm_v4(
                    m, trl, vl,
                    epochs=int(MODE_CFG['hpo_epochs']), lr=lr, weight_decay=wd,
                    lambda_concept=lc, lambda_sparse=ls,
                    kd_temperature=T_kd,
                    kd_alpha_hard=a_kd,
                    cw_tensor=cw_tensor,
                    trial=trial, silent=True)
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
                if _is_retryable_gpu_error_text(repr(_e)):
                    raise
                raise optuna.exceptions.TrialPruned(str(_e))
            finally:
                del m; gc.collect()
            return ret

        import importlib.util as _iutil
        _cmaes_available = _iutil.find_spec('cmaes') is not None
        if not _cmaes_available:
            print("  [HPO] package 'cmaes' unavailable -> TPE "
                  "(pip install cmaes to enable CMA-ES)", flush=True)

        _sampler = None
        if _cmaes_available:
            try:
                _sampler = optuna.samplers.CmaEsSampler(
                    seed=SEED,
                    n_startup_trials=min(25, max(4, int(MODE_CFG['n_trials']) // 3)),
                    warn_independent_sampling=False,
                )
            except Exception as _ce:
                print(f"  [HPO] CmaEsSampler error: {_ce} -> TPE", flush=True)
                _sampler = None

        if _sampler is None:
            _sampler = optuna.samplers.TPESampler(
                seed=SEED, n_startup_trials=min(25, max(4, int(MODE_CFG['n_trials']) // 3)))

        _ctx = getattr(self, '_resilience_context', None) or {}
        _task_dir = _ctx.get('task_dir') or _rule_v25_task_dir()
        os.makedirs(_task_dir, exist_ok=True)
        _study_db = os.path.join(_task_dir, 'optuna.sqlite3')
        _study_name = 'rulegen_' + hashlib.sha256(json.dumps({
            'mode': RUN_MODE, 'seed': SEED, 'task_dir': _task_dir,
            'teacher': self.mode, 'n_trials': int(self.n_trials),
        }, sort_keys=True).encode()).hexdigest()[:24]
        try:
            _storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{_study_db}",
                engine_kwargs={'connect_args': {'timeout': 120}},
                heartbeat_interval=60, grace_period=300)
        except TypeError:
            _storage = f"sqlite:///{_study_db}"
        study = optuna.create_study(
            study_name=_study_name, storage=_storage, load_if_exists=True,
            direction='maximize', sampler=_sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=min(15, max(4, int(MODE_CFG['n_trials']) // 3)),
                n_warmup_steps=5 if RUN_MODE != 'normal' else 20))
        try: optuna.storages.fail_stale_trials(study)
        except Exception: pass
        _done_trials = sum(t.state in (optuna.trial.TrialState.COMPLETE,
                                       optuna.trial.TrialState.PRUNED)
                           for t in study.trials)
        _remaining = max(0, int(self.n_trials)-int(_done_trials))
        if _done_trials:
            print(f"  [V25-HPO-RESUME] {_done_trials}/{self.n_trials}; remaining={_remaining}", flush=True)
        _hpo_to = _CBM_HPO_TIMEOUT if _CBM_HPO_TIMEOUT > 0 else None
        if _remaining:
            study.optimize(objective, n_trials=_remaining,
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
        print(f"  [CBM] Final training: {MODE_CFG['final_epochs']} epochs × "
              f"{MODE_CFG['final_restarts']} restart(s), M={best_k}, "
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
        _task_dir = _ctx.get('task_dir') or _rule_v25_task_dir()
        os.makedirs(_task_dir, exist_ok=True)
        _interval = int(os.environ.get('LDRV25_EPOCH_CHECKPOINT_INTERVAL', '10'))
        _meta_base = {'mode': RUN_MODE, 'seed': SEED, 'best_params': bp,
                      'final_epochs': int(MODE_CFG['final_epochs'])}
        for _restart in range(int(MODE_CFG['final_restarts'])):
            _seed_r = SEED + _restart * 37
            _complete = os.path.join(_task_dir, f'restart_{_restart}.pt')
            _epoch_ck = os.path.join(_task_dir, f'restart_{_restart}_epoch.pt')
            _meta = dict(_meta_base, restart=_restart, restart_seed=_seed_r)
            _concept_names_k = self._concept_names_all[:best_k]
            _m_try = None; _val_fin = float('-inf')
            if os.path.isfile(_complete):
                try:
                    saved=torch.load(_complete,map_location=device,weights_only=False)
                    if saved.get('metadata') != _meta: raise ValueError('metadata mismatch')
                    _m_try=CBMP1TSModelV4(d,best_k,n_cls,best_hd,best_dr,
                        concept_dropout_p=best_cdrop,concept_names=_concept_names_k).to(device)
                    _m_try.load_state_dict(saved['model_state'])
                    _val_fin=float(saved['val_acc'])
                    print(f"  [V25-RESTART-RESUME] {_restart+1}: {_val_fin:.4f}",flush=True)
                except Exception:
                    try: os.remove(_complete)
                    except OSError: pass
                    _m_try=None
            if _m_try is None:
                torch.manual_seed(_seed_r); np.random.seed(_seed_r%(2**31)); random.seed(_seed_r)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(_seed_r)
                _m_try=CBMP1TSModelV4(d,best_k,n_cls,best_hd,best_dr,
                    concept_dropout_p=best_cdrop,concept_names=_concept_names_k).to(device)
                _m_try,_val_fin=_train_cbm_v4(
                    _m_try,trl_fin,vl_fin,epochs=int(MODE_CFG['final_epochs']),lr=best_lr,
                    lambda_concept=best_lc,lambda_sparse=best_ls,
                    kd_temperature=best_T_kd,kd_alpha_hard=best_a_kd,
                    weight_decay=best_wd,cw_tensor=cw_tensor,silent=(_restart>0),
                    checkpoint_path=_epoch_ck,checkpoint_interval=_interval,
                    checkpoint_metadata=_meta)
                _atomic_torch_save({'metadata':_meta,'val_acc':float(_val_fin),
                    'model_state':_state_dict_cpu(_m_try.state_dict())},_complete)
                try: os.remove(_epoch_ck)
                except OSError: pass
            print(f"  [CBM] Restart {_restart+1}/{int(MODE_CFG['final_restarts'])}: val_acc={_val_fin:.4f}",flush=True)
            if _val_fin > _best_val_fin:
                _best_val_fin=_val_fin; self.model_=copy.deepcopy(_m_try)
            del _m_try; gc.collect()
        torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        print(f"  [CBM] Best restart val_acc={_best_val_fin:.4f}", flush=True)

        _t_fin_end = time.time()
        print(f"  [CBM] Final training: {_t_fin_end-_t_fin_start:.1f}s",
              flush=True)
        kept = self.model_.keep_balanced_rules(k_total=best_rb)
        print(f"  [CBM] Active rules after pruning: {kept} "
              f"(target={best_rb}, K={n_cls})", flush=True)

        n_hedge = optimize_linguistic_hedges(
            self.model_, vl_fin, n_cls,
            max_iter=int(MODE_CFG['hedge_iter']), silent=False)
        print(f"  [HEDGE] Modified exponents: {n_hedge}", flush=True)

        try:
            self._temperature = _temperature_scaling(
                self.model_, X_val, y_val)
        except Exception as e:
            print(f"  [TempScale] ERROR: {e} -> T=1.0", flush=True)
            self._temperature = 1.0
        self._concept_names = self.model_.concept_names
        self._ramp_meta = []
        for i in range(best_k):
            orig_idx = idx_top[i]
            typ, a_scaled, b_scaled = ramp_par[orig_idx]
            nm = self._concept_names_all[i]
            if typ != 'binary':
                inv = self._inverse_transform_params.get(orig_idx)
                if inv is not None:
                    rs_c, rs_s = inv
                    a_orig = a_scaled * rs_s + rs_c
                    b_orig = b_scaled * rs_s + rs_c
                    self._ramp_meta.append({
                        "type": "numeric",
                        "name": nm,
                        "a": a_orig,
                        "b": b_orig,
                    })
                else:
                    self._ramp_meta.append({
                        "type": "numeric",
                        "name": nm,
                        "a": a_scaled,
                        "b": b_scaled,
                    })
            else:
                self._ramp_meta.append({
                    "type": "binary",
                    "name": nm,
                })
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

# --------------------------------------------------------------------
# TabPFN-3 teacher / baseline configuration
# --------------------------------------------------------------------
_TABPFN_MODE = 'tabpfn-3-explicit-checkpoint'
_TABPFN_PACKAGE_VERSION = 'unknown'
if _TABPFN:
    try:
        import tabpfn as _tabpfn_pkg
        _TABPFN_PACKAGE_VERSION = getattr(_tabpfn_pkg, '__version__', 'unknown')
    except Exception:
        pass

# Publication runs pin the exact checkpoint explicitly. The
# official TabPFN API supports TabPFNClassifier(model_path=...). Guessing the
# largest file in a cache is forbidden because it may not be the file used.
_TABPFN_MODEL_PATH = os.environ.get(
    'LDRV25_TABPFN_MODEL_PATH', os.environ.get('TABPFN_MODEL_PATH', '')
).strip()
if _TABPFN_MODEL_PATH:
    _TABPFN_MODEL_PATH = os.path.abspath(os.path.expanduser(_TABPFN_MODEL_PATH))
_TABPFN_REQUIRE_PINNED_MODEL = (
    os.environ.get('LDRV25_REQUIRE_PINNED_TABPFN_MODEL', '1') != '0'
)
_TABPFN_REQUIRE_TOKEN = (
    os.environ.get('LDRV25_REQUIRE_TABPFN_TOKEN',
                   os.environ.get('LDRV20_REQUIRE_TABPFN_TOKEN', '1')) != '0'
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

_TABPFN_N_MAX = int(os.environ.get('LDRV25_TABPFN_N_MAX', os.environ.get('LDRV20_TABPFN_N_MAX', '1000000')))
_TABPFN_D_MAX = int(os.environ.get('LDRV25_TABPFN_D_MAX', os.environ.get('LDRV20_TABPFN_D_MAX', '500')))
_TABPFN_K_MAX = int(os.environ.get('LDRV25_TABPFN_K_MAX', os.environ.get('LDRV20_TABPFN_K_MAX', '1000')))
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
                  f"(n={n},d={d},K={K}) -> fallback LR", flush=True)
            self._mode = 'fallback'
            if _DISABLED_TREE_TEACHER:
                self._clf = _DISABLED_TREE_TEACHER_CLASSIFIER(
                    n_estimators=300, learning_rate=0.05,
                    class_weight='balanced', random_state=SEED, verbose=-1)
            else:
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
            print(f"  [IDS] fit error: {e} -> fallback DT", flush=True)
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
    def wrap(factory):
        return lambda n_i, n_c: ProbaCachingClassifier(factory(n_i, n_c))

    clfs = []

    # clfs.append(('CBM',
        # wrap(lambda n_i, n_c: CBMClassifier(mode='standalone'))))
    clfs.append(('CBM_KE',
        wrap(lambda n_i, n_c: CBMClassifier(mode='ke'))))

    # clfs.append(('DT_depth5',
        # wrap(lambda n_i, n_c: DecisionTreeClassifier(
            # max_depth=5, random_state=SEED, class_weight='balanced'))))
    # clfs.append(('DT_depth3',
        # wrap(lambda n_i, n_c: DecisionTreeClassifier(
            # max_depth=3, random_state=SEED, class_weight='balanced'))))

    # clfs.append(('LogReg',
        # wrap(lambda n_i, n_c: LogisticRegression(
            # max_iter=1000, random_state=SEED,
            # class_weight='balanced', solver='lbfgs'))))

    # clfs.append(('KNN_k5',
        # wrap(lambda n_i, n_c: KNeighborsClassifier(n_neighbors=5, n_jobs=1))))

    if _IMODELS:
        clfs.append(('FIGS',
            wrap(lambda n_i, n_c: FIGSClassifier(
                max_rules=12, random_state=SEED))))
        clfs.append(('RuleFit',
            wrap(lambda n_i, n_c: RuleFitClassifier(
                random_state=SEED, max_rules=50))))
    else:
        print("[INFO] FIGS/RuleFit skipped (missing 'imodels'). "
              "Installation: pip install imodels")

    if _CORELS:
        clfs.append(('CORELS',
            wrap(lambda n_i, n_c: BinarizingWrapper(
                _CORElSWrapper(max_card=2, c=0.01, n_iter=10000)))))
    else:
        print("[INFO] CORELS skipped (pip install corels).")

    if _BRCG:
        clfs.append(('BRCG',
            wrap(lambda n_i, n_c: BinarizingWrapper(
                _BRCGWrapper(lambda0=0.001, lambda1=0.001)))))
    else:
        print("[INFO] BRCG skipped (pip install aix360).")

    if _IMLI:
        clfs.append(('IMLI',
            wrap(lambda n_i, n_c: BinarizingWrapper(
                _IMLIWrapper(num_clauses=4, rule_length=3),
                timeout_s=FOLD_TIMEOUT_S))))
    else:
        print("[INFO] IMLI skipped (pip install imli).")

    if _BRS:
        clfs.append(('BRS',
            wrap(lambda n_i, n_c: BinarizingWrapper(
                _BRSWrapper(n_iter=2000, n_chains=2),
                timeout_s=FOLD_TIMEOUT_S))))
    else:
        print("[INFO] BRS skipped (pip install aix360).")

    if _BRL:
        clfs.append(('BRL',
            wrap(lambda n_i, n_c: BinarizingWrapper(
                _BRLWrapper(n_chains=3, n_iter=30000),
                timeout_s=FOLD_TIMEOUT_S))))
    else:
        print("[INFO] BRL skipped (pip install pysbrl).")

    if _IDS:
        clfs.append(('IDS',
            wrap(lambda n_i, n_c: TimeoutWrapper(
                _IDSWrapper(), timeout_s=FOLD_TIMEOUT_S))))
    else:
        print("[INFO] IDS skipped (pip install pyids).")

    if _DL85:
        clfs.append(('DL8.5',
            wrap(lambda n_i, n_c: BinarizingWrapper(
                _DL85Wrapper(max_depth=4, time_limit=120)))))
    else:
        print("[INFO] DL8.5 skipped (pip install dl8.5).")

    if _EBM:
        clfs.append(('EBM',
            wrap(lambda n_i, n_c: ExplainableBoostingClassifier(
                random_state=SEED))))
    else:
        print("[INFO] EBM skipped (missing 'interpret').")

    if _RIPPER:
        clfs.append(('RIPPER',
            wrap(lambda n_i, n_c: RIPPERMulticlassWrapper(
                k=2, random_state=SEED))))
    else:
        print("[INFO] RIPPER skipped (missing 'wittgenstein').")

    if _TABPFN and MODE_CFG.get('standalone_tabpfn', True):
        clfs.append(('TabPFN',
            wrap(lambda n_i, n_c: SafeTabPFN(device_=device))))
    elif _TABPFN:
        print("[INFO] FAST mode: standalone TabPFN classifier skipped; TabPFN-3 is used only as CBM_KE teacher.")
    else:
        print("[INFO] TabPFN skipped (missing 'tabpfn').")

    if _DISABLED_TREE_TEACHER:
        clfs.append(('disabled optional tree teacher',
            wrap(lambda n_i, n_c: _DISABLED_TREE_TEACHER_CLASSIFIER(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                num_leaves=31, class_weight='balanced',
                random_state=SEED, verbose=-1))))
    if _CATBOOST:
        clfs.append(('CatBoost',
            wrap(lambda n_i, n_c: CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6,
                random_seed=SEED, verbose=False,
                auto_class_weights='Balanced'))))
    if _XGBOOST:
        clfs.append(('XGBoost',
            wrap(lambda n_i, n_c: XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                random_state=SEED, eval_metric='mlogloss',
                verbosity=0, use_label_encoder=False))))

    # clfs.append(('RandomForest',
        # wrap(lambda n_i, n_c: RandomForestClassifier(
            # n_estimators=300, max_depth=None,
            # class_weight='balanced', random_state=SEED, n_jobs=1))))

    # clfs.append(('MLP',
        # wrap(lambda n_i, n_c: MLPClassifier(
            # hidden_layer_sizes=(256, 128), max_iter=500,
            # random_state=SEED, early_stopping=True,
            # validation_fraction=0.1))))

    # clfs.append(('SVM_RBF',
        # wrap(lambda n_i, n_c: SVC(
            # kernel='rbf', probability=True,
            # class_weight='balanced', random_state=SEED, C=1.0))))

    print(f"\n>>> Loaded classifiers: {len(clfs)}", flush=True)
    for name, _ in clfs:
        print(f"    - {name}", flush=True)
    return clfs

CBM_NAMES = {'CBM_KE'}  # 'CBM', 'CBM_KE'}

SOTA_INTERPRETABLE_NAMES = set()
    # {
    # 'EBM', 'FIGS', 'RuleFit', 'RIPPER', 'LogReg',
    # 'CORELS', 'BRCG', 'IMLI', 'BRS', 'BRL', 'IDS', 'DL8.5',
    # 'DT_depth5', 'DT_depth3',
# }
SOTA_INTERPRETABLE_TOP5_COUNT = 5

SOTA_NONINTERPRETABLE_NAMES = {'TabPFN'}


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
        print(f"  [SEPARATE] comparison.csv empty ({csv_path}) -> skipping.",
              flush=True)
        return
    except Exception as e:
        print(f"  [SEPARATE] Error reading comparison.csv: {e} -> skipping.",
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

        print(f"  [SEPARATE] {subdir_name}/ -> {out_dir}", flush=True)

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
                print(f"  [MERGE] WARN: {f0} empty/corrupted -> skipping.", flush=True)
                df0 = pd.DataFrame()
            try:
                df1 = pd.read_csv(f1_)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                print(f"  [MERGE] WARN: {f1_} empty/corrupted -> skipping.", flush=True)
                df1 = pd.DataFrame()
            if df0.empty and df1.empty:
                print(f"  [MERGE] WARN: both files empty -> skipping merge.", flush=True)
                continue
            merged = pd.concat([df0, df1], ignore_index=True)
            out_path = os.path.join(merged_dir, fname)
            merged.to_csv(out_path, index=False)
            print(f"  [MERGE] {fname}: {len(df0)}+{len(df1)} -> {len(merged)} rows",
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
        print(f"  [MERGE] comparison.csv empty or missing -> skipping CACP.",
              flush=True)

    try:
        _clfs_all = build_classifiers()
        post_process_separate_comparisons(
            merged_dir, _clfs_all, CACP_METRICS)
        print(f"  [MERGE] ✓ interpretable/ + noninterpretable/", flush=True)
    except Exception as e:
        print(f"  [MERGE] ✗ separate comparisons: {e}", flush=True)

    print(f"  [MERGE] Merged results -> {merged_dir}", flush=True)

from concurrent.futures import ThreadPoolExecutor, as_completed

# PyTorch/NumPy RNGs are process-global. Publication-integrity mode
# serializes complete GPU tasks so parallel threads cannot overwrite one
# another's seeds during model construction, shuffling or training.
_SERIALIZE_GPU_TASKS = os.environ.get('LDRV25_SERIALIZE_GPU_TASKS', '1') != '0'
_GPU_TASK_LOCK = threading.RLock()

_GPU_CLF_NAMES = {'CBM', 'CBM_KE', 'TabPFN'}

def _is_gpu_clf(name):
    return any(name == g or name.startswith(g + '_') for g in _GPU_CLF_NAMES)

def _run_one_task_once(ds_name, fold, clf_name, clf_factory, metrics_list,
                       fold_seed=None):
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
                _inner._resilience_context = {
                    'task_dir': _rule_v25_task_dir(ds_name, clf_name, fold.index)
                }
                _inner.feature_names = getattr(fold, 'feature_names', None)
                _inner._raw_input = True
                _inner._raw_cat_idx = getattr(fold, 'cat_idx', [])
                _inner._raw_num_idx = getattr(fold, 'num_idx', [])
                _inner._raw_feature_names = getattr(
                    fold, 'raw_feature_names', getattr(fold, 'feature_names', None))
                X_train_input = fold.x_train_raw
                X_test_input = fold.x_test_raw
        except Exception:
            pass

        _t_train = time.time()
        clf.fit(X_train_input, fold.y_train)

        try:
            _inner = clf.clf if hasattr(clf, 'clf') else clf
            if isinstance(_inner, CBMClassifier) and _inner.model_ is not None:
                _model = _inner.model_
                _concept_names = getattr(_inner, '_concept_names',
                                         _model.concept_names)
                _ramp_meta = getattr(_inner, '_ramp_meta', None)
                _class_names = getattr(fold, 'class_names_orig',
                        [str(c) for c in fold.labels])
                _rules_data = extract_metarules(_model, _class_names, _concept_names)
                _rules_report = format_aggregated_rules_report(
                    _rules_data, _ramp_meta, _concept_names)
                print(_rules_report, flush=True)
        except Exception as _re:
            print(f"  [RULES] Rule extraction error: {_re}", flush=True)

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



def _run_one_task(ds_name, fold, clf_name, clf_factory, metrics_list,
                  fold_seed=None):
    """Retry transient GPU resource failures indefinitely with backoff.

    Durable Optuna/restart/epoch state guarantees monotonic progress.
    """
    attempt = 0
    while True:
        result = _run_one_task_once(ds_name, fold, clf_name, clf_factory,
                                    metrics_list, fold_seed=fold_seed)
        if str(result.get('Status','FAILED')).upper() in ('OK','PARTIAL'):
            return result
        error = result.get('Error','')
        if not _is_gpu_clf(clf_name) or not _is_retryable_gpu_error_text(error):
            return result
        attempt += 1
        delay = min(900, 30 * (2 ** min(attempt-1, 10)))
        print(f"  [V25-RETRY] {clf_name} fold={fold.index}: {error}; sleep={delay}s",
              flush=True)
        gc.collect()
        if torch.cuda.is_available():
            try: torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            except Exception: pass
        time.sleep(delay)


def _progress_bar(pct, width=20):
    filled = int(width * pct / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"


def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_progress(ds_idx, n_ds, n_done_results, n_total_results,
                    t_start, ds_name, ds_time):
    elapsed   = time.time() - t_start
    pct_ds    = 100.0 * (ds_idx + 1) / n_ds
    pct_res   = 100.0 * n_done_results / max(1, n_total_results)
    pct       = (pct_ds + pct_res) / 2.0

    bar       = _progress_bar(pct)
    t_elapsed = _fmt_time(elapsed)

    if pct > 0.5:
        eta_s = elapsed * (100.0 - pct) / pct
    else:
        eta_s = 0.0
    t_eta = _fmt_time(eta_s)

    speed = (n_done_results / elapsed * 60) if elapsed > 1 else 0.0

    t_total_est = elapsed + eta_s
    t_total_str = _fmt_time(t_total_est)

    print(f"\n{'═'*62}", flush=True)
    print(f"  PROGRESS {bar} {pct:5.1f}%  |  dataset {ds_idx+1}/{n_ds}",
          flush=True)
    print(f"  Results:   {n_done_results}/{n_total_results} CSV rows",
          flush=True)
    print(f"  Elapsed:  {t_elapsed}  |  ETA: ~{t_eta}  "
          f"|  Total est.: ~{t_total_str}", flush=True)
    print(f"  Speed:    {speed:.1f} results/min  "
          f"|  Last dataset ({ds_name}): {ds_time:.0f}s", flush=True)
    print(f"{'═'*62}\n", flush=True)

# =====================================================================
# RESOURCE MONITOR: CPU / RAM / GPU / progress / STALL
# =====================================================================
_STALL_THRESHOLD_S = 1800  # 30 min without a finished task => warning


class _MonState:
    """Minimal shared progress state for the resource monitor."""
    def __init__(self):
        self.lock        = threading.Lock()
        self.t_start     = time.time()
        self.t_last_task = time.time()
        self.tasks_done  = 0
        self.tasks_total = 0
        self.ds_name     = ""

    def reset(self, total):
        with self.lock:
            self.t_start     = time.time()
            self.t_last_task = time.time()
            self.tasks_done  = 0
            self.tasks_total = int(total)

    def task_done(self):
        with self.lock:
            self.tasks_done += 1
            self.t_last_task = time.time()

    def set_ds(self, name):
        with self.lock:
            self.ds_name = str(name)

    def snapshot(self):
        with self.lock:
            return dict(tasks_done=self.tasks_done,
                        tasks_total=self.tasks_total,
                        ds_name=self.ds_name,
                        t_last_task=self.t_last_task,
                        elapsed=time.time() - self.t_start)


_MON = _MonState()
_MON_STARTED = [False]


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
                info[k] = int(v.split()[0])  # kB
        tot   = info.get('MemTotal', 0)     / 1048576.0
        avail = info.get('MemAvailable', 0) / 1048576.0
        return tot, tot - avail
    except Exception:
        return 0.0, 0.0


def _resource_monitor_worker(interval: int = 60):
    import multiprocessing as _mp2
    n_cores = _mp2.cpu_count()
    while True:
        time.sleep(max(5, interval))
        try:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            try:
                la1, _, _ = os.getloadavg()
            except OSError:
                la1 = 0.0
            ram_tot, ram_used = _read_ram_gib()
            gpu_rows = _read_gpu_stats()
            gpu_str = "  ".join(
                f"GPU{r[0]}: {r[1]}% util, {r[2]}/{r[3]} MiB"
                for r in gpu_rows) or "GPU: n/a"
            snap = _MON.snapshot()
            tt   = snap['tasks_total']
            pct  = 100.0 * snap['tasks_done'] / tt if tt else 0.0
            eta  = ((snap['elapsed'] * (tt - snap['tasks_done'])
                     / snap['tasks_done'])
                    if snap['tasks_done'] > 0 and tt else 0.0)
            print(f"  [MON {ts}] CPU load1m={la1:.1f}/{n_cores}"
                  f" | RAM {ram_used:.1f}/{ram_tot:.0f} GiB"
                  f" | {gpu_str}"
                  f" | tasks {snap['tasks_done']}/{tt} ({pct:.1f}%)"
                  f" | ETA ~{_fmt_time(eta)}", flush=True)
            stall_s = time.time() - snap['t_last_task']
            # STALL fires ALSO with zero completed tasks: the
            # observed v01 failure stalled BEFORE the first task finished
            # (startup network call) and stayed invisible for 10 days.
            if stall_s > _STALL_THRESHOLD_S:
                _where = ("startup/dataset-loading/pre-warm phase"
                          if snap['tasks_done'] == 0
                          else f"dataset='{snap['ds_name']}'")
                print(f"  [MON][STALL] No task finished for "
                      f"{stall_s/60:.0f} min ({_where}). "
                      f"Check: py-spy dump --pid {os.getpid()}", flush=True)
        except Exception:
            pass


def _record_versions(out_dir):
    """Write a v24 environment and exact-teacher manifest."""
    import json as _json
    import platform as _platform
    info = {
        'driver': 'rule_gen_v25_1.py',
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
        'run_mode': RUN_MODE,
        'mode_config': MODE_CFG,
        'tf32': os.environ.get('LDRV2_TF32', '1'),
        'serialize_gpu_tasks': _SERIALIZE_GPU_TASKS,
        'tabpfn': _TABPFN_PACKAGE_VERSION,
        'tabpfn_fingerprint': (_record_tabpfn_fingerprint()
                               if _TABPFN else None),
        'v24_fixes': [
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
        print(f"  [V24] Environment manifest -> {_p}", flush=True)
    except Exception as _e:
        print(f"  [V24] manifest write error: {_e}", flush=True)

def _start_resource_monitor(interval: int = 60):
    if _MON_STARTED[0]:
        return
    _MON_STARTED[0] = True
    t = threading.Thread(target=_resource_monitor_worker,
                         args=(interval,), daemon=True)
    t.start()
    print(f"  [V02] Resource monitor started (every {interval}s)", flush=True)


def parallel_run_experiment(datasets, classifiers, results_directory, metrics,
                            n_gpu_workers=5, n_cpu_workers=14):
    os.makedirs(results_directory, exist_ok=True)
    _start_resource_monitor(interval=_MON_INTERVAL)
    _record_versions(results_directory)

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

    _n_folds       = int(MODE_CFG['n_folds'])
    _n_clf         = len(classifiers)
    _n_ds          = len(datasets)
    _n_total_res   = _n_ds * _n_folds * _n_clf
    _n_done_res    = 0
    print(f"  [PROGRESS] Estimated number of results: "
          f"{_n_ds} datasets × {_n_folds} folds × {_n_clf} clf "
          f"= {_n_total_res} CSV rows", flush=True)
    _MON.reset(total=_n_total_res)

    # ----------------------------------------------------------------
    # CHECKPOINTING: every completed (Dataset, Algorithm, fold)
    # is appended immediately to checkpoint_rows.csv; on restart these
    # tasks are skipped (set LDRV2_NO_RESUME=1 to recompute everything).
    # ----------------------------------------------------------------
    _ckpt_path = os.path.join(results_directory, 'checkpoint_rows.csv')
    _ckpt_lock = threading.Lock()
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
            for _ in range(len(_df_ok)):
                _MON.task_done()
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
        # Pre-warm in a watchdog thread: even if a network layer
        # ignores the global socket timeout, the main flow continues after
        # LDRV2_PREWARM_TIMEOUT seconds (default 600).
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
        _pw_t = threading.Thread(target=_prewarm_tabpfn, daemon=True)
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
        print(f"{'─'*60}", flush=True)

        _t_folds = time.time()
        folds = list(ds.folds(n_folds=int(MODE_CFG['n_folds'])))
        print(f"  Folds: {len(folds)} (pre-computed in "
              f"{time.time()-_t_folds:.1f}s)", flush=True)

        _MON.set_ds(ds.name)

        # ---------------------- CPU TASKS ---------------------
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
        _cpu_lock    = threading.Lock()
        _t_cpu_start = time.time()

        def _on_cpu_result(res):
            with _cpu_lock:
                _cpu_results.append(res)
                _n_now = len(_cpu_results)
            _ckpt_append(res)
            _MON.task_done()
            if _n_now % 5 == 0 or _n_now == n_cpu_tasks:
                _dt_cpu = time.time() - _t_cpu_start
                _pct2 = 100.0 * _n_now / max(1, n_cpu_tasks)
                _per_task2 = _dt_cpu / max(1, _n_now)
                _eta_cpu = _per_task2 * (n_cpu_tasks - _n_now)
                print(f"    CPU [{_pct2:5.1f}%] {_n_now}/{n_cpu_tasks}  "
                      f"{res['Algorithm']} f{res['CV index']}  "
                      f"avg/task={_per_task2:.1f}s  "
                      f"eta_cpu={_fmt_time(_eta_cpu)}", flush=True)

        def _cpu_threads_run(task_args):
            ex = ThreadPoolExecutor(max_workers=n_cpu_workers,
                                    thread_name_prefix=f'cpu_{ds.name}')
            futs = {ex.submit(_run_one_task, a0, a1, a2, a3, a4,
                              fold_seed=a5): (a2, a1.index)
                    for (a0, a1, a2, a3, a4, a5) in task_args}
            # no result(timeout=...): after as_completed it could
            # never fire; hangs are reported by the [STALL] monitor.
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
            """Runs CPU classifiers in PROCESSES (joblib/loky); isolates
            runaway native threads left by TimeoutWrapper from this
            process. Automatic fallback to the v01 thread pool."""
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
                for res in _gen:
                    _on_cpu_result(res)
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

        _cpu_thread = threading.Thread(target=_cpu_collector, daemon=True)
        _cpu_thread.start()

        # ---------------------- GPU TASKS -----------------------------
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
        _t_fold_start = time.time()

        # NOTE: in v01, fut.result(timeout=3600) after as_completed
        # could never raise TimeoutError (as_completed yields only ALREADY
        # completed futures); a hung task is detected by [STALL] instead.
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
            _MON.task_done()
            n_done += 1
            # After every fold show the progress + ETA for this dataset
            _dt_gpu = time.time() - _t_fold_start
            _pct = 100.0 * n_done / max(1, n_gpu_tasks)
            _per_task = _dt_gpu / max(1, n_done)
            _eta_gpu = _per_task * (n_gpu_tasks - n_done)
            print(f"    GPU [{_pct:5.1f}%] {n_done}/{n_gpu_tasks}  "
                  f"{clf_n} f{fold_i} train={res['Train time [s]']:.0f}s  "
                  f"avg/task={_per_task:.0f}s  eta_gpu={_fmt_time(_eta_gpu)}",
                  flush=True)

        gpu_executor.shutdown(wait=True)
        _t_gpu_done = time.time()
        print(f"  GPU done: {_t_gpu_done - _t_ds_start:.1f}s", flush=True)

        _cpu_thread.join()
        with _cpu_lock:
            ds_results.extend(_cpu_results)

        all_results.extend(ds_results)
        _n_done_res += len(ds_results)
        _dt_ds = time.time() - _t_ds_start
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
        print(f"  [CACP-SKIP] No results (len(df)==0) -> skipping post-processing.",
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


if __name__ == '__main__':

    _role = f"WORKER-GPU{_GPU_WORKER_ID}" if _GPU_WORKER_ID else "MASTER"
    print(f"\n{'#'*70}", flush=True)
    print(f"#  rule_gen.py v5.0-standalone-interactive  |  role={_role}  |  PID={os.getpid()}",
          flush=True)
    print(f"#  Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)
    print(f"{'#'*70}\n", flush=True)

    # -----------------------------------------------------------------
    # AUTOMATIC HARDWARE DETECTION
    # -----------------------------------------------------------------
    import multiprocessing as _mp
    n_gpus       = torch.cuda.device_count()
    n_cpu_cores  = _mp.cpu_count()

    print(f">>> HARDWARE DETECTION:", flush=True)
    print(f"    CPU cores (logical): {n_cpu_cores}", flush=True)
    print(f"    CUDA GPUs detected:  {n_gpus}", flush=True)
    if n_gpus > 0:
        for _i in range(n_gpus):
            _nm  = torch.cuda.get_device_name(_i)
            _mem = torch.cuda.get_device_properties(_i).total_memory / 1e9
            print(f"      GPU {_i}: {_nm} ({_mem:.1f} GB)", flush=True)

    # -----------------------------------------------------------------
    # DATASET LOADING (needed to know whether it is diabetes only)
    # -----------------------------------------------------------------
    datasets    = get_all_datasets()
    classifiers = build_classifiers()
    print_dataset_summary(datasets)

    n_ds = len(datasets)
    if n_ds == 0:
        print("\n[ERROR] No datasets loaded. Check network access to "
              "OpenML and contents of DATASETS_BINARY/DATASETS_MULTI.",
              flush=True)
        sys.exit(1)

    # -----------------------------------------------------------------
    # AUTOMATIC SELECTION OF THE NUMBER OF WORKERS
    # -----------------------------------------------------------------
    # Heuristic:
    # For 1 dataset (e.g. diabetes) there is no need for many workers
    # overlapping on the same GPU. Each worker loads the TabPFN
    # weights into VRAM (~1.5 GB); an excess of workers would cause
    # OOM or thrash the GPU. 2 GPU workers are optimal for 1-3
    # datasets.
    # For > 3 datasets it pays off to launch more workers so that
    # folds of different datasets are processed in parallel.
    # CPU workers: we leave ~75% of the cores for the CPU-based
    # classifiers (RF, DT, SVM, KNN, LogReg, MLP); but for 1 dataset
    # with 2 classifiers, 4 CPU workers are enough.
    if n_gpus == 0:
        # No-GPU mode: TabPFN will compute on CPU (very slow, but works)
        n_gpu_workers = 1
        n_cpu_workers = max(1, n_cpu_cores - 2)
        print(f">>> NO GPU MODE: n_cpu_workers={n_cpu_workers}, "
              f"GPU-flagged classifiers will run on CPU", flush=True)
    elif n_ds <= 3:
        # Small set of datasets (e.g. diabetes only) -> few workers
        n_gpu_workers = int(MODE_CFG['n_gpu_workers']) if MODE_CFG.get('n_gpu_workers') is not None else max(2, n_gpus)
        n_cpu_workers = int(MODE_CFG['n_cpu_workers']) if MODE_CFG.get('n_cpu_workers') is not None else max(2, min(8, n_cpu_cores // 2))
        print(f">>> SMALL-BENCHMARK MODE (n_ds={n_ds}): "
              f"n_gpu_workers={n_gpu_workers}, n_cpu_workers={n_cpu_workers}",
              flush=True)
    else:
        # Full benchmark - more workers for parallelism
        # GPU: ~3 workers per GPU (TabPFN model ~1.5GB, H200 has 141 GB)
        n_gpu_workers = max(2, 3 * n_gpus)
        # CPU: ~75% of the cores, max 32
        n_cpu_workers = max(4, min(32, int(0.75 * n_cpu_cores)))
        print(f">>> FULL-BENCHMARK MODE (n_ds={n_ds}): "
              f"n_gpu_workers={n_gpu_workers}, n_cpu_workers={n_cpu_workers}",
              flush=True)

    # -----------------------------------------------------------------
    # SUBPROCESS SPLIT ONLY FOR THE FULL BENCHMARK ON 2+ GPUs
    # For 1-3 datasets we launch a single process (single mode).
    # -----------------------------------------------------------------
    if _GPU_WORKER_ID is not None:
        # Worker mode (launched via subprocess with LDR_GPU_ID=N)
        gpu_id   = int(_GPU_WORKER_ID)
        res_dir  = f'./results_LDR/gpu{gpu_id}'
        os.makedirs(res_dir, exist_ok=True)

        _start_heartbeat(interval=120)

        _partitions = _load_balanced_split(datasets, n_splits=2)
        datasets = _partitions[gpu_id] if gpu_id < len(_partitions) else []
        label    = f"GPU-{gpu_id} (load-balanced, {len(datasets)} datasets)"

        n_folds   = int(MODE_CFG['n_folds'])
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

        t_exp_start = time.time()

        parallel_run_experiment(
            datasets,
            classifiers,
            results_directory=res_dir,
            metrics=CACP_METRICS,
            n_gpu_workers=n_gpu_workers,
            n_cpu_workers=n_cpu_workers,
        )

        t_exp_end = time.time()
        t_total   = t_exp_end - t_exp_start
        _banner(f"WORKER GPU-{gpu_id} FINISHED")
        print(f"    Results:       {res_dir}", flush=True)
        print(f"    Total time:  {str(datetime.timedelta(seconds=int(t_total)))}",
              flush=True)
        print(f"    Time/fold:    {t_total/max(1,total_exp):.1f} s", flush=True)

    elif n_gpus >= 2 and n_ds > 3:
        # Full benchmark + 2 GPUs -> launch 2 workers (one per GPU)
        print(f">>> MULTI-GPU MODE: {n_gpus} GPUs detected and "
              f"{n_ds} datasets to process. Launching 2 workers.",
              flush=True)

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
            p0 = subprocess.Popen(
                [sys.executable, __file__],
                env=_make_env(0), stdout=f0, stderr=subprocess.STDOUT)
            p1 = subprocess.Popen(
                [sys.executable, __file__],
                env=_make_env(1), stdout=f1, stderr=subprocess.STDOUT)

            print(f"  PID worker-0: {p0.pid}", flush=True)
            print(f"  PID worker-1: {p1.pid}", flush=True)
            print(f"", flush=True)
            print(f"  Progress monitoring:", flush=True)
            print(f"    tail -f {log_gpu0}", flush=True)
            print(f"    tail -f {log_gpu1}", flush=True)
            print(f"    watch -n 10 nvidia-smi", flush=True)
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
            print(f"\n>>> DONE. Final results: {res_dir}/merged/",
                  flush=True)
        else:
            print(f"[ERROR] Worker ended with an error. "
                  f"Check: {log_gpu0}, {log_gpu1}", flush=True)

    else:
        # Single mode (1 GPU or CPU, or 2+ GPUs but few datasets)
        if n_gpus == 0:
            print(f">>> SINGLE-CPU MODE (no GPU detected).", flush=True)
        elif n_gpus == 1:
            print(f">>> SINGLE-GPU MODE (1 GPU detected).", flush=True)
        else:
            print(f">>> SINGLE-PROCESS MODE ({n_gpus} GPUs, but "
                  f"only {n_ds} datasets -- no subprocess split).",
                  flush=True)

        res_dir = './results_LDR/single'
        os.makedirs(res_dir, exist_ok=True)

        print(f"\n>>> START EXPERIMENT rule_gen v4.9", flush=True)
        print(f"    Datasets:       {len(datasets)}", flush=True)
        print(f"    Classifiers:    {len(classifiers)}", flush=True)
        print(f"    n_gpu_workers:  {n_gpu_workers}", flush=True)
        print(f"    n_cpu_workers:  {n_cpu_workers}", flush=True)
        print(f"    Metrics:        {[m[0] for m in CACP_METRICS]}", flush=True)
        print(f"    Results ->      {res_dir}\n", flush=True)

        t_exp_start = time.time()

        parallel_run_experiment(
            datasets,
            classifiers,
            results_directory=res_dir,
            metrics=CACP_METRICS,
            n_gpu_workers=n_gpu_workers,
            n_cpu_workers=n_cpu_workers,
        )

        t_exp_end = time.time()
        t_total   = t_exp_end - t_exp_start
        print(f"\n>>> EXPERIMENT FINISHED in "
              f"{str(datetime.timedelta(seconds=int(t_total)))}",
              flush=True)
        print(f">>> Results: {res_dir}", flush=True)
