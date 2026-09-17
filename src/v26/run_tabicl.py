# run_tabicl.py -- R1.2: TabICLv2 (ICML 2026) on exactly the v25/v26 folds. No HPO (in-context model),
# following TabArena, which evaluates tabular foundation models without tuning.
# Folds: StratifiedKFold(n_splits=10, shuffle=True, random_state=seed) on the same cached CSV files,
# i.e. identical to the driver, because the split depends only on the labels and the seed.
# usage: python run_tabicl.py <tasks.csv> <shard> <n_shards>     env: V26_OUT, DATA_DIR
import os, sys, json, time, hashlib, traceback, platform
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef)
TASKS = sys.argv[1]
SH, NSH = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
DATA = os.environ.get('DATA_DIR', 'DATASETS_LDR')
OUT = os.path.join(os.path.abspath(os.environ.get('V26_OUT', 'results_v26')), 'tabicl')
os.makedirs(OUT, exist_ok=True)
from tabicl import TabICLClassifier
import torch, sklearn, importlib.metadata

man = os.path.join(OUT, 'variant_manifest.json')
if not os.path.isfile(man):
    json.dump(dict(variant='tabicl', description='R1.2: TabICLv2 evaluated in-context (no HPO) on the same '
                   'stratified folds, seeds and metrics as the main benchmark.',
                   tabicl=importlib.metadata.version("tabicl"), torch=torch.__version__, sklearn=sklearn.__version__,
                   python=platform.python_version(),
                   runner_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                   started=time.strftime('%Y-%m-%d %H:%M:%S')), open(man, 'w'), indent=2)

def load(dsn):
    df = pd.read_csv(f'{DATA}/{dsn}_ldr.csv', low_memory=False)
    X, y = df.iloc[:, :-1], df.iloc[:, -1].astype(str).values
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat:
        X = X.copy()
        X[cat] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[cat].astype(str))
    return X.astype(float).values, y

def metrics(y, yh, P, labels):
    out = dict(Accuracy=accuracy_score(y, yh),
               Precision=precision_score(y, yh, average='macro', zero_division=0),
               Recall=recall_score(y, yh, average='macro', zero_division=0),
               F1_macro=f1_score(y, yh, average='macro', zero_division=0),
               MCC=matthews_corrcoef(y, yh))
    try:
        out['AUC_ROC'] = (roc_auc_score((y == labels[1]).astype(int), P[:, 1]) if len(labels) == 2
                          else roc_auc_score(y, P, multi_class='ovr', average='macro', labels=labels))
    except Exception:
        out['AUC_ROC'] = None
    return out

DS = {}
for i, t in pd.read_csv(TASKS).iterrows():
    if i % NSH != SH:
        continue
    seed, dsn, f = int(t.seed), str(t.dataset), int(t.fold)
    done = os.path.join(OUT, f's{seed}_{dsn}_TabICLv2_f{f}.json')
    if os.path.isfile(done):
        continue
    if dsn not in DS:
        DS[dsn] = load(dsn)
    X, y = DS[dsn]
    tr, te = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=seed).split(X, y))[f]
    t0 = time.time()
    try:
        m = TabICLClassifier(device='cuda', random_state=seed).fit(X[tr], y[tr])
        P = m.predict_proba(X[te]); yh = m.classes_[P.argmax(1)]
        res = dict(Status='OK', **metrics(y[te], yh, P, list(m.classes_)))
    except Exception as e:
        res = dict(Status='FAILED', Error=repr(e)[:400]); traceback.print_exc()
    res.update(Seed=seed, Dataset=dsn, Algorithm='TabICLv2', **{'CV index': f},
               n_train=len(tr), n_test=len(te), wall_s=round(time.time() - t0, 1))
    target = done if res['Status'] == 'OK' else done.replace('.json', '.failed.json')
    with open(target + '.tmp', 'w') as fh:
        json.dump(res, fh, default=str, indent=1)
    os.replace(target + '.tmp', target)
    print(f"[TABICL] s{seed} {dsn} f{f} {res['Status']} acc={res.get('Accuracy')} auc={res.get('AUC_ROC')} "
          f"wall={res['wall_s']}s", flush=True)
print(f'[TABICL] shard {SH}/{NSH} finished', flush=True)
