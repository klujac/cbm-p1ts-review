# run_stability.py -- R2.4: subsampling stability. For bank, magic and pendigits we draw independent subsamples of
# 10000 rows from the FULL OpenML data: 3 stratified and 3 non-stratified (the v25 scheme), then evaluate with
# stratified 5-fold CV. This separates the effect of stratification from the variance of the draw itself.
# usage: python run_stability.py <alg> <tasks.csv> <shard> <n_shards>   env: DRIVER_DIR, V26_OUT, LARGE_DIR
import os, sys, json, time, hashlib, traceback, platform
ALG, TASKS = sys.argv[1], sys.argv[2]
SH, NSH = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
N_TRIALS, N_FOLDS, N_SUB = 40, 5, 10000
sys.argv = ['benchmark_driver_v25_3.py', '--no-menu', '--datasets-group', '2', '--trials', str(N_TRIALS),
            '--epochs-hpo', '80', '--patience', '40', '--epochs-final', '250', '--restarts', '2',
            '--seeds', '42', '--run-id', 'v26stab']
sys.path.insert(0, os.environ.get('DRIVER_DIR', 'src'))
import numpy as np, pandas as pd, torch
import benchmark_driver_v25_3 as D
from sklearn.model_selection import train_test_split

SRC = {'bank': 'bank_full', 'magic': 'magic_full', 'pendigits': None}   # pendigits: full file = DATASETS_LDR copy
LARGE = os.environ.get('LARGE_DIR', 'DATASETS_LARGE')

def full_frame(dsn):
    if SRC.get(dsn):
        return pd.read_csv(f'{LARGE}/{SRC[dsn]}_ldr.csv', low_memory=False)
    return pd.read_csv(f'DATASETS_LDR/{dsn}_ldr.csv', low_memory=False)   # pendigits: 10992 rows, cached copy

def subsample(dsn, mode, sub_seed):
    df = full_frame(dsn)
    if len(df) <= N_SUB:
        return df.reset_index(drop=True)
    if mode == 'strat':
        idx, _ = train_test_split(np.arange(len(df)), train_size=N_SUB,
                                  stratify=df.iloc[:, -1].astype(str), random_state=sub_seed)
        return df.iloc[np.sort(idx)].reset_index(drop=True)
    return df.sample(n=N_SUB, random_state=sub_seed).reset_index(drop=True)   # v25 scheme

OUT = os.path.join(os.path.abspath(os.environ.get('V26_OUT', 'results_v26')), 'stability')
os.makedirs(OUT, exist_ok=True)
DSHA = hashlib.sha256(open(D.__file__, 'rb').read()).hexdigest()
PH = hashlib.sha256(('v26stab|' + DSHA).encode()).hexdigest()
man = os.path.join(OUT, 'variant_manifest.json')
if not os.path.isfile(man):
    json.dump(dict(variant='stability', protocol_hash=PH, n_sub=N_SUB, n_folds=N_FOLDS, n_trials=N_TRIALS,
                   description='R2.4: 3 stratified and 3 non-stratified subsamples of 10000 rows per dataset '
                               '(bank, magic, pendigits), stratified 5-fold CV, outer seed 42.',
                   driver_sha256=DSHA, runner_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                   python=platform.python_version(), torch=torch.__version__,
                   started=time.strftime('%Y-%m-%d %H:%M:%S')), open(man, 'w'), indent=2)

DS = {}
for i, t in pd.read_csv(TASKS).iterrows():
    if i % NSH != SH:
        continue
    dsn, mode, ss, f = str(t['dataset']), str(t['mode']), int(t['sub_seed']), int(t['fold'])
    tag = f'{dsn}_{mode}{ss}'
    done = os.path.join(OUT, f'{tag}_{ALG}_f{f}.json')
    if os.path.isfile(done):
        continue
    D._set_run_seed(42)
    if tag not in DS:
        df = subsample(dsn, mode, ss)
        ds = D.LocalDataset(tag, df.iloc[:, :-1], df.iloc[:, -1])
        DS[tag] = list(ds.folds(n_folds=N_FOLDS, random_state=42))
    fold = DS[tag][f]
    tdir = os.path.join(OUT, 'state', tag, ALG, f'fold{f}'); os.makedirs(tdir, exist_ok=True)
    ctx = {'protocol_hash': PH, 'seed': 42, 'dataset': tag, 'algorithm': ALG, 'fold': f, 'task_dir': tdir,
           'epoch_checkpoint_interval': int(D._ARGS.epoch_checkpoint_interval)}
    t0 = time.time()
    try:
        res = D._run_one_task(tag, fold, ALG, dict(D.build_classifiers())[ALG], D.CACP_METRICS,
                              fold_seed=42 + f, task_context=(ctx if ALG.startswith('CBM') else None))
    except Exception as e:
        res = {'Status': 'FAILED', 'Error': repr(e)[:400]}; traceback.print_exc()
    res.update(Dataset=dsn, mode=mode, sub_seed=ss, Algorithm=ALG, **{'CV index': f}, variant='stability',
               wall_s=round(time.time() - t0, 1))
    target = done if res.get('Status') in ('OK', 'PARTIAL') else done.replace('.json', '.failed.json')
    with open(target + '.tmp', 'w') as fh:
        json.dump(res, fh, default=str, indent=1)
    os.replace(target + '.tmp', target)
    print(f"[STAB] {tag} {ALG} f{f} {res.get('Status')} acc={res.get('Accuracy')} auc={res.get('AUC_ROC')} "
          f"wall={res['wall_s']}s", flush=True)
print(f'[STAB] {ALG} shard {SH}/{NSH} finished', flush=True)
