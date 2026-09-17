# run_v26_large.py -- R1.1: large-scale track on top of the UNCHANGED v25.3 driver.
# Datasets are read from ./DATASETS_LARGE (no subsampling); protocol: stratified 5-fold, seed 42.
# Documented deviation (compute budget): 15 HPO trials instead of 40 and 5 folds instead of 10. Everything else
# (preprocessing, concepts, restarts, pruning, hedges, calibration, metrics) is identical to the main benchmark.
# usage: python run_v26_large.py <alg> <tasks.csv> <shard_index> <n_shards>
import os, sys, json, time, hashlib, traceback, platform
os.environ.setdefault('LDRV25_TABPFN_N_MAX', '200000')
os.environ.setdefault('LDRV25_TABPFN_D_MAX', '500')
ALG, TASKS = sys.argv[1], sys.argv[2]
SH, NSH = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
N_TRIALS, N_FOLDS = 15, 5
sys.argv = ['benchmark_driver_v25_3.py', '--no-menu', '--datasets-group', '2', '--trials', str(N_TRIALS),
            '--epochs-hpo', '80', '--patience', '40', '--epochs-final', '250', '--restarts', '2',
            '--seeds', '42', '--run-id', 'v26large']
sys.path.insert(0, os.environ.get('DRIVER_DIR', 'src'))
import pandas as pd, torch
import benchmark_driver_v25_3 as D

class Large:                                   # same interface as the driver's LDRDataset
    def __init__(self, key):
        df = pd.read_csv(f'DATASETS_LARGE/{key}_ldr.csv', low_memory=False)
        self.ds = D.LocalDataset(key, df.iloc[:, :-1], df.iloc[:, -1])
    def folds(self):
        return self.ds.folds(n_folds=N_FOLDS, random_state=42)

OUT = os.path.join(os.path.abspath(os.environ.get('V26_OUT', 'results_v26')), 'large')
os.makedirs(OUT, exist_ok=True)
DSHA = hashlib.sha256(open(D.__file__, 'rb').read()).hexdigest()
PH = hashlib.sha256(('v26large|' + DSHA).encode()).hexdigest()
man = os.path.join(OUT, 'variant_manifest.json')
if not os.path.isfile(man):
    json.dump(dict(variant='large', protocol_hash=PH, n_trials=N_TRIALS, n_folds=N_FOLDS, seed=42,
                   description='R1.1: six large datasets (19k-130k rows), stratified 5-fold, seed 42, no '
                               'subsampling. Documented deviation for compute: 15 HPO trials (vs 40) and 5 folds '
                               '(vs 10); TabPFN row limit raised to 200000. All other protocol elements identical.',
                   driver_sha256=DSHA, runner_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                   python=platform.python_version(), torch=torch.__version__,
                   started=time.strftime('%Y-%m-%d %H:%M:%S')), open(man, 'w'), indent=2)

tasks = pd.read_csv(TASKS); DS = {}
for i, t in tasks.iterrows():
    if i % NSH != SH:
        continue
    dsn, f = str(t.dataset), int(t.fold)
    done = os.path.join(OUT, f's42_{dsn}_{ALG}_f{f}.json')
    if os.path.isfile(done):
        continue
    D._set_run_seed(42)
    if dsn not in DS:
        DS[dsn] = list(Large(dsn).folds())
    fold = DS[dsn][f]
    tdir = os.path.join(OUT, 'state', dsn, ALG, f'fold{f}'); os.makedirs(tdir, exist_ok=True)
    ctx = {'protocol_hash': PH, 'seed': 42, 'dataset': dsn, 'algorithm': ALG, 'fold': f, 'task_dir': tdir,
           'epoch_checkpoint_interval': int(D._ARGS.epoch_checkpoint_interval)}
    t0 = time.time()
    try:
        facs = dict(D.build_classifiers())
        res = D._run_one_task(dsn, fold, ALG, facs[ALG], D.CACP_METRICS, fold_seed=42 + f,
                              task_context=(ctx if ALG.startswith('CBM') else None))
    except Exception as e:
        res = {'Status': 'FAILED', 'Error': repr(e)[:500]}; traceback.print_exc()
    res.update(Seed=42, Dataset=dsn, Algorithm=ALG, **{'CV index': f}, variant='large',
               n_train=int(len(fold.y_train)), n_test=int(len(fold.y_test)),
               wall_s=round(time.time() - t0, 1), gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    target = done if res.get('Status') in ('OK', 'PARTIAL') else done.replace('.json', '.failed.json')
    with open(target + '.tmp', 'w') as fh:
        json.dump(res, fh, default=str, indent=1)
    os.replace(target + '.tmp', target)
    print(f"[V26-L] {dsn} {ALG} f{f} status={res.get('Status')} acc={res.get('Accuracy')} "
          f"auc={res.get('AUC_ROC')} n_train={res['n_train']} wall={res['wall_s']}s", flush=True)
print(f'[V26-L] {ALG} shard {SH}/{NSH} finished', flush=True)
