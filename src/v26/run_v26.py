# run_v26.py -- v26 experiments on top of the UNCHANGED v25.3 driver: every variant changes exactly one factor.
# usage: python run_v26.py <variant> <tasks.csv> <shard_index> <n_shards>
# env  : DRIVER_DIR (directory of benchmark_driver_v25_3.py), V26_OUT (results root, default ./results_v26)
# Resumable: a finished task leaves <out>/<variant>/s<seed>_<dataset>_<alg>_f<fold>.json and is skipped on restart;
# an interrupted task resumes from its own Optuna database and restart checkpoints (native v25 resilience).
import os, sys, json, time, hashlib, traceback, platform
import pandas as pd, torch
VARIANT, TASKS = sys.argv[1], sys.argv[2]
SH, NSH = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
sys.argv = ['benchmark_driver_v25_3.py', '--no-menu', '--datasets-group', '2', '--trials', '40',
            '--epochs-hpo', '80', '--patience', '40', '--epochs-final', '250', '--restarts', '2',
            '--seeds', '42,43,44', '--run-id', 'v26']                 # identical protocol constants to v25
sys.path.insert(0, os.environ.get('DRIVER_DIR', 'src'))
import benchmark_driver_v25_3 as D                                    # importing does not start a run
import optuna

DESCRIPTION = {
    'main_pc0': 'R2.3: HPO search space without concept dropout (p_c = 0; concepts stay in [0,1] and the concept '
                'loss acts on the undropped sigmoid output). Everything else identical to v25.3.',
}
if VARIANT not in DESCRIPTION:
    raise SystemExit(f'unknown variant {VARIANT}; known: {sorted(DESCRIPTION)}')

# ---------------- variant patches (exactly one factor changed) ----------------
if VARIANT == 'main_pc0':
    _orig_sc = optuna.trial.Trial.suggest_categorical
    def _sc(self, name, choices):
        return _orig_sc(self, name, [0.0] if name == 'concept_dropout_p' else choices)
    optuna.trial.Trial.suggest_categorical = _sc

# ---------------- bookkeeping ----------------
OUT = os.path.join(os.path.abspath(os.environ.get('V26_OUT', 'results_v26')), VARIANT)
os.makedirs(OUT, exist_ok=True)
DRIVER_SHA = hashlib.sha256(open(D.__file__, 'rb').read()).hexdigest()
PH = hashlib.sha256(f'v26|{VARIANT}|{DRIVER_SHA}'.encode()).hexdigest()
man = os.path.join(OUT, 'variant_manifest.json')
if not os.path.isfile(man):
    json.dump(dict(variant=VARIANT, description=DESCRIPTION[VARIANT], protocol_hash=PH, driver=D.__file__,
                   driver_sha256=DRIVER_SHA, runner_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                   python=platform.python_version(), torch=torch.__version__, optuna=optuna.__version__,
                   started=time.strftime('%Y-%m-%d %H:%M:%S')), open(man, 'w'), indent=2)

def study_summary(task_dir):
    db = os.path.join(task_dir, 'optuna.sqlite3')
    if not os.path.isfile(db):
        return {}
    st = f'sqlite:///{db}'; out = {}
    for name in optuna.study.get_all_study_names(st):
        tr = optuna.load_study(study_name=name, storage=st).trials
        out = dict(n_trials=len(tr),
                   n_complete=sum(t.state == optuna.trial.TrialState.COMPLETE for t in tr),
                   n_pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in tr),
                   pc_values_tried=sorted({t.params.get('concept_dropout_p') for t in tr if 'concept_dropout_p' in t.params}))
    return out

tasks = pd.read_csv(TASKS)
ds_map, clf_map = D._v25_build_cache()
FOLDS = {}
for i, t in tasks.iterrows():
    if i % NSH != SH:
        continue
    seed, dsn, alg, f = int(t.seed), str(t.dataset), str(t.algorithm), int(t.fold)
    done = os.path.join(OUT, f's{seed}_{dsn}_{alg}_f{f}.json')
    if os.path.isfile(done):
        continue
    D._set_run_seed(seed)                                             # exactly as _v25_execute_task
    if (seed, dsn) not in FOLDS:
        FOLDS[(seed, dsn)] = list(ds_map[dsn].folds(n_folds=int(D._ARGS.n_folds), random_state=seed))
    fold = FOLDS[(seed, dsn)][f]
    tdir = os.path.join(OUT, 'state', f'seed{seed}', dsn, alg, f'fold{f}')
    os.makedirs(tdir, exist_ok=True)
    ctx = {'protocol_hash': PH, 'seed': seed, 'dataset': dsn, 'algorithm': alg, 'fold': f, 'task_dir': tdir,
           'epoch_checkpoint_interval': int(D._ARGS.epoch_checkpoint_interval)}
    t0 = time.time()
    try:
        res = D._run_one_task(dsn, fold, alg, clf_map[alg], D.CACP_METRICS, fold_seed=seed + f, task_context=ctx)
    except Exception as e:                                            # should not happen: _run_one_task catches
        res = {'Status': 'FAILED', 'Error': repr(e)[:500]}; traceback.print_exc()
    res.update(Seed=seed, Dataset=dsn, Algorithm=alg, **{'CV index': f}, variant=VARIANT,
               wall_s=round(time.time() - t0, 1), gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    try:
        res['best_params'] = torch.load(os.path.join(tdir, 'restart_0.pt'), map_location='cpu',
                                        weights_only=False)['metadata']['best_params']
    except Exception:
        res['best_params'] = None
    try:
        res['study'] = study_summary(tdir)
    except Exception as e:
        res['study'] = {'error': repr(e)[:200]}
    target = done if res.get('Status') in ('OK', 'PARTIAL') else done.replace('.json', '.failed.json')
    with open(target + '.tmp', 'w') as fh:
        json.dump(res, fh, default=str, indent=1)
    os.replace(target + '.tmp', target)
    print(f"[V26] {VARIANT} s{seed} {dsn} {alg} f{f} status={res.get('Status')} acc={res.get('Accuracy')} "
          f"auc={res.get('AUC_ROC')} wall={res['wall_s']}s", flush=True)
print(f'[V26] shard {SH}/{NSH} finished', flush=True)
