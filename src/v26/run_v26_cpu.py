# run_v26_cpu.py -- R2.2: six baselines tuned with the CBM budget, on exactly the v25 folds (CPU only).
# usage: python run_v26_cpu.py <tasks.csv> <shard_index> <n_shards>     (algorithm names end with '_T')
# env  : DRIVER_DIR, V26_OUT (default ./results_v26), V26_HPO_TIMEOUT (seconds per fold, default 7200)
# Resumable at task granularity: finished tasks leave <out>/tuned6/s<seed>_<dataset>_<alg>_f<fold>.json.
import os, sys, json, time, hashlib, platform
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import pandas as pd
TASKS = sys.argv[1]
SH, NSH = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
sys.argv = ['benchmark_driver_v25_3.py', '--no-menu', '--datasets-group', '2', '--seeds', '42,43,44', '--run-id', 'v26cpu']
sys.path.insert(0, os.environ.get('DRIVER_DIR', 'src'))
import benchmark_driver_v25_3 as D                                    # same data loading, folds and metrics as v25
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tuned_baselines import Tuned, N_TRIALS
import optuna

VARIANT = 'tuned6'
OUT = os.path.join(os.path.abspath(os.environ.get('V26_OUT', 'results_v26')), VARIANT)
os.makedirs(OUT, exist_ok=True)
TIMEOUT = int(os.environ.get('V26_HPO_TIMEOUT', '7200'))
man = os.path.join(OUT, 'variant_manifest.json')
if not os.path.isfile(man):
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(dict(variant=VARIANT, description='R2.2: XGBoost, LightGBM, CatBoost, EBM, RuleFit, FIGS tuned with '
                   f'Optuna TPE ({N_TRIALS} trials, per-fold timeout {TIMEOUT}s) plus the exact v25 default as an extra candidate, '
                   'selection on inner-validation accuracy (ties -> default), inner split and criterion identical to '
                   'CBM-P1TS, refit on the full outer training fold; base settings identical to v25.',
                   driver_sha256=hashlib.sha256(open(D.__file__, 'rb').read()).hexdigest(),
                   runner_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                   tuned_baselines_sha256=hashlib.sha256(open(os.path.join(here, 'tuned_baselines.py'), 'rb').read()).hexdigest(),
                   python=platform.python_version(), optuna=optuna.__version__,
                   started=time.strftime('%Y-%m-%d %H:%M:%S')), open(man, 'w'), indent=2)

LAST = {}
def factory_for(base, seed, f):
    def fac(n_i, n_c):
        est = Tuned(base, n_c, seed, f, timeout=TIMEOUT)
        LAST['est'] = est
        return D.ProbaCachingClassifier(est)
    return fac

tasks = pd.read_csv(TASKS)
ds_map = {d.name: d for d in D.get_datasets_for_group(D._ARGS.datasets_group or '2')}
FOLDS = {}
for i, t in tasks.iterrows():
    if i % NSH != SH:
        continue
    seed, dsn, alg, f = int(t.seed), str(t.dataset), str(t.algorithm), int(t.fold)
    done = os.path.join(OUT, f's{seed}_{dsn}_{alg}_f{f}.json')
    if os.path.isfile(done):
        continue
    D._set_run_seed(seed)
    if (seed, dsn) not in FOLDS:
        FOLDS[(seed, dsn)] = list(ds_map[dsn].folds(n_folds=int(D._ARGS.n_folds), random_state=seed))
    fold = FOLDS[(seed, dsn)][f]
    LAST.clear(); t0 = time.time()
    res = D._run_one_task(dsn, fold, alg, factory_for(alg[:-2], seed, f), D.CACP_METRICS,
                          fold_seed=seed + f, task_context=None)
    res.update(Seed=seed, Dataset=dsn, Algorithm=alg, **{'CV index': f}, variant=VARIANT,
               wall_s=round(time.time() - t0, 1), hpo=getattr(LAST.get('est'), 'hpo_', None))
    target = done if res.get('Status') in ('OK', 'PARTIAL') else done.replace('.json', '.failed.json')
    with open(target + '.tmp', 'w') as fh:
        json.dump(res, fh, default=str, indent=1)
    os.replace(target + '.tmp', target)
    h = res.get('hpo') or {}
    print(f"[V26-CPU] s{seed} {dsn} {alg} f{f} status={res.get('Status')} acc={res.get('Accuracy')} "
          f"trials={h.get('n_trials')} err_trials={h.get('n_error_trials')} default={h.get('chose_default')} wall={res['wall_s']}s", flush=True)
print(f'[V26-CPU] shard {SH}/{NSH} finished', flush=True)
