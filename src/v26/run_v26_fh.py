# run_v26_fh.py -- one-factor ablations with FROZEN hyper-parameters (FH) on top of the UNCHANGED v25.3 driver.
# The finished HPO study is rebuilt in memory from a fixed parameter set, so only the final training runs
# (same restarts, seeds, pruning, hedges, calibration and metrics as in the benchmark).
# usage: python run_v26_fh.py <variant> <tasks.csv> <shard_index> <n_shards>
# env  : DRIVER_DIR, OLD_RUN (v25 archive, read-only), V26_OUT (default ./results_v26)
import os, sys, json, time, hashlib, traceback, platform
import numpy as np, pandas as pd, torch
VARIANT, TASKS = sys.argv[1], sys.argv[2]
SH, NSH = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
sys.argv = ['benchmark_driver_v25_3.py', '--no-menu', '--datasets-group', '2', '--trials', '40',
            '--epochs-hpo', '80', '--patience', '40', '--epochs-final', '250', '--restarts', '2',
            '--seeds', '42,43,44', '--run-id', 'v26fh']
sys.path.insert(0, os.environ.get('DRIVER_DIR', 'src'))
import benchmark_driver_v25_3 as D
import optuna

V = {  # source of hyper-parameters: 'v25' archive or 'v26' corrected model (main_pc0 where re-tuned, else v25)
    'fh_ref_v25':  dict(src='v25', desc='FH reference: v25 hyper-parameters unchanged (determinism check)'),
    'fh_pc0':      dict(src='v25', set={'concept_dropout_p': 0.0}, desc='R2.3: v25 hyper-parameters, concept dropout p_c set to 0'),
    'fh_clamp':    dict(src='v25', patch='clamp', desc='R4.5: v25 hyper-parameters, concept dropout followed by clamp(max=1)'),
    'fh_ref_v26':  dict(src='v26', desc='FH reference of the corrected (v26) model'),
    'fh_noresid':  dict(src='v26', ablate='residual', desc='R4.2: residual S=0 and b_skip=0 (native v25 ablation)'),
    'fh_norules':  dict(src='v26', patch='norules', desc='R4.2: rule weights zeroed and frozen (linear concept head only)'),
    'fh_nokl':     dict(src='v26', patch='nokl', ke_only=True, desc='R2.5: KL term removed, same alpha_hard (a*CE + lc*MSE + ls*L1)'),
    'fh_ls0':      dict(src='v26', ablate='smoothing', desc='R4.4: label smoothing 0 (native v25 ablation)'),
    'fh_tmean':    dict(src='v26', patch='teacher_mean', ke_only=True, desc='K3: sample-independent teacher (mean soft label)'),
    'fh_thard':    dict(src='v26', patch='teacher_hard', ke_only=True, desc='K4: teacher hard labels'),
    'fh_fixedppl': dict(src='v26', patch='fixedppl', desc='R4.1: concepts fixed to PPL memberships (no learned concept predictor)'),
    'fh_lc10':     dict(src='v26', scale={'lambda_concept': 10.0}, desc='R4.1: concept-loss weight x10'),
    'fh_lc100':    dict(src='v26', scale={'lambda_concept': 100.0}, desc='R4.1: concept-loss weight x100'),
}
if VARIANT not in V:
    raise SystemExit(f'unknown variant {VARIANT}; known: {sorted(V)}')
CFG = V[VARIANT]; PATCH = CFG.get('patch')

# ---------------- frozen hyper-parameters: in-memory finished study ----------------
_REAL_CREATE_STUDY = optuna.create_study
_ARCH = {}
def _study_from_params(*a, **k):
    st = _REAL_CREATE_STUDY(direction='maximize')
    bp = dict(_ARCH['bp']); dist = {n: optuna.distributions.CategoricalDistribution([v]) for n, v in bp.items()}
    for _ in range(int(D.CBM_N_TRIALS)):
        st.add_trial(optuna.trial.create_trial(params=bp, distributions=dist, value=1.0))
    return st
optuna.create_study = _study_from_params

# ---------------- one-factor patches ----------------
_PPL = {}
class _ClampDrop(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d
    def forward(self, x):
        return torch.clamp(self.d(x), max=1.0)
class _FixedPPL(torch.nn.Module):                     # exactly _apply_fuzzy_concepts_test, in torch
    def __init__(self, ramp_par, idx):
        super().__init__()
        lo, sc, deg = [], [], []
        for j in idx:
            typ, a, b = ramp_par[int(j)]
            if typ == 'binary':
                lo.append(0.0); sc.append(1.0); deg.append(False)
            elif abs(b - a) < 1e-10:
                lo.append(0.0); sc.append(1.0); deg.append(True)
            else:
                lo.append(float(a)); sc.append(1.0 / (float(b) - float(a))); deg.append(False)
        self.register_buffer('idx', torch.as_tensor([int(j) for j in idx], dtype=torch.long))
        self.register_buffer('lo', torch.tensor(lo, dtype=torch.float32))
        self.register_buffer('sc', torch.tensor(sc, dtype=torch.float32))
        self.register_buffer('deg', torch.tensor(deg, dtype=torch.bool))
    def forward(self, x):
        c = torch.clamp((x[:, self.idx] - self.lo) * self.sc, 0.0, 1.0)
        return torch.where(self.deg, torch.full_like(c, 0.5), c)
if PATCH in ('clamp', 'norules', 'fixedppl'):
    _orig_init = D.CBMP1TSModelV4.__init__
    def _init(self, *a, **k):
        _orig_init(self, *a, **k)
        if PATCH == 'clamp':
            self._concept_drop = _ClampDrop(self._concept_drop)
        elif PATCH == 'norules':
            with torch.no_grad():
                self.rule_weights.zero_()
            self.rule_weights.requires_grad_(False)
        elif PATCH == 'fixedppl':
            self.concept_predictor = _FixedPPL(_PPL['ramp_par'], _PPL['idx_top'][:self.num_concepts])
    D.CBMP1TSModelV4.__init__ = _init
if PATCH == 'fixedppl':
    _orig_bfc = D._build_fuzzy_concepts_train
    def _bfc(*a, **k):
        out = _orig_bfc(*a, **k); _PPL['ramp_par'], _PPL['idx_top'] = out[1], out[2]
        return out
    D._build_fuzzy_concepts_train = _bfc
if PATCH == 'nokl':
    torch.nn.functional.kl_div = lambda inp, target, *a, **k: inp.sum() * 0.0      # contributes exactly 0
if PATCH in ('teacher_mean', 'teacher_hard'):
    _orig_mk = D._make_tabpfn_v25
    class _TeacherWrap:
        def __init__(self, t):
            self.t = t
        def fit(self, X, y):
            self.t.fit(X, y); return self
        def predict_proba(self, X):
            p = np.asarray(self.t.predict_proba(X), dtype=np.float64)
            if PATCH == 'teacher_mean':
                return np.tile(p.mean(0, keepdims=True), (p.shape[0], 1))
            q = np.full_like(p, 1e-4); q[np.arange(len(p)), p.argmax(1)] = 1.0
            return q / q.sum(1, keepdims=True)
        def __getattr__(self, name):
            return getattr(self.t, name)
    D._make_tabpfn_v25 = lambda *a, **k: _TeacherWrap(_orig_mk(*a, **k))

# ---------------- hyper-parameter sources ----------------
OLD_RUN = os.path.abspath(os.environ['OLD_RUN'])
PH25 = json.load(open(os.path.join(OLD_RUN, 'protocol.json')))['protocol_hash']
ROOT = os.path.abspath(os.environ.get('V26_OUT', 'results_v26'))
def bp_v25(seed, dsn, alg, f):
    tdir = D._v25_task_dir(OLD_RUN, dict(protocol_hash=PH25, seed=seed, dataset=dsn, algorithm=alg, fold=f))
    return torch.load(os.path.join(tdir, 'restart_0.pt'), map_location='cpu', weights_only=False)['metadata']['best_params']
def bp_source(seed, dsn, alg, f):
    b25 = bp_v25(seed, dsn, alg, f)
    if CFG['src'] == 'v25':
        return dict(b25), 'v25'
    if float(b25.get('concept_dropout_p', 0.0)) == 0.0:
        return dict(b25), 'v25(p_c=0)'
    j = os.path.join(ROOT, 'main_pc0', f's{seed}_{dsn}_{alg}_f{f}.json')
    if os.path.isfile(j):
        bp = json.load(open(j)).get('best_params')
        if bp:
            return dict(bp), 'v26 main_pc0'
    return None, 'v26 not yet available'

OUT = os.path.join(ROOT, VARIANT); os.makedirs(OUT, exist_ok=True)
DRIVER_SHA = hashlib.sha256(open(D.__file__, 'rb').read()).hexdigest()
PH = hashlib.sha256(f'v26fh|{VARIANT}|{DRIVER_SHA}'.encode()).hexdigest()
man = os.path.join(OUT, 'variant_manifest.json')
if not os.path.isfile(man):
    json.dump(dict(variant=VARIANT, description=CFG['desc'], config={k: v for k, v in CFG.items() if k != 'desc'},
                   protocol_hash=PH, driver_sha256=DRIVER_SHA,
                   runner_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                   python=platform.python_version(), torch=torch.__version__,
                   started=time.strftime('%Y-%m-%d %H:%M:%S')), open(man, 'w'), indent=2)

tasks = pd.read_csv(TASKS)
ds_map = {d.name: d for d in D.get_datasets_for_group(D._ARGS.datasets_group or '2')}
FOLDS = {}
for i, t in tasks.iterrows():
    if i % NSH != SH:
        continue
    seed, dsn, alg, f = int(t.seed), str(t.dataset), str(t.algorithm), int(t.fold)
    if CFG.get('ke_only') and alg != 'CBM_KE':
        continue
    done = os.path.join(OUT, f's{seed}_{dsn}_{alg}_f{f}.json')
    if os.path.isfile(done):
        continue
    bp, src = bp_source(seed, dsn, alg, f)
    if bp is None:
        print(f'[V26-FH] {VARIANT} s{seed} {dsn} {alg} f{f} SKIP ({src})', flush=True); continue
    for k2, v2 in CFG.get('set', {}).items():
        bp[k2] = v2
    for k2, v2 in CFG.get('scale', {}).items():
        bp[k2] = float(bp[k2]) * v2
    _ARCH['bp'] = bp
    D._set_run_seed(seed)
    if (seed, dsn) not in FOLDS:
        FOLDS[(seed, dsn)] = list(ds_map[dsn].folds(n_folds=int(D._ARGS.n_folds), random_state=seed))
    fold = FOLDS[(seed, dsn)][f]
    tdir = os.path.join(OUT, 'state', f'seed{seed}', dsn, alg, f'fold{f}'); os.makedirs(tdir, exist_ok=True)
    ctx = {'protocol_hash': PH, 'seed': seed, 'dataset': dsn, 'algorithm': alg, 'fold': f, 'task_dir': tdir,
           'epoch_checkpoint_interval': int(D._ARGS.epoch_checkpoint_interval)}
    mode = 'ke' if alg == 'CBM_KE' else 'standalone'
    fac = lambda n_i, n_c: D.ProbaCachingClassifier(D.CBMClassifier(mode=mode, n_trials=D.CBM_N_TRIALS, ablate=CFG.get('ablate')))
    t0 = time.time()
    try:
        res = D._run_one_task(dsn, fold, alg, fac, D.CACP_METRICS, fold_seed=seed + f, task_context=ctx)
    except Exception as e:
        res = {'Status': 'FAILED', 'Error': repr(e)[:500]}; traceback.print_exc()
    res.update(Seed=seed, Dataset=dsn, Algorithm=alg, **{'CV index': f}, variant=VARIANT, bp_source=src,
               bp_used=bp, wall_s=round(time.time() - t0, 1), gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    target = done if res.get('Status') in ('OK', 'PARTIAL') else done.replace('.json', '.failed.json')
    with open(target + '.tmp', 'w') as fh:
        json.dump(res, fh, default=str, indent=1)
    os.replace(target + '.tmp', target)
    print(f"[V26-FH] {VARIANT} s{seed} {dsn} {alg} f{f} status={res.get('Status')} acc={res.get('Accuracy')} "
          f"auc={res.get('AUC_ROC')} src={src} wall={res['wall_s']}s", flush=True)
print(f'[V26-FH] {VARIANT} shard {SH}/{NSH} finished', flush=True)
