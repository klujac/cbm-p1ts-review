# replay_v25.py  --  exact reconstruction of archived CBM/CBM_KE models, NO training.
# usage: python replay_v25.py <run_root> <shard_index> <n_shards>
# optional env: DRIVER_DIR (dir of benchmark_driver_v25_3.py), REPLAY_DATASETS=a,b  REPLAY_SEEDS=42  REPLAY_MAXTASKS=2
# v2 (2026-09-10): the per-task optuna.sqlite3 files are absent in the v25_1_publication archive, so the finished
#     study is reconstructed IN MEMORY from 'best_params' stored in restart_0.pt metadata (the exact values used
#     for final training). Nothing is written to the archive; training remains blocked.
# v3 (2026-09-11): D7 uses the KD teacher exactly as in training (TabPFN-3 fitted on the inner-fit split);
#     R2 and purity are NA for constant targets / insensitive concepts; cancellation index kappa and signed
#     margin terms added (D3); the post-hoc "don't care" hedge search (former D5) removed from the plan.
# v4 (2026-09-11): the temporary task copy inherits the read-only mode of the locked archive, so it is made
#     writable (the driver may create an empty optuna.sqlite3 there; it is deleted with the copy).
# v5 (2026-09-11): the v4 no-op replacement of RDBStorage was removed: optuna.storages.get_storage() calls
#     isinstance(storage, RDBStorage), which fails when RDBStorage is not a class.
# v6 (2026-09-11): integrated gradients with 128 midpoint steps (completeness error reported as max and median);
#     histograms of predicted and true classes stored for the collapse analysis (R4.4).
import os, sys, json, glob, time, traceback, tempfile, shutil, numpy as np, pandas as pd, torch
ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'results_LDR/v25_publication')
SH, NSH = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
sys.argv = ['benchmark_driver_v25_3.py', '--no-menu', '--datasets-group', '2', '--trials', '40',
            '--epochs-hpo', '80', '--patience', '40', '--epochs-final', '250', '--restarts', '2',
            '--seeds', '42,43,44', '--run-id', 'v25_1_publication']    # must match the v25 run
sys.path.insert(0, os.environ.get('DRIVER_DIR', 'src'))
import benchmark_driver_v25_3 as D                                    # importing does not start a run
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def _no_training(*a, **k):                                            # replay must never train
    raise RuntimeError('replay attempted training: archived restart/metadata mismatch')
D._train_cbm_v4 = _no_training
import optuna
_REAL_CREATE_STUDY = optuna.create_study
_ARCH = {}                                                            # best params of the task being replayed
def _study_from_archive(*args, **kwargs):                             # replaces the missing optuna.sqlite3
    st = _REAL_CREATE_STUDY(direction='maximize')                     # in-memory study, nothing on disk
    bp = dict(_ARCH['bp']); dist = {k: optuna.distributions.CategoricalDistribution([v]) for k, v in bp.items()}
    for _ in range(int(D.CBM_N_TRIALS)):                              # 40 COMPLETE trials => no HPO is launched
        st.add_trial(optuna.trial.create_trial(params=bp, distributions=dist, value=float(_ARCH['val'])))
    return st
optuna.create_study = _study_from_archive
PH = json.load(open(os.path.join(ROOT, 'protocol.json')))['protocol_hash']
_ref = os.path.join(ROOT, 'comparison.csv')
REF = pd.read_csv(_ref if os.path.isfile(_ref) else 'results/comparison.csv')
ds_map, clf_map = D._v25_build_cache()
dev, EPS = D.device, 1e-7

def auc_of(y, P):
    try:
        return roc_auc_score(y, P[:, 1]) if P.shape[1] == 2 else \
               roc_auc_score(y, P, multi_class='ovr', average='macro')
    except Exception:
        return float('nan')

def logits_from_c(m, c):                     # identical to CBMP1TSModelV4.forward after g_theta
    mem = (c[:, None, :] * m.rule_matrix[None] + (1 - c[:, None, :]) * (1 - m.rule_matrix[None])).clamp(min=EPS)
    g = (mem ** m.hedge_exponents[None]).prod(2) * m._rule_mask_f[None]
    return g @ m.rule_weights + m.class_bias + m.concept_to_class(c), g

def _med(x):                                 # median of a (possibly empty) tensor
    return float(x.median()) if x.numel() else float('nan')

def replay_one(seed, dsn, alg, f, fold, task, tmp):
    fs = seed + f; torch.manual_seed(fs); np.random.seed(fs % 2**31)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(fs)
    clf = clf_map[alg](fold.x_train.shape[1], len(fold.labels)); cbm = clf.clf
    cbm._resilience_context = dict(task, task_dir=tmp, epoch_checkpoint_interval=10)
    cbm._raw_input = True; cbm._raw_cat_idx, cbm._raw_num_idx = fold.cat_idx, fold.num_idx
    cbm._raw_feature_names = fold.raw_feature_names
    _saved = torch.load(os.path.join(tmp, 'restart_0.pt'), map_location='cpu', weights_only=False)
    _ARCH.update(bp=_saved['metadata']['best_params'], val=_saved.get('val_acc', 1.0))
    clf.fit(fold.x_train_raw, fold.y_train)          # in-memory study from metadata + archived restarts (no training)
    # ---- (0) reproduction check against comparison.csv
    yte = fold.y_test; P = cbm.predict_proba(fold.x_test_raw); yhat = P.argmax(1)
    ref = REF[(REF.Seed == seed) & (REF.Dataset == dsn) & (REF.Algorithm == alg) & (REF['CV index'] == f)].iloc[0]
    r = dict(seed=seed, dataset=dsn, algorithm=alg, fold=f,
             dev_acc=abs(accuracy_score(yte, yhat) - ref.Accuracy), dev_auc=abs(auc_of(yte, P) - ref.AUC_ROC))
    meta = torch.load(os.path.join(tmp, 'restart_0.pt'), map_location='cpu', weights_only=False)['metadata']
    bp = meta['best_params']; pc = float(bp.get('concept_dropout_p', 0.0))
    m = cbm.model_.eval(); M = m.num_concepts
    Xall = cbm._transform_inputs(fold.x_train_raw)
    itr, iva = train_test_split(np.arange(len(fold.y_train)), test_size=0.2, stratify=fold.y_train, random_state=seed)
    Xte = cbm._transform_inputs(fold.x_test_raw)
    Cpre = D._apply_fuzzy_concepts_test(Xte, cbm._ramp_par, cbm._idx_top)[:, :M]
    xt = torch.tensor(Xte, dtype=torch.float32, device=dev)
    with torch.no_grad():
        logit, c, g = m(xt, return_details=True)
    cn = c.cpu().numpy()
    # ---- D1 fidelity
    ss = ((Cpre - Cpre.mean(0)) ** 2).sum(0)
    r2 = np.where(ss > 1e-12, 1 - ((cn - Cpre) ** 2).sum(0) / np.maximum(ss, 1e-12), np.nan)   # NA for a constant target
    beta = [np.polyfit(Cpre[:, j], cn[:, j], 1)[0] if Cpre[:, j].std() > 1e-9 else np.nan for j in range(M)]
    r.update(M=M, pc=pc, kd_alpha=bp.get('kd_alpha_hard', np.nan), kd_T=bp.get('kd_temperature', np.nan),
             fid_mse=float(((cn - Cpre) ** 2).mean()), fid_r2_med=float(np.nanmedian(r2)) if np.isfinite(r2).any() else float('nan'),
             n_const_targets=int((ss <= 1e-12).sum()), slope_med=float(np.nanmedian(beta)) if np.isfinite(beta).any() else float('nan'))
    # ---- D2 purity (Jacobian at x, integrated gradients from the inner-fit median)
    src = torch.as_tensor(cbm._idx_top[:M], device=dev); ar = torch.arange(M, device=dev)
    xs = xt.clone().requires_grad_(True); cs = m.concept_predictor(xs)
    Jab = torch.stack([torch.autograd.grad(cs[:, j].sum(), xs, retain_graph=True)[0].abs().mean(0) for j in range(M)])
    x0 = torch.tensor(np.median(Xall[itr], 0), dtype=torch.float32, device=dev)
    IG = torch.zeros(xt.shape[0], M, xt.shape[1], device=dev); S = 128
    for s in range(1, S + 1):
        xi = (x0 + ((s - 0.5) / S) * (xt - x0)).requires_grad_(True); ci = m.concept_predictor(xi)   # midpoint rule
        for j in range(M):
            IG[:, j, :] += torch.autograd.grad(ci[:, j].sum(), xi, retain_graph=(j < M - 1))[0] / S
    IG *= (xt - x0)[:, None, :]
    with torch.no_grad():
        cerr = (IG.sum(2) - (m.concept_predictor(xt) - m.concept_predictor(x0[None]))).abs()
    aIG = IG.abs().sum(0)
    dJ, dI = Jab.sum(1), aIG.sum(1)
    pj = (Jab[ar, src] / dJ.clamp_min(1e-12))[dJ > 1e-12]           # NA (dropped) for a locally insensitive concept
    pg = (aIG[ar, src] / dI.clamp_min(1e-12))[dI > 1e-12]
    r.update(pur_jac_med=_med(pj), pur_ig_med=_med(pg), n_insensitive=int((dJ <= 1e-12).sum()),
             ig_compl_err_max=float(cerr.max()), ig_compl_err_med=float(cerr.median()))
    # ---- D3 attribution shares + numerical check of Theorem 1
    with torch.no_grad():
        A = g @ m.rule_weights; B = c @ m.concept_to_class.weight.T; C0 = (m.class_bias + m.concept_to_class.bias).expand_as(A)
        ix = torch.arange(len(logit), device=dev)
        kh = logit.argmax(1); L2 = logit.clone(); L2[ix, kh] = -float('inf'); q = L2.argmax(1)
        dA, dB, dC = A[ix, kh] - A[ix, q], B[ix, kh] - B[ix, q], C0[ix, kh] - C0[ix, q]
        tot = dA.abs() + dB.abs() + dC.abs(); ok = tot > 1e-12; tt = tot.clamp_min(1e-12)
        kap = 1 - (dA + dB + dC).abs() / tt                                   # cancellation index
        r.update(recon_err=float((A + B + C0 - logit).abs().max()),
                 share_rule_med=_med((dA.abs() / tt)[ok]), share_res_med=_med((dB.abs() / tt)[ok]),
                 share_bias_med=_med((dC.abs() / tt)[ok]), kappa_med=_med(kap[ok]),
                 dA_med=_med(dA), dB_med=_med(dB), dC_med=_med(dC),
                 rule_suff=float((dA > 0).float().mean()), acc_rule_only=accuracy_score(yte, A.argmax(1).cpu().numpy()))
        # ---- D4 complexity and firing sparsity
        act = torch.where(m.rule_mask)[0]; ga = g[:, act]
        r.update(kstar=len(act), n_ante=int(len(act) * M), eff_rules_tau01_mean=float((ga >= 0.1).sum(1).float().mean()),
                 eff_rules_tau01_p95=float((ga >= 0.1).sum(1).float().quantile(0.95)), fire_sum_max=float(ga.sum(1).max()),
                 n_dilation=int((m.hedge_exponents[act] == 0.5).sum()))
        # ---- D6 share of training-time concept values above one under inverted dropout (v25)
        cfit = m.concept_predictor(torch.tensor(Xall[itr], dtype=torch.float32, device=dev)).cpu().numpy()
        r['frac_gt1'] = (1 - pc) * float((cfit > 1 - pc).mean()) if pc > 0 else 0.0
        # ---- D8 oracle concept substitution
        lo, _ = logits_from_c(m, torch.tensor(Cpre, dtype=torch.float32, device=dev))
        r['d_oracle_acc'] = accuracy_score(yte, lo.argmax(1).cpu().numpy()) - accuracy_score(yte, yhat)
    r.update(acc_v25=accuracy_score(yte, yhat), auc_v25=auc_of(yte, P),
             pred_hist=json.dumps(np.bincount(yhat, minlength=P.shape[1]).tolist()),
             true_hist=json.dumps(np.bincount(np.asarray(yte, dtype=int), minlength=P.shape[1]).tolist()))
    # ---- D7 agreement with the KD teacher: TabPFN-3 fitted on the inner-fit split, exactly as in CBM_KE training
    tch = D._make_tabpfn_v25(dev); tch.fit(Xall[itr], fold.y_train[itr]); Pt = np.asarray(tch.predict_proba(Xte)); del tch
    if Pt.shape == P.shape:
        r.update(agree_teacher=float((Pt.argmax(1) == yhat).mean()), acc_teacher=accuracy_score(yte, Pt.argmax(1)),
                 kl_teacher=float(np.mean(np.sum(Pt * (np.log(Pt + 1e-12) - np.log(P + 1e-12)), 1))))
    else:
        r.update(agree_teacher=float('nan'), acc_teacher=float('nan'), kl_teacher=float('nan'))
    return r

FOLDS = {}
_DS = [d for d in os.environ.get('REPLAY_DATASETS', '').split(',') if d]          # optional filters (smoke tests)
_SEEDS = [int(x) for x in os.environ.get('REPLAY_SEEDS', '42,43,44').split(',') if x]
_MAXT = int(os.environ.get('REPLAY_MAXTASKS', '0'))
tasks = [(s, d, a, f) for s in _SEEDS for d in sorted(ds_map) if (not _DS or d in _DS)
         for f in range(10) for a in ('CBM', 'CBM_KE')]
if _MAXT: tasks = tasks[:_MAXT]
OUT = f'replay_v25_diag_{SH}of{NSH}.csv'; out = []
for t_idx, (seed, dsn, alg, f) in enumerate(tasks):
    if t_idx % NSH != SH: continue
    task = dict(protocol_hash=PH, seed=seed, dataset=dsn, algorithm=alg, fold=f)
    tdir = D._v25_task_dir(ROOT, task)
    if not glob.glob(os.path.join(tdir, 'restart_*.pt')):
        out.append(dict(seed=seed, dataset=dsn, algorithm=alg, fold=f, error='no archived restarts', sec=0.0)); continue
    t0 = time.time(); tmp = tempfile.mkdtemp(prefix='replay_')
    try:
        shutil.copytree(tdir, tmp, dirs_exist_ok=True)                  # the archive itself stays untouched
        os.chmod(tmp, 0o700)                                            # the copy inherits read-only mode from the archive
        for _f in os.listdir(tmp): os.chmod(os.path.join(tmp, _f), 0o600)
        D._set_run_seed(seed)
        if (seed, dsn) not in FOLDS:
            FOLDS[(seed, dsn)] = list(ds_map[dsn].folds(n_folds=10, random_state=seed))
        r = replay_one(seed, dsn, alg, f, FOLDS[(seed, dsn)][f], task, tmp)
    except Exception as e:
        r = dict(seed=seed, dataset=dsn, algorithm=alg, fold=f, error=repr(e)[:300]); traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    r['sec'] = round(time.time() - t0, 1); out.append(r)
    print(t_idx, dsn, alg, f, 'dev_acc=%s' % r.get('dev_acc'), 'sec=%.1f' % r['sec'], flush=True)
    if len(out) % 10 == 0: pd.DataFrame(out).to_csv(OUT, index=False)
pd.DataFrame(out).to_csv(OUT, index=False)
