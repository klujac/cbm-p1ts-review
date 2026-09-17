# verify_paper_numbers.py -- recomputes every aggregate number of the revised manuscript
# from the released fold-level results and prints it next to the value printed in the paper.
# No GPU, no training: reads results/v26/*.csv only. Run from the repository root.
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon

R = 'results/v26'
MET = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1_macro', 'MCC']
K = ['Seed', 'Dataset', 'Algorithm', 'CV index']
ok = []

def check(label, got, paper, tol):
    good = abs(got - paper) <= tol
    ok.append(good)
    print(f"  {'OK ' if good else 'FAIL'} {label:52s} computed {got:8.4f}   paper {paper:8.4f}")

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj

A = pd.read_csv(f'{R}/v26_comparison_23methods.csv')
ds = A.groupby(['Algorithm', 'Dataset'])[MET].mean()

print('\n== Table 4: dataset-level means (Section 4.2)')
for alg, paper_auc, paper_acc in [('TabPFN', 0.958, 0.924), ('TabICLv2', 0.957, 0.923),
                                  ('CBM_KE', 0.935, 0.904), ('CBM', 0.891, 0.863)]:
    check(f'{alg} AUC-ROC', ds.loc[alg].AUC_ROC.mean(), paper_auc, 0.0006)
    check(f'{alg} accuracy', ds.loc[alg].Accuracy.mean(), paper_acc, 0.0006)
rank = ds['AUC_ROC'].unstack(0).rank(axis=1, ascending=False).mean().sort_values()
pos = list(ds.groupby('Algorithm').AUC_ROC.mean().sort_values(ascending=False).index).index('CBM_KE') + 1
check('CBM-KE position by mean AUC-ROC', pos, 10, 0)
check('CBM-KE mean rank', rank['CBM_KE'], 12.05, 0.2)

print('\n== Tuning effect on the baselines (Section 4.2)')
for b, paper in [('XGBoost', 0.0102), ('LightGBM', 0.0081), ('RuleFit', 0.0081)]:
    x, y = ds.loc[b + '_T'].AUC_ROC, ds.loc[b].AUC_ROC.reindex(ds.loc[b + '_T'].index)
    check(f'{b}: AUC-ROC gain from tuning', x.mean() - y.mean(), paper, 0.0006)

print('\n== Table 7: one-factor ablations (Section 4.3)')
full = pd.read_csv(f'{R}/v26_comparison_all.csv')
fm = full[full.Algorithm.isin(['CBM', 'CBM_KE'])].groupby(['Dataset', 'Algorithm'])[MET].mean()
for f, alg, met, paper in [('v26_fh_noresid', 'CBM_KE', 'Accuracy', -0.029),
                           ('v26_fh_noresid', 'CBM_KE', 'AUC_ROC', -0.019),
                           ('v26_fh_norules', 'CBM_KE', 'Accuracy', -0.003),
                           ('v26_fh_nokl', 'CBM_KE', 'Accuracy', -0.085),
                           ('v26_fh_nokl', 'CBM_KE', 'AUC_ROC', -0.038),
                           ('v26_fh_nokl', 'CBM_KE', 'MCC', -0.099),
                           ('v26_fh_fixedppl', 'CBM_KE', 'Accuracy', -0.135),
                           ('v26_fh_fixedppl', 'CBM', 'Accuracy', -0.153)]:
    a = pd.read_csv(f'{R}/{f}.csv')
    a = a[a.Algorithm == alg].groupby('Dataset')[MET].mean()
    b = fm.xs(alg, level=1).reindex(a.index)
    check(f'{f.replace("v26_fh_","")}, {alg}, {met}', a[met].mean() - b[met].mean(), paper, 0.0015)

print('\n== Distillation isolated: KL term removed (Section 4.3c)')
nokl = pd.read_csv(f'{R}/v26_fh_nokl.csv').groupby('Dataset')[MET].mean()
ke = fm.xs('CBM_KE', level=1).reindex(nokl.index)
praw = [wilcoxon(ke[m], nokl[m]).pvalue for m in MET]
for m, p in zip(MET, holm(praw)):
    better = int((ke[m] > nokl[m]).sum())
    print(f"  {'OK ' if p < 0.0001 and better >= 19 else 'FAIL'} {m:52s} "
          f"better on {better}/20, Holm p = {p:.2e}   paper p < 1e-4")
    ok.append(p < 0.0001 and better >= 19)

print('\n== Proposition 10: collapse threshold, eps_ls = 0.05 (Section 4.3d)')
counts = {'page-blocks': [4913, 329, 28, 88, 115], 'wine-quality': [20, 163, 1457, 2198, 880, 175, 5],
          'car': [1210, 384, 69, 65], 'balance-scale': [288, 49, 288], 'bank': [39922, 5289]}
for name, c in counts.items():
    c = np.array(c, float); Kc = len(c); ir = c.max() / c.min(); thr = 1 + Kc * 0.95 / 0.05
    flag = ir > thr
    expect = name in ('page-blocks', 'wine-quality')
    print(f"  {'OK ' if flag == expect else 'FAIL'} {name:20s} IR={ir:7.1f} threshold={thr:6.1f} "
          f"flagged={flag}   paper={expect}")
    ok.append(flag == expect)

print('\n== Label smoothing removed (Section 4.3d)')
ls0 = pd.read_csv(f'{R}/v26_fh_ls0.csv')
for dsn, paper in [('page-blocks', 0.727), ('wine-quality', 0.106)]:
    g = ls0[(ls0.Dataset == dsn) & (ls0.Algorithm == 'CBM')]
    check(f'{dsn}: standalone MCC with eps_ls = 0', g.MCC.mean(), paper, 0.002)

print('\n== Table 8: large-scale track (Section 4.4)')
L = pd.read_csv(f'{R}/v26_large.csv').groupby(['Dataset', 'Algorithm'])[['AUC_ROC', 'Accuracy']].mean()
for dsn, alg, met, paper in [('MiniBooNE', 'CBM', 'AUC_ROC', 0.979), ('shuttle', 'CBM', 'Accuracy', 0.340),
                             ('shuttle', 'CBM_KE', 'Accuracy', 0.081), ('shuttle', 'CBM_KE', 'AUC_ROC', 0.808),
                             ('bank_full', 'CBM_KE', 'AUC_ROC', 0.923), ('magic_full', 'CBM_KE', 'AUC_ROC', 0.939)]:
    check(f'{dsn} / {alg} / {met}', L.loc[(dsn, alg), met], paper, 0.0015)

print('\n== Section 4.5: subsampling stability')
S = pd.read_csv(f'{R}/v26_stability.csv')
for alg, dsn, mode, paper in [('CBM_KE', 'bank', 'strat', 0.0023), ('CBM_KE', 'bank', 'nonstrat', 0.0069)]:
    g = S[(S.Algorithm == alg) & (S.Dataset == dsn) & (S['mode'] == mode)]
    per = g.groupby('sub_seed').AUC_ROC.mean()
    check(f'{alg} / {dsn} / {mode}: s.d. across subsamples', per.std(ddof=0), paper, 0.0004)

print('\n== Section 4.1: tuned baselines run')
T = pd.read_csv(f'{R}/v26_tuned6.csv')
check('share of tasks in which the default configuration won', T.chose_default.mean(), 0.12, 0.005)
check('number of tuning tasks', len(T), 3600, 0)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
