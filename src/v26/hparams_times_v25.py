# hparams_times_v25.py
import glob, os, re, sys, collections, torch, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'results_LDR/v25_publication'
rows = []
for f in glob.glob(os.path.join(R, 'state', '*', 'seed*', '*', 'CBM*', 'fold*', 'restart_0.pt')):
    p = f.split(os.sep); seed, ds, alg, fold = p[-5][4:], p[-4], p[-3], p[-2][4:]
    meta = torch.load(f, map_location='cpu', weights_only=False)['metadata']
    bp = meta['best_params']
    rows.append(dict(seed=int(seed), dataset=ds, algorithm=alg, fold=int(fold), **bp))
H = pd.DataFrame(rows)
H.to_csv('hparams_v25.csv', index=False)
print(H.groupby('algorithm')[['num_concepts', 'rule_budget', 'concept_dropout_p']].describe().T)
print('share of folds with concept_dropout_p == 0:', (H.concept_dropout_p == 0).mean())
# times from worker logs
T = collections.defaultdict(dict); cur = None
for fn in sorted(glob.glob(os.path.join(R, 'logs', 'gpu-*.log'))):
    for line in open(fn, errors='ignore'):
        m = re.search(r'\[V25-CLAIM\] \S+: (s\d+:[^:]+:[^:]+:f\d+)', line)
        if m: cur = m.group(1); continue
        if cur is None: continue
        m = re.search(r'\[CBM\] HPO done: ([\d.]+)s', line)
        if m: T[cur]['hpo_s'] = float(m.group(1))
        m = re.search(r'\[CBM\] Final training: ([\d.]+)s', line)
        if m: T[cur]['final_s'] = float(m.group(1))
        m = re.search(r'Active rules after pruning: (\d+)', line)
        if m: T[cur]['kstar'] = int(m.group(1))
        m = re.search(r'changed positions: (\d+)/(\d+)', line)
        if m: T[cur]['hedge_changed'], T[cur]['hedge_total'] = int(m.group(1)), int(m.group(2))
TT = pd.DataFrame.from_dict(T, orient='index'); TT.index.name = 'task'
TT.to_csv('times_v25.csv')
if {'hpo_s', 'final_s'} <= set(TT.columns):
    rho = TT.final_s / (TT.hpo_s + TT.final_s)
    print('rho = t_final/t_total: median %.3f, IQR [%.3f, %.3f]' % (rho.median(), rho.quantile(.25), rho.quantile(.75)))
