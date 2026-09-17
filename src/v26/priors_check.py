# priors_check.py : realised vs. population class proportions of the v25 subsamples
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from scipy.stats import chi2
rows = []
for key, name, ver in [('bank','bank-marketing',1), ('magic','MagicTelescope',1), ('pendigits','pendigits',1)]:
    full = fetch_openml(name=name, version=ver, as_frame=True, parser='liac-arff').frame
    sub  = pd.read_csv(f'DATASETS_LDR/{key}_ldr.csv')
    yf, ys = full.iloc[:, -1].astype(str), sub.iloc[:, -1].astype(str)
    N, n = len(yf), len(ys)
    pi = yf.value_counts(normalize=True).sort_index()
    nk = ys.value_counts().reindex(pi.index, fill_value=0)
    fpc = (N - n) / (N - 1)
    sd = np.sqrt(pi * (1 - pi) / n * fpc)
    z = (nk / n - pi) / sd
    stat = float((((nk - n * pi) ** 2) / (n * pi)).sum() / fpc)
    p = float(chi2.sf(stat, len(pi) - 1))
    for c in pi.index:
        rows.append(dict(dataset=key, cls=c, N=N, n=n, pi_full=pi[c], pi_sub=nk[c] / n,
                         diff_pp=100 * (nk[c] / n - pi[c]), z=z[c], chi2_fpc=stat, p_value=p))
pd.DataFrame(rows).to_csv('priors_check.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
# imbalance ratios of all cached datasets (replaces the 'documented' values in the paper)
for f in sorted(__import__('glob').glob('DATASETS_LDR/*_ldr.csv')):
    y = pd.read_csv(f).iloc[:, -1].astype(str).value_counts()
    print(f.split('/')[-1], 'K=%d IR=%.1f min=%d' % (len(y), y.max() / y.min(), y.min()))
