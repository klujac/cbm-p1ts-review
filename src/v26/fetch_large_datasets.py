# fetch_large_datasets.py -- downloads the six datasets of the large-scale track (Section 4.4)
# and the full releases used by the stability experiment (Section 4.5) into ./DATASETS_LARGE.
# The label column is placed last, as expected by the driver's LocalDataset.
import os

from sklearn.datasets import fetch_openml

OUT = 'DATASETS_LARGE'
SPEC = [('adult', 'adult', 2, None), ('electricity', 'electricity', 1, None),
        ('magic_full', 'MagicTelescope', 1, None), ('bank_full', 'bank-marketing', 1, None),
        ('shuttle', 'shuttle', 1, None), ('MiniBooNE', 'MiniBooNE', 1, 'signal')]

os.makedirs(OUT, exist_ok=True)
for key, name, version, label in SPEC:
    path = f'{OUT}/{key}_ldr.csv'
    if os.path.isfile(path):
        print(f'{key:12s} already present')
        continue
    df = fetch_openml(name=name, version=version, as_frame=True, parser='liac-arff').frame.dropna(axis=0)
    if label:                                     # MiniBooNE ships the label first
        df = df[[c for c in df.columns if c != label] + [label]]
    df.to_csv(path, index=False)
    y = df.iloc[:, -1].astype(str); v = y.value_counts()
    print(f'{key:12s} n={len(df):7d} d={df.shape[1]-1:3d} K={y.nunique():2d} '
          f'IR={v.max()/v.min():8.1f} label={df.columns[-1]}')
