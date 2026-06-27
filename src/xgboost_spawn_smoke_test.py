#!/usr/bin/env python3
import multiprocessing as mp
import os
import sys

def worker(index: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import numpy as np
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier

    for repeat in range(4):
        X, y = make_classification(
            n_samples=1200,
            n_features=24,
            n_informative=12,
            n_redundant=4,
            n_classes=2,
            random_state=1000 + 100 * index + repeat,
        )
        clf = XGBClassifier(
            n_estimators=80,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42 + repeat,
            n_jobs=1,
            verbosity=0,
        )
        clf.fit(X, y)
        pred = clf.predict(X[:32])
        if len(pred) != 32:
            raise RuntimeError("Niepoprawny wynik testu XGBoost.")

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(i,)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(300)
    bad = [(p.pid, p.exitcode) for p in procs if p.exitcode != 0]
    if bad:
        print(f"XGBOOST_SPAWN_TEST_FAILED: {bad}", flush=True)
        raise SystemExit(1)
    print("XGBOOST_SPAWN_TEST_OK", flush=True)
