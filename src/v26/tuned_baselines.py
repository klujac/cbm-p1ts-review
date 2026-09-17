# tuned_baselines.py -- R2.2: six baselines tuned with the SAME budget and criterion as CBM-P1TS
# (Optuna TPE, 40 trials, identical inner split test_size=0.2 / stratified / random_state=seed, objective = inner
# validation accuracy), then refitted on the full outer training fold (conservative w.r.t. our claims).
# Base settings identical to the v25 defaults (threads = 1, same seeds); only the searched hyper-parameters differ.
# v2: the EXACT v25 default configuration is evaluated as a 41st candidate on the same inner split; the final
#     configuration (default or best trial) is chosen on inner-validation accuracy only (ties -> default).
import inspect, time
import numpy as np, optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

N_TRIALS = 40

def space(base, t):
    f, i = t.suggest_float, t.suggest_int
    if base == 'XGBoost':
        return dict(n_estimators=i('n_estimators', 100, 1000), learning_rate=f('learning_rate', 1e-3, 0.3, log=True),
                    max_depth=i('max_depth', 2, 12), min_child_weight=f('min_child_weight', 1e-2, 20, log=True),
                    subsample=f('subsample', 0.5, 1.0), colsample_bytree=f('colsample_bytree', 0.5, 1.0),
                    reg_lambda=f('reg_lambda', 1e-3, 10, log=True), reg_alpha=f('reg_alpha', 1e-3, 10, log=True))
    if base == 'LightGBM':
        return dict(n_estimators=i('n_estimators', 100, 1000), learning_rate=f('learning_rate', 1e-3, 0.3, log=True),
                    num_leaves=i('num_leaves', 8, 256), min_child_samples=i('min_child_samples', 5, 100),
                    subsample=f('subsample', 0.5, 1.0), colsample_bytree=f('colsample_bytree', 0.5, 1.0),
                    reg_lambda=f('reg_lambda', 1e-3, 10, log=True))
    if base == 'CatBoost':
        return dict(iterations=i('iterations', 100, 1000), learning_rate=f('learning_rate', 1e-3, 0.3, log=True),
                    depth=i('depth', 4, 10), l2_leaf_reg=f('l2_leaf_reg', 1.0, 10.0, log=True),
                    bagging_temperature=f('bagging_temperature', 0.0, 1.0),
                    random_strength=f('random_strength', 1e-3, 10, log=True))
    if base == 'EBM':
        return dict(learning_rate=f('learning_rate', 1e-3, 0.1, log=True), max_leaves=i('max_leaves', 2, 3),
                    min_samples_leaf=i('min_samples_leaf', 2, 50), interactions=i('interactions', 0, 20),
                    max_bins=t.suggest_categorical('max_bins', [128, 256, 512]))
    if base == 'RuleFit':
        return dict(n_estimators=i('n_estimators', 50, 300), tree_size=i('tree_size', 2, 8),
                    max_rules=i('max_rules', 10, 100), memory_par=f('memory_par', 1e-3, 0.1, log=True))
    if base == 'FIGS':
        return dict(max_rules=i('max_rules', 5, 60),
                    max_trees=t.suggest_categorical('max_trees', [1, 2, 3, 5, 0]),       # 0 means None
                    min_impurity_decrease=f('min_impurity_decrease', 1e-5, 1e-1, log=True))
    raise ValueError(base)

def _filtered(cls, params):
    ok = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in params.items() if k in ok}, sorted(set(params) - ok)

def make(base, n_c, params, seed, final=False):
    p = dict(params)
    if base == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=seed, n_jobs=1, verbosity=0, **p), []
    if base == 'LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1, **p), []
    if base == 'CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False, **p), []
    if base == 'EBM':
        from interpret.glassbox import ExplainableBoostingClassifier
        extra = {} if final else {'outer_bags': 1}              # HPO with 1 bag (15x faster); final fit = v25 default
        return ExplainableBoostingClassifier(random_state=seed, n_jobs=1, **extra, **p), []
    if base in ('RuleFit', 'FIGS'):
        from imodels import RuleFitClassifier, FIGSClassifier
        cls = RuleFitClassifier if base == 'RuleFit' else FIGSClassifier
        if base == 'FIGS' and p.get('max_trees') == 0:
            p['max_trees'] = None
        q, dropped = _filtered(cls, dict(p, random_state=seed))
        est = cls(**q)
        return (OneVsRestClassifier(est, n_jobs=1) if n_c > 2 else est), dropped
    raise ValueError(base)

def make_default(base, n_c, seed):
    """Exact v25 constructors (build_classifiers in benchmark_driver_v25_3.py)."""
    if base == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, tree_method='hist', eval_metric='logloss', random_state=seed, n_jobs=1, verbosity=0)
    if base == 'LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, random_state=seed, n_jobs=1, verbose=-1)
    if base == 'CatBoost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False)
    if base == 'EBM':
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(random_state=seed, n_jobs=1)
    if base in ('RuleFit', 'FIGS'):
        from imodels import RuleFitClassifier, FIGSClassifier
        cls = RuleFitClassifier if base == 'RuleFit' else FIGSClassifier
        q, _ = _filtered(cls, {'random_state': seed})
        est = cls(**q)
        return OneVsRestClassifier(est, n_jobs=1) if n_c > 2 else est
    raise ValueError(base)

class Tuned(BaseEstimator, ClassifierMixin):
    def __init__(self, base, n_c, seed, fold, n_trials=N_TRIALS, timeout=7200):
        self.base, self.n_c, self.seed, self.fold, self.n_trials, self.timeout = base, n_c, seed, fold, n_trials, timeout

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        itr, iva = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=self.seed)
        dropped_all = set()
        def objective(trial):
            try:
                m, dropped = make(self.base, self.n_c, space(self.base, trial), self.seed)
                dropped_all.update(dropped)
                m.fit(X[itr], y[itr])
                return float((np.asarray(m.predict(X[iva])).ravel() == y[iva]).mean())
            except Exception as e:                              # an invalid configuration counts as a (worst) trial
                trial.set_user_attr('error', repr(e)[:300])
                return 0.0
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        st = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=1000 * self.seed + self.fold))
        t0 = time.time()
        st.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        hpo_s = time.time() - t0
        try:                                                    # 41st candidate: the exact v25 default
            md = make_default(self.base, self.n_c, self.seed); md.fit(X[itr], y[itr])
            def_acc = float((np.asarray(md.predict(X[iva])).ravel() == y[iva]).mean())
        except Exception:
            def_acc = -1.0
        chose_default = def_acc >= st.best_value
        self.model_ = (make_default(self.base, self.n_c, self.seed) if chose_default
                       else make(self.base, self.n_c, st.best_params, self.seed, final=True)[0])
        t1 = time.time(); self.model_.fit(X, y); refit_s = time.time() - t1
        self.classes_ = getattr(self.model_, 'classes_', np.unique(y))
        errs = [t.user_attrs['error'] for t in st.trials if 'error' in t.user_attrs]
        self.hpo_ = dict(n_trials=len(st.trials), best_inner_acc=st.best_value, best_params=st.best_params,
                         default_inner_acc=def_acc, chose_default=bool(chose_default),
                         hpo_s=round(hpo_s, 1), refit_s=round(refit_s, 1), timeout_hit=len(st.trials) < self.n_trials,
                         n_error_trials=len(errs), first_error=errs[0] if errs else '', dropped_params=sorted(dropped_all))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X))).ravel()
