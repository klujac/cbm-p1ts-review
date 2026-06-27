"""Local, self-contained replacement for the small CACP reporting surface.

This module intentionally implements only the functions used by the v25
benchmark scripts.  It has no dependency on the external ``cacp`` package.
Raw fold-level results remain the source of truth in ``comparison.csv``.
"""
from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__version__ = "1.0.0-local"

_META_COLS = {
    "Dataset", "Algorithm", "CV index", "Seed", "Number of classes",
    "Train size", "Test size", "Train time [s]", "Prediction time [s]",
    "Status", "Error", "Protocol hash", "Attempts",
}


def _root(result_dir: os.PathLike[str] | str) -> Path:
    p = Path(result_dir)
    return p.parent if p.is_file() else p


def _read(result_dir: os.PathLike[str] | str) -> tuple[Path, pd.DataFrame]:
    root = _root(result_dir)
    csv_path = Path(result_dir) if Path(result_dir).is_file() else root / "comparison.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing comparison.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"comparison.csv is empty: {csv_path}")
    return root, df


def _metric_names(metrics: Sequence[Any] | None, df: pd.DataFrame) -> list[str]:
    requested: list[str] = []
    if metrics:
        for item in metrics:
            if isinstance(item, (tuple, list)) and item:
                requested.append(str(item[0]))
            elif isinstance(item, str):
                requested.append(item)
    names = [c for c in requested if c in df.columns]
    if names:
        return names
    return [
        c for c in df.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Status" not in df.columns:
        return df.copy()
    # PARTIAL rows are retained per metric; undefined values are already NaN.
    return df[df["Status"].isin(["OK", "PARTIAL"])].copy()


def _escape_latex(s: Any) -> str:
    text = str(s)
    for old, new in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(old, new)
    return text


def _write_tex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tex = df.to_latex(
            index=False,
            escape=True,
            float_format=lambda x: f"{x:.6g}" if pd.notna(x) else "--",
            caption=caption,
            label=label,
            longtable=True,
        )
    except TypeError:
        tex = df.to_latex(index=False, escape=True)
    path.write_text(tex, encoding="utf-8")


def process_comparison_results(result_dir: os.PathLike[str] | str,
                               metrics: Sequence[Any] | None = None,
                               *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Aggregate fold/seed results by dataset and algorithm.

    Creates:
      comparison_result.csv
      comparison_result.tex
      comparison_result_overall.csv
      comparison_result_overall.tex
    """
    root, raw = _read(result_dir)
    df = _valid_rows(raw)
    metric_names = _metric_names(metrics, df)
    if not metric_names:
        raise ValueError("No metric columns found in comparison.csv")

    group_cols = ["Dataset", "Algorithm"]
    grouped = df.groupby(group_cols, dropna=False)[metric_names].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(root / "comparison_result.csv", index=False)
    _write_tex(grouped, root / "comparison_result.tex",
               "Dataset-level benchmark results.",
               "tab:comparison-result")

    overall = df.groupby("Algorithm", dropna=False)[metric_names].agg(["mean", "std", "count"])
    overall.columns = [f"{metric}_{stat}" for metric, stat in overall.columns]
    overall = overall.reset_index()
    overall.to_csv(root / "comparison_result_overall.csv", index=False)
    _write_tex(overall, root / "comparison_result_overall.tex",
               "Overall benchmark summary.",
               "tab:comparison-overall")
    return grouped


def process_comparison_results_plots(result_dir: os.PathLike[str] | str,
                                     metrics: Sequence[Any] | None = None,
                                     *args: Any, **kwargs: Any) -> None:
    """Create distribution and average-rank plots using Matplotlib."""
    root, raw = _read(result_dir)
    df = _valid_rows(raw)
    metric_names = _metric_names(metrics, df)
    out = root / "plot"
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in metric_names:
        sub = df[["Algorithm", metric]].dropna()
        if sub.empty:
            continue
        order = (sub.groupby("Algorithm")[metric].median()
                   .sort_values(ascending=False).index.tolist())
        data = [sub.loc[sub["Algorithm"] == alg, metric].to_numpy() for alg in order]
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(order)), 5.5))
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_title(f"Distribution of {metric}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=70)
        fig.tight_layout()
        fig.savefig(out / f"{metric}_boxplot.png", dpi=180)
        fig.savefig(out / f"{metric}_boxplot.pdf")
        plt.close(fig)

        ds_alg = sub.join(df[["Dataset"]]).groupby(["Dataset", "Algorithm"])[metric].mean().unstack()
        ranks = ds_alg.rank(axis=1, ascending=False, method="average")
        avg = ranks.mean().sort_values()
        if not avg.empty:
            fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(avg)), 5.5))
            ax.bar(np.arange(len(avg)), avg.to_numpy())
            ax.set_xticks(np.arange(len(avg)), avg.index, rotation=70, ha="right")
            ax.set_ylabel("Average rank (lower is better)")
            ax.set_title(f"Average dataset rank: {metric}")
            fig.tight_layout()
            fig.savefig(out / f"{metric}_average_rank.png", dpi=180)
            fig.savefig(out / f"{metric}_average_rank.pdf")
            plt.close(fig)


def process_comparison_result_winners(result_dir: os.PathLike[str] | str,
                                      metrics: Sequence[Any] | None = None,
                                      *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Count dataset wins and compute average ranks, including ties."""
    root, raw = _read(result_dir)
    df = _valid_rows(raw)
    metric_names = _metric_names(metrics, df)
    out = root / "winner"
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        tab = df.groupby(["Dataset", "Algorithm"])[metric].mean().unstack()
        ranks = tab.rank(axis=1, ascending=False, method="average")
        wins = pd.Series(0.0, index=tab.columns)
        for _, row in tab.iterrows():
            valid = row.dropna()
            if valid.empty:
                continue
            best = valid.max()
            tied = valid.index[np.isclose(valid.to_numpy(), best, rtol=1e-12, atol=1e-12)]
            if len(tied):
                wins.loc[tied] += 1.0 / len(tied)
        summary = pd.DataFrame({
            "Algorithm": tab.columns,
            "Metric": metric,
            "Datasets_available": tab.notna().sum().reindex(tab.columns).to_numpy(),
            "Fractional_wins": wins.reindex(tab.columns).to_numpy(),
            "Average_rank": ranks.mean().reindex(tab.columns).to_numpy(),
        }).sort_values(["Average_rank", "Fractional_wins"], ascending=[True, False])
        summary.to_csv(out / f"{metric}_winners.csv", index=False)
        _write_tex(summary, out / f"{metric}_winners.tex",
                   f"Wins and average ranks for {metric}.",
                   f"tab:winners-{metric.lower().replace('_', '-')}")
        all_rows.extend(summary.to_dict("records"))

    combined = pd.DataFrame(all_rows)
    combined.to_csv(out / "all_metrics_winners.csv", index=False)
    return combined


def process_times(result_dir: os.PathLike[str] | str,
                  *args: Any, **kwargs: Any) -> pd.DataFrame:
    root, raw = _read(result_dir)
    df = _valid_rows(raw)
    cols = [c for c in ["Train time [s]", "Prediction time [s]"] if c in df.columns]
    if not cols:
        raise ValueError("No timing columns found")
    out = root / "time"
    out.mkdir(parents=True, exist_ok=True)
    agg = df.groupby("Algorithm")[cols].agg(["mean", "median", "std", "count"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "times.csv", index=False)
    _write_tex(agg, out / "times.tex", "Training and prediction times.", "tab:times")
    return agg


def _holm(pvals: Sequence[float]) -> list[float]:
    finite = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    out = [math.nan] * len(pvals)
    order = sorted(finite, key=lambda i: pvals[i])
    running = 0.0
    m = len(order)
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * float(pvals[i])))
        out[i] = running
    return out


def process_wilcoxon(classifiers: Sequence[Any],
                     result_dir: os.PathLike[str] | str,
                     metrics: Sequence[Any] | None = None,
                     *args: Any, **kwargs: Any) -> None:
    """Pairwise two-sided Wilcoxon tests on dataset-level means.

    P-values are Holm-adjusted separately within each metric.
    """
    root, raw = _read(result_dir)
    df = _valid_rows(raw)
    metric_names = _metric_names(metrics, df)
    out = root / "wilcoxon"
    out.mkdir(parents=True, exist_ok=True)

    from scipy.stats import wilcoxon

    names = []
    for item in classifiers or []:
        if isinstance(item, (tuple, list)) and item:
            names.append(str(item[0]))
        elif isinstance(item, str):
            names.append(item)
    available = sorted(df["Algorithm"].dropna().astype(str).unique())
    names = [n for n in names if n in available] or available

    for metric in metric_names:
        tab = df.groupby(["Dataset", "Algorithm"])[metric].mean().unstack()
        rows: list[dict[str, Any]] = []
        pvals: list[float] = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a not in tab.columns or b not in tab.columns:
                    continue
                pair = tab[[a, b]].dropna()
                diff = pair[a] - pair[b]
                try:
                    if len(pair) == 0 or np.allclose(diff.to_numpy(), 0.0):
                        p = 1.0
                        statistic = 0.0
                    else:
                        res = wilcoxon(pair[a], pair[b], zero_method="pratt",
                                       alternative="two-sided", method="auto")
                        p = float(res.pvalue)
                        statistic = float(res.statistic)
                except Exception:
                    p = math.nan
                    statistic = math.nan
                rows.append({
                    "Algorithm_A": a,
                    "Algorithm_B": b,
                    "Metric": metric,
                    "N_datasets": int(len(pair)),
                    "Mean_A_minus_B": float(diff.mean()) if len(pair) else math.nan,
                    "Median_A_minus_B": float(diff.median()) if len(pair) else math.nan,
                    "Wilcoxon_statistic": statistic,
                    "p_raw": p,
                })
                pvals.append(p)
        adjusted = _holm(pvals)
        for row, p_adj in zip(rows, adjusted):
            row["p_holm"] = p_adj
            row["significant_at_0.05"] = bool(np.isfinite(p_adj) and p_adj < 0.05)
        pd.DataFrame(rows).to_csv(out / f"wilcoxon_{metric}.csv", index=False)


def dataset_info(datasets: Iterable[Any], result_dir: os.PathLike[str] | str,
                 *args: Any, **kwargs: Any) -> pd.DataFrame:
    root = _root(result_dir)
    out = root / "info"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for ds in datasets:
        row: dict[str, Any] = {"Dataset": getattr(ds, "name", type(ds).__name__)}
        try:
            X, y, *_ = ds.get_data()
            row.update({
                "N_samples": int(len(y)),
                "N_features": int(np.asarray(X).shape[1]),
                "N_classes": int(len(np.unique(y))),
            })
        except Exception as exc:
            try:
                row["N_samples"] = int(len(ds))
            except Exception:
                row["N_samples"] = math.nan
            row["Error"] = repr(exc)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "datasets.csv", index=False)
    _write_tex(df, out / "datasets.tex", "Datasets used in the benchmark.", "tab:datasets")
    return df


def classifier_info(classifiers: Iterable[Any], result_dir: os.PathLike[str] | str,
                    *args: Any, **kwargs: Any) -> pd.DataFrame:
    root = _root(result_dir)
    out = root / "info"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in classifiers:
        if isinstance(item, (tuple, list)) and item:
            name = str(item[0])
            factory = item[1] if len(item) > 1 else None
        else:
            name, factory = str(item), None
        module = getattr(factory, "__module__", None)
        qualname = getattr(factory, "__qualname__", getattr(factory, "__name__", None))
        try:
            source_file = inspect.getsourcefile(factory) if factory is not None else None
        except Exception:
            source_file = None
        rows.append({
            "Algorithm": name,
            "Factory_module": module,
            "Factory_name": qualname,
            "Source_file": source_file,
            "Factory_repr": repr(factory),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "classifiers.csv", index=False)
    _write_tex(df, out / "classifiers.tex", "Classifiers used in the benchmark.", "tab:classifiers")
    return df
