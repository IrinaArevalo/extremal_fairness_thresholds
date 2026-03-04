"""
1) Scatter of t* by dataset × metric (with jitter)
2) Scatter with error bars (t* mean ± std)
3) Scatter of test disparity by dataset × metric (with jitter)
4) Scatter with error bars (test_mean ± test_std)
5) Violin plot of t* distribution across metrics per dataset (uses the per-seed CSVs)
6) Violin plot of test disparity distribution across metrics per dataset (per-seed)
7) Per-dataset panels: t* by metric + test disparity by metric (two-panel figure)

Requirements:
  pip install pandas matplotlib numpy

Inputs:
- master_summary.csv (one row per dataset×metric)
- per-seed files:
    results_all_metrics/<metric>/<dataset>_per_seed.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from metrics import fairness_metric_at_threshold 

# ----------------------------
# Helpers
# ----------------------------
def _set_dataset_order(df: pd.DataFrame, dataset_order: list[str] | None) -> pd.DataFrame:
    df = df.copy()
    if dataset_order is None:
        dataset_order = sorted(df["dataset"].unique().tolist())
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    return df.sort_values("dataset")


def _jittered_x(codes: np.ndarray, seed: int = 0, scale: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return codes + rng.normal(0, scale, size=len(codes))


def _nice_metric_label(m: str) -> str:
    return m.replace("_", " ")


# ----------------------------
# 1) t* scatter (jitter)
# ----------------------------
def plot_thresholds_scatter(
    master_csv: str = "results_all_metrics/master_summary.csv",
    dataset_order: list[str] | None = None,
    savepath: str | None = None,
):
    df = pd.read_csv(master_csv)
    df = _set_dataset_order(df, dataset_order)
    metrics = df["metric"].unique()

    plt.figure(figsize=(9, 4.8))
    for i, m in enumerate(metrics):
        sub = df[df["metric"] == m]
        x = _jittered_x(sub["dataset"].cat.codes.to_numpy(), seed=123 + i)
        plt.scatter(x, sub["t_star_mean"], s=50, label=m)

    plt.xticks(range(df["dataset"].cat.categories.size), df["dataset"].cat.categories, rotation=30, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Extremal age threshold $t^*$ (mean over seeds)")
    plt.title("Extremal thresholds by dataset and metric")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------
# 2) t* with error bars (mean ± std)
# ----------------------------
def plot_thresholds_errorbars(
    master_csv: str = "results_all_metrics/master_summary.csv",
    dataset_order: list[str] | None = None,
    savepath: str | None = None,
):
    df = pd.read_csv(master_csv)
    df = _set_dataset_order(df, dataset_order)
    metrics = df["metric"].unique()

    plt.figure(figsize=(9, 4.8))
    for i, m in enumerate(metrics):
        sub = df[df["metric"] == m]
        x = _jittered_x(sub["dataset"].cat.codes.to_numpy(), seed=555 + i)
        plt.errorbar(
            x,
            sub["t_star_mean"],
            yerr=sub["t_star_std"],
            fmt="o",
            capsize=3,
            markersize=5,
            label=m,
        )

    plt.xticks(range(df["dataset"].cat.categories.size), df["dataset"].cat.categories, rotation=30, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Extremal age threshold $t^*$ (mean ± std)")
    plt.title("Extremal thresholds with uncertainty")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------
# 3) test disparity scatter (jitter)
# ----------------------------
def plot_test_disparity_scatter(
    master_csv: str = "results_all_metrics/master_summary.csv",
    dataset_order: list[str] | None = None,
    savepath: str | None = None,
):
    df = pd.read_csv(master_csv)
    df = _set_dataset_order(df, dataset_order)
    metrics = df["metric"].unique()

    plt.figure(figsize=(9, 4.8))
    for i, m in enumerate(metrics):
        sub = df[df["metric"] == m]
        x = _jittered_x(sub["dataset"].cat.codes.to_numpy(), seed=777 + i)
        plt.scatter(x, sub["test_mean"], s=50, label=m)

    plt.axhline(0.0, linewidth=1)
    plt.xticks(range(df["dataset"].cat.categories.size), df["dataset"].cat.categories, rotation=30, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Test disparity at $t^*$ (mean over seeds)")
    plt.title("Held-out test disparity at extremal threshold $t^*$")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------
# 4) test disparity with error bars (mean ± std)
# ----------------------------
def plot_test_disparity_errorbars(
    master_csv: str = "results_all_metrics/master_summary.csv",
    dataset_order: list[str] | None = None,
    savepath: str | None = None,
):
    df = pd.read_csv(master_csv)
    df = _set_dataset_order(df, dataset_order)
    metrics = df["metric"].unique()

    plt.figure(figsize=(9, 4.8))
    for i, m in enumerate(metrics):
        sub = df[df["metric"] == m]
        x = _jittered_x(sub["dataset"].cat.codes.to_numpy(), seed=999 + i)
        plt.errorbar(
            x,
            sub["test_mean"],
            yerr=sub["test_std"],
            fmt="o",
            capsize=3,
            markersize=5,
            label=m,
        )

    plt.axhline(0.0, linewidth=1)
    plt.xticks(range(df["dataset"].cat.categories.size), df["dataset"].cat.categories, rotation=30, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Test disparity at $t^*$ (mean ± std)")
    plt.title("Held-out test disparity with uncertainty")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


# ----------------------------
# Per-seed loading for violin plots
# ----------------------------
def _load_per_seed_all_metrics(
    results_root: str,
    dataset_name: str,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Expect per-seed files at:
      {results_root}/{metric}/{dataset_name}_per_seed.csv
    with columns: seed, t_star, search_value, test_value
    """
    frames = []
    for m in metrics:
        p = os.path.join(results_root, m, f"{dataset_name}_per_seed.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing per-seed file: {p}")
        d = pd.read_csv(p)
        d["metric"] = m
        d["dataset"] = dataset_name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ----------------------------
# 5) Violin plot: distribution of t_star across metrics (per dataset)
# ----------------------------
def plot_violin_tstar_by_metric(
    results_root: str,
    master_csv: str = "results_all_metrics/master_summary.csv",
    dataset_order: list[str] | None = None,
    metrics: list[str] | None = None,
    savepath: str | None = None,
):
    master = pd.read_csv(master_csv)
    master = _set_dataset_order(master, dataset_order)

    if metrics is None:
        metrics = sorted(master["metric"].unique().tolist())

    all_frames = []
    for ds in master["dataset"].cat.categories:
        if ds in master["dataset"].unique():
            all_frames.append(_load_per_seed_all_metrics(results_root, ds, metrics))
    per_seed = pd.concat(all_frames, ignore_index=True)

    # One figure per dataset (cleanest + readable)
    for ds in master["dataset"].cat.categories:
        sub = per_seed[per_seed["dataset"] == ds]
        if sub.empty:
            continue

        data = [sub[sub["metric"] == m]["t_star"].dropna().to_numpy() for m in metrics]

        plt.figure(figsize=(9, 4.2))
        parts = plt.violinplot(data, showmeans=True, showextrema=True)
        # No manual colors (per your style constraints)

        plt.xticks(range(1, len(metrics) + 1), [_nice_metric_label(m) for m in metrics], rotation=25, ha="right")
        plt.ylabel("Extremal threshold $t^*$ (per-seed)")
        plt.title(f"{ds}: distribution of selected $t^*$ across metrics")
        plt.tight_layout()

        if savepath:
            base, ext = os.path.splitext(savepath)
            out = f"{base}_{ds}{ext or '.png'}"
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()


# ----------------------------
# 6) Violin plot: distribution of test_value across metrics (per dataset)
# ----------------------------
def plot_violin_test_by_metric(
    results_root: str,
    master_csv: str = "results_all_metrics/master_summary.csv",
    dataset_order: list[str] | None = None,
    metrics: list[str] | None = None,
    savepath: str | None = None,
):
    master = pd.read_csv(master_csv)
    master = _set_dataset_order(master, dataset_order)

    if metrics is None:
        metrics = sorted(master["metric"].unique().tolist())

    all_frames = []
    for ds in master["dataset"].cat.categories:
        if ds in master["dataset"].unique():
            all_frames.append(_load_per_seed_all_metrics(results_root, ds, metrics))
    per_seed = pd.concat(all_frames, ignore_index=True)

    for ds in master["dataset"].cat.categories:
        sub = per_seed[per_seed["dataset"] == ds]
        if sub.empty:
            continue

        data = [sub[sub["metric"] == m]["test_value"].dropna().to_numpy() for m in metrics]

        plt.figure(figsize=(9, 4.2))
        plt.violinplot(data, showmeans=True, showextrema=True)

        plt.axhline(0.0, linewidth=1)
        plt.xticks(range(1, len(metrics) + 1), [_nice_metric_label(m) for m in metrics], rotation=25, ha="right")
        plt.ylabel("Test disparity at $t^*$ (per-seed)")
        plt.title(f"{ds}: distribution of held-out test disparity across metrics")
        plt.tight_layout()

        if savepath:
            base, ext = os.path.splitext(savepath)
            out = f"{base}_{ds}{ext or '.png'}"
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()


# ----------------------------
# 7) Per-dataset panels: (A) t* mean±std by metric, (B) test mean±std by metric
# ----------------------------
def plot_panels_per_dataset(
    master_csv: str = "results_all_metrics/master_summary.csv",
    dataset_order: list[str] | None = None,
    metrics: list[str] | None = None,
    savepath: str | None = None,
):
    df = pd.read_csv(master_csv)
    df = _set_dataset_order(df, dataset_order)

    if metrics is None:
        metrics = sorted(df["metric"].unique().tolist())

    for ds in df["dataset"].cat.categories:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        sub = sub.set_index("metric").reindex(metrics).reset_index()

        fig = plt.figure(figsize=(10, 4.2))

        # Panel A: t*
        ax1 = fig.add_subplot(1, 2, 1)
        x = np.arange(len(metrics))
        ax1.errorbar(x, sub["t_star_mean"], yerr=sub["t_star_std"], fmt="o", capsize=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels([_nice_metric_label(m) for m in metrics], rotation=30, ha="right")
        ax1.set_ylabel("Extremal threshold $t^*$")
        ax1.set_title(f"{ds}: $t^*$ (mean ± std)")

        # Panel B: test disparity
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.errorbar(x, sub["test_mean"], yerr=sub["test_std"], fmt="o", capsize=3)
        ax2.axhline(0.0, linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels([_nice_metric_label(m) for m in metrics], rotation=30, ha="right")
        ax2.set_ylabel("Test disparity at $t^*$")
        ax2.set_title(f"{ds}: test disparity (mean ± std)")

        fig.suptitle("Extremal threshold selection and held-out disparity", y=1.02)
        fig.tight_layout()

        if savepath:
            base, ext = os.path.splitext(savepath)
            out = f"{base}_{ds}{ext or '.png'}"
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    
    dataset_order = [
        "adult_uci",
        "compas_db",
        "german_uci",
        "give_me_credit",
        "synthetic_tb35_bs0.7",
        "taiwan_xls",
    ]

    # 1
    plot_thresholds_scatter("results_all_metrics/master_summary.csv", dataset_order, savepath="fig_thresholds_scatter.png")

    # 2
    plot_thresholds_errorbars("results_all_metrics/master_summary.csv", dataset_order, savepath="fig_thresholds_errorbars.png")

    # 3
    plot_test_disparity_scatter("results_all_metrics/master_summary.csv", dataset_order, savepath="fig_test_disparity_scatter.png")

    # 4
    plot_test_disparity_errorbars("results_all_metrics/master_summary.csv", dataset_order, savepath="fig_test_disparity_errorbars.png")

    # 5 & 6 require per-seed CSVs from run_all_metrics outputs:
    results_root = "results_all_metrics"
    metrics = [
        "auc_difference",
        "average_odds_difference",
        "equal_opportunity_difference",
        "log_disparate_impact",
        "statistical_parity_difference",
    ]

    # 5
    plot_violin_tstar_by_metric(
        results_root=results_root,
        master_csv="results_all_metrics/master_summary.csv",
        dataset_order=dataset_order,
        metrics=metrics,
        savepath="fig_violin_tstar.png",
    )

    # 6
    plot_violin_test_by_metric(
        results_root=results_root,
        master_csv="results_all_metrics/master_summary.csv",
        dataset_order=dataset_order,
        metrics=metrics,
        savepath="fig_violin_test.png",
    )

    # 7
    plot_panels_per_dataset(
        master_csv="results_all_metrics/master_summary.csv",
        dataset_order=dataset_order,
        metrics=metrics,
        savepath="fig_panels.png",
    )


def compute_fairness_sensitivity_curve(
    y_true,
    y_score,
    age,
    metric,
    decision_threshold=0.5,
    min_group_size=50,
):
    """
    Compute fairness metric Φ(t) across thresholds t.

    Returns
    -------
    thresholds : np.ndarray
    values : np.ndarray
    """
    ages = np.asarray(age)
    thresholds = np.unique(ages)
    thresholds = (thresholds[:-1] + thresholds[1:]) / 2  # midpoints

    values = []

    for t in thresholds:
        mask_left = ages < t
        mask_right = ~mask_left

        if mask_left.sum() < min_group_size or mask_right.sum() < min_group_size:
            values.append(np.nan)
            continue

        val, *_ = fairness_metric_at_threshold(
            y_true=y_true,
            y_score=y_score,
            age=ages,
            t=t,
            metric=metric,
            decision_threshold=decision_threshold,
        )
        values.append(val)

    return thresholds, np.array(values)


def plot_curve_from_saved(ds_dir, metric, savepath=None):
    curve = pd.read_csv(f"{ds_dir}/curves/{metric}.csv")
    plt.figure(figsize=(6,4))
    plt.plot(curve["t"], curve["phi"])
    plt.axhline(0)
    plt.xlabel("Age threshold $t$")
    plt.ylabel(r"Disparity $\Phi(t)$")
    plt.title(f"{os.path.basename(ds_dir)} — {metric}")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_two_curves_pdf(ds1, ds2, metric, out_pdf="sensitivity_curves.pdf"):
    c1 = pd.read_csv(f"{ds1}/curves/{metric}.csv")
    c2 = pd.read_csv(f"{ds2}/curves/{metric}.csv")

    plt.figure(figsize=(7,3))

    plt.subplot(1,2,1)
    plt.plot(c1["t"], c1["phi"])
    plt.axhline(0)
    plt.title(os.path.basename(ds1))
    plt.xlabel("$t$")
    plt.ylabel(r"$\Phi(t)$")

    plt.subplot(1,2,2)
    plt.plot(c2["t"], c2["phi"])
    plt.axhline(0)
    plt.title(os.path.basename(ds2))
    plt.xlabel("$t$")

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.show()


def plot_curve_with_tstar(ds_dir, master_csv, metric, out_pdf=None):
    curve = pd.read_csv(f"{ds_dir}/curves/{metric}.csv")
    master = pd.read_csv(master_csv)

    ds_name = os.path.basename(ds_dir)
    row = master[(master["dataset"] == ds_name) & (master["metric"] == metric)].iloc[0]
    t_star = float(row["t_star_mean"])

    # find curve value near t_star
    idx = (curve["t"] - t_star).abs().idxmin()
    phi_star = float(curve.loc[idx, "phi"])

    plt.figure(figsize=(6,4))
    plt.plot(curve["t"], curve["phi"])
    plt.axhline(0)
    plt.scatter([t_star], [phi_star], s=80)
    plt.axvline(t_star, linestyle="--")
    plt.xlabel("Age threshold $t$")
    plt.ylabel(r"Disparity $\Phi(t)$")
    plt.title(f"{ds_name} — {metric} (marker at $t^*$)")
    plt.tight_layout()
    if out_pdf:
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.show()


def plot_all_curves_with_tstar(
    saved_root,
    master_csv,
    metric,
    out_dir="curve_plots",
):
    """
    Plot sensitivity curve with t* marker for every dataset.

    Parameters
    ----------
    saved_root : str
        Root folder (e.g., "saved_predictions")
    master_csv : str
        master_summary.csv path
    metric : str
        fairness metric name
    out_dir : str
        where PDFs will be saved
    """
    os.makedirs(out_dir, exist_ok=True)
    master = pd.read_csv(master_csv)

    datasets = [
        d for d in os.listdir(saved_root)
        if os.path.isdir(os.path.join(saved_root, d))
    ]

    for ds in datasets:
        ds_dir = os.path.join(saved_root, ds)
        curve_path = os.path.join(ds_dir, "curves", f"{metric}.csv")

        if not os.path.exists(curve_path):
            continue

        curve = pd.read_csv(curve_path)

        row = master[(master["dataset"] == ds) & (master["metric"] == metric)]
        if row.empty:
            continue

        t_star = float(row.iloc[0]["t_star_mean"])

        # nearest curve value
        idx = (curve["t"] - t_star).abs().idxmin()
        phi_star = float(curve.loc[idx, "phi"])

        plt.figure(figsize=(6,4))
        plt.plot(curve["t"], curve["phi"])
        plt.axhline(0)
        plt.scatter([t_star], [phi_star], s=80)
        plt.axvline(t_star, linestyle="--")

        plt.xlabel("Age threshold $t$")
        plt.ylabel(r"$\Phi(t)$")
        plt.title(f"{ds} — {metric}")

        plt.tight_layout()
        out_pdf = os.path.join(out_dir, f"{ds}_{metric}.pdf")
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved plots to {out_dir}")



def plot_all_curves_grid(
    saved_root,
    master_csv,
    metric,
    out_pdf="all_sensitivity_curves.pdf",
    ncols=3,
):
    master = pd.read_csv(master_csv)

    datasets = [
        d for d in os.listdir(saved_root)
        if os.path.isdir(os.path.join(saved_root, d))
    ]

    valid = []
    for ds in datasets:
        curve_path = os.path.join(saved_root, ds, "curves", f"{metric}.csv")
        if os.path.exists(curve_path):
            valid.append(ds)

    n = len(valid)
    nrows = math.ceil(n / ncols)

    plt.figure(figsize=(4*ncols, 3*nrows))

    for i, ds in enumerate(valid, 1):
        ds_dir = os.path.join(saved_root, ds)
        curve = pd.read_csv(os.path.join(ds_dir, "curves", f"{metric}.csv"))

        row = master[(master["dataset"] == ds) & (master["metric"] == metric)]
        if row.empty:
            continue

        t_star = float(row.iloc[0]["t_star_mean"])
        idx = (curve["t"] - t_star).abs().idxmin()
        phi_star = float(curve.loc[idx, "phi"])

        plt.subplot(nrows, ncols, i)
        plt.plot(curve["t"], curve["phi"])
        plt.axhline(0)
        plt.scatter([t_star], [phi_star], s=50)
        plt.axvline(t_star, linestyle="--")

        plt.title(ds)
        plt.xlabel("$t$")
        if i % ncols == 1:
            plt.ylabel(r"$\Phi(t)$")

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.show()