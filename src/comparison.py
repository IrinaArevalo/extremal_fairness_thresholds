import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from metrics import fairness_metric_at_threshold, candidate_thresholds  
from model import make_model_pipeline
from extremal_threshold_search import run_all_datasets
from data import load_uci_german_credit, load_adult_uci, load_taiwan_credit_xls, load_compas_sqlite, load_give_me_some_credit_kaggle


@dataclass
class NaiveRunResult:
    seed: int
    t_naive: float
    naive_value: float


def _as_scalar(val):
    if isinstance(val, (tuple, list, np.ndarray)):
        return float(val[0])
    return float(val)


def run_naive_threshold_search_once(
    X: pd.DataFrame,
    y: pd.Series,
    age: pd.Series,
    metric: str = "equal_opportunity_difference",
    decision_threshold: float = 0.5,
    train_frac: float = 0.7,
    min_group_size: int = 50,
    seed: int = 0,
) -> NaiveRunResult:
    """
    Naive baseline:
      - split data into fit (train_frac) and eval (1-train_frac)
      - train model on fit
      - on eval: choose t maximizing |Φ_eval(t)| AND report Φ_eval(t) on same eval split

    This overestimates due to threshold selection on the same sample.
    """
    y_np = y.to_numpy(dtype=int)
    idx_all = np.arange(len(X))

    idx_fit, idx_eval = train_test_split(
        idx_all,
        test_size=(1 - train_frac),
        random_state=seed,
        stratify=y_np,
    )

    X_fit, y_fit = X.iloc[idx_fit], y_np[idx_fit]
    X_eval, y_eval = X.iloc[idx_eval], y_np[idx_eval]
    age_eval = age.iloc[idx_eval].to_numpy(dtype=float)

    model = make_model_pipeline(X)
    model.fit(X_fit, y_fit)

    y_score_eval = model.predict_proba(X_eval)[:, 1]

    Ts = candidate_thresholds(age_eval)
    best_t, best_val, best_abs = None, None, -np.inf

    for t in Ts:
        # group sizes
        nL = int((age_eval < t).sum())
        nR = int((age_eval >= t).sum())
        if nL < min_group_size or nR < min_group_size:
            continue

        val = fairness_metric_at_threshold(
            y_true=y_eval,
            y_score=y_score_eval,
            age=age_eval,
            t=float(t),
            metric=metric,
            decision_threshold=decision_threshold,
        )
        val = _as_scalar(val)
        if np.isnan(val) or np.isinf(val):
            continue

        if abs(val) > best_abs:
            best_abs = abs(val)
            best_t = float(t)
            best_val = float(val)

    if best_t is None:
        raise ValueError("No valid threshold found; lower min_group_size or check age distribution.")

    return NaiveRunResult(seed=seed, t_naive=best_t, naive_value=best_val)


def run_naive_threshold_search_repeated(
    X: pd.DataFrame,
    y: pd.Series,
    age: pd.Series,
    metric: str = "equal_opportunity_difference",
    decision_threshold: float = 0.5,
    train_frac: float = 0.7,
    min_group_size: int = 50,
    n_repeats: int = 20,
    base_seed: int = 0,
) -> pd.DataFrame:
    """
    Repeat naive baseline across seeds.
    Returns per-seed DataFrame with mean/std you can compare to your sample-splitting results.
    """
    rows = []
    for r in range(n_repeats):
        seed = base_seed + r
        res = run_naive_threshold_search_once(
            X=X, y=y, age=age,
            metric=metric,
            decision_threshold=decision_threshold,
            train_frac=train_frac,
            min_group_size=min_group_size,
            seed=seed,
        )
        rows.append({"seed": res.seed, "t_naive": res.t_naive, "naive_value": res.naive_value})

    df = pd.DataFrame(rows)
    print(f"[Naive baseline] {metric}:")
    print(f"  t_naive mean±std: {df['t_naive'].mean():.2f} ± {df['t_naive'].std(ddof=1):.2f}")
    print(f"  naive_value mean±std: {df['naive_value'].mean():+.4f} ± {df['naive_value'].std(ddof=1):.4f}")
    return df

def _mean_std_str(x: pd.Series, decimals: int = 3) -> str:
    m = float(x.mean())
    s = float(x.std(ddof=1))
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"

def _mean_std_str_abs(x: pd.Series, decimals: int = 3) -> str:
    m = float(x.abs().mean())
    s = float(x.abs().std(ddof=1))
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def build_naive_vs_ours_table(
    paths: dict,
    *,
    metric: str = "equal_opportunity_difference",
    decision_threshold: float = 0.5,
    # ours
    fit_frac: float = 0.5,
    search_frac: float = 0.25,
    test_frac: float = 0.25,
    # naive
    naive_train_frac: float = 0.7,
    min_group_size: int = 50,
    n_repeats: int = 20,
    base_seed: int = 0,
    save_csv: str | None = "naive_vs_ours.csv",
):
    """
    Returns a DataFrame comparing naive threshold search vs sample-splitting estimator.
    """

    # --- Run OURS (sample splitting) across all datasets ---
    ours_summary, _ = run_all_datasets(
        paths,
        metric=metric,
        decision_threshold=decision_threshold,
        fit_frac=fit_frac,
        search_frac=search_frac,
        test_frac=test_frac,
        min_group_size=min_group_size,
        n_repeats=n_repeats,
        base_seed=base_seed,
        save_dir=None,
        verbose=False,
        run_synthetic=False,  # keep comparable unless you want it too
    )
    ours_summary = ours_summary.set_index("dataset")

    # --- Helper: load datasets one-by-one to run NAIVE ---
    # You may need to adapt these names if your loader names differ.
    def load_by_key(key: str):
        if key == "german_uci":
            return load_uci_german_credit(paths[key])
        if key == "adult_uci":
            return load_adult_uci(paths[key])
        if key == "taiwan_xls":
            return load_taiwan_credit_xls(paths[key])
        if key == "compas_db":
            return load_compas_sqlite(paths[key])
        if key == "give_me_credit_train":
            X, y, age = load_give_me_some_credit_kaggle(paths[key])
            return X, y, age
        raise KeyError(key)

    dataset_keys = [k for k in ["german_uci", "adult_uci", "taiwan_xls", "compas_db", "give_me_credit_train"] if k in paths]

    rows = []
    for key in dataset_keys:
        # Map path-key -> dataset name used in your summaries
        ds_name = "give_me_credit" if key == "give_me_credit_train" else key

        X, y, age = load_by_key(key)

        # NAIVE repeated
        naive_df = run_naive_threshold_search_repeated(
            X=X, y=y, age=age,
            metric=metric,
            decision_threshold=decision_threshold,
            train_frac=naive_train_frac,
            min_group_size=min_group_size,
            n_repeats=n_repeats,
            base_seed=base_seed,
        )

        # OURS row
        if ds_name not in ours_summary.index:
            raise ValueError(f"Dataset '{ds_name}' not found in ours_summary index.")

        o = ours_summary.loc[ds_name]

        # Build comparison row
        rows.append({
            "dataset": ds_name,
            "n": int(len(X)),
            # ours
            "ours_t*": f"{o['t_star_mean']:.2f} ± {o['t_star_std']:.2f}",
            "ours_test": f"{o['test_mean']:+.3f} ± {o['test_std']:.3f}",
            "ours_|test|": f"{abs(float(o['test_mean'])):.3f} (mean)  ± {float(o['test_std']):.3f} (std)",
            # naive
            "naive_t*": _mean_std_str(naive_df["t_naive"], decimals=2),
            "naive_value": _mean_std_str(naive_df["naive_value"], decimals=3),
            "naive_|value|": _mean_std_str_abs(naive_df["naive_value"], decimals=3),
        })

    out = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

    if save_csv:
        out.to_csv(save_csv, index=False)

    return out

import numpy as np
import pandas as pd
from scipy import stats

# Assumes you already have:
# - run_all_datasets(paths, metric=..., ...) -> (summary_df, details)
# - run_naive_threshold_search_repeated(...) -> per-seed df: t_naive, naive_value
# - loaders: load_uci_german_data, load_adult_uci, load_taiwan_credit_xls,
#           load_compas_sqlite, load_give_me_some_credit_kaggle

DEFAULT_METRICS = [
    "equal_opportunity_difference",
    "statistical_parity_difference",
    "average_odds_difference",
    "auc_difference",
    "log_disparate_impact",
]


def _ci_and_p_from_samples(samples: np.ndarray, alpha: float = 0.05) -> dict:
    """
    95% CI and p-value for H0: mean = 0 using a t-test.
    Works on per-seed samples (independent splits).
    """
    x = np.asarray(samples, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {
            "n": n,
            "mean": float(np.nanmean(x)) if n else np.nan,
            "std": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se

    t_stat = mean / se if se > 0 else np.nan
    p_value = float(2 * stats.t.sf(np.abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else np.nan

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_stat": float(t_stat),
        "p_value": p_value,
    }


def build_naive_vs_ours_table_all_metrics_with_ci(
    paths: dict,
    *,
    metrics: list[str] = DEFAULT_METRICS,
    decision_threshold: float = 0.5,
    # OURS (sample splitting)
    fit_frac: float = 0.5,
    search_frac: float = 0.25,
    test_frac: float = 0.25,
    # NAIVE (train/eval; search+eval on same eval)
    naive_train_frac: float = 0.7,
    min_group_size: int = 50,
    n_repeats: int = 20,
    base_seed: int = 0,
    out_csv: str = "naive_vs_ours_all_metrics_with_ci.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Long-form CSV with one row per (dataset, metric) including:
      - Ours: mean/std + 95% CI + p-value (H0: mean disparity = 0)
      - Naive: mean/std + 95% CI + p-value (H0: mean disparity = 0)
    """

    def _log(msg: str):
        if verbose:
            print(msg)

    # Loader mapping
    def load_by_key(key: str):
        if key == "german_uci":
            return load_uci_german_credit(paths[key])
        if key == "adult_uci":
            return load_adult_uci(paths[key])
        if key == "taiwan_xls":
            return load_taiwan_credit_xls(paths[key])
        if key == "compas_db":
            return load_compas_sqlite(paths[key])
        if key == "give_me_credit_train":
            return load_give_me_some_credit_kaggle(paths[key])
        raise KeyError(key)

    dataset_keys = [k for k in ["german_uci", "adult_uci", "taiwan_xls", "compas_db", "give_me_credit_train"] if k in paths]
    ds_name_map = {"give_me_credit_train": "give_me_credit"}

    rows = []

    for metric in metrics:
        _log(f"\n=== Metric: {metric} ===")

        # Run OURS across all datasets for this metric
        ours_summary, ours_details = run_all_datasets(
            paths,
            metric=metric,
            decision_threshold=decision_threshold,
            fit_frac=fit_frac,
            search_frac=search_frac,
            test_frac=test_frac,
            min_group_size=min_group_size,
            n_repeats=n_repeats,
            base_seed=base_seed,
            save_dir=None,
            verbose=False,
            run_synthetic=False,
        )
        ours_summary = ours_summary.set_index("dataset")

        for key in dataset_keys:
            ds_name = ds_name_map.get(key, key)
            _log(f"  - Dataset: {ds_name}")

            X, y, age = load_by_key(key)

            # NAIVE repeated (per-seed values)
            naive_df = run_naive_threshold_search_repeated(
                X=X,
                y=y,
                age=age,
                metric=metric,
                decision_threshold=decision_threshold,
                train_frac=naive_train_frac,
                min_group_size=min_group_size,
                n_repeats=n_repeats,
                base_seed=base_seed,
            )

            naive_stats = _ci_and_p_from_samples(naive_df["naive_value"].to_numpy(), alpha=0.05)
            naive_t_stats = _ci_and_p_from_samples(naive_df["t_naive"].to_numpy(), alpha=0.05)

            # OURS per-seed test values: pulled from details saved by run_all_datasets
            # (it stores rep.all_results per dataset)
            if ds_name not in ours_details or "all_results" not in ours_details[ds_name]:
                # fallback: use mean/std only (no per-seed) if not available
                ours_vals = None
            else:
                ours_vals = ours_details[ds_name]["all_results"]["test_value"].to_numpy()

            if ours_vals is None:
                # approximate CI using summary (less ideal, but keeps the function robust)
                o = ours_summary.loc[ds_name]
                # treat as normal approx with df=n_repeats-1
                se = float(o["test_std"]) / np.sqrt(n_repeats)
                tcrit = float(stats.t.ppf(0.975, df=n_repeats - 1))
                ours_ci_low = float(o["test_mean"]) - tcrit * se
                ours_ci_high = float(o["test_mean"]) + tcrit * se
                t_stat = float(o["test_mean"]) / se if se > 0 else np.nan
                p_val = float(2 * stats.t.sf(np.abs(t_stat), df=n_repeats - 1)) if np.isfinite(t_stat) else np.nan
                ours_stats = {
                    "n": n_repeats,
                    "mean": float(o["test_mean"]),
                    "std": float(o["test_std"]),
                    "se": se,
                    "ci_low": ours_ci_low,
                    "ci_high": ours_ci_high,
                    "t_stat": float(t_stat),
                    "p_value": p_val,
                }
            else:
                ours_stats = _ci_and_p_from_samples(ours_vals, alpha=0.05)

            # Ours t* distribution isn't stored in ours_details by default; use summary stats for t*
            o = ours_summary.loc[ds_name]
            t_se = float(o["t_star_std"]) / np.sqrt(n_repeats)
            tcrit = float(stats.t.ppf(0.975, df=n_repeats - 1))
            ours_t_ci_low = float(o["t_star_mean"]) - tcrit * t_se
            ours_t_ci_high = float(o["t_star_mean"]) + tcrit * t_se

            rows.append({
                "dataset": ds_name,
                "metric": metric,
                "n": int(len(X)),

                # OURS (sample splitting) - disparity
                "ours_test_mean": ours_stats["mean"],
                "ours_test_std": ours_stats["std"],
                "ours_test_ci_low": ours_stats["ci_low"],
                "ours_test_ci_high": ours_stats["ci_high"],
                "ours_test_p_value": ours_stats["p_value"],

                # OURS - threshold (from summary normal/t CI)
                "ours_t_star_mean": float(o["t_star_mean"]),
                "ours_t_star_std": float(o["t_star_std"]),
                "ours_t_star_ci_low": ours_t_ci_low,
                "ours_t_star_ci_high": ours_t_ci_high,

                # NAIVE - disparity
                "naive_value_mean": naive_stats["mean"],
                "naive_value_std": naive_stats["std"],
                "naive_value_ci_low": naive_stats["ci_low"],
                "naive_value_ci_high": naive_stats["ci_high"],
                "naive_value_p_value": naive_stats["p_value"],

                # NAIVE - threshold
                "naive_t_star_mean": naive_t_stats["mean"],
                "naive_t_star_std": naive_t_stats["std"],
                "naive_t_star_ci_low": naive_t_stats["ci_low"],
                "naive_t_star_ci_high": naive_t_stats["ci_high"],
            })

    out = pd.DataFrame(rows).sort_values(["dataset", "metric"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    _log(f"\nSaved: {out_csv}")
    return out
