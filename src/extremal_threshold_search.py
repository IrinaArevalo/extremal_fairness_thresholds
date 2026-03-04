"""
Extremal fairness-threshold search over a continuous sensitive attribute (age)

- Load common bias datasets
- Preprocess (one-hot for categoricals, impute, scale)
- Train LogisticRegression
- 3-way sample splitting: fit / search / test
- Sweep age thresholds on SEARCH split to pick t* maximizing |fairness metric|
- Evaluate the chosen t* on TEST split
- Repeat across many random seeds and report mean±std for t*, search metric, test metric

Metrics implemented:
- statistical_parity_difference
- disparate_impact  (also provides log_disparate_impact, which is easier to maximize)
- equal_opportunity_difference (TPR gap)
- average_odds_difference (avg of TPR and FPR gaps)
- auc_difference

Dependencies:
  pip install numpy pandas scikit-learn matplotlib readxls
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
from dataclasses import asdict

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from data import SingleRunResult, RepeatedSummary
from config import MetricName
from model import make_model_pipeline
from metrics import fairness_metric_at_threshold, candidate_thresholds
from data import load_uci_german_credit, load_taiwan_credit_xls, load_adult_uci, load_compas_sqlite, load_give_me_some_credit_kaggle, make_synthetic_age_bias


def find_extremal_threshold_on_split(
    y_true: np.ndarray,
    y_score: np.ndarray,
    age: np.ndarray,
    metric: MetricName,
    decision_threshold: float,
    min_group_size: int,
) -> Tuple[float, float]:
    Ts = candidate_thresholds(age, use_midpoints=True, max_thresholds=None)

    best_t = None
    best_val = None
    best_abs = -np.inf

    for t in Ts:
        val, nL, nR = fairness_metric_at_threshold(y_true, y_score, age, float(t), metric, decision_threshold)
        if np.isnan(val) or np.isinf(val):
            continue
        if nL < min_group_size or nR < min_group_size:
            continue

        score = abs(val) if metric != "disparate_impact" else abs(np.log(val)) if val > 0 else np.inf
        # For DI we usually want to maximize |log(DI)|; but if metric=="disparate_impact" we still pick by |log(DI)|
        if score > best_abs:
            best_abs = score
            best_t = float(t)
            best_val = float(val)

    if best_t is None or best_val is None:
        raise ValueError("No valid threshold found. Try lowering min_group_size or check age column.")

    return best_t, best_val


def run_sample_splitting_once(
    X: pd.DataFrame,
    y: pd.Series,
    age: pd.Series,
    metric: MetricName = "equal_opportunity_difference",
    decision_threshold: float = 0.5,
    fit_frac: float = 0.5,
    search_frac: float = 0.25,
    test_frac: float = 0.25,
    min_group_size: int = 50,
    seed: int = 0,
) -> SingleRunResult:
    """
    Perform one 3-way sample-splitting run of extremal fairness-threshold estimation.

    Estimates the age threshold t* that maximizes disparity of a
    specified fairness metric for a trained classifier, while avoiding fairness
    overfitting via data splitting.

    Protocol
    --------
    The dataset is partitioned into three disjoint subsets using the given random seed:

        • Fit split   : train the predictive model f
        • Search split: sweep candidate age thresholds t and select
                        t* = argmax_t |Φ(t)|
        • Test split  : evaluate Φ(t*) on unseen data

    where Φ(t) is the fairness disparity between groups defined by:
        younger = {age < t}
        older   = {age ≥ t}

    The selected threshold t* therefore identifies the age boundary that induces
    the largest observed model disparity on the search data, and its fairness impact
    is then estimated on the independent test data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (all predictors except label). Mixed numeric/categorical allowed.

    y : pd.Series
        Binary labels encoded as {0,1}, where 1 denotes the favorable outcome
        (e.g., good credit).

    age : pd.Series
        Continuous sensitive attribute aligned with X (e.g., age in years).

    metric : MetricName, default="equal_opportunity_difference"
        Fairness disparity functional Φ(t). Supported:
            - "statistical_parity_difference"
            - "equal_opportunity_difference"
            - "average_odds_difference"
            - "auc_difference"
            - "log_disparate_impact"

    decision_threshold : float, default=0.5
        Probability cutoff used to convert model scores into binary predictions.

    fit_frac : float, default=0.5
        Fraction of samples used for model training.

    search_frac : float, default=0.25
        Fraction used to select the extremal threshold t*.

    test_frac : float, default=0.25
        Fraction used for unbiased fairness evaluation at t*.

    min_group_size : int, default=50
        Minimum number of samples required in each age group for a threshold
        to be considered during search.

    seed : int, default=0
        Random seed controlling all data splits.

    Returns
    -------
    SingleRunResult
        Dataclass with fields:
            seed : int
                Random seed used for the splits.
            t_star : float
                Selected extremal age threshold.
            search_value : float
                Fairness disparity Φ(t*) measured on the search split.
            test_value : float
                Fairness disparity Φ(t*) measured on the independent test split.

    Notes
    -----
    • The search_value is optimistically biased because t* is optimized on the
      search data. The test_value provides an unbiased estimate of extremal
      disparity.

    • Repeating this function across multiple seeds and reporting mean ± std
      yields a statistically stable estimate of the extremal fairness boundary.

    • Thresholds are evaluated at midpoints between unique observed ages,
      producing interpretable boundaries (e.g., 24.5 ≈ ≤24 vs ≥25).
    """


    if not np.isclose(fit_frac + search_frac + test_frac, 1.0):
        raise ValueError("fit_frac + search_frac + test_frac must sum to 1.")

    # Make y binary 0/1 robustly if it's categorical like "good"/"bad"
    y_arr = y.to_numpy()
    if y_arr.dtype.kind not in "iu":
        # map to 0/1 based on sorted unique
        classes = pd.unique(y_arr)
        if len(classes) != 2:
            raise ValueError(f"Expected binary label; got classes={classes}")
        # common for credit-g: 'good'/'bad' -> map good=1
        if set(map(str, classes)) == {"good", "bad"}:
            y_bin = (y_arr.astype(str) == "good").astype(int)
        else:
            # fallback: first unique -> 0, second -> 1
            y_bin = (y_arr == classes[1]).astype(int)
    else:
        y_bin = y_arr.astype(int)

    idx_all = np.arange(len(X))

    # Split fit vs tmp
    idx_fit, idx_tmp = train_test_split(
        idx_all,
        test_size=(search_frac + test_frac),
        random_state=seed,
        stratify=y_bin
    )

    # Split tmp into search vs test
    rel_test = test_frac / (search_frac + test_frac)
    idx_search, idx_test = train_test_split(
        idx_tmp,
        test_size=rel_test,
        random_state=seed + 1,
        stratify=y_bin[idx_tmp]
    )

    X_fit, y_fit = X.iloc[idx_fit], y_bin[idx_fit]
    X_search, y_search = X.iloc[idx_search], y_bin[idx_search]
    X_test, y_test = X.iloc[idx_test], y_bin[idx_test]

    age_search = age.iloc[idx_search].to_numpy(dtype=float)
    age_test = age.iloc[idx_test].to_numpy(dtype=float)

    model = make_model_pipeline(X)
    model.fit(X_fit, y_fit)

    s_search = model.predict_proba(X_search)[:, 1]
    s_test = model.predict_proba(X_test)[:, 1]

    # Search on SEARCH split
    t_star, search_val = find_extremal_threshold_on_split(
        y_true=y_search,
        y_score=s_search,
        age=age_search,
        metric=metric,
        decision_threshold=decision_threshold,
        min_group_size=min_group_size,
    )

    # Evaluate on TEST split at fixed t*
    test_val, nL, nR = fairness_metric_at_threshold(
        y_true=y_test,
        y_score=s_test,
        age=age_test,
        t=t_star,
        metric=metric,
        decision_threshold=decision_threshold,
    )

    return SingleRunResult(
        seed=seed,
        t_star=float(t_star),
        search_value=float(search_val),
        test_value=float(test_val),
    )


# -----------------------------
# Repetition across seeds
# -----------------------------


def repeat_over_seeds(
    X: pd.DataFrame,
    y: pd.Series,
    age: pd.Series,
    metric: MetricName = "equal_opportunity_difference",
    decision_threshold: float = 0.5,
    fit_frac: float = 0.5,
    search_frac: float = 0.25,
    test_frac: float = 0.25,
    min_group_size: int = 50,
    n_repeats: int = 20,
    base_seed: int = 0,
) -> RepeatedSummary:
    rows = []
    for r in range(n_repeats):
        seed = base_seed + r
        res = run_sample_splitting_once(
            X=X, y=y, age=age,
            metric=metric,
            decision_threshold=decision_threshold,
            fit_frac=fit_frac,
            search_frac=search_frac,
            test_frac=test_frac,
            min_group_size=min_group_size,
            seed=seed,
        )
        rows.append({"seed": res.seed, "t_star": res.t_star, "search_value": res.search_value, "test_value": res.test_value})

    df = pd.DataFrame(rows)

    return RepeatedSummary(
        metric=metric,
        n_repeats=n_repeats,
        t_star_mean=float(df["t_star"].mean()),
        t_star_std=float(df["t_star"].std(ddof=1)),
        search_mean=float(df["search_value"].mean()),
        search_std=float(df["search_value"].std(ddof=1)),
        test_mean=float(df["test_value"].mean()),
        test_std=float(df["test_value"].std(ddof=1)),
        all_results=df,
    )

# -----------------------------
# Repetition across datasets
# -----------------------------

def run_all_datasets(
    paths: Dict[str, str],
    *,
    metric: str = "equal_opportunity_difference",
    decision_threshold: float = 0.5,
    fit_frac: float = 0.5,
    search_frac: float = 0.25,
    test_frac: float = 0.25,
    min_group_size: int = 50,
    n_repeats: int = 20,
    base_seed: int = 0,
    save_dir: str | None = None,
    verbose: bool = True,
    # ---- Synthetic config ----
    run_synthetic: bool = True,
    synthetic_n: int = 20000,
    synthetic_true_boundary: float = 35.0,
    synthetic_bias_strength: float = 0.7,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run extremal-threshold experiments across datasets and save results progressively.

    If save_dir is provided:
        - summary.csv updated after each dataset
        - {dataset}_per_seed.csv saved for each dataset
        - summary_final.csv written at the end

    Includes an optional synthetic dataset with known unfairness boundary.

    Parameters
    ----------
    paths : dict
        Keys can include (any subset of):
          - "german_uci": path to german.data
          - "adult_uci": path to adult.data
          - "taiwan_xls": path to 'default of credit card clients.xls'
          - "compas_db": path to compas.db
          - "give_me_credit_train": path to cs-training.csv
        (cs-test.csv is unlabeled, not used.)

    run_synthetic : bool
        If True, also runs a synthetic dataset generated by make_synthetic_age_bias.
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    details: Dict[str, Any] = {}
    rows = []

    def _log(msg: str):
        if verbose:
            print(msg)

    def _save_progress(dataset_name: str, rep):
        """Save per-dataset and cumulative summary."""
        if save_dir is None:
            return

        # Save per-seed results
        per_seed_path = os.path.join(save_dir, f"{dataset_name}_per_seed.csv")
        rep.all_results.to_csv(per_seed_path, index=False)

        # Save cumulative summary
        summary_df = pd.DataFrame(rows)
        summary_path = os.path.join(save_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

    # ---------- helper to process one dataset ----------
    def _process_dataset(name: str, X, y, age):
        _log(f"\n[{name}] running extremal-threshold…")

        rep = repeat_over_seeds(
            X=X,
            y=y,
            age=age,
            metric=metric,
            decision_threshold=decision_threshold,
            fit_frac=fit_frac,
            search_frac=search_frac,
            test_frac=test_frac,
            min_group_size=min_group_size,
            n_repeats=n_repeats,
            base_seed=base_seed,
        )

        row = {
            "dataset": name,
            "n": int(len(X)),
            "metric": rep.metric,
            "t_star_mean": rep.t_star_mean,
            "t_star_std": rep.t_star_std,
            "search_mean": rep.search_mean,
            "search_std": rep.search_std,
            "test_mean": rep.test_mean,
            "test_std": rep.test_std,
            "age_min": float(age.min()),
            "age_max": float(age.max()),
            "y_mean": float(y.mean()),
        }

        rows.append(row)

        details[name] = {
            "summary": asdict(rep),
            "all_results": rep.all_results,
            "n_samples": int(len(X)),
            "age_min": row["age_min"],
            "age_max": row["age_max"],
            "y_mean": row["y_mean"],
        }

        _save_progress(name, rep)

        _log(
            f"[{name}] done | "
            f"test={rep.test_mean:+.4f}±{rep.test_std:.4f} | "
            f"t*={rep.t_star_mean:.2f}±{rep.t_star_std:.2f}"
        )

    # ---------- German ----------
    if "german_uci" in paths:
        X, y, age = load_uci_german_credit(paths["german_uci"])
        _process_dataset("german_uci", X, y, age)

    # ---------- Adult ----------
    if "adult_uci" in paths:
        X, y, age = load_adult_uci(paths["adult_uci"])
        _process_dataset("adult_uci", X, y, age)

    # ---------- Taiwan ----------
    if "taiwan_xls" in paths:
        X, y, age = load_taiwan_credit_xls(paths["taiwan_xls"])
        _process_dataset("taiwan_xls", X, y, age)

    # ---------- COMPAS ----------
    if "compas_db" in paths:
        X, y, age = load_compas_sqlite(paths["compas_db"])
        _process_dataset("compas_db", X, y, age)

    # ---------- GiveMeCredit ----------
    if "give_me_credit_train" in paths:
        X, y, age = load_give_me_some_credit_kaggle(paths["give_me_credit_train"])
        _process_dataset("give_me_credit", X, y, age)

    # ---------- Synthetic ----------
    if run_synthetic:
        # Use a fixed seed for data generation so only the split randomness changes across repeats.
        Xs, ys, ages = make_synthetic_age_bias(
            n=synthetic_n,
            true_boundary=synthetic_true_boundary,
            bias_strength=synthetic_bias_strength,
            seed=12345,
        )
        _process_dataset(
            f"synthetic_tb{synthetic_true_boundary:g}_bs{synthetic_bias_strength:g}",
            Xs, ys, ages
        )

    summary_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

    # final save
    if save_dir is not None:
        summary_df.to_csv(os.path.join(save_dir, "summary_final.csv"), index=False)

    return summary_df, details

# -------------------------
# Example usage
# -------------------------
# paths = {
#     "german_uci": "data/german.data",
#     "adult_uci": "data/adult.data",
#     "taiwan_xls": "data/default of credit card clients.xls",
#     "compas_db": "data/compas.db",
#     "give_me_credit_train": "data/cs-training.csv",
# }
#
#summary, details = run_all_datasets(
#    paths,
#    metric="equal_opportunity_difference",
#    n_repeats=20,
#    save_dir="results",   
#    run_synthetic=True,
#)
# print(summary_df)

# -----------------------------
# Repetition across all metrics
# -----------------------------

def run_all_metrics(
    paths: Dict[str, str],
    *,
    metrics: List[str] = MetricName,
    # Shared experiment config
    decision_threshold: float = 0.5,
    fit_frac: float = 0.5,
    search_frac: float = 0.25,
    test_frac: float = 0.25,
    min_group_size: int = 50,
    n_repeats: int = 20,
    base_seed: int = 0,
    # Saving
    save_root: str | None = None,
    save_master_summary: bool = True,
    verbose: bool = True,
    # Synthetic passthrough
    run_synthetic: bool = True,
    synthetic_n: int = 20000,
    synthetic_true_boundary: float = 35.0,
    synthetic_bias_strength: float = 0.7,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full multi-dataset experiment for multiple fairness metrics.

    For each metric in `metrics`, this calls run_all_datasets(..., metric=metric)
    and (optionally) saves:
        {save_root}/{metric}/summary.csv (progress)
        {save_root}/{metric}/summary_final.csv
        {save_root}/{metric}/*_per_seed.csv

    It then concatenates all per-metric summary_final outputs into a single
    master DataFrame (one row per dataset × metric) and optionally writes:
        {save_root}/master_summary.csv

    Returns
    -------
    master_df : pd.DataFrame
        Columns include: dataset, metric, t_star_mean/std, test_mean/std, etc.

    details_by_metric : dict
        Mapping metric -> details dict returned by run_all_datasets.
    """
    details_by_metric: Dict[str, Any] = {}
    master_rows: List[pd.DataFrame] = []

    def _log(msg: str):
        if verbose:
            print(msg)

    if save_root is not None:
        os.makedirs(save_root, exist_ok=True)

    for m in metrics:
        _log(f"\n==============================")
        _log(f"Running metric: {m}")
        _log(f"==============================")

        metric_dir = os.path.join(save_root, m) if save_root is not None else None

        summary_df, details = run_all_datasets(
            paths=paths,
            metric=m,
            decision_threshold=decision_threshold,
            fit_frac=fit_frac,
            search_frac=search_frac,
            test_frac=test_frac,
            min_group_size=min_group_size,
            n_repeats=n_repeats,
            base_seed=base_seed,
            save_dir=metric_dir,
            verbose=verbose,
            run_synthetic=run_synthetic,
            synthetic_n=synthetic_n,
            synthetic_true_boundary=synthetic_true_boundary,
            synthetic_bias_strength=synthetic_bias_strength,
        )

        # Ensure the metric column is correct (some summaries already include it)
        summary_df = summary_df.copy()
        summary_df["metric"] = m

        master_rows.append(summary_df)
        details_by_metric[m] = details

    master_df = pd.concat(master_rows, ignore_index=True).sort_values(["dataset", "metric"]).reset_index(drop=True)

    if save_root is not None and save_master_summary:
        master_path = os.path.join(save_root, "master_summary.csv")
        master_df.to_csv(master_path, index=False)
        _log(f"\nSaved master summary to: {master_path}")

    return master_df, details_by_metric

