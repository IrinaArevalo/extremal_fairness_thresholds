"""
Train a model for every dataset, generate predictions, and save all data needed
to plot fairness sensitivity curves later — WITHOUT retraining.

What it saves per dataset:
- X (features) as parquet
- y (labels) as csv
- age (sensitive attribute) as csv
- y_score = model predicted probabilities as .npy
- (optional) a fitted model via joblib
- (optional) precomputed sensitivity curves for a list of metrics as csv

Dependencies:
  pip install numpy pandas scikit-learn pyarrow joblib

Assumes you already have these loaders in your codebase:
- load_uci_german_data(path)
- load_adult_uci(path)
- load_taiwan_credit_xls(path)
- load_compas_sqlite(path)
- load_give_me_some_credit_kaggle(path)
- make_synthetic_age_bias(...)
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump


from model import make_model_pipeline
from data import (
    load_uci_german_credit,
    load_adult_uci,
    load_taiwan_credit_xls,
    load_compas_sqlite,
    load_give_me_some_credit_kaggle,
    make_synthetic_age_bias,
)
from metrics import fairness_metric_at_threshold, candidate_thresholds

def compute_sensitivity_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    age: np.ndarray,
    metric: str,
    decision_threshold: float = 0.5,
    min_group_size: int = 50,
) -> pd.DataFrame:
    ages = np.asarray(age, dtype=float)
    ts = candidate_thresholds(ages)

    vals: list[float] = []
    n_left: list[int] = []
    n_right: list[int] = []

    for t in ts:
        left = ages < t
        right = ~left
        nL, nR = int(left.sum()), int(right.sum())
        n_left.append(nL)
        n_right.append(nR)

        if nL < min_group_size or nR < min_group_size:
            vals.append(np.nan)
            continue

        val = fairness_metric_at_threshold(
            y_true=y_true,
            y_score=y_score,
            age=ages,
            t=float(t),
            metric=metric,
            decision_threshold=decision_threshold,
        )

        if isinstance(val, (tuple, list, np.ndarray)):
            val = val[0]

        vals.append(float(val) if val is not None else np.nan)

    return pd.DataFrame({"t": ts, "phi": vals, "n_left": n_left, "n_right": n_right})


def train_and_save_dataset(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    age: pd.Series,
    out_dir: str,
    *,
    save_model: bool = True,
    save_features: bool = True,
    precompute_curves: bool = True,
    metrics_for_curves=None,
    decision_threshold: float = 0.5,
    min_group_size: int = 50,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)

    # ---- fit model
    model = make_model_pipeline(X)
    model.fit(X, y.to_numpy(dtype=int))
    y_score = model.predict_proba(X)[:, 1].astype(np.float64)

    # ---- save essentials for curves (always)
    y.to_csv(os.path.join(ds_dir, "y.csv"), index=False, header=True)
    age.to_csv(os.path.join(ds_dir, "age.csv"), index=False, header=True)
    np.save(os.path.join(ds_dir, "y_score.npy"), y_score)

    # ---- metadata dict (we’ll update it with what we actually saved)
    meta = {
        "dataset": name,
        "n": int(len(X)),
        "age_min": float(np.nanmin(age.to_numpy(dtype=float))),
        "age_max": float(np.nanmax(age.to_numpy(dtype=float))),
        "y_mean": float(y.mean()),
        "decision_threshold": decision_threshold,
        "min_group_size": min_group_size,
    }

    if save_features:
        try:
            X.to_parquet(os.path.join(ds_dir, "X.parquet"), index=False)
            meta["X_saved_as"] = "parquet"
        except Exception as e:
            X.to_pickle(os.path.join(ds_dir, "X.pkl"))
            meta["X_saved_as"] = f"pickle (parquet failed: {type(e).__name__})"
        # ---- save model (now picklable because no lambda)
        if save_model:
            dump(model, os.path.join(ds_dir, "model.joblib"))
            meta["model_saved_as"] = "joblib"

    # ---- write metadata
    with open(os.path.join(ds_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ---- precompute curves (optional)
    if precompute_curves:
        if metrics_for_curves is None:
            metrics_for_curves = [
                "equal_opportunity_difference",
                "statistical_parity_difference",
                "average_odds_difference",
                "auc_difference",
                "log_disparate_impact",
            ]
        curves_dir = os.path.join(ds_dir, "curves")
        os.makedirs(curves_dir, exist_ok=True)

        y_np = y.to_numpy(dtype=int)
        age_np = age.to_numpy(dtype=float)

        for m in metrics_for_curves:
            curve = compute_sensitivity_curve(
                y_true=y_np,
                y_score=y_score,
                age=age_np,
                metric=m,
                decision_threshold=decision_threshold,
                min_group_size=min_group_size,
            )
            curve.to_csv(os.path.join(curves_dir, f"{m}.csv"), index=False)



def train_and_save_all_datasets(
    paths: Dict[str, str],
    out_dir: str = "saved_predictions",
    *,
    include_synthetic: bool = True,
    synthetic_n: int = 20000,
    synthetic_true_boundary: float = 35.0,
    synthetic_bias_strength: float = 0.7,
    save_model: bool = True,
    save_features: bool = True,
    precompute_curves: bool = True,
    metrics_for_curves: Optional[List[str]] = None,
    decision_threshold: float = 0.5,
    min_group_size: int = 50,
) -> None:
    """
    Train on FULL data for each dataset and save:
    X, y, age, y_score, model, curves.

    The `paths` dict can include keys:
      - "german_uci"
      - "adult_uci"
      - "taiwan_xls"
      - "compas_db"
      - "give_me_credit_train"
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- German ---
    if "german_uci" in paths:
        X, y, age = load_uci_german_credit(paths["german_uci"])
        train_and_save_dataset(
            "german_uci", X, y, age, out_dir,
            save_model=save_model, save_features=save_features,
            precompute_curves=precompute_curves, metrics_for_curves=metrics_for_curves,
            decision_threshold=decision_threshold, min_group_size=min_group_size,
        )

    # --- Adult ---
    if "adult_uci" in paths:
        X, y, age = load_adult_uci(paths["adult_uci"])
        train_and_save_dataset(
            "adult_uci", X, y, age, out_dir,
            save_model=save_model, save_features=save_features,
            precompute_curves=precompute_curves, metrics_for_curves=metrics_for_curves,
            decision_threshold=decision_threshold, min_group_size=min_group_size,
        )

    # --- Taiwan ---
    if "taiwan_xls" in paths:
        X, y, age = load_taiwan_credit_xls(paths["taiwan_xls"])
        train_and_save_dataset(
            "taiwan_xls", X, y, age, out_dir,
            save_model=save_model, save_features=save_features,
            precompute_curves=precompute_curves, metrics_for_curves=metrics_for_curves,
            decision_threshold=decision_threshold, min_group_size=min_group_size,
        )

    # --- COMPAS ---
    if "compas_db" in paths:
        X, y, age = load_compas_sqlite(paths["compas_db"])
        train_and_save_dataset(
            "compas_db", X, y, age, out_dir,
            save_model=save_model, save_features=save_features,
            precompute_curves=precompute_curves, metrics_for_curves=metrics_for_curves,
            decision_threshold=decision_threshold, min_group_size=min_group_size,
        )

    # --- GiveMeCredit ---
    if "give_me_credit_train" in paths:
        X, y, age = load_give_me_some_credit_kaggle(paths["give_me_credit_train"])
        train_and_save_dataset(
            "give_me_credit", X, y, age, out_dir,
            save_model=save_model, save_features=save_features,
            precompute_curves=precompute_curves, metrics_for_curves=metrics_for_curves,
            decision_threshold=decision_threshold, min_group_size=min_group_size,
        )

    # --- Synthetic ---
    if include_synthetic:
        Xs, ys, ages = make_synthetic_age_bias(
            n=synthetic_n,
            true_boundary=synthetic_true_boundary,
            bias_strength=synthetic_bias_strength,
            seed=12345,  # fixed data generation seed
        )
        train_and_save_dataset(
            f"synthetic_tb{synthetic_true_boundary:g}_bs{synthetic_bias_strength:g}",
            Xs, ys, ages, out_dir,
            save_model=save_model, save_features=save_features,
            precompute_curves=precompute_curves, metrics_for_curves=metrics_for_curves,
            decision_threshold=decision_threshold, min_group_size=min_group_size,
        )

