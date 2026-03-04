from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import sqlite3


# =========================
# 0) UCI German Credit 
# =========================
UCI_GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
    "label",
]

def load_uci_german_credit(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    UCI german.data (space-separated). label: 1=good, 2=bad.
    Returns: X, y (good=1), age
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=UCI_GERMAN_COLUMNS)
    y = (df["label"].astype(int) == 1).astype(int)
    age = df["age"].astype(float)
    X = df.drop(columns=["label"])
    return X, y, age


# =========================
# 1) Adult (Census Income) via OpenML
# =========================
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_adult_uci(
    path: str = 'data/adult.data',
    drop_weight: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load UCI Adult dataset from adult.data (no header, comma-separated).

    income column:
        '>50K'  → favorable (1)
        '<=50K' → unfavorable (0)

    Returns
    -------
    X : pd.DataFrame
        Features (categorical + numeric)

    y : pd.Series
        Binary label (1 if income >50K)

    sensitive : pd.Series
        Age (continuous)

    Notes
    -----
    adult.data has spaces after commas → use skipinitialspace=True
    """
    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )

    # Drop rows with missing values (standard Adult preprocessing)
    df = df.dropna().reset_index(drop=True)

    # Label: income >50K
    y = (df["income"].str.strip() == ">50K").astype(int)

    sensitive = df["age"].astype(float)

    drop_cols = ["income"]
    if drop_weight:
        drop_cols.append("fnlwgt")  # sampling weight often removed in ML papers

    X = df.drop(columns=drop_cols)

    return X, y, sensitive


# =========================
# 2) Taiwanese Credit Default (UCI)
# =========================

def load_taiwan_credit_xls(
    path: str = 'data/default of credit card clients.xls',
    sheet_name: str = "Data",
    label_col: str = "default payment next month",
    age_col: str = "AGE",
    drop_id: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the UCI 'Default of Credit Card Clients' (Taiwan) dataset from the original .xls.

    This file has two header rows:
      - Row 0: variable codes (X1..X23, Y)
      - Row 1: actual column names
    so we use header=1.

    Parameters
    ----------
    path : str
        Path to the .xls file (e.g., 'default of credit card clients.xls').

    sheet_name : str, default="Data"
        Sheet name (in the UCI file it's typically "Data").

    label_col : str, default="default payment next month"
        Target column. In the UCI dataset:
          1 = default
          0 = no default

    age_col : str, default="AGE"
        Continuous sensitive attribute column.

    drop_id : bool, default=True
        Drop the 'ID' column from features if present.

    Returns
    -------
    X : pd.DataFrame
        Features (all columns except label, and optionally ID).

    y : pd.Series
        Binary label where 1 is the favorable outcome (NO default), 0 otherwise.

    sensitive : pd.Series
        AGE as float.

    Notes
    -----
    Reading .xls requires xlrd>=2.0.1:
      pip install xlrd
    """
    # header=1 uses the second row as the header (skips the X1..Y code row)
    df = pd.read_excel(path, sheet_name=sheet_name, header=1)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    if age_col not in df.columns:
        raise ValueError(f"Age column '{age_col}' not found. Columns: {list(df.columns)}")

    y_default = df[label_col].astype(int)          # 1=default, 0=no default
    y = (y_default == 0).astype(int)               # favorable outcome = no default

    sensitive = df[age_col].astype(float)

    drop_cols = [label_col]
    if drop_id and "ID" in df.columns:
        drop_cols.append("ID")

    X = df.drop(columns=drop_cols)
    return X, y, sensitive


# =========================
# 3) COMPAS (ProPublica-style CSV)
# =========================

def load_compas_sqlite(
    path: str = 'data/compas.db',
    table: str = "people",
    label_col: str = "is_recid",
    favorable_if_no_recid: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load COMPAS data from SQLite DB (e.g., compas.db).

    Uses the 'people' table by default, which contains:
        - age (continuous)
        - is_recid (1 recidivated, 0 did not)
        - demographics and priors

    Parameters
    ----------
    path : str
        Path to compas.db

    table : str, default="people"
        Table to load (usually 'people')

    label_col : str, default="is_recid"
        Binary recidivism indicator

    favorable_if_no_recid : bool, default=True
        If True: y=1 means no recidivism (favorable)
        If False: y=1 means recidivism

    Returns
    -------
    X : pd.DataFrame
        Features (excluding label)

    y : pd.Series
        Binary outcome

    sensitive : pd.Series
        Age (continuous)
    """
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()

    if "age" not in df.columns:
        raise ValueError(f"'age' column not found in {table}. Columns: {list(df.columns)}")

    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' not found in {table}. Columns: {list(df.columns)}")

    age = df["age"].astype(float)

    y_raw = df[label_col].astype(int)
    if favorable_if_no_recid:
        y = (y_raw == 0).astype(int)  # 1 = no recidivism
    else:
        y = y_raw

    X = df.drop(columns=[label_col])

    return X, y, age


# =========================
# 4) Give Me Some Credit (Kaggle) — local CSV
# =========================

def load_give_me_some_credit_kaggle(
    train_path: str = 'data/cs-training.csv',
    target_col: str = "SeriousDlqin2yrs",
    age_col: str = "age",
    drop_id: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load Kaggle 'Give Me Some Credit' training data (cs-training.csv).

    Target:
        SeriousDlqin2yrs
            1 = serious delinquency (bad)
            0 = no delinquency (good)

    We map:
        y = 1 if NO delinquency (favorable)
        y = 0 if delinquent

    Parameters
    ----------
    train_path : str
        Path to cs-training.csv

    target_col : str, default="SeriousDlqin2yrs"
        Target column name

    age_col : str, default="age"
        Continuous sensitive attribute

    drop_id : bool, default=True
        Drop unnamed ID column if present

    Returns
    -------
    X : pd.DataFrame
        Features

    y : pd.Series
        Binary label (1 = good credit)

    sensitive : pd.Series
        Age (continuous)
    """
    df = pd.read_csv(train_path)

    # Kaggle file has unnamed index column
    if drop_id and df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found. Columns: {list(df.columns)}")

    if age_col not in df.columns:
        raise ValueError(f"Age column '{age_col}' not found. Columns: {list(df.columns)}")

    y_bad = df[target_col].astype(int)
    y = (y_bad == 0).astype(int)  # favorable = no delinquency

    sensitive = df[age_col].astype(float)

    X = df.drop(columns=[target_col])

    return X, y, sensitive


# =========================
# 5) ACSIncome / Folktables
# =========================
def load_acs_income_folktables(
    states: Optional[list[str]] = None,
    year: int = 2018,
    horizon: str = "1-Year",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    ACSIncome task from Folktables.

    Requires:
      pip install folktables

    Output is already a ML task:
      y = 1 if income > 50k (favorable)
      sensitive = AGEp (continuous age in years)

    Notes:
    - If the Census API/data source is unavailable in your environment, Folktables may fail.
      In that case, you can use a cached/local copy or a dataset mirror.
    """
    try:
        from folktables import ACSDataSource, ACSIncome  # type: ignore
    except Exception as e:
        raise ImportError("Install folktables: pip install folktables") from e

    if states is None:
        # single state to keep it small by default (change to multiple for bigger experiments)
        states = ["CA"]

    data_source = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    acs_data = data_source.get_data(states=states, download=True)

    features, labels, _ = ACSIncome.df_to_pandas(acs_data)

    # Folktables uses AGEp for age (years)
    if "AGEp" not in features.columns:
        age_like = [c for c in features.columns if "age" in c.lower()]
        if not age_like:
            raise ValueError(f"Couldn't find AGEp in features. Columns: {list(features.columns)}")
        age_col = age_like[0]
    else:
        age_col = "AGEp"

    sensitive = features[age_col].astype(float)
    y = labels.astype(int)  # already 0/1 where 1 is income>50k
    X = features.copy()

    return X, y, sensitive


# =========================
# 6) Synthetic dataset with known bias boundary
# =========================
def make_synthetic_age_bias(
    n: int = 10000,
    true_boundary: float = 35.0,
    bias_strength: float = 0.7,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a synthetic binary classification task where the true unfairness boundary is known.

    Mechanism:
      - features: x1, x2 ~ N(0,1)
      - age ~ Uniform(18,70)
      - label probability depends on x1, x2 and age,
        with an additional shift for age < true_boundary to induce disparity.

    Returns:
      X with columns ['x1','x2','age']
      y in {0,1} (1 favorable)
      sensitive = age
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    age = rng.uniform(18, 70, size=n)

    # base logit + bias term
    logit = 0.8 * x1 + 0.6 * x2 + 0.02 * (age - 40)
    logit += -bias_strength * (age < true_boundary)  # depress favorable outcome for younger

    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)

    X = pd.DataFrame({"x1": x1, "x2": x2, "age": age})
    return X, pd.Series(y, name="y"), pd.Series(age, name="age")



@dataclass
class SingleRunResult:
    seed: int
    t_star: float
    search_value: float
    test_value: float

@dataclass
class RepeatedSummary:
    metric: str
    n_repeats: int
    t_star_mean: float
    t_star_std: float
    search_mean: float
    search_std: float
    test_mean: float
    test_std: float
    all_results: pd.DataFrame