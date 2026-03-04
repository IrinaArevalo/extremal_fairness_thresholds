
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

# -----------------------------
# Model pipeline
# -----------------------------

def to_str_array(z):
    return z.astype(str)


def make_model_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(to_str_array, feature_names_out="one-to-one")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=4000, solver="lbfgs")

    return Pipeline([("pre", pre), ("clf", clf)])
