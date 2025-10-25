"""
Data preparation and model training utilities for the Lending Club project.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG, MODEL_CATEGORICAL_FEATURES, MODEL_FEATURES, MODEL_NUMERIC_FEATURES

DEFAULT_BAD_STATUS_KEYWORDS = (
    "charged off",
    "default",
    "late (31-120 days)",
    "late (16-30 days)",
    "in grace period",
)

DEFAULT_GOOD_STATUS_KEYWORDS = ("fully paid",)


def label_loan_status(status: str) -> float:
    """Map Lending Club loan_status strings to a binary default label."""
    if not isinstance(status, str):
        return np.nan
    status_lower = status.strip().lower()
    if any(keyword in status_lower for keyword in DEFAULT_BAD_STATUS_KEYWORDS):
        return 1.0
    if any(keyword in status_lower for keyword in DEFAULT_GOOD_STATUS_KEYWORDS):
        return 0.0
    return np.nan


def convert_term_to_months(term: str) -> float:
    """Convert string terms like '36 months' into a numeric month count."""
    if pd.isna(term):
        return np.nan
    digits = "".join(ch for ch in str(term) if ch.isdigit())
    return float(digits) if digits else np.nan


def clean_emp_length(emp_length: str) -> float:
    """Normalize employment length strings into numeric year counts."""
    if pd.isna(emp_length):
        return np.nan
    text = str(emp_length).strip().lower()
    if text in {"nan", "n/a", ""}:
        return np.nan
    if text == "< 1 year":
        return 0.5
    if text in {"10+ years", "10 years"}:
        return 10.0
    digits = "".join(ch for ch in text if ch.isdigit())
    return float(digits) if digits else np.nan


def clean_percentage(series: pd.Series) -> pd.Series:
    """Convert percentage strings to decimal floats."""
    return (
        pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")
        / 100.0
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer additional features required by the modeling pipeline."""
    df = df.copy()
    df["term_months"] = df["term"].apply(convert_term_to_months)
    df["int_rate_decimal"] = clean_percentage(df["int_rate"])
    df["revol_util_decimal"] = clean_percentage(df["revol_util"])
    df["emp_length_years"] = df["emp_length"].apply(clean_emp_length)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line_dt"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (df["issue_d"] - df["earliest_cr_line_dt"]).dt.days / 365.25
    df["fico_average"] = df.loc[:, ["fico_range_low", "fico_range_high"]].mean(axis=1)
    df["last_fico_average"] = df.loc[:, ["last_fico_range_low", "last_fico_range_high"]].mean(axis=1)
    df["installment_to_income_ratio"] = df["installment"] / (df["annual_inc"] / 12)
    df["installment_to_income_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop(columns=["earliest_cr_line_dt"], inplace=True)
    return df


def prepare_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data, engineer features, and retain modeling columns."""
    work_df = df.copy()
    work_df["loan_status"] = work_df["loan_status"].astype(str)
    work_df[CONFIG["target_column"]] = work_df["loan_status"].apply(label_loan_status)
    work_df = work_df[work_df[CONFIG["target_column"]].isin([0, 1])].copy()
    work_df = engineer_features(work_df)

    required_columns = MODEL_FEATURES + [CONFIG["date_column"], CONFIG["target_column"]]
    missing = set(MODEL_FEATURES) - set(work_df.columns)
    if missing:
        raise KeyError(f"Missing engineered feature columns: {missing}")

    model_df = work_df[required_columns].copy()
    model_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    model_df.dropna(subset=[CONFIG["date_column"], CONFIG["target_column"]], inplace=True)
    model_df.sort_values(CONFIG["date_column"], inplace=True)
    model_df.reset_index(drop=True, inplace=True)
    return model_df


def time_based_split(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Timestamp]]:
    """Split dataset chronologically into train/validation/test partitions."""
    date_col = CONFIG["date_column"]
    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    total_n = len(df_sorted)
    train_end = int(total_n * CONFIG["train_fraction"])
    val_end = int(total_n * (CONFIG["train_fraction"] + CONFIG["val_fraction"]))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    if train.empty or val.empty or test.empty:
        raise ValueError("One of the data splits is empty. Adjust the fraction settings.")

    cutoffs = {
        "train_end": train.iloc[-1][date_col],
        "val_end": val.iloc[-1][date_col],
        "test_end": test.iloc[-1][date_col],
    }

    return {"train": train, "val": val, "test": test}, cutoffs


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Return class imbalance weight for XGBoost."""
    positive_rate = y.mean()
    if positive_rate <= 0:
        return 1.0
    return float((1 - positive_rate) / positive_rate)


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing transformer for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, MODEL_CATEGORICAL_FEATURES),
        ]
    )


def build_xgb_classifier(scale_pos_weight: float | None = None) -> xgb.XGBClassifier:
    """Instantiate the XGBoost classifier with optional class weight override."""
    params = CONFIG["model_params"].copy()
    if scale_pos_weight is not None:
        params = params.copy()
        params["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(**params)


def build_pipeline(scale_pos_weight: float | None = None) -> Pipeline:
    """Combine preprocessing and classifier into a single pipeline."""
    preprocessor = build_preprocessor()
    classifier = build_xgb_classifier(scale_pos_weight=scale_pos_weight)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def run_random_search(
    X,
    y,
    scale_pos_weight: float | None = None,
    param_distributions: Dict[str, list] | None = None,
) -> RandomizedSearchCV:
    """Perform time-series aware RandomizedSearchCV for pipeline tuning."""
    if param_distributions is None:
        param_distributions = CONFIG["xgb_param_distributions"]

    pipeline = build_pipeline(scale_pos_weight=scale_pos_weight)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=CONFIG["n_iter"],
        scoring="roc_auc",
        n_jobs=CONFIG["n_jobs"],
        cv=TimeSeriesSplit(n_splits=CONFIG["time_series_splits"]),
        verbose=1,
        random_state=CONFIG["random_state"],
        refit=True,
    )
    search.fit(X, y)
    return search


def build_lightgbm_classifier(scale_pos_weight: float | None = None) -> LGBMClassifier:
    """Instantiate a LightGBM classifier with optional class weight override."""
    params = CONFIG["lightgbm_params"].copy()
    if scale_pos_weight is not None:
        params = params.copy()
        params["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**params)


def build_lightgbm_pipeline(scale_pos_weight: float | None = None) -> Pipeline:
    """Pipeline wrapper that uses LightGBM as the estimator."""
    preprocessor = build_preprocessor()
    classifier = build_lightgbm_classifier(scale_pos_weight=scale_pos_weight)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def run_lightgbm_random_search(
    X,
    y,
    scale_pos_weight: float | None = None,
    param_distributions: Dict[str, list] | None = None,
) -> RandomizedSearchCV:
    """RandomizedSearchCV configured for LightGBM pipelines."""
    if param_distributions is None:
        param_distributions = CONFIG["lightgbm_param_distributions"]

    pipeline = build_lightgbm_pipeline(scale_pos_weight=scale_pos_weight)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=CONFIG["n_iter"],
        scoring="roc_auc",
        n_jobs=CONFIG["n_jobs"],
        cv=TimeSeriesSplit(n_splits=CONFIG["time_series_splits"]),
        verbose=1,
        random_state=CONFIG["random_state"],
        refit=True,
    )
    search.fit(X, y)
    return search
