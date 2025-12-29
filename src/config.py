"""
Project-level configuration, feature definitions, and helper utilities for the
Lending Club default risk pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

ACCEPTED_FILE_PATTERN = "accepted_*.csv"

MODEL_NUMERIC_FEATURES: List[str] = [
    "loan_amnt",
    "term_months",
    "int_rate_decimal",
    "installment",
    "emp_length_years",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util_decimal",
    "total_acc",
    "collections_12_mths_ex_med",
    "chargeoff_within_12_mths",
    "pub_rec_bankruptcies",
    "tax_liens",
    "fico_average",
    # "last_fico_average",
    "credit_history_years",
    "installment_to_income_ratio",
    "mths_since_recent_inq",
    "mort_acc",
    "tot_cur_bal",
    "total_rev_hi_lim",
]

MODEL_CATEGORICAL_FEATURES: List[str] = [
    # "grade",
    # "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

MODEL_FEATURES: List[str] = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES

RAW_FEATURES: List[str] = sorted(
    {
        "loan_status",
        "issue_d",
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "emp_length",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "collections_12_mths_ex_med",
        "chargeoff_within_12_mths",
        "pub_rec_bankruptcies",
        "tax_liens",
        "fico_range_low",
        "fico_range_high",
        "last_fico_range_low",
        "last_fico_range_high",
        "earliest_cr_line",
        "mths_since_recent_inq",
        "mort_acc",
        "tot_cur_bal",
        "total_rev_hi_lim",
        "grade",
        "sub_grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "addr_state",
        "application_type",
    }
)

CONFIG: Dict[str, object] = {
    "target_column": "is_default",
    "date_column": "issue_d",
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "max_training_rows": 600_000,
    "tuning_sample_size": 250_000,
    "time_series_splits": 3,
    "n_iter": 15,
    "random_state": 42,
    "n_jobs": max(os.cpu_count() - 1, 1),
    "enable_shap": True,
    "model_params": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": max(os.cpu_count() - 1, 1),
    },
    "xgb_param_distributions": {
        "classifier__n_estimators": [300, 400, 500, 600],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__gamma": [0.0, 0.1, 0.3],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__reg_lambda": [0.5, 1.0, 2.0],
        "classifier__reg_alpha": [0.0, 0.5, 1.0],
    },
    "lightgbm_params": {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": max(os.cpu_count() - 1, 1),
    },
    "lightgbm_param_distributions": {
        "classifier__n_estimators": [300, 400, 500, 600],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__num_leaves": [15, 31, 63, 127],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__reg_lambda": [0.5, 1.0, 2.0],
        "classifier__reg_alpha": [0.0, 0.5, 1.0],
        "classifier__min_child_samples": [10, 20, 40, 60],
    },
}


def get_accepted_data_path() -> Path:
    """Return the first accepted loan CSV path matching the configured pattern."""
    matches = sorted(DATA_DIR.glob(ACCEPTED_FILE_PATTERN))
    if not matches:
        raise FileNotFoundError(
            f"No files matching pattern '{ACCEPTED_FILE_PATTERN}' found in {DATA_DIR}"
        )
    return matches[0]
