"""
Evaluation, interpretation, and artifact export helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    fbeta_score,
)

from .config import ARTIFACT_DIR, CONFIG, MODEL_FEATURES


def evaluate_split(
    pipeline,
    X,
    y,
    label: str,
    threshold: float | None = None,
    fbeta_beta: float | None = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute metrics, predictions, and probabilities for a dataset split.

    If `threshold` is provided, labels are derived from probabilities using that cutoff.
    Otherwise the pipeline's own `predict` is used.
    """
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int) if threshold is not None else pipeline.predict(X)
    metrics = {
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
        "average_precision": average_precision_score(y, proba),
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds),
    }
    if fbeta_beta is not None:
        metrics[f"f{fbeta_beta}"] = fbeta_score(y, preds, beta=fbeta_beta)
    print(f"\n=== {label.upper()} ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y, preds, digits=4))
    return metrics, preds, proba


def pick_threshold_by_fbeta(y_true, proba, beta: float = 2.0):
    """Return the threshold that maximizes F-beta on a validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    best_idx = fbeta.argmax()
    return float(thresholds[best_idx]), float(fbeta[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def apply_threshold(proba, threshold: float) -> np.ndarray:
    """Convert probabilities into class labels using the provided threshold."""
    return (np.asarray(proba) >= threshold).astype(int)


def confusion_matrix_dataframe(y_true, y_pred) -> pd.DataFrame:
    """Return a labelled confusion matrix dataframe."""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])


def plot_diagnostics(pipeline, X_test, y_test):
    """Plot ROC and PR curves for the provided split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[0])
    axes[0].set_title("Test ROC Curve")
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[1])
    axes[1].set_title("Test Precision-Recall Curve")
    plt.tight_layout()
    plt.show()


def permutation_importance_summary(pipeline, X_val, y_val) -> pd.Series:
    """Compute permutation importance over the validation split."""
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=5,
        random_state=CONFIG["random_state"],
        n_jobs=CONFIG["n_jobs"],
        scoring="roc_auc",
    )
    return pd.Series(result.importances_mean, index=MODEL_FEATURES).sort_values(ascending=False)


def aggregate_shap_by_feature(shap_values: np.ndarray, feature_names: np.ndarray) -> pd.Series:
    """Aggregate SHAP importances back to base feature names."""

    def base_feature(name: str) -> str:
        if name.startswith("num__"):
            return name.split("num__", 1)[1]
        if name.startswith("cat__"):
            remainder = name.split("cat__", 1)[1]
            return remainder.split("_", 1)[0]
        return name

    base_names = np.array([base_feature(name) for name in feature_names])
    unique_bases = np.unique(base_names)
    aggregated = []
    for base in unique_bases:
        mask = base_names == base
        aggregated.append(np.abs(shap_values[:, mask]).mean())
    return pd.Series(aggregated, index=unique_bases).sort_values(ascending=False)


def compute_shap_importances(pipeline, X: pd.DataFrame, max_samples: int = 2000):
    """Compute SHAP values for a sample of X and return aggregated importances."""
    sample = X.sample(min(max_samples, len(X)), random_state=CONFIG["random_state"])
    transformed = pipeline.named_steps["preprocessor"].transform(sample)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.array(shap_values)
    shap_agg = aggregate_shap_by_feature(shap_values, np.array(feature_names))
    return shap_agg, shap_values, transformed, feature_names


def save_metrics(metrics: Dict[str, Dict[str, float]], filename: str = "metrics.json") -> Path:
    """Persist evaluation metrics to JSON."""
    path = ARTIFACT_DIR / filename
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_best_params(params: Dict[str, object], filename: str = "best_params.json") -> Path:
    """Persist tuned hyperparameters to JSON."""
    path = ARTIFACT_DIR / filename
    with path.open("w") as f:
        json.dump(params, f, indent=2)
    return path


def save_predictions(df: pd.DataFrame, filename: str = "test_predictions.csv") -> Path:
    """Persist prediction dataframe to CSV."""
    path = ARTIFACT_DIR / filename
    df.to_csv(path, index=False)
    return path


def save_feature_importances(series: pd.Series, filename: str) -> Path:
    """Persist feature importance series to CSV."""
    path = ARTIFACT_DIR / filename
    series.to_frame(series.name or "value").to_csv(path)
    return path


def save_pipeline(pipeline, filename: str = "xgb_pipeline.joblib") -> Path:
    """Persist the trained pipeline using joblib."""
    path = ARTIFACT_DIR / filename
    import joblib

    joblib.dump(pipeline, path)
    return path
