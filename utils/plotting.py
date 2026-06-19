"""
Shared plotting helpers so the notebooks stay focused on ideas, not matplotlib
boilerplate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A consistent colour per library across all notebooks.
MODEL_COLORS = {
    "XGBoost": "#1f77b4",
    "LightGBM": "#2ca02c",
    "CatBoost": "#d62728",
    "EBM": "#9467bd",
    "Scratch GBDT": "#7f7f7f",
}


def plot_training_curve(history: dict, title: str = "Training curve", ax=None):
    """
    history : {label -> array of per-iteration metric values}
    Plots metric vs boosting iteration. Good for spotting overfitting and the
    effect of learning rate / early stopping.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for label, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=label)
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_importance_vs_truth(importances: dict, ground_truth, model_name="model",
                             ax=None):
    """
    Bar chart of model feature importances, coloured by the feature's TRUE role
    (linear / interaction / nonlinear / categorical / noise). This is the money
    plot: noise features should be near zero.

    importances : {feature_name -> importance_value}
    ground_truth : GroundTruth from utils.data
    """
    truth = ground_truth.describe().set_index("feature")["true_role"].to_dict()
    role_color = {
        "linear": "#1f77b4", "interaction": "#ff7f0e",
        "nonlinear": "#2ca02c", "categorical": "#9467bd", "noise": "#cccccc",
    }
    feats = list(importances.keys())
    vals = np.array([importances[f] for f in feats], dtype=float)
    if vals.sum() > 0:
        vals = vals / vals.sum()
    order = np.argsort(vals)
    feats = [feats[i] for i in order]
    vals = vals[order]
    colors = [role_color.get(truth.get(f, "noise"), "#cccccc") for f in feats]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.barh(feats, vals, color=colors)
    ax.set_xlabel("Normalized importance")
    ax.set_title(f"{model_name}: importance vs. true feature role")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in role_color.values()]
    ax.legend(handles, role_color.keys(), title="true role", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    return ax


def plot_pred_vs_actual(y_true, y_pred, title="Predicted vs actual", ax=None):
    """Scatter of predictions against truth with the y=x reference line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_partial_dependence_1d(model, X, feature, n_points=50, ax=None,
                               predict_fn=None):
    """
    Simple 1-D partial dependence: sweep `feature` across its range (holding the
    rest at their observed values) and average the model's prediction.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    grid = np.linspace(X[feature].min(), X[feature].max(), n_points)
    preds = []
    Xtmp = X.copy()
    pred = predict_fn if predict_fn is not None else model.predict
    for v in grid:
        Xtmp[feature] = v
        preds.append(np.mean(pred(Xtmp)))
    ax.plot(grid, preds, color="#1f77b4")
    ax.set_xlabel(feature)
    ax.set_ylabel("Avg. prediction")
    ax.set_title(f"Partial dependence: {feature}")
    ax.grid(alpha=0.3)
    return ax


def benchmark_bar(results: pd.DataFrame, metric_col: str, ax=None,
                  lower_is_better=False):
    """
    Horizontal bar chart comparing models on a single metric.
    results : DataFrame indexed by model name with a column `metric_col`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    s = results[metric_col].sort_values(ascending=lower_is_better)
    colors = [MODEL_COLORS.get(m, "#888888") for m in s.index]
    ax.barh(s.index, s.values, color=colors)
    ax.set_xlabel(metric_col)
    ax.set_title(f"Model comparison: {metric_col}")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.4g}", va="center", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    return ax
