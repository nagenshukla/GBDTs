"""
Synthetic data generators with a KNOWN ground truth.

The whole point of this notebook series is that we plant the signal ourselves,
so we can later check whether each model recovered it. Every generated dataset
comes with a `GroundTruth` describing exactly how the target was built:

    y = 3.0*x0  -  2.0*x1  +  1.5*x2          (linear main effects)
        + 2.5*(x3 * x4)                       (a planted pairwise INTERACTION)
        + 2.0*sin(3*x5)                       (a planted NONLINEAR effect)
        + category_effect[cat]                (a categorical effect)
        + noise                               (irrelevant gaussian noise)

Features x6, x7, x8 are pure NOISE columns — they have zero true effect and a
good model / explanation should give them ~zero importance.

For classification, the same structured signal is passed through a logistic
link to produce class probabilities and then sampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Names of the feature columns we generate -------------------------------
NUMERIC_FEATURES = [f"x{i}" for i in range(9)]   # x0..x8
CATEGORICAL_FEATURE = "category"
ALL_FEATURES = NUMERIC_FEATURES + [CATEGORICAL_FEATURE]

# Effect attached to each category level (the "true" categorical signal).
CATEGORY_EFFECTS = {"A": 3.0, "B": -1.0, "C": 0.5, "D": -2.5}


@dataclass
class GroundTruth:
    """Everything we deliberately planted in the data."""
    linear_coefs: dict = field(default_factory=dict)      # feature -> coefficient
    interaction: tuple = ("x3", "x4")                      # planted interaction pair
    interaction_strength: float = 2.5
    nonlinear_feature: str = "x5"                          # 2*sin(3*x5)
    nonlinear_strength: float = 2.0
    category_effects: dict = field(default_factory=lambda: dict(CATEGORY_EFFECTS))
    noise_features: list = field(default_factory=lambda: ["x6", "x7", "x8"])
    noise_std: float = 1.0
    task: str = "regression"

    def describe(self) -> pd.DataFrame:
        """A tidy table of the true per-feature role, handy for display."""
        rows = []
        for f in NUMERIC_FEATURES:
            role, strength = "noise", 0.0
            if f in self.linear_coefs:
                role, strength = "linear", self.linear_coefs[f]
            if f in self.interaction:
                role, strength = "interaction", self.interaction_strength
            if f == self.nonlinear_feature:
                role, strength = "nonlinear", self.nonlinear_strength
            rows.append({"feature": f, "true_role": role, "strength": strength})
        rows.append({"feature": CATEGORICAL_FEATURE, "true_role": "categorical",
                     "strength": np.ptp(list(self.category_effects.values()))})
        return pd.DataFrame(rows)


def _structured_signal(df: pd.DataFrame, gt: GroundTruth) -> np.ndarray:
    """The deterministic (noise-free) part of the target."""
    signal = np.zeros(len(df))
    for feat, coef in gt.linear_coefs.items():
        signal += coef * df[feat].to_numpy()
    a, b = gt.interaction
    signal += gt.interaction_strength * df[a].to_numpy() * df[b].to_numpy()
    signal += gt.nonlinear_strength * np.sin(3.0 * df[gt.nonlinear_feature].to_numpy())
    signal += df[CATEGORICAL_FEATURE].map(gt.category_effects).to_numpy()
    return signal


def make_dataset(
    task: str = "regression",
    n_samples: int = 4000,
    noise_std: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, GroundTruth]:
    """
    Generate a synthetic dataset with a known ground truth.

    Parameters
    ----------
    task : "regression" or "classification"
    n_samples : number of rows
    noise_std : std-dev of the additive gaussian noise (signal-to-noise knob)
    seed : RNG seed for reproducibility

    Returns
    -------
    X : DataFrame  (x0..x8 numeric + 'category' string column)
    y : Series     (float target for regression, 0/1 for classification)
    gt : GroundTruth
    """
    rng = np.random.default_rng(seed)

    gt = GroundTruth(
        linear_coefs={"x0": 3.0, "x1": -2.0, "x2": 1.5},
        noise_std=noise_std,
        task=task,
    )

    # Numeric features ~ N(0, 1).
    X = pd.DataFrame(
        rng.standard_normal((n_samples, len(NUMERIC_FEATURES))),
        columns=NUMERIC_FEATURES,
    )
    # Categorical feature with a deliberately uneven class balance.
    levels = list(CATEGORY_EFFECTS.keys())
    X[CATEGORICAL_FEATURE] = rng.choice(levels, size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    X[CATEGORICAL_FEATURE] = X[CATEGORICAL_FEATURE].astype("category")

    signal = _structured_signal(X, gt)

    if task == "regression":
        y = signal + rng.normal(0.0, noise_std, size=n_samples)
        y = pd.Series(y, name="target")
    elif task == "classification":
        # Logistic link. Scale the signal so probabilities aren't saturated.
        logit = (signal - signal.mean()) / (signal.std() + 1e-9) * 1.6
        logit += rng.normal(0.0, noise_std * 0.5, size=n_samples)
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = pd.Series((rng.random(n_samples) < prob).astype(int), name="target")
    else:
        raise ValueError(f"Unknown task: {task!r}")

    return X, y, gt


def encode_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode the 'category' column (for models that need numeric input,
    e.g. the from-scratch GBDT and plain XGBoost). Returns a copy.
    """
    X = X.copy()
    X[CATEGORICAL_FEATURE] = X[CATEGORICAL_FEATURE].cat.codes.astype("int32")
    return X


def train_test_split_df(X, y, test_size=0.25, seed=42):
    """Thin wrapper around sklearn's split that preserves DataFrame dtypes."""
    from sklearn.model_selection import train_test_split

    stratify = y if y.nunique() <= 10 else None
    return train_test_split(X, y, test_size=test_size, random_state=seed,
                            stratify=stratify)
