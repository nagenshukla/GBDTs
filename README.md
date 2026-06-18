# Understanding Gradient Boosted Decision Trees (GBDTs)

A hands-on notebook series that takes apart four of the best modern gradient
boosting libraries and shows **how they actually work**, not just how to call
`.fit()`.

The four models covered:

| Library | Headline idea | Why it's here |
|---|---|---|
| **XGBoost** | Second-order (Newton) boosting with strong regularization | The benchmark workhorse |
| **LightGBM** | Leaf-wise growth + histogram binning (GOSS / EFB) | Fastest on large data |
| **CatBoost** | Ordered boosting + native categorical handling + symmetric trees | Best out-of-the-box on categoricals |
| **EBM** (Explainable Boosting Machine) | Glass-box additive GA2M boosting | A modern, fully-interpretable GBDT |

## The core idea behind these notebooks

We **don't** use a messy real dataset. Instead we *generate* synthetic data
where **we define the true signal ourselves**: a few linear effects, one
deliberately planted feature interaction, additive noise, and several totally
useless features. Because we know the ground truth, every notebook can ask a
sharper question than "what's the accuracy?":

> **Did the model actually recover the signal we planted - and did it correctly
> ignore the noise features?**

That turns "interpretability" from hand-waving into something we can verify.

## Notebooks

Run them in order:

| # | Notebook | Focus |
|---|---|---|
| 00 | `00_setup_and_synthetic_data.ipynb` | Environment check, generate & visualize the synthetic ground truth |
| 01 | `01_boosting_from_scratch.ipynb` | Build a tiny GBDT by hand - residuals, learning rate, additive trees |
| 02 | `02_xgboost.ipynb` | Newton boosting, regularization, training curves, importance |
| 03 | `03_lightgbm.ipynb` | Leaf-wise growth, histogram binning, speed |
| 04 | `04_catboost.ipynb` | Ordered boosting, symmetric trees, native categoricals |
| 05 | `05_ebm_explainable_boosting.ipynb` | Glass-box explanations that recover our planted effects |
| 06 | `06_comparison_and_tuning.ipynb` | Head-to-head accuracy/speed, tuning sweeps, SHAP cross-comparison |

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Tested on Python 3.14. All four libraries install from prebuilt wheels - no
compiler needed.

## Layout

```
GBDTs/
├── requirements.txt
├── README.md
├── utils/
│   ├── data.py        # synthetic generators (regression + classification)
│   └── plotting.py    # shared training-curve / importance / SHAP plots
├── data/              # cached generated datasets (created by notebook 00)
└── notebooks/
```
