"""Cost-sensitive sample weighting for training.

WHAT THIS CHANGES
-----------------
RingWatch already knows that a missed fraud and a wrongly-declined customer cost different
amounts — `core/evaluate.py` uses exactly that asymmetry to choose an operating threshold.
But the *model* has never known it. Training treats every row as equally important and the
cost only enters afterwards, when a threshold is picked from the finished scores.

This module lets the asymmetry enter during training instead, by weighting each row by what
getting it wrong would actually cost:

    a fraud row       ->  amount + chargeback fee     (the money lost by missing it)
    a legitimate row  ->  amount x gross margin       (the margin lost by declining it)

Those are the **same named constants** `core/evaluate.py` already uses. That matters: two
different cost models — one for training, one for thresholding — would be a quiet
inconsistency that made any comparison meaningless.

WHY THIS MIGHT NOT HELP, SAID IN ADVANCE
----------------------------------------
Weighting changes what the model optimises, not what it can see. If the features that
separate expensive fraud are the same ones that separate cheap fraud, reweighting buys
nothing and mostly adds variance. The value-weighted ablation already found that
graph-linked fraud carries 3.57% of value against 3.35% of count — barely any value
concentration — which is weak prior evidence that value structure in this dataset is thin.

So a null result is the expected outcome and will be reported as plainly as the graph
result was. The comparison worth watching is not AUC-PR but **total expected cost at the
insult-constrained operating point**, since cost is the thing being optimised: a variant
can lose on AUC-PR and still win on cost, and that would be the genuinely interesting case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.evaluate import CHARGEBACK_FEE_INR, GROSS_MARGIN_RATE, amount_inr


def cost_weights(
    y_true: np.ndarray | pd.Series,
    amounts_usd: np.ndarray | pd.Series,
    normalise: bool = True,
) -> np.ndarray:
    """Per-row misclassification cost, usable directly as a LightGBM `sample_weight`.

    NORMALISATION
    -------------
    Weights are scaled to mean 1.0 by default. LightGBM accepts any positive weights, but
    the learning rate, `min_child_samples` and the L2 term in `BASE_PARAMS` were all chosen
    against uniform weights averaging 1. Feeding raw rupee magnitudes (a mean around 12,000)
    would rescale the effective gradient and regularisation together, so a measured
    difference could not be attributed to the cost structure rather than to an accidental
    change in learning dynamics. Normalising keeps the *relative* asymmetry — which is the
    entire point — while leaving everything else comparable to the baseline.
    """
    y_true = np.asarray(y_true).astype(int)
    amounts = amount_inr(np.asarray(amounts_usd, dtype=np.float64))

    # Cost of getting each row wrong, in the same rupee terms as the threshold logic.
    missed_fraud = amounts + CHARGEBACK_FEE_INR
    wrongly_declined = amounts * GROSS_MARGIN_RATE
    weights = np.where(y_true == 1, missed_fraud, wrongly_declined)

    # Guard: a zero-weight row is invisible to training, and a negative one is meaningless.
    weights = np.clip(weights, a_min=1e-6, a_max=None)

    if normalise:
        mean = weights.mean()
        if mean > 0:
            weights = weights / mean
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2).

    THE DIAGNOSTIC THAT EXPLAINS THE RESULT
    ---------------------------------------
    Weighting does not add information, it redistributes attention -- and skewed weights
    redistribute it into a smaller effective dataset. This is the standard measure of how
    much: with uniform weights it equals n, and it collapses toward the count of heavy rows
    as the spread widens.

    On this data it lands at **55,864 against 472,432 nominal rows -- 11.8%**. Cost
    weighting discards roughly seven-eighths of the effective training set, because a
    legitimate row's weight is its amount times a 0.12 margin while a fraud row's is its
    amount plus a fixed fee, and that produces a 172,000x spread. The heaviest 1% of rows
    end up holding 20% of all the weight.

    That number is why the cost-sensitive variant lost on both axes, and it is worth
    computing before concluding anything about cost-sensitivity as an idea: what was
    measured here is one weighting scheme on one dataset destroying more information than
    its cost signal was worth, not a general verdict.
    """
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / (weights**2).sum())


def weight_summary(weights: np.ndarray, y_true: np.ndarray | pd.Series) -> dict:
    """Descriptive stats, so the asymmetry actually applied is reported rather than assumed."""
    y_true = np.asarray(y_true).astype(int)
    fraud = weights[y_true == 1]
    legit = weights[y_true == 0]
    return {
        "mean_weight": float(weights.mean()),
        "effective_sample_size": effective_sample_size(weights),
        "effective_fraction": effective_sample_size(weights) / len(weights) if len(weights) else 0.0,
        "mean_fraud_weight": float(fraud.mean()) if fraud.size else 0.0,
        "mean_legit_weight": float(legit.mean()) if legit.size else 0.0,
        "asymmetry_ratio": float(fraud.mean() / legit.mean()) if legit.size and legit.mean() else float("nan"),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
    }
