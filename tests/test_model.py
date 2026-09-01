"""Tests for the classifier — determinism above all.

WHY DETERMINISM IS THE THING WORTH TESTING HERE
------------------------------------------------
RingWatch's headline finding is a *negative* result argued from small differences: the
graph layer moves AUC-PR by −0.0011 to −0.0064, and the project's claim is that some of
those deltas are noise and one of them is not. That argument collapses entirely if
training is itself nondeterministic, because then a −0.0064 difference between two
variants could be nothing more than two different runs of the same model.

So the ablation's credibility rests on this file. `core/model.py` sets `seed=42`,
`deterministic=True` and `force_row_wise=True` (LightGBM requires one of the force_*
options for its deterministic mode to actually hold); these tests confirm that
configuration does what it claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import TARGET, TIME_COL
from core.model import inner_temporal_validation_split, train_model

# Small model params so the test trains in about a second on 1,200 rows. The defaults in
# BASE_PARAMS (num_leaves=96, min_child_samples=100) are tuned for 470k rows and would
# produce a degenerate single-leaf model at this scale.
FAST_PARAMS = {"min_child_samples": 5, "num_leaves": 8, "learning_rate": 0.1}

FEATURES = ["signal", "noise"]


def make_training_frame(n: int = 1_200, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame shaped like the real one.

    TransactionDT must be sorted and spread out, because `train_model` carves its
    early-stopping validation set off the END of the training period by quantile. The
    positive class is generated from `signal` so that both sides of that inner split
    contain fraud — a validation set with no positives makes average_precision undefined
    and early stopping meaningless.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(1.6 * signal - 1.9)))
    return pd.DataFrame(
        {
            TIME_COL: np.arange(n, dtype=np.int64) * 3_600,
            TARGET: (rng.random(n) < probability).astype(int),
            "signal": signal,
            "noise": rng.normal(size=n),
        }
    )


def test_synthetic_frame_has_positives_on_both_sides_of_inner_split():
    """Guard: if this fails, the determinism tests below are testing a degenerate model."""
    frame = make_training_frame()
    fit, validation = inner_temporal_validation_split(frame)
    assert fit[TARGET].sum() > 0
    assert validation[TARGET].sum() > 0


def test_training_is_deterministic():
    """Two runs on identical data must produce byte-identical predictions.

    This is the assertion `core/model.py`'s docstring refers to.
    """
    train = make_training_frame()
    holdout = make_training_frame(n=300, seed=99)

    first = train_model(train, FEATURES, params=FAST_PARAMS)
    second = train_model(train, FEATURES, params=FAST_PARAMS)

    predictions_a = first.predict(holdout)
    predictions_b = second.predict(holdout)

    assert np.array_equal(predictions_a, predictions_b), (
        "LightGBM produced different predictions across two identical runs. The ablation's "
        "small deltas cannot be distinguished from run-to-run noise if this fails."
    )


def test_predictions_are_not_constant():
    """Keeps the determinism test from passing vacuously.

    A model that predicts one number for every row is trivially deterministic and would
    satisfy the test above while proving nothing.
    """
    train = make_training_frame()
    holdout = make_training_frame(n=300, seed=99)
    predictions = train_model(train, FEATURES, params=FAST_PARAMS).predict(holdout)

    assert predictions.std() > 0.0
    assert len(np.unique(predictions)) > 1


def test_training_metadata_is_reproducible():
    """Best iteration and validation score must also match, not just the predictions."""
    train = make_training_frame()
    first = train_model(train, FEATURES, params=FAST_PARAMS)
    second = train_model(train, FEATURES, params=FAST_PARAMS)

    assert first.best_iteration == second.best_iteration
    assert first.validation_ap == second.validation_ap


def test_inner_validation_split_has_no_leakage():
    """Early stopping must not see the future either.

    The outer temporal split protects the test set; this protects the model-selection
    decision. A random inner split would let the choice of boosting rounds be informed by
    later transactions than the ones being fitted.
    """
    frame = make_training_frame()
    fit, validation = inner_temporal_validation_split(frame)

    assert fit[TIME_COL].max() < validation[TIME_COL].min()
    assert len(fit) + len(validation) == len(frame)


def test_predict_uses_only_declared_features():
    """Extra columns in the frame must not disturb prediction."""
    train = make_training_frame()
    model = train_model(train, FEATURES, params=FAST_PARAMS)

    holdout = make_training_frame(n=300, seed=99)
    baseline = model.predict(holdout)

    noisy = holdout.copy()
    noisy["an_unrelated_column"] = np.arange(len(noisy))
    assert np.array_equal(model.predict(noisy), baseline)
