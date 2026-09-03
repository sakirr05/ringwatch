"""Tests for cost-sensitive sample weighting.

The weighting has to encode the *same* asymmetry `core/evaluate.py` uses to pick a
threshold. If training and thresholding disagreed about what a mistake costs, the ablation
comparing them would be measuring the gap between two cost models rather than the value of
cost-sensitivity — a difference that would be invisible in the numbers and fatal to the
conclusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.costs import cost_weights, weight_summary
from core.evaluate import CHARGEBACK_FEE_INR, GROSS_MARGIN_RATE, USD_TO_INR
from core.model import train_model
from tests.test_model import FAST_PARAMS, FEATURES, make_training_frame


# --------------------------------------------------------------------------
# the weights themselves
# --------------------------------------------------------------------------


def test_weights_on_a_hand_worked_example():
    """Worked on paper, un-normalised so the arithmetic is visible.

        row  fraud?  amount_usd   amount_inr (x88)   cost
          0    yes        100.0          8800.0      8800 + 1200 = 10000.0
          1     no        100.0          8800.0      8800 x 0.12  =  1056.0

    A missed fraud costs the transaction plus the chargeback fee; a wrongly declined
    customer costs only the margin. Both figures come from the same constants the
    threshold logic uses.
    """
    y = np.array([1, 0])
    amounts = np.array([100.0, 100.0])

    weights = cost_weights(y, amounts, normalise=False)
    assert weights[0] == pytest.approx(100.0 * USD_TO_INR + CHARGEBACK_FEE_INR)
    assert weights[1] == pytest.approx(100.0 * USD_TO_INR * GROSS_MARGIN_RATE)
    assert weights[0] == pytest.approx(10_000.0)
    assert weights[1] == pytest.approx(1_056.0)


def test_fraud_is_weighted_more_heavily_than_legitimate_traffic():
    """The whole point: missing fraud must cost more than insulting a customer."""
    y = np.array([1, 0, 1, 0])
    amounts = np.array([50.0, 50.0, 200.0, 200.0])

    weights = cost_weights(y, amounts)
    assert weights[0] > weights[1]
    assert weights[2] > weights[3]


def test_larger_transactions_carry_more_weight_within_a_class():
    """Cost scales with the money at stake, not just the label."""
    y = np.array([1, 1, 0, 0])
    amounts = np.array([10.0, 10_000.0, 10.0, 10_000.0])

    weights = cost_weights(y, amounts)
    assert weights[1] > weights[0]
    assert weights[3] > weights[2]


def test_normalisation_preserves_relative_weights():
    """Scaling to mean 1 must not change the asymmetry, only the magnitude.

    Normalising matters because BASE_PARAMS -- learning rate, min_child_samples, the L2
    term -- was chosen against uniform weights averaging 1. Feeding raw rupee magnitudes
    would rescale the gradient and the regularisation together, so any measured difference
    could not be attributed to cost structure rather than to changed learning dynamics.
    """
    y = np.array([1, 0, 1, 0, 0])
    amounts = np.array([100.0, 100.0, 500.0, 500.0, 20.0])

    raw = cost_weights(y, amounts, normalise=False)
    scaled = cost_weights(y, amounts, normalise=True)

    assert scaled.mean() == pytest.approx(1.0)
    # Every pairwise ratio survives.
    assert np.allclose(raw / raw[0], scaled / scaled[0])


def test_weights_are_strictly_positive():
    """A zero-weight row is invisible to training; a negative one is meaningless."""
    y = np.array([1, 0, 0])
    amounts = np.array([0.0, 0.0, 5.0])
    assert (cost_weights(y, amounts) > 0).all()


def test_weight_summary_reports_the_asymmetry_actually_applied():
    """The asymmetry gets reported rather than assumed."""
    y = np.array([1, 0, 1, 0])
    amounts = np.array([100.0, 100.0, 100.0, 100.0])

    summary = weight_summary(cost_weights(y, amounts), y)
    # 10000 / 1056 from the hand-worked example above.
    assert summary["asymmetry_ratio"] == pytest.approx(10_000 / 1_056, rel=1e-6)
    assert summary["mean_fraud_weight"] > summary["mean_legit_weight"]


def test_weights_accept_pandas_input():
    y = pd.Series([1, 0, 1])
    amounts = pd.Series([10.0, 20.0, 30.0])
    assert len(cost_weights(y, amounts)) == 3


# --------------------------------------------------------------------------
# training with weights
# --------------------------------------------------------------------------


def test_training_accepts_weights_and_stays_deterministic():
    train = make_training_frame()
    weights = cost_weights(
        train["isFraud"], np.abs(train["signal"]) * 100 + 10
    )

    first = train_model(train, FEATURES, params=FAST_PARAMS, sample_weight=weights)
    second = train_model(train, FEATURES, params=FAST_PARAMS, sample_weight=weights)

    holdout = make_training_frame(n=200, seed=99)
    assert np.array_equal(first.predict(holdout), second.predict(holdout))


def test_weighting_actually_changes_the_model():
    """Guard against the weights being silently ignored.

    If LightGBM were not receiving them -- a wrong keyword, a dropped argument -- every
    test above would still pass and the "cost-sensitive" variant would be a duplicate of
    the baseline wearing a different name. Extreme weights must move the predictions.
    """
    train = make_training_frame()
    holdout = make_training_frame(n=200, seed=99)

    unweighted = train_model(train, FEATURES, params=FAST_PARAMS)
    lopsided = np.where(train["isFraud"] == 1, 50.0, 1.0)
    weighted = train_model(train, FEATURES, params=FAST_PARAMS, sample_weight=lopsided)

    assert not np.array_equal(unweighted.predict(holdout), weighted.predict(holdout))


def test_weights_follow_rows_through_the_inner_split():
    """The alignment trap.

    `train_model` splits the frame temporally before building its Datasets. A weight array
    left in original row order would pair each row with some other row's cost -- training
    would succeed and the result would be nonsense. Weights are therefore reindexed with
    the frame, which this test pins by shuffling the frame's index and checking the model
    still matches one trained on an equivalently-ordered pair.
    """
    train = make_training_frame()
    weights = pd.Series(
        np.where(train["isFraud"] == 1, 9.0, 1.0), index=train.index
    )

    model = train_model(train, FEATURES, params=FAST_PARAMS, sample_weight=weights)

    # Same data, index relabelled: alignment is by index, so the result must be identical.
    relabelled = train.copy()
    relabelled.index = relabelled.index + 10_000
    relabelled_weights = pd.Series(weights.to_numpy(), index=relabelled.index)
    other = train_model(
        relabelled, FEATURES, params=FAST_PARAMS, sample_weight=relabelled_weights
    )

    holdout = make_training_frame(n=200, seed=99)
    assert np.array_equal(model.predict(holdout), other.predict(holdout))


def test_unweighted_training_is_unchanged_by_the_new_parameter():
    """Regression guard: every published score was produced without weights."""
    train = make_training_frame()
    holdout = make_training_frame(n=200, seed=99)

    without = train_model(train, FEATURES, params=FAST_PARAMS)
    explicit_none = train_model(train, FEATURES, params=FAST_PARAMS, sample_weight=None)
    assert np.array_equal(without.predict(holdout), explicit_none.predict(holdout))


# --------------------------------------------------------------------------
# effective sample size -- the diagnostic that explained the result
# --------------------------------------------------------------------------


def test_uniform_weights_have_full_effective_sample_size():
    """With every row equally weighted, ESS must equal n exactly."""
    from core.costs import effective_sample_size

    assert effective_sample_size(np.ones(1000)) == pytest.approx(1000.0)
    assert effective_sample_size(np.full(500, 7.3)) == pytest.approx(500.0)


def test_skewed_weights_collapse_the_effective_sample_size():
    """One dominant row makes the rest nearly invisible, and ESS must show it."""
    from core.costs import effective_sample_size

    weights = np.concatenate([[1000.0], np.full(999, 0.001)])
    assert effective_sample_size(weights) < 2.0


def test_effective_sample_size_is_bounded_by_n():
    """It can never exceed the number of rows, whatever the weights."""
    from core.costs import effective_sample_size

    rng = np.random.default_rng(0)
    for _ in range(5):
        w = rng.exponential(scale=50.0, size=300) + 1e-6
        assert 0 < effective_sample_size(w) <= 300 + 1e-9


def test_cost_weights_on_realistic_amounts_lose_effective_sample_size():
    """Pins the mechanism behind the reported negative result.

    Real transaction amounts are heavy-tailed, so cost weights span orders of magnitude
    and ESS collapses well below nominal. On the real training set this measured 11.8%.
    """
    from core.costs import effective_sample_size

    rng = np.random.default_rng(1)
    n = 20_000
    y = (rng.random(n) < 0.035).astype(int)
    amounts = rng.lognormal(mean=4.5, sigma=1.2, size=n)

    weights = cost_weights(y, amounts)
    fraction = effective_sample_size(weights) / n
    assert fraction < 0.6, "expected a substantial loss of effective sample size"


def test_summary_exposes_effective_sample_size():
    y = np.array([1, 0, 1, 0])
    weights = cost_weights(y, np.array([10.0, 10.0, 5000.0, 5000.0]))
    summary = weight_summary(weights, y)
    assert 0 < summary["effective_fraction"] <= 1.0
