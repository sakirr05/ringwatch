"""Tests for value-weighted evaluation.

The most important test here is `test_value_weighting_can_change_the_ranking`. Everything
else checks the metric is consistent with the existing count-weighted one; that test checks
the metric is *capable of detecting the effect it is being used to look for*. Without it a
null result would be uninformative — you could not distinguish "there is no value effect"
from "this metric never moves".
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from core.evaluate import bootstrap_auc_pr_delta
from core.value_metrics import (
    value_concentration,
    value_detection_rate,
    value_weighted_average_precision,
)


# --------------------------------------------------------------------------
# consistency with the existing count-weighted metric
# --------------------------------------------------------------------------


def test_equal_weights_reproduce_the_unweighted_metric():
    """With uniform amounts, the value-weighted AP must equal the ordinary AP exactly.

    This is what makes the two numbers comparable: any difference between them on real data
    is attributable to the weighting, not to a different estimator.
    """
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    scores = rng.random(500)
    ones = np.ones(500)

    assert value_weighted_average_precision(y, scores, ones) == pytest.approx(
        average_precision_score(y, scores)
    )


def test_value_weighting_is_scale_invariant():
    """Multiplying every amount by a constant must change nothing.

    This is why no currency conversion is applied anywhere in this module: VDR and the
    value-weighted AP are ratios, so an exchange rate would cancel. The value-weighted
    result therefore rests on strictly fewer assumptions than the rupee costing.
    """
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.08).astype(int)
    scores = rng.random(400)
    amounts = rng.exponential(scale=120.0, size=400)

    base = value_weighted_average_precision(y, scores, amounts)
    for factor in (0.01, 88.0, 1_000.0):
        assert value_weighted_average_precision(y, scores, amounts * factor) == pytest.approx(
            base
        )


def test_negative_amounts_are_rejected():
    y = np.array([1, 0, 1])
    scores = np.array([0.9, 0.5, 0.1])
    with pytest.raises(ValueError, match="non-negative"):
        value_weighted_average_precision(y, scores, np.array([1.0, -5.0, 1.0]))


# --------------------------------------------------------------------------
# the metric must be able to detect what it is looking for
# --------------------------------------------------------------------------


def test_value_weighting_can_change_the_ranking():
    """A model that catches the EXPENSIVE frauds must beat one that catches cheap ones.

    Both models below have identical count-weighted AP by construction — each catches
    exactly one of two frauds. Only the value weighting can tell them apart. If this test
    fails, a null result from the real ablation would mean nothing.
    """
    # Two frauds — one cheap, one expensive — buried in 100 legitimate rows. The legitimate
    # bulk matters: with only a handful of rows, precision stays high even when the
    # expensive fraud is ranked last, and the weighting has nowhere to show itself.
    n_legit = 100
    y = np.array([1, 1] + [0] * n_legit)
    amounts = np.array([10.0, 10_000.0] + [50.0] * n_legit)

    middle = np.linspace(0.4, 0.6, n_legit)
    catches_cheap = np.concatenate([[0.99, 0.01], middle])
    catches_expensive = np.concatenate([[0.01, 0.99], middle])

    # Identical when every fraud counts the same.
    assert average_precision_score(y, catches_cheap) == pytest.approx(
        average_precision_score(y, catches_expensive)
    )

    # Very different once value matters.
    cheap = value_weighted_average_precision(y, catches_cheap, amounts)
    expensive = value_weighted_average_precision(y, catches_expensive, amounts)
    assert expensive > cheap + 0.2


# --------------------------------------------------------------------------
# value detection rate
# --------------------------------------------------------------------------


def test_value_detection_rate_on_a_hand_worked_example():
    """Worked on paper.

        idx  y  score  amount   caught at threshold 0.5?
          0   1   0.90    1000   yes  -> counted
          1   1   0.30    3000   no
          2   0   0.80    9999   yes, but legitimate -> not in either total
          3   1   0.70    1000   yes  -> counted

        total fraud value      = 1000 + 3000 + 1000 = 5000
        caught fraud value     = 1000 + 1000        = 2000
        VDR                    = 2000 / 5000        = 0.40

    Note the contrast with recall: 2 of 3 frauds caught is 0.667, but only 40% of the money.
    The gap between those two numbers is the entire reason this metric exists.
    """
    y = np.array([1, 1, 0, 1])
    scores = np.array([0.9, 0.3, 0.8, 0.7])
    amounts = np.array([1000.0, 3000.0, 9999.0, 1000.0])

    assert value_detection_rate(y, scores, amounts, threshold=0.5) == pytest.approx(0.40)


def test_vdr_is_one_when_everything_is_caught():
    y = np.array([1, 1, 0])
    scores = np.array([0.9, 0.8, 0.1])
    amounts = np.array([100.0, 900.0, 5.0])
    assert value_detection_rate(y, scores, amounts, threshold=0.5) == pytest.approx(1.0)


def test_vdr_is_zero_when_nothing_is_caught():
    y = np.array([1, 1, 0])
    scores = np.array([0.2, 0.1, 0.9])
    amounts = np.array([100.0, 900.0, 5.0])
    assert value_detection_rate(y, scores, amounts, threshold=0.5) == 0.0


def test_vdr_handles_a_set_with_no_fraud():
    """No fraud value present: return 0 rather than dividing by zero."""
    y = np.zeros(5, dtype=int)
    assert value_detection_rate(y, np.random.rand(5), np.ones(5), 0.5) == 0.0


# --------------------------------------------------------------------------
# value concentration — the mechanism check
# --------------------------------------------------------------------------


def test_concentration_detects_a_high_value_subgroup():
    """A subgroup holding a disproportionate share of fraud value must show enrichment > 1."""
    y = np.array([1, 1, 1, 1, 0, 0])
    amounts = np.array([9000.0, 9000.0, 100.0, 100.0, 500.0, 500.0])
    mask = np.array([True, True, False, False, True, False])

    result = value_concentration(y, amounts, mask, subgroup="rich")
    assert result.count_share == pytest.approx(0.5)
    assert result.value_share == pytest.approx(18_000 / 18_200)
    assert result.enrichment > 1.9


def test_concentration_reports_no_enrichment_when_there_is_none():
    """Equal amounts everywhere: value share must equal count share exactly."""
    y = np.array([1, 1, 1, 1, 0, 0])
    amounts = np.full(6, 250.0)
    mask = np.array([True, True, False, False, True, False])

    result = value_concentration(y, amounts, mask)
    assert result.value_share == pytest.approx(result.count_share)
    assert result.enrichment == pytest.approx(1.0)


def test_concentration_summary_mentions_enrichment():
    y = np.array([1, 0, 1])
    result = value_concentration(y, np.ones(3), np.array([True, False, False]))
    assert any("enrichment" in line for line in result.summary_lines())


# --------------------------------------------------------------------------
# weighted bootstrap
# --------------------------------------------------------------------------


def test_weighted_bootstrap_is_zero_for_identical_scores():
    """The null case must still hold once weights are involved."""
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.06).astype(int)
    scores = rng.random(400)
    amounts = rng.exponential(scale=100.0, size=400)

    delta = bootstrap_auc_pr_delta(
        y, scores, scores, name="self", n_resamples=40, seed=0, sample_weight=amounts
    )
    assert abs(delta.delta) < 1e-12
    assert not delta.significant


def test_unweighted_bootstrap_is_unchanged_by_the_new_parameter():
    """Regression guard: omitting sample_weight must reproduce the old behaviour exactly.

    Every confidence interval this project has already published was produced without
    weights. Adding the parameter must not move any of them.
    """
    rng = np.random.default_rng(4)
    y = (rng.random(300) < 0.1).astype(int)
    baseline = rng.random(300)
    variant = rng.random(300)

    without = bootstrap_auc_pr_delta(y, baseline, variant, name="a", n_resamples=30, seed=7)
    explicit_none = bootstrap_auc_pr_delta(
        y, baseline, variant, name="a", n_resamples=30, seed=7, sample_weight=None
    )
    assert without.delta == explicit_none.delta
    assert without.ci_low == explicit_none.ci_low
    assert without.ci_high == explicit_none.ci_high


def test_weighted_bootstrap_detects_a_value_effect():
    """Paired with the ranking test: the weighted bootstrap must fire on a real effect."""
    rng = np.random.default_rng(5)
    n = 600
    y = (rng.random(n) < 0.1).astype(int)
    amounts = np.where(y == 1, rng.exponential(5_000, n), rng.exponential(50, n)) + 1.0

    noise = rng.random(n)
    informed = y * 0.8 + rng.random(n) * 0.2

    delta = bootstrap_auc_pr_delta(
        y, noise, informed, name="informed", n_resamples=40, seed=0, sample_weight=amounts
    )
    assert delta.delta > 0
    assert delta.ci_low > 0
