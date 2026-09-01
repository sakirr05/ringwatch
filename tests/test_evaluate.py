"""Tests for the evaluation harness.

WHY THIS FILE EXISTS
--------------------
`core/evaluate.py` produces every number the submission's honesty claim rests on: AUC-PR,
the paired bootstrap that decides whether a difference is real, and the entire
insult-rate costing. Until now it had no tests at all — the project asserted rigour while
its most load-bearing arithmetic was unverified, which is precisely the posture it
criticises elsewhere.

These tests are deliberately hand-worked rather than property-based. A confusion matrix
computed by a second implementation could repeat the same misunderstanding; a confusion
matrix computed on paper, with the working written into the test, cannot.
"""

from __future__ import annotations

import numpy as np

from core.evaluate import (
    CHARGEBACK_FEE_INR,
    GROSS_MARGIN_RATE,
    bootstrap_auc_pr_delta,
    cost_at_threshold,
)


def test_cost_at_threshold_confusion_matrix_on_known_example():
    """Hand-worked example. Every expected value below was computed on paper.

    Inputs (six transactions, three fraudulent):

        idx  y_true  y_score  amount_inr   predicted (>= 0.5)   outcome
        ---  ------  -------  ----------   ------------------   -------
          0     1      0.90       1000            True            TP
          1     1      0.30       2000            False           FN
          2     0      0.80       5000            True            FP
          3     0      0.20        300            False           TN
          4     0      0.10        400            False           TN
          5     1      0.95       1500            True            TP

    Confusion matrix:  tp = 2 (idx 0, 5)   fp = 1 (idx 2)
                       fn = 1 (idx 1)      tn = 2 (idx 3, 4)      total = 6 OK

    precision   = tp / (tp + fp) = 2 / 3
    recall      = tp / (tp + fn) = 2 / 3
    insult_rate = fp / n_legit   = 1 / 3      (three rows have y_true == 0)

    fraud_caught_inr = amounts[TP]                    = 1000 + 1500        = 2500
    fraud_missed_inr = amounts[FN] + fn * 1200        = 2000 + 1200        = 3200
    insult_cost_inr  = amounts[FP] * 0.12             = 5000 * 0.12        =  600
    total_cost_inr   = fraud_missed + insult_cost     = 3200 + 600         = 3800
    """
    y_true = np.array([1, 1, 0, 0, 0, 1])
    y_score = np.array([0.90, 0.30, 0.80, 0.20, 0.10, 0.95])
    amounts_inr = np.array([1000.0, 2000.0, 5000.0, 300.0, 400.0, 1500.0])

    report = cost_at_threshold(y_true, y_score, amounts_inr, threshold=0.5)

    assert report.true_positives == 2
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.true_negatives == 2
    # The four cells must account for every row, or something is being double-counted.
    assert (
        report.true_positives
        + report.false_positives
        + report.false_negatives
        + report.true_negatives
        == len(y_true)
    )

    assert report.precision == 2 / 3
    assert report.recall == 2 / 3
    assert report.insult_rate == 1 / 3

    assert report.fraud_caught_inr == 2500.0
    assert report.fraud_missed_inr == 3200.0
    assert report.insult_cost_inr == 600.0
    assert report.total_cost_inr == 3800.0

    # Guard the constants the arithmetic above assumes. If someone changes the economic
    # assumptions, this test should fail loudly rather than silently check stale numbers.
    assert CHARGEBACK_FEE_INR == 1_200.0
    assert GROSS_MARGIN_RATE == 0.12


def test_threshold_is_inclusive_at_the_boundary():
    """`>=` not `>`: a score exactly equal to the threshold is a decline."""
    y_true = np.array([1, 0])
    y_score = np.array([0.5, 0.5])
    amounts = np.array([100.0, 100.0])

    report = cost_at_threshold(y_true, y_score, amounts, threshold=0.5)
    assert report.true_positives == 1
    assert report.false_positives == 1
    assert report.false_negatives == 0


def test_declining_nothing_catches_no_fraud():
    """A threshold above every score: no declines, so no insults and no fraud caught."""
    y_true = np.array([1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.3])
    amounts = np.array([100.0, 200.0, 300.0])

    report = cost_at_threshold(y_true, y_score, amounts, threshold=0.99)
    assert report.true_positives == 0
    assert report.false_positives == 0
    assert report.insult_rate == 0.0
    assert report.insult_cost_inr == 0.0
    # Both frauds missed: their value plus two chargeback fees.
    assert report.fraud_missed_inr == 100.0 + 300.0 + 2 * CHARGEBACK_FEE_INR


def test_bootstrap_delta_is_zero_when_scores_are_identical():
    """The null case: comparing a model against itself must yield exactly no difference.

    Each resample computes average_precision(y[idx], scores[idx]) minus the very same
    quantity, so every bootstrap draw is exactly 0.0 and the interval collapses onto zero.
    If this ever reports a non-zero delta or a non-degenerate interval, the resampling is
    not paired — the two models are being evaluated on different rows — and every CI the
    project reports would be too wide.
    """
    rng = np.random.default_rng(0)
    n = 500
    y_true = (rng.random(n) < 0.05).astype(int)
    scores = rng.random(n)

    delta = bootstrap_auc_pr_delta(
        y_true, scores, scores, name="self-comparison", n_resamples=50, seed=0
    )

    epsilon = 1e-12
    assert abs(delta.delta) < epsilon
    assert abs(delta.ci_low) < epsilon
    assert abs(delta.ci_high) < epsilon
    assert not delta.significant


def test_bootstrap_detects_a_genuinely_better_model():
    """Sanity check in the other direction, so the null test above isn't the only signal.

    A score vector that perfectly ranks the positives must beat pure noise, and the
    interval should exclude zero.
    """
    rng = np.random.default_rng(1)
    n = 800
    y_true = (rng.random(n) < 0.1).astype(int)
    perfect = y_true.astype(float) + rng.normal(scale=0.01, size=n)
    noise = rng.random(n)

    delta = bootstrap_auc_pr_delta(
        y_true, noise, perfect, name="perfect vs noise", n_resamples=50, seed=0
    )

    assert delta.delta > 0
    assert delta.ci_low > 0
    assert delta.significant
    assert delta.verdict() == "SIGNIFICANTLY BETTER"


def test_delta_report_verdict_wording():
    """The verdict string is quoted in the README, so its wording is part of the contract."""
    rng = np.random.default_rng(2)
    y_true = (rng.random(400) < 0.1).astype(int)
    scores = rng.random(400)

    identical = bootstrap_auc_pr_delta(
        y_true, scores, scores, name="same", n_resamples=25, seed=0
    )
    assert identical.verdict() == "not significant (CI spans 0)"
