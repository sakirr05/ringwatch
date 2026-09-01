"""Tests for the calibration diagnostics.

The construction below is the standard way to build a known-calibrated case: draw a
probability p uniformly, then draw the label from Bernoulli(p) and use p itself as the
score. By construction the score IS the true probability, so a correct implementation must
report a near-diagonal reliability curve. Distorting that score monotonically then gives a
case that is equally well *ranked* but badly calibrated — which is exactly the distinction
this module exists to measure, and which AUC-PR cannot see.
"""

from __future__ import annotations

import numpy as np

from core.calibration import calibration_report, expected_calibration_error


def make_calibrated(n: int = 20_000, seed: int = 0):
    """score == true probability, so the model is perfectly calibrated by construction."""
    rng = np.random.default_rng(seed)
    probability = rng.random(n)
    labels = (rng.random(n) < probability).astype(int)
    return labels, probability


def make_overconfident(n: int = 20_000, seed: int = 0):
    """Same labels and same ranking, but the scores are pushed toward 1.

    p ** 0.3 is monotonic, so the AUC-PR of these scores is identical to the calibrated
    case. Only the calibration changes. This is the case a ranking metric is blind to.
    """
    labels, probability = make_calibrated(n=n, seed=seed)
    return labels, probability**0.3


def test_calibrated_scores_lie_near_the_diagonal():
    labels, scores = make_calibrated()
    report = calibration_report("calibrated", labels, scores)

    assert report.ece < 0.02
    assert report.max_bin_error < 0.05
    # Every bin's observed rate should track its predicted rate.
    assert np.allclose(report.prob_true, report.prob_pred, atol=0.05)


def test_calibrated_scores_have_a_low_brier_score():
    labels, scores = make_calibrated()
    report = calibration_report("calibrated", labels, scores)

    # For p ~ U(0,1) with y ~ Bernoulli(p), the expected Brier score is E[p(1-p)] = 1/6.
    # That is the floor for this data-generating process, not zero -- a well-calibrated
    # model on genuinely uncertain data still carries irreducible error.
    assert 0.14 < report.brier < 0.19


def test_miscalibrated_scores_are_worse_on_both_metrics():
    """The headline assertion: distortion must be detected by ECE and by Brier."""
    calibrated_labels, calibrated_scores = make_calibrated()
    skewed_labels, skewed_scores = make_overconfident()

    good = calibration_report("calibrated", calibrated_labels, calibrated_scores)
    bad = calibration_report("overconfident", skewed_labels, skewed_scores)

    assert bad.ece > good.ece
    assert bad.brier > good.brier
    # The distortion is substantial, so it should be obvious, not marginal.
    assert bad.ece > 0.05


def test_ranking_is_unchanged_by_the_distortion():
    """Confirms the fixture isolates calibration from discrimination.

    If AUC-PR moved between the two cases, the test above would not be measuring
    calibration alone.
    """
    from sklearn.metrics import average_precision_score

    labels, calibrated_scores = make_calibrated()
    _, skewed_scores = make_overconfident()

    assert average_precision_score(labels, calibrated_scores) == (
        average_precision_score(labels, skewed_scores)
    )


def test_ece_is_weighted_by_bin_population():
    """A sparse, badly-off bin must not dominate a densely populated, accurate one."""
    prob_true = np.array([0.10, 0.90])
    prob_pred = np.array([0.10, 0.10])  # second bin is off by 0.8
    heavy_first = np.array([9_999.0, 1.0])

    weighted = expected_calibration_error(prob_true, prob_pred, heavy_first)
    unweighted = np.mean(np.abs(prob_true - prob_pred))

    assert weighted < 0.01
    assert unweighted == 0.4
    assert weighted < unweighted


def test_report_is_deterministic():
    labels, scores = make_calibrated()
    first = calibration_report("a", labels, scores)
    second = calibration_report("a", labels, scores)

    assert first.brier == second.brier
    assert first.ece == second.ece
    assert np.array_equal(first.prob_true, second.prob_true)


def test_summary_lines_flag_the_brier_caveat():
    """The caveat must travel with the number wherever it is printed."""
    labels, scores = make_calibrated(n=2_000)
    text = "\n".join(calibration_report("x", labels, scores).summary_lines())

    assert "Brier" in text
    assert "NOT a pure" in text
    assert "ECE" in text
