"""Tests for the precomputed threshold curve behind the dashboard slider.

The sweep exists so the browser never computes a confusion matrix. That makes its
correctness a *boundary* property, not just a numerical one: if these points disagreed with
`cost_at_threshold`, the dashboard would be showing a second, unvalidated implementation of
the costing model while claiming to render committed output.

So the first test is an exact agreement check against the function that produced both
published operating points, and the last group asserts the published points land exactly on
the curve rather than near it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.evaluate import (
    MAX_ACCEPTABLE_INSULT_RATE,
    choose_threshold_by_cost,
    choose_threshold_under_insult_cap,
    cost_at_threshold,
    threshold_sweep,
)


@pytest.fixture
def fixture_data():
    """A skewed, realistic-shaped problem: rare positives, scores bunched near zero."""
    rng = np.random.default_rng(0)
    n = 4000
    y_true = (rng.random(n) < 0.05).astype(int)
    # Fraud scores higher on average, with heavy overlap -- as in the real data.
    scores = rng.beta(1.2, 30, size=n)
    scores[y_true == 1] = rng.beta(2.5, 6, size=int(y_true.sum()))
    amounts = rng.lognormal(8.0, 1.1, size=n)
    return y_true, np.clip(scores, 0, 1), amounts


# --------------------------------------------------------------------------
# agreement with the single-threshold oracle
# --------------------------------------------------------------------------


def test_every_point_matches_cost_at_threshold_exactly(fixture_data):
    """The sweep must be the same function, evaluated many times. Nothing re-derived."""
    y_true, scores, amounts = fixture_data
    for point in threshold_sweep(y_true, scores, amounts, n_points=40):
        oracle = cost_at_threshold(y_true, scores, amounts, point.threshold)
        assert point == oracle


def test_the_sweep_is_deterministic(fixture_data):
    y_true, scores, amounts = fixture_data
    first = threshold_sweep(y_true, scores, amounts, n_points=30)
    second = threshold_sweep(y_true, scores, amounts, n_points=30)
    assert first == second


# --------------------------------------------------------------------------
# shape of the curve
# --------------------------------------------------------------------------


def test_thresholds_are_sorted_and_unique(fixture_data):
    """The slider indexes this array positionally; duplicates would be dead stops."""
    y_true, scores, amounts = fixture_data
    thresholds = [p.threshold for p in threshold_sweep(y_true, scores, amounts)]
    assert thresholds == sorted(thresholds)
    assert len(thresholds) == len(set(thresholds))


def test_raising_the_threshold_never_increases_what_is_declined(fixture_data):
    """Monotonicity: a stricter threshold cannot flag more rows, catch more fraud, or
    insult more customers. Anything else means the sweep is not a threshold sweep."""
    y_true, scores, amounts = fixture_data
    points = threshold_sweep(y_true, scores, amounts)
    for earlier, later in zip(points, points[1:]):
        assert later.true_positives <= earlier.true_positives
        assert later.false_positives <= earlier.false_positives
        assert later.recall <= earlier.recall + 1e-12
        assert later.insult_rate <= earlier.insult_rate + 1e-12


def test_the_confusion_matrix_totals_the_dataset_at_every_point(fixture_data):
    y_true, scores, amounts = fixture_data
    for p in threshold_sweep(y_true, scores, amounts, n_points=25):
        total = p.true_positives + p.false_positives + p.true_negatives + p.false_negatives
        assert total == len(y_true)


def test_cost_is_the_sum_of_its_two_declared_parts(fixture_data):
    """The dashboard shows the parts and the total; they must not be able to disagree."""
    y_true, scores, amounts = fixture_data
    for p in threshold_sweep(y_true, scores, amounts, n_points=25):
        assert p.total_cost_inr == pytest.approx(p.fraud_missed_inr + p.insult_cost_inr)


def test_n_points_controls_the_resolution(fixture_data):
    y_true, scores, amounts = fixture_data
    assert len(threshold_sweep(y_true, scores, amounts, n_points=20)) <= 20
    assert len(threshold_sweep(y_true, scores, amounts, n_points=200)) > 20


# --------------------------------------------------------------------------
# the published operating points land ON the curve
# --------------------------------------------------------------------------


def test_extra_thresholds_are_included_exactly(fixture_data):
    """A marked point sitting between two samples would contradict the table above it."""
    y_true, scores, amounts = fixture_data
    extras = [0.0438046535, 0.216658042]
    thresholds = [
        p.threshold for p in threshold_sweep(y_true, scores, amounts, extra_thresholds=extras)
    ]
    for extra in extras:
        assert extra in thresholds


def test_both_published_operating_points_are_reproduced_on_the_curve(fixture_data):
    """End to end: sweep, look the published thresholds up, get the published figures."""
    y_true, scores, amounts = fixture_data
    cost_point = choose_threshold_by_cost(y_true, scores, amounts)
    capped_point = choose_threshold_under_insult_cap(y_true, scores, amounts)

    points = threshold_sweep(
        y_true, scores, amounts,
        extra_thresholds=[cost_point.threshold, capped_point.threshold],
    )
    by_threshold = {p.threshold: p for p in points}

    assert by_threshold[cost_point.threshold] == cost_point
    assert by_threshold[capped_point.threshold] == capped_point


def test_the_insult_cap_partitions_the_curve(fixture_data):
    """The 'operationally unshippable' region must be identifiable from the data alone.

    The dashboard labels a dragged-to threshold unshippable when its insult rate exceeds
    the cap. That is only meaningful if the property is monotone in threshold — otherwise
    the warning would flicker on and off as the slider moves.
    """
    y_true, scores, amounts = fixture_data
    points = threshold_sweep(y_true, scores, amounts)
    unshippable = [p.insult_rate > MAX_ACCEPTABLE_INSULT_RATE for p in points]
    # Once it becomes shippable it must stay shippable, as the threshold only rises.
    assert unshippable == sorted(unshippable, reverse=True)


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------


def test_a_single_distinct_score_does_not_crash():
    y_true = np.array([0, 1, 0, 1])
    scores = np.full(4, 0.3)
    amounts = np.array([100.0, 200.0, 300.0, 400.0])
    points = threshold_sweep(y_true, scores, amounts, n_points=10)
    assert len(points) == 1  # every quantile collapses to the same threshold


def test_all_legitimate_traffic_gives_a_zero_insult_rate_only_when_nothing_is_declined():
    y_true = np.zeros(100, dtype=int)
    scores = np.linspace(0, 1, 100)
    amounts = np.full(100, 500.0)
    points = threshold_sweep(y_true, scores, amounts, n_points=20)
    assert all(p.true_positives == 0 for p in points)
    assert all(p.recall == 0.0 for p in points)  # no fraud to catch
