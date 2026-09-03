"""Tests for the drift diagnostics.

PSI has no implementation in any standard library, so there is no obvious oracle to check
it against — and testing it against a second copy of my own reasoning would prove nothing.
The way out is an identity: PSI is exactly the **Jeffreys divergence**, KL(a‖b) + KL(b‖a),
and `scipy.stats.entropy` is an independent implementation of KL. That makes a real oracle
available, and it is the same discipline used for k-core against networkx.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy

from core.data import TARGET, TIME_COL
from core.drift import (
    drift_report,
    ranking_quality_trend,
    population_stability_index,
    psi_verdict,
    trend_is_distinguishable_from_noise,
)


# --------------------------------------------------------------------------
# PSI against an independent implementation
# --------------------------------------------------------------------------


def test_psi_matches_jeffreys_divergence_from_scipy():
    """The oracle test. PSI = KL(a||b) + KL(b||a), computed by scipy independently."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        reference = rng.normal(size=5_000)
        comparison = rng.normal(loc=0.4, scale=1.3, size=5_000)

        psi = population_stability_index(reference, comparison, n_bins=10)

        # Rebuild the same binning to hand scipy the identical discrete distributions.
        edges = np.unique(np.percentile(reference, np.linspace(0, 100, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(comparison, bins=edges)[0].astype(float)
        b = np.histogram(reference, bins=edges)[0].astype(float)
        a = np.maximum(a / a.sum(), 1e-6)
        b = np.maximum(b / b.sum(), 1e-6)

        jeffreys = entropy(a, b) + entropy(b, a)
        assert psi == pytest.approx(jeffreys, rel=1e-9)


def test_identical_distributions_have_zero_psi():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=4_000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_symmetric_for_a_fixed_binning_but_not_under_argument_swap():
    """A subtlety worth pinning, because the obvious test is wrong.

    The *formula* sum (a-b) ln(a/b) is symmetric in a and b. The *implementation* is not,
    because it bins on the REFERENCE quantiles: swapping the arguments also swaps which
    sample defines the bin edges, so the two calls compare different discrete
    distributions and legitimately differ.

    That asymmetry is intended, not a defect. The question PSI answers in production is
    "how far has the incoming population moved relative to the one the model was fitted
    on", which requires the training distribution to define the bins. Binning on the new
    data would answer a different question.

    So the symmetry that must hold is symmetry *given fixed bins* — which is exactly what
    the scipy oracle test above verifies, since it builds one binning and compares both
    directions through it. Here we assert the swap difference is real but small for
    similar distributions, documenting the behaviour rather than pretending it away.
    """
    rng = np.random.default_rng(2)
    a = rng.normal(size=3_000)
    b = rng.normal(loc=0.6, size=3_000)

    forward = population_stability_index(a, b)
    reverse = population_stability_index(b, a)

    assert forward != reverse  # different binnings, so genuinely different numbers
    assert abs(forward - reverse) < 0.1 * max(forward, reverse)  # same order of magnitude

    # Symmetry given ONE binning, computed directly from the shared bin edges.
    edges = np.unique(np.percentile(a, np.linspace(0, 100, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    pa, pb = np.maximum(pa, 1e-6), np.maximum(pb, 1e-6)
    assert np.sum((pa - pb) * np.log(pa / pb)) == pytest.approx(
        np.sum((pb - pa) * np.log(pb / pa)), rel=1e-9
    )


def test_psi_grows_with_the_size_of_the_shift():
    """Monotonicity: a bigger move must not score as less drift."""
    rng = np.random.default_rng(3)
    reference = rng.normal(size=6_000)

    scores = [
        population_stability_index(reference, rng.normal(loc=shift, size=6_000))
        for shift in (0.0, 0.25, 0.75, 2.0)
    ]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]


def test_psi_is_never_negative():
    """A divergence cannot be negative; a negative value would mean a sign error."""
    rng = np.random.default_rng(4)
    for _ in range(10):
        a = rng.exponential(size=1_000)
        b = rng.exponential(scale=rng.uniform(0.5, 3.0), size=1_000)
        assert population_stability_index(a, b) >= -1e-12


def test_constant_feature_cannot_drift():
    """One unique value gives no bins to compare; report no drift, not a crash."""
    assert population_stability_index(np.full(500, 7.0), np.full(500, 7.0)) == 0.0


def test_psi_handles_nan_and_empty_input():
    rng = np.random.default_rng(5)
    with_nans = np.concatenate([rng.normal(size=500), np.full(50, np.nan)])
    assert np.isfinite(population_stability_index(with_nans, rng.normal(size=500)))
    assert np.isnan(population_stability_index(np.array([]), np.array([1.0, 2.0])))


def test_psi_verdicts_follow_the_conventional_bands():
    assert psi_verdict(0.05) == "stable"
    assert psi_verdict(0.15) == "moderate shift"
    assert psi_verdict(0.40) == "significant shift"
    assert psi_verdict(float("nan")) == "undefined"


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


def make_frame(n: int = 12_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(1.5 * signal - 2.5)))
    return pd.DataFrame(
        {
            TIME_COL: np.sort(rng.integers(0, 42 * 86_400, size=n)),
            TARGET: (rng.random(n) < probability).astype(int),
            "signal": signal,
            "noise": rng.normal(size=n),
        }
    )


def test_windows_partition_the_period_without_overlap():
    frame = make_frame()
    scores = 1.0 / (1.0 + np.exp(-frame["signal"].to_numpy()))
    windows = drift_report(frame, scores, frame, ["signal", "noise"], n_windows=6)

    assert len(windows) == 6
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.end_day <= later.start_day
    assert sum(w.n_rows for w in windows) == len(frame)


def test_windows_are_cut_on_time_not_row_count():
    """Equal-count windows would vary in duration with traffic, so 'week 3' would drift.

    Durations must be equal; row counts need not be.
    """
    frame = make_frame()
    scores = np.random.default_rng(0).random(len(frame))
    windows = drift_report(frame, scores, frame, ["signal"], n_windows=6)

    durations = [w.end_day - w.start_day for w in windows]
    assert max(durations) - min(durations) < 1e-6


def test_each_window_reports_an_interval_around_its_auc_pr():
    """Point estimates on ~680 positives are noisy; the interval is what makes that legible."""
    frame = make_frame()
    scores = 1.0 / (1.0 + np.exp(-frame["signal"].to_numpy()))
    for window in drift_report(frame, scores, frame, ["signal"], n_windows=6):
        low, high = window.auc_pr_ci
        assert low <= window.auc_pr <= high
        assert high > low  # a degenerate interval would hide the imprecision


def test_no_drift_is_reported_when_the_distribution_is_stable():
    """Reference and windows drawn from one distribution: PSI must stay in the stable band."""
    frame = make_frame()
    scores = np.random.default_rng(1).random(len(frame))
    windows = drift_report(frame, scores, frame, ["signal", "noise"], n_windows=4)

    for window in windows:
        assert all(value < 0.25 for value in window.psi.values())


def test_injected_drift_is_detected():
    """A feature that genuinely shifts late in the period must show elevated PSI."""
    frame = make_frame()
    n = len(frame)
    # Shift the last third of the period hard.
    frame.loc[frame.index[int(n * 0.66) :], "signal"] += 3.0

    scores = np.random.default_rng(2).random(n)
    reference = frame.iloc[: int(n * 0.33)]
    windows = drift_report(frame, scores, reference, ["signal"], n_windows=3)

    assert windows[-1].psi["signal"] > windows[0].psi["signal"]
    assert windows[-1].psi["signal"] > 0.25


def test_windows_without_fraud_are_skipped_not_crashed():
    """AUC-PR is undefined with no positives; drop the window rather than raise."""
    frame = make_frame()
    frame.loc[frame[TIME_COL] < 7 * 86_400, TARGET] = 0
    scores = np.random.default_rng(3).random(len(frame))

    windows = drift_report(frame, scores, frame, ["signal"], n_windows=6)
    assert all(w.n_fraud > 0 for w in windows)
    assert len(windows) < 6


def test_worst_psi_feature_is_identified():
    frame = make_frame()
    n = len(frame)
    frame.loc[frame.index[int(n * 0.5) :], "noise"] += 4.0
    scores = np.random.default_rng(4).random(n)

    windows = drift_report(
        frame, scores, frame.iloc[: int(n * 0.4)], ["signal", "noise"], n_windows=2
    )
    name, value = windows[-1].worst_psi_feature
    assert name == "noise"
    assert value > 0


# --------------------------------------------------------------------------
# the guard against reading a trend into noise
# --------------------------------------------------------------------------


def test_overlapping_intervals_are_not_a_trend():
    """The honest default: if the intervals overlap, there is no measurable trend."""
    frame = make_frame()
    scores = 1.0 / (1.0 + np.exp(-frame["signal"].to_numpy()))
    windows = drift_report(frame, scores, frame, ["signal"], n_windows=6)
    # Stationary data by construction, so no trend may be claimed.
    assert trend_is_distinguishable_from_noise(windows) is False


def test_a_real_collapse_is_distinguishable():
    """A model that genuinely stops working late must be detectable, or the guard is useless."""
    frame = make_frame()
    n = len(frame)
    good = 1.0 / (1.0 + np.exp(-frame["signal"].to_numpy()))
    scores = good.copy()
    late = frame.index[int(n * 0.5) :]
    scores[late] = np.random.default_rng(5).random(len(late))  # pure noise later on

    windows = drift_report(frame, scores, frame, ["signal"], n_windows=2)
    assert trend_is_distinguishable_from_noise(windows) is True


# --------------------------------------------------------------------------
# the prevalence confound
# --------------------------------------------------------------------------


def test_lift_is_reported_but_is_not_the_trend_measure():
    """Lift is kept for context only; the trend verdict deliberately does not use it.

    `lift_ci` was removed along with the ratio-based trend test, because dividing AP by
    prevalence is valid only at low AP and this project runs at ~0.5. The number itself is
    still worth showing beside the raw score.
    """
    frame = make_frame()
    scores = 1.0 / (1.0 + np.exp(-frame["signal"].to_numpy()))
    for window in drift_report(frame, scores, frame, ["signal"], n_windows=4):
        assert window.lift_over_prevalence == pytest.approx(
            window.auc_pr / window.prevalence
        )
    assert not hasattr(window, "lift_ci"), "the ratio-based interval should be gone"


def test_a_pure_prevalence_change_is_not_reported_as_a_model_trend():
    """The catch this encodes: a rising fraud rate must not read as improving performance.

    Scores carry identical information in every window; only the base rate moves. AUC-PR
    will rise anyway, because its floor IS prevalence. AUC-ROC must not, because it is
    invariant to class balance -- which is exactly why the trend verdict uses it.

    Dividing AP by prevalence was tried first and rejected: it holds only at low AP and
    over-corrects at the ~0.5 this project operates at. That is measured, not assumed --
    a fixed-quality strong ranker's lift falls 29.9x -> 12.5x as prevalence goes 2% -> 6%.
    """
    rng = np.random.default_rng(7)
    n = 24_000
    times = np.sort(rng.integers(0, 42 * 86_400, size=n))
    position = np.linspace(0.0, 1.0, n)
    y = (rng.random(n) < (0.02 + 0.04 * position)).astype(int)
    scores = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))

    frame = pd.DataFrame({TIME_COL: times, TARGET: y, "signal": rng.normal(size=n)})
    windows = drift_report(frame, scores, frame, ["signal"], n_windows=4)

    assert windows[-1].prevalence > windows[0].prevalence * 1.5  # base rate really moved
    assert not ranking_quality_trend(windows), (
        "a pure prevalence change was reported as a change in model performance"
    )


def test_a_genuine_degradation_still_registers():
    """The verdict must not be blind to real decay, or it is useless.

    Prevalence is held flat and the scores go to noise halfway through.
    """
    rng = np.random.default_rng(8)
    n = 24_000
    times = np.sort(rng.integers(0, 42 * 86_400, size=n))
    y = (rng.random(n) < 0.05).astype(int)

    scores = np.where(y == 1, rng.beta(6, 2, n), rng.beta(2, 6, n))
    late = times >= times[int(n * 0.5)]
    scores[late] = rng.random(late.sum())

    frame = pd.DataFrame({TIME_COL: times, TARGET: y, "signal": rng.normal(size=n)})
    windows = drift_report(frame, scores, frame, ["signal"], n_windows=2)

    assert ranking_quality_trend(windows) is True


def test_auc_roc_interval_brackets_its_point_estimate():
    frame = make_frame()
    scores = 1.0 / (1.0 + np.exp(-frame["signal"].to_numpy()))
    for window in drift_report(frame, scores, frame, ["signal"], n_windows=4):
        low, high = window.auc_roc_ci
        assert low <= window.auc_roc <= high
        assert 0.0 <= window.auc_roc <= 1.0
