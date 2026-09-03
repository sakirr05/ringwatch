"""Drift diagnostics across the temporal test period.

WHY
---
The model is trained once on the first 141 days and evaluated on the next 42. A single
aggregate score over that whole span hides whether performance is steady or decaying — and
decay is the normal condition for a deployed fraud model, because fraud adapts and
population mix moves. Splitting the held-out period into sequential windows shows which.

This pairs directly with the calibration finding. The model is systematically
under-confident overall; the question this module can answer is whether that
under-confidence is stable or gets worse as the test period runs on.

WHAT THIS IS NOT
----------------
A diagnostic, not a retraining pipeline. Nothing here triggers a refit, and no threshold
adapts. It measures and reports; acting on it would be a separate system with its own
failure modes.

THE HONEST CAVEAT, STATED BEFORE THE NUMBERS
--------------------------------------------
Six windows over 118,108 test rows gives roughly 20,000 rows and **~680 fraud cases each**.
AUC-PR on 680 positives is noisy, so a window-to-window wobble of a few points is
indistinguishable from sampling variation. Every window's AUC-PR is therefore reported with
a bootstrap interval, and any trend claim has to survive those intervals overlapping.
Drawing a line through six noisy points and calling it decay is the failure mode this
caveat exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from core.calibration import calibration_report
from core.data import TARGET, TIME_COL

DEFAULT_WINDOWS = 6
PSI_BINS = 10

# Conventional PSI reading, from the credit-scoring literature where the measure
# originates. These are rules of thumb, not thresholds with a theoretical basis, and are
# labelled as such wherever they are displayed.
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25


def population_stability_index(
    reference: np.ndarray, comparison: np.ndarray, n_bins: int = PSI_BINS
) -> float:
    """PSI between two samples of one feature, binned on the reference's quantiles.

        PSI = sum over bins of (p_comparison - p_reference) * ln(p_comparison / p_reference)

    This is exactly the **Jeffreys divergence**, KL(a||b) + KL(b||a), which is what
    `tests/test_drift.py` validates it against using `scipy.stats.entropy` as an
    independent implementation. There is no PSI in any standard library, so that identity
    is what makes an oracle available at all rather than testing the function against a
    second copy of my own reasoning.

    Bins come from the REFERENCE quantiles, because the question is how far the new
    population has moved relative to the one the model was trained on. A consequence worth
    knowing: the formula is symmetric in its two arguments, but this function is NOT, since
    swapping them also swaps which sample defines the bin edges. That is intended -- the
    training distribution has to define the bins for the comparison to mean what it says --
    and `tests/test_drift.py` pins the behaviour rather than assuming it away. Empty bins would
    make the logarithm diverge, so both sides get a small floor; that biases PSI slightly
    downward for sparse bins, which errs toward under-reporting drift rather than
    inventing it.
    """
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]

    if reference.size == 0 or comparison.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 2:
        return 0.0  # a constant feature cannot drift

    edges[0], edges[-1] = -np.inf, np.inf

    reference_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    comparison_counts = np.histogram(comparison, bins=edges)[0].astype(np.float64)

    floor = 1e-6
    p_reference = np.maximum(reference_counts / reference_counts.sum(), floor)
    p_comparison = np.maximum(comparison_counts / comparison_counts.sum(), floor)

    return float(np.sum((p_comparison - p_reference) * np.log(p_comparison / p_reference)))


def psi_verdict(psi: float) -> str:
    """Conventional reading. Rules of thumb, not theory -- labelled as such on display."""
    if not np.isfinite(psi):
        return "undefined"
    if psi < PSI_NO_SHIFT:
        return "stable"
    if psi < PSI_MODERATE_SHIFT:
        return "moderate shift"
    return "significant shift"


@dataclass
class WindowMetrics:
    """One sequential slice of the held-out period."""

    index: int
    start_day: float
    end_day: float
    n_rows: int
    n_fraud: int
    prevalence: float
    auc_pr: float
    auc_pr_ci: tuple[float, float]
    auc_roc: float
    auc_roc_ci: tuple[float, float]
    brier: float
    ece: float
    psi: dict[str, float] = field(default_factory=dict)

    @property
    def lift_over_prevalence(self) -> float:
        """AUC-PR divided by the window's base rate. REPORTED, BUT NOT USED FOR TRENDS.

        A random classifier scores AUC-PR equal to prevalence, so AUC-PR moves with the base
        rate whether or not the model changed. Dividing it out looks like the obvious fix and
        is not one: AP scales with prevalence only when AP is SMALL. Measured on synthetic
        data with a fixed-quality ranker and prevalence rising 2% -> 6%:

            weak ranker  (AP 0.03-0.07):  lift 1.3x -> 1.2x   -- roughly flat, ratio works
            strong ranker(AP 0.60-0.75):  lift 29.9x -> 12.5x -- ratio badly over-corrects

        This project's AP is around 0.5, squarely in the regime where the ratio is wrong.
        So the number is kept for context and the trend question is answered with AUC-ROC,
        which is prevalence-invariant by construction rather than by approximation.
        """
        return self.auc_pr / self.prevalence if self.prevalence > 0 else float("nan")

    @property
    def worst_psi_feature(self) -> tuple[str, float]:
        if not self.psi:
            return ("none", float("nan"))
        name = max(self.psi, key=lambda k: self.psi[k])
        return (name, self.psi[name])


def _bootstrap_metric(
    y_true: np.ndarray, scores: np.ndarray, metric, n_resamples: int = 200, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Point estimate and percentile interval for one metric within one window.

    Not a comparison against another model, so this cannot reuse `bootstrap_delta` -- that
    measures a paired difference. Here the point is the width of a single estimate, which
    is what tells a reader whether a window-to-window wobble means anything at all.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = y_true[idx]
        if resampled.sum() == 0 or resampled.sum() == len(resampled):
            continue
        values.append(metric(resampled, scores[idx]))
    if not values:
        return (float("nan"), (float("nan"), float("nan")))
    low, high = np.percentile(values, [2.5, 97.5])
    return (float(metric(y_true, scores)), (float(low), float(high)))


def drift_report(
    test: pd.DataFrame,
    scores: np.ndarray,
    reference: pd.DataFrame,
    features: list[str],
    n_windows: int = DEFAULT_WINDOWS,
) -> list[WindowMetrics]:
    """Split the test period into sequential windows and measure each.

    Windows are cut on TIME, not on row count, so each one is a genuine calendar slice of
    the held-out period. Equal-count windows would silently vary in duration with traffic
    volume, and "performance in week 3" would stop meaning week 3.

    `reference` is the training frame: PSI compares each window against the distribution
    the model was actually fitted on.
    """
    times = test[TIME_COL].to_numpy()
    y_true = test[TARGET].to_numpy().astype(int)

    edges = np.linspace(times.min(), times.max() + 1, n_windows + 1)

    windows: list[WindowMetrics] = []
    for i in range(n_windows):
        mask = (times >= edges[i]) & (times < edges[i + 1])
        if mask.sum() == 0 or y_true[mask].sum() == 0:
            continue  # a window with no fraud makes AUC-PR undefined

        window_y = y_true[mask]
        window_scores = scores[mask]

        calibration = calibration_report(f"window {i + 1}", window_y, window_scores)
        roc, roc_ci = _bootstrap_metric(window_y, window_scores, roc_auc_score)

        psi = {
            feature: population_stability_index(
                reference[feature].to_numpy(dtype=np.float64, na_value=np.nan),
                test.loc[mask, feature].to_numpy(dtype=np.float64, na_value=np.nan),
            )
            for feature in features
            if feature in test.columns and feature in reference.columns
        }

        windows.append(
            WindowMetrics(
                index=i + 1,
                start_day=float(edges[i] / 86_400),
                end_day=float(edges[i + 1] / 86_400),
                n_rows=int(mask.sum()),
                n_fraud=int(window_y.sum()),
                prevalence=float(window_y.mean()),
                auc_pr=float(average_precision_score(window_y, window_scores)),
                auc_pr_ci=_bootstrap_metric(window_y, window_scores,
                                            average_precision_score)[1],
                auc_roc=roc,
                auc_roc_ci=roc_ci,
                brier=calibration.brier,
                ece=calibration.ece,
                psi=psi,
            )
        )
    return windows


def _intervals_disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] > b[1] or b[0] > a[1]


def trend_is_distinguishable_from_noise(windows: list[WindowMetrics]) -> bool:
    """Do the first and last windows' AUC-PR intervals fail to overlap?

    The guard against reading a trend into six noisy points. If the intervals overlap, the
    honest statement is "no measurable trend", however suggestive the point estimates look
    when plotted.

    NOTE: this is the RAW comparison and is confounded by prevalence. Use
    `prevalence_adjusted_trend` for the question anyone actually means.
    """
    if len(windows) < 2:
        return False
    return _intervals_disjoint(windows[0].auc_pr_ci, windows[-1].auc_pr_ci)


def ranking_quality_trend(windows: list[WindowMetrics]) -> bool:
    """Did the model's ranking quality actually change? Answered with AUC-ROC.

    THE ONE TO REPORT. AUC-ROC is invariant to class balance by construction, so it
    separates "the model changed" from "the fraud rate changed" without relying on an
    approximation. The raw AUC-PR comparison conflates the two, and dividing AP by
    prevalence -- the obvious correction -- is only valid at low AP and over-corrects badly
    at the ~0.5 this project operates at. Both of those were tried before landing here.
    """
    if len(windows) < 2:
        return False
    return _intervals_disjoint(windows[0].auc_roc_ci, windows[-1].auc_roc_ci)
