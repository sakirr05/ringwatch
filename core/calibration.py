"""Calibration diagnostics: are the predicted probabilities trustworthy as probabilities?

WHY THIS EXISTS SEPARATELY FROM AUC-PR
--------------------------------------
AUC-PR answers "does the model rank fraud above legitimate traffic?" It is invariant to
any monotonic transformation of the scores, so a model can have excellent AUC-PR while its
outputs are useless as probabilities — every score squashed into 0.01-0.05, say. That
distinction matters here specifically because `core/evaluate.py` chooses its operating
threshold by minimising expected rupee cost, and that computation treats the score as a
probability. If the probabilities are badly calibrated, the threshold is being placed by
arithmetic built on a number that does not mean what it claims to mean.

So this module asks the separate question: when the model says 0.30, does fraud actually
occur about 30% of the time?

WHY quantile BINNING, NOT uniform
---------------------------------
`calibration_curve` defaults to equal-width bins. At a 3.44% positive rate the predicted
probabilities pile up near zero, so equal-width bins put almost every row in the first bin
or two and leave the upper bins holding a handful of points each — producing a curve whose
right-hand end is dominated by noise. Quantile binning puts an equal number of rows in
each bin instead, which is the right choice for a skewed score distribution and makes each
plotted point carry comparable weight.

ON READING THE BRIER SCORE
--------------------------
The Brier score is reported because it is standard, but it is not a pure calibration
metric: by its decomposition it conflates calibration with discrimination/refinement, so a
model can improve its Brier score by getting better at separating classes without becoming
any better calibrated. Expected Calibration Error is reported alongside it for that reason
-- ECE measures only the vertical distance from the diagonal, which is the quantity this
module is actually about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

DEFAULT_N_BINS = 10
DEFAULT_STRATEGY = "quantile"


@dataclass
class CalibrationReport:
    """Calibration of one model's scores on one held-out set."""

    name: str
    n: int
    prevalence: float
    brier: float
    ece: float
    max_bin_error: float
    prob_true: np.ndarray = field(repr=False)
    prob_pred: np.ndarray = field(repr=False)
    bin_counts: np.ndarray = field(repr=False)
    n_bins: int = DEFAULT_N_BINS
    strategy: str = DEFAULT_STRATEGY

    def summary_lines(self) -> list[str]:
        lines = [
            f"--- {self.name} ---",
            f"  rows                  {self.n:,}  (prevalence {self.prevalence:.4f})",
            f"  Brier score           {self.brier:.6f}   (lower is better; NOT a pure "
            "calibration metric -- see module docstring)",
            f"  ECE                   {self.ece:.6f}   (mean |observed - predicted|, "
            "bin-count weighted)",
            f"  worst bin deviation   {self.max_bin_error:.6f}",
            f"  reliability curve ({self.n_bins} {self.strategy} bins):",
            f"    {'predicted':>12} {'observed':>12} {'rows':>8}",
        ]
        for predicted, observed, count in zip(
            self.prob_pred, self.prob_true, self.bin_counts
        ):
            lines.append(f"    {predicted:>12.4f} {observed:>12.4f} {int(count):>8,}")
        return lines


def expected_calibration_error(
    prob_true: np.ndarray, prob_pred: np.ndarray, bin_counts: np.ndarray
) -> float:
    """Bin-count-weighted mean absolute distance from the diagonal.

    Weighting by bin population matters: an unweighted mean lets a bin holding twelve rows
    count as much as one holding twelve thousand, which on skewed score distributions
    reports a number driven almost entirely by the sparsest bins.
    """
    total = bin_counts.sum()
    if total == 0:
        return float("nan")
    return float(np.sum(bin_counts * np.abs(prob_true - prob_pred)) / total)


def _bin_counts(
    y_score: np.ndarray, n_bins: int, strategy: str
) -> np.ndarray:
    """Rows per bin, matching how calibration_curve formed its bins.

    sklearn does not return bin populations, and ECE needs them, so the binning is
    reproduced here using the same rules sklearn applies.
    """
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.percentile(y_score, quantiles * 100)
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    edges = np.unique(edges)
    # np.digitize with right=True matches sklearn's binning convention closely enough for
    # counting purposes; empty bins are dropped below to stay aligned with the curve.
    indices = np.digitize(y_score, edges[1:-1], right=True)
    counts = np.bincount(indices, minlength=len(edges) - 1)
    return counts[counts > 0]


def calibration_report(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = DEFAULT_STRATEGY,
) -> CalibrationReport:
    """Reliability curve, Brier score and ECE for one set of scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    prob_true, prob_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy=strategy
    )

    counts = _bin_counts(y_score, n_bins, strategy)
    if len(counts) != len(prob_true):
        # Bin alignment can drift when scores are heavily tied. Fall back to equal
        # weighting rather than silently pairing mismatched bins.
        counts = np.full(len(prob_true), len(y_score) / max(len(prob_true), 1))

    deviations = np.abs(prob_true - prob_pred)

    return CalibrationReport(
        name=name,
        n=len(y_true),
        prevalence=float(y_true.mean()),
        brier=float(brier_score_loss(y_true, y_score)),
        ece=expected_calibration_error(prob_true, prob_pred, counts),
        max_bin_error=float(deviations.max()) if len(deviations) else float("nan"),
        prob_true=prob_true,
        prob_pred=prob_pred,
        bin_counts=counts,
        n_bins=n_bins,
        strategy=strategy,
    )
