"""Honest evaluation: AUC-PR, a defended operating threshold, and insult-rate costing.

WHY NOT ACCURACY
----------------
The test set is 3.44% fraud. A model that returns "legitimate" for every single
transaction scores 96.56% accuracy and catches nothing. Accuracy is not merely a weak
metric on this data, it is an actively misleading one, and any submission reporting it as
a headline number is reporting noise. RingWatch reports **AUC-PR** (average precision),
which is the area under the precision-recall curve and is sensitive to exactly the thing
that matters: performance on the rare positive class.

The right reference point for AUC-PR is not 0.5 (that is the AUC-ROC baseline). A random
classifier scores AUC-PR equal to the positive-class prevalence, so ~0.034 here. Every
reported AUC-PR below is accompanied by its lift over that prevalence floor.

WHY A THRESHOLD MUST BE DEFENDED
--------------------------------
AUC-PR is threshold-free, but a deployed fraud system is not: it must decide. Declining a
legitimate customer ("insulting" them) is not free, and neither is missing a fraud. This
module therefore picks the operating threshold that minimises *expected rupee cost* under
explicitly named assumptions, rather than defaulting to 0.5 (arbitrary) or to max-F1
(which silently asserts that a false positive and a false negative cost the same, and
they do not).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# ECONOMIC ASSUMPTIONS
#
# These are ASSUMPTIONS, not measurements. They are declared here as named constants,
# in one place, so that every rupee figure this project reports is traceable to a stated
# premise a reviewer can disagree with and re-run. Changing one number here changes every
# downstream cost consistently.
# ---------------------------------------------------------------------------

# IEEE-CIS is a US e-commerce dataset (Vesta); TransactionAmt is denominated in USD.
# The buildathon context is Indian payments, so amounts are converted for reporting.
# This is a reporting convenience, NOT a claim that this is Indian transaction data --
# see the README limitations section.
USD_TO_INR = 88.0

# Gross margin retained on a completed order. When a legitimate order is wrongly
# declined, the merchant loses this margin, not the full order value.
GROSS_MARGIN_RATE = 0.12

# Fixed acquirer/processor fee levied per chargeback, on top of losing the goods and the
# transaction value. Order of magnitude for the Indian market.
CHARGEBACK_FEE_INR = 1_200.0

# A wrongly-declined customer may never return. That lifetime-value damage is real and is
# deliberately NOT monetised here -- putting a number on churn would be inventing data.
# It is reported as a raw count ("N legitimate customers wrongly declined") instead, and
# it means every insult cost below is an UNDERESTIMATE. Stated in the README.


def amount_inr(amount_usd: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(amount_usd, dtype=np.float64) * USD_TO_INR


@dataclass
class ThresholdReport:
    """Confusion outcome and rupee cost at one operating threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    insult_rate: float
    fraud_caught_inr: float
    fraud_missed_inr: float
    insult_cost_inr: float
    total_cost_inr: float

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("threshold", f"{self.threshold:.4f}"),
            ("precision", f"{self.precision:.4f}"),
            ("recall", f"{self.recall:.4f}"),
            ("fraud caught / missed", f"{self.true_positives} / {self.false_negatives}"),
            ("legit customers declined", f"{self.false_positives:,}"),
            ("insult rate", f"{self.insult_rate*100:.3f}% of legitimate traffic"),
            ("fraud value missed", f"Rs {self.fraud_missed_inr:,.0f}"),
            ("insult cost", f"Rs {self.insult_cost_inr:,.0f}"),
            ("total expected cost", f"Rs {self.total_cost_inr:,.0f}"),
        ]


# An operationally realistic ceiling on how much legitimate traffic a merchant will
# tolerate having declined. The pure cost-minimising threshold ignores this and, on this
# data, lands around a 5% insult rate -- a number no real payments team would ship,
# because the churn it causes is precisely the cost the model refuses to monetise.
# Both operating points are therefore reported, and the tension between them is discussed
# in the README rather than resolved silently in favour of the prettier one.
MAX_ACCEPTABLE_INSULT_RATE = 0.01


@dataclass
class EvaluationReport:
    """Everything measured for one model on one held-out set."""

    name: str
    n: int
    n_fraud: int
    prevalence: float
    auc_pr: float
    auc_roc: float
    auc_pr_lift_over_random: float
    operating_point: ThresholdReport
    constrained_operating_point: ThresholdReport
    pr_curve: tuple[np.ndarray, np.ndarray, np.ndarray] = field(repr=False)

    def summary_lines(self) -> list[str]:
        lines = [
            f"--- {self.name} ---",
            f"  rows                {self.n:,}  ({self.n_fraud:,} fraud, "
            f"{self.prevalence*100:.3f}%)",
            f"  AUC-PR              {self.auc_pr:.4f}",
            f"  random-baseline PR  {self.prevalence:.4f}  "
            f"(lift x{self.auc_pr_lift_over_random:.1f})",
            f"  AUC-ROC             {self.auc_roc:.4f}  (reported for reference only)",
            "  [A] cost-minimising operating point:",
        ]
        lines += [f"    {k:26s} {v}" for k, v in self.operating_point.as_rows()]
        lines += [
            f"  [B] insult-constrained operating point "
            f"(<= {MAX_ACCEPTABLE_INSULT_RATE*100:.1f}% of legitimate traffic):"
        ]
        lines += [
            f"    {k:26s} {v}" for k, v in self.constrained_operating_point.as_rows()
        ]
        return lines


@dataclass
class DeltaReport:
    """A difference in AUC-PR between two models, with an honest uncertainty interval.

    `n_comparisons` records how many tests the interval is corrected for. Running eight
    comparisons at 95% each gives roughly a one-in-three chance of at least one false
    positive, so an uncorrected interval from a family of tests overstates its own
    confidence. Bonferroni is conservative, which is the right direction to err when the
    alternative is announcing an effect that is not there.
    """

    name: str
    delta: float
    ci_low: float
    ci_high: float
    n_resamples: int
    n_comparisons: int = 1

    @property
    def significant(self) -> bool:
        """True only if the 95% interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    def verdict(self) -> str:
        if not self.significant:
            corrected = "" if self.n_comparisons == 1 else f", {self.n_comparisons}-way corrected"
            return f"not significant (CI spans 0{corrected})"
        return "SIGNIFICANTLY BETTER" if self.delta > 0 else "SIGNIFICANTLY WORSE"

    def line(self) -> str:
        return (
            f"  {self.name:<16} {self.delta:+.4f}  "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  {self.verdict()}"
        )


def bootstrap_delta(
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    variant_scores: np.ndarray,
    metric,
    name: str,
    n_resamples: int = 400,
    seed: int = 0,
    n_comparisons: int = 1,
    baseline_metric=None,
) -> DeltaReport:
    """Paired bootstrap CI for the difference in ANY metric between two models.

    The resampling machinery is identical whichever metric is being compared, so it lives
    here once and takes the metric as a parameter. `metric(y_resampled, scores_resampled)`
    returns a float, or None when the resample is degenerate (no positives, or no positive
    value) and the comparison is undefined -- those draws are scored as a zero difference
    rather than dropped, which keeps the interval conservative.

    WHY PAIRED
    ----------
    Both models are evaluated on the SAME resampled indices every draw. Test-set variation
    that affects them equally cancels, so the interval reflects the difference between the
    models rather than the noisiness of the test set. Unpaired resampling would produce
    intervals wide enough to hide any real effect.
    """
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_y = y_true[idx]
        variant_value = metric(resampled_y, variant_scores[idx], idx)
        baseline_value = (baseline_metric or metric)(
            resampled_y, baseline_scores[idx], idx
        )
        deltas[i] = (
            0.0
            if variant_value is None or baseline_value is None
            else variant_value - baseline_value
        )

    # Bonferroni: split the 5% error budget across the whole family of comparisons, so an
    # interval taken from one of eight tests is not read as though it were the only one.
    tail = 5.0 / (2 * max(n_comparisons, 1))
    low, high = np.percentile(deltas, [tail, 100 - tail])
    return DeltaReport(
        name=name,
        delta=float(deltas.mean()),
        ci_low=float(low),
        ci_high=float(high),
        n_resamples=n_resamples,
        n_comparisons=n_comparisons,
    )


def bootstrap_auc_pr_delta(
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    variant_scores: np.ndarray,
    name: str,
    n_resamples: int = 400,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
    n_comparisons: int = 1,
) -> DeltaReport:
    """Paired bootstrap CI for the AUC-PR difference between two models.

    WHY THIS IS NOT OPTIONAL
    ------------------------
    A single-run AUC-PR difference of 0.002 is not a result, it is a number. Both models
    are scored on the same 118,108 rows containing 4,064 fraud cases, and resampling those
    rows shows how much of any observed gap is just which transactions happened to land in
    the test period.

    Reporting a raw delta without this interval is how projects claim lifts that do not
    exist. Here it is what separates "the graph layer helps a little" from "the graph
    layer does nothing measurable", and the honest answer turned out to be the second.

    `sample_weight` turns this into the value-weighted comparison without changing the
    estimator, the pairing, or the resampling. Default None reproduces the count-weighted
    behaviour exactly, so every previously reported interval is unaffected.
    """
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    def metric(resampled_y, resampled_scores, idx):
        if resampled_y.sum() == 0:  # degenerate resample, no positives
            return None
        resampled_weights = None if weights is None else weights[idx]
        if resampled_weights is not None and resampled_weights[resampled_y == 1].sum() <= 0:
            # No positive VALUE in this resample: the value-weighted metric is undefined.
            return None
        return average_precision_score(
            resampled_y, resampled_scores, sample_weight=resampled_weights
        )

    return bootstrap_delta(
        y_true,
        baseline_scores,
        variant_scores,
        metric,
        name=name,
        n_resamples=n_resamples,
        seed=seed,
        n_comparisons=n_comparisons,
    )


def cost_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts_inr: np.ndarray,
    threshold: float,
) -> ThresholdReport:
    """Confusion matrix and rupee cost if we declined everything scoring >= threshold."""
    predicted_fraud = y_score >= threshold
    actual_fraud = y_true == 1

    tp_mask = predicted_fraud & actual_fraud
    fp_mask = predicted_fraud & ~actual_fraud
    fn_mask = ~predicted_fraud & actual_fraud
    tn_mask = ~predicted_fraud & ~actual_fraud

    tp, fp, fn, tn = (
        int(tp_mask.sum()),
        int(fp_mask.sum()),
        int(fn_mask.sum()),
        int(tn_mask.sum()),
    )

    # Missed fraud costs the transaction value plus a fixed chargeback fee.
    fraud_missed_inr = float(amounts_inr[fn_mask].sum() + fn * CHARGEBACK_FEE_INR)
    fraud_caught_inr = float(amounts_inr[tp_mask].sum())
    # A wrongly declined legitimate order costs the merchant its margin.
    insult_cost_inr = float(amounts_inr[fp_mask].sum() * GROSS_MARGIN_RATE)

    n_legit = int((~actual_fraud).sum())

    return ThresholdReport(
        threshold=float(threshold),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        insult_rate=fp / n_legit if n_legit else 0.0,
        fraud_caught_inr=fraud_caught_inr,
        fraud_missed_inr=fraud_missed_inr,
        insult_cost_inr=insult_cost_inr,
        total_cost_inr=fraud_missed_inr + insult_cost_inr,
    )


def choose_threshold_by_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts_inr: np.ndarray,
    n_candidates: int = 200,
) -> ThresholdReport:
    """Pick the threshold minimising total expected rupee cost.

    Swept over score quantiles rather than a uniform 0-1 grid, because predicted
    probabilities on a 3.4% positive class bunch up near zero and a uniform grid would
    spend most of its candidates in a region containing no data.
    """
    quantiles = np.linspace(0.50, 0.9999, n_candidates)
    candidates = np.unique(np.quantile(y_score, quantiles))

    best: ThresholdReport | None = None
    for threshold in candidates:
        report = cost_at_threshold(y_true, y_score, amounts_inr, threshold)
        if best is None or report.total_cost_inr < best.total_cost_inr:
            best = report

    assert best is not None
    return best


def choose_threshold_under_insult_cap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts_inr: np.ndarray,
    max_insult_rate: float = MAX_ACCEPTABLE_INSULT_RATE,
    n_candidates: int = 200,
) -> ThresholdReport:
    """Cheapest threshold that keeps the insult rate at or below the cap.

    This is the operating point a payments team could actually deploy. If no candidate
    satisfies the cap (possible for a very weak model), the strictest threshold tried is
    returned rather than silently exceeding the constraint.
    """
    quantiles = np.linspace(0.50, 0.99999, n_candidates)
    candidates = np.unique(np.quantile(y_score, quantiles))

    feasible: ThresholdReport | None = None
    strictest: ThresholdReport | None = None
    for threshold in candidates:
        report = cost_at_threshold(y_true, y_score, amounts_inr, threshold)
        if strictest is None or report.insult_rate < strictest.insult_rate:
            strictest = report
        if report.insult_rate <= max_insult_rate:
            if feasible is None or report.total_cost_inr < feasible.total_cost_inr:
                feasible = report

    assert strictest is not None
    return feasible if feasible is not None else strictest


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts_inr: np.ndarray,
    n_points: int = 150,
    extra_thresholds: Iterable[float] = (),
) -> list[ThresholdReport]:
    """The full cost/precision/recall curve, evaluated once so nothing recomputes it later.

    This exists so the dashboard's threshold slider has something to *read*. Every point is
    produced here, by the same `cost_at_threshold` that produced both published operating
    points — the web layer looks values up in a committed artifact and does no arithmetic of
    its own. A slider that recomputed a confusion matrix in the browser would be a second,
    unvalidated implementation of the costing model, which is exactly the kind of quiet
    divergence this project's structure is meant to prevent.

    Swept over score quantiles for the reason `choose_threshold_by_cost` already documents:
    predicted probabilities on a 3.4% positive class bunch near zero, so a uniform 0-1 grid
    would spend most of its points where there is no data. Equal steps in quantile are equal
    steps in *number of transactions declined*, which is also the axis an operator actually
    reasons about.

    `extra_thresholds` is merged in so the published operating points land exactly on the
    curve rather than near it — a slider whose marked point sits between two samples would
    display figures that disagree with the table above it.
    """
    quantiles = np.linspace(0.85, 0.99995, n_points)
    candidates = np.quantile(np.asarray(y_score, dtype=np.float64), quantiles)
    candidates = np.concatenate([candidates, np.asarray(list(extra_thresholds), dtype=np.float64)])
    candidates = np.unique(candidates)  # sorts ascending and de-duplicates

    return [
        cost_at_threshold(y_true, y_score, amounts_inr, float(threshold))
        for threshold in candidates
    ]


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts_usd: pd.Series | np.ndarray,
) -> EvaluationReport:
    """Full honest evaluation of one model's scores on a held-out set."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    amounts = amount_inr(amounts_usd)

    prevalence = float(y_true.mean())
    auc_pr = float(average_precision_score(y_true, y_score))
    auc_roc = float(roc_auc_score(y_true, y_score))

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    return EvaluationReport(
        name=name,
        n=len(y_true),
        n_fraud=int(y_true.sum()),
        prevalence=prevalence,
        auc_pr=auc_pr,
        auc_roc=auc_roc,
        auc_pr_lift_over_random=auc_pr / prevalence if prevalence else float("nan"),
        operating_point=choose_threshold_by_cost(y_true, y_score, amounts),
        constrained_operating_point=choose_threshold_under_insult_cap(
            y_true, y_score, amounts
        ),
        pr_curve=(precision, recall, thresholds),
    )
