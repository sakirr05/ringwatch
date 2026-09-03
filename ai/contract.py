"""The contract between the deterministic engine and the narrative layer.

This module is the ONLY thing the two halves of RingWatch share, and the dependency runs
strictly one way:

    core/  ---> ai.contract <--- ai/

`core/` builds a ClusterEvidence and hands it over. `ai/` consumes it. `ai/` never
imports `core/` — not the model, not the scorer, not the graph — so there is no code path
by which the language model can reach a score, recompute a number, or alter a flag. This
is enforced by a test that walks the import graph, not by a comment asking nicely.

Everything in here is frozen. The narrative layer physically cannot mutate the evidence
it was given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# What the deterministic layer is allowed to tell the model about a cluster.
Confidence = Literal["high", "medium", "low"]

PROBABLE_CAUSES: tuple[str, ...] = (
    "CARD_TESTING",
    "COORDINATED_RING",
    "SHARED_CREDENTIAL_REUSE",
    "SYNTHETIC_IDENTITY",
    "BENIGN_COINCIDENCE",
    "UNEXPLAINED",
)

CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")


@dataclass(frozen=True)
class ClusterEvidence:
    """Deterministically-computed facts about one flagged cluster.

    Every number here was produced by `core/`. The narrative layer may quote these
    numbers and may not produce any others — see `allowed_numbers`.
    """

    cluster_id: int
    entity_count: int
    transaction_count: int
    flagged_transaction_count: int
    component_size: int
    core_number: int
    max_degree: int
    shared_attributes: tuple[str, ...]
    distinct_cards: int
    distinct_addresses: int
    distinct_email_domains: int
    span_days: int
    total_amount_inr: float
    max_risk_score: float
    mean_risk_score: float

    def allowed_numbers(self) -> set[str]:
        """Every numeric token the model is permitted to use.

        Any figure in the model's output that is not in this set was invented, and the
        response is rejected. Small integers 0-10 are permitted unconditionally because
        they appear in ordinary prose ("two of the three cards") and banning them would
        reject valid narratives without preventing any meaningful fabrication.
        """
        allowed = {str(n) for n in range(0, 11)}
        for value in (
            self.cluster_id,
            self.entity_count,
            self.transaction_count,
            self.flagged_transaction_count,
            self.component_size,
            self.core_number,
            self.max_degree,
            self.distinct_cards,
            self.distinct_addresses,
            self.distinct_email_domains,
            self.span_days,
        ):
            allowed.add(str(int(value)))
        for value in (self.total_amount_inr, self.max_risk_score, self.mean_risk_score):
            allowed.add(_normalise_number(str(value)))
            allowed.add(str(int(value)))
            allowed.add(f"{value:.2f}")
            allowed.add(f"{value:.1f}")
            allowed.add(f"{value:.0f}")
        return {_normalise_number(a) for a in allowed}

    def as_prompt_facts(self) -> str:
        """The evidence, rendered for the prompt. No scores are recomputed here."""
        return "\n".join(
            [
                f"cluster_id: {self.cluster_id}",
                f"entities_in_cluster: {self.entity_count}",
                f"transactions: {self.transaction_count}",
                f"transactions_flagged_by_the_model: {self.flagged_transaction_count}",
                f"connected_component_size: {self.component_size}",
                f"k_core_number: {self.core_number}",
                f"max_entity_degree: {self.max_degree}",
                f"shared_attributes: {', '.join(self.shared_attributes)}",
                f"distinct_cards: {self.distinct_cards}",
                f"distinct_addresses: {self.distinct_addresses}",
                f"distinct_email_domains: {self.distinct_email_domains}",
                f"activity_span_days: {self.span_days}",
                f"total_amount_inr: {self.total_amount_inr:.2f}",
                f"max_risk_score: {self.max_risk_score:.4f}",
                f"mean_risk_score: {self.mean_risk_score:.4f}",
            ]
        )


@dataclass(frozen=True)
class ClusterNarrative:
    """A validated narrative. Prose only — it carries no authority over any decision."""

    cluster_id: int
    probable_cause: str
    confidence: str
    human_summary: str
    suggested_action: str
    status: str = "OK"
    rejection_reason: str | None = None


def unavailable(cluster_id: int, reason: str) -> ClusterNarrative:
    """The honest fallback. Used when the model cannot produce a valid narrative.

    RingWatch never degrades to a guessed narrative: an analyst reading
    NARRATIVE_UNAVAILABLE knows there is nothing here, whereas an analyst reading a
    plausible invented paragraph does not.
    """
    return ClusterNarrative(
        cluster_id=cluster_id,
        probable_cause="UNEXPLAINED",
        confidence="low",
        human_summary="NARRATIVE_UNAVAILABLE",
        suggested_action="NARRATIVE_UNAVAILABLE",
        status="NARRATIVE_UNAVAILABLE",
        rejection_reason=reason,
    )


_NUMBER_PATTERN = re.compile(r"\d[\d,]*\.?\d*")


def _normalise_number(token: str) -> str:
    """Strip thousands separators and trailing zeros so 1,200 == 1200 == 1200.00."""
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token or "0"


def extract_numbers(text: str) -> set[str]:
    """Every numeric token in a piece of prose, normalised for comparison."""
    return {_normalise_number(m.group()) for m in _NUMBER_PATTERN.finditer(text)}


# ---------------------------------------------------------------------------
# INVESTIGATION ORCHESTRATOR CONTRACT
#
# Same one-way shape as ClusterEvidence above: `core/investigation.py` assembles a frozen
# CaseFile and hands it over; `ai/disposition.py` reads it and writes prose. The AI side
# still imports nothing from `core/`, which `tests/test_ai_boundary.py` enforces by walking
# the import graph.
#
# The orchestrator's autonomy is deliberately bounded. It cannot compute or alter a score,
# decide a match, or execute anything at all -- there is no action surface in this contract
# for it to reach. Its output is a RECOMMENDATION that a human either approves or rejects,
# and the approval gate is the feature rather than an obstacle to it.
# ---------------------------------------------------------------------------

DISPOSITIONS: tuple[str, ...] = ("confirm", "dismiss", "escalate")


@dataclass(frozen=True)
class CaseFile:
    """Everything deterministically known about one flagged cluster, frozen for review.

    Richer than `ClusterEvidence` on purpose: an investigation needs comparative context a
    narrative does not. Rank, percentile and cross-cluster overlap all come from `core/`,
    and `corroborating` / `contradicting` are factual statements the deterministic engine
    derived — not the model's opinions, which is what makes them safe to reason from.
    """

    case_id: str
    cluster: ClusterEvidence
    rank: int
    total_flagged_clusters: int
    risk_percentile: float
    entities_in_other_clusters: int
    transactions_per_entity: float
    flagged_share: float
    population_mean_risk: float
    corroborating: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()

    def allowed_numbers(self) -> set[str]:
        """Every figure the drafter may quote. Inherits the cluster's own allowances."""
        allowed = set(self.cluster.allowed_numbers())
        for value in (self.rank, self.total_flagged_clusters, self.entities_in_other_clusters):
            allowed.add(_normalise_number(str(int(value))))
        for value in (
            self.risk_percentile,
            self.transactions_per_entity,
            self.flagged_share,
            self.population_mean_risk,
        ):
            for rendering in (f"{value:.4f}", f"{value:.2f}", f"{value:.1f}", f"{value:.0f}"):
                allowed.add(_normalise_number(rendering))
        # Figures quoted inside the deterministic findings are quotable too.
        for statement in (*self.corroborating, *self.contradicting):
            allowed |= extract_numbers(statement)
        return allowed

    def as_prompt_facts(self) -> str:
        """The case file, rendered for the drafter. Nothing is recomputed here."""
        lines = [
            f"case_id: {self.case_id}",
            self.cluster.as_prompt_facts(),
            f"rank_by_risk: {self.rank} of {self.total_flagged_clusters}",
            f"risk_percentile: {self.risk_percentile:.2f}",
            f"entities_also_in_other_flagged_clusters: {self.entities_in_other_clusters}",
            f"transactions_per_entity: {self.transactions_per_entity:.2f}",
            f"share_of_transactions_flagged: {self.flagged_share:.2f}",
            f"population_mean_risk_score: {self.population_mean_risk:.4f}",
        ]
        if self.corroborating:
            lines.append("findings_supporting_concern:")
            lines += [f"  - {item}" for item in self.corroborating]
        if self.contradicting:
            lines.append("findings_arguing_against_concern:")
            lines += [f"  - {item}" for item in self.contradicting]
        return "\n".join(lines)


@dataclass(frozen=True)
class Disposition:
    """A recommended disposition. Advisory only — nothing acts on it without a human."""

    case_id: str
    recommendation: str          # confirm | dismiss | escalate
    confidence: str              # high | medium | low
    rationale: str
    key_factors: tuple[str, ...] = ()
    status: str = "OK"
    rejection_reason: str | None = None

    @property
    def requires_human_approval(self) -> bool:
        """Always true. Present so the invariant is expressed in the type, not just the UI."""
        return True


def disposition_unavailable(case_id: str, reason: str) -> Disposition:
    """The honest fallback, matching `unavailable()` for narratives.

    An analyst reading DISPOSITION_UNAVAILABLE knows there is nothing here. One reading a
    plausible invented recommendation does not, and might act on it.
    """
    return Disposition(
        case_id=case_id,
        recommendation="escalate",
        confidence="low",
        rationale="DISPOSITION_UNAVAILABLE",
        key_factors=(),
        status="DISPOSITION_UNAVAILABLE",
        rejection_reason=reason,
    )
