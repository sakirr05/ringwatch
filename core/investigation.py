"""Assemble an investigation case file. Entirely deterministic.

This is the last thing on the deterministic side of the orchestrator boundary. It gathers
what an analyst would want before forming a view on a flagged cluster — comparative
context, cross-cluster overlap, and factual findings both for and against concern — and
freezes it into a `CaseFile` that the drafting layer reads and cannot alter.

WHY THE FINDINGS ARE COMPUTED HERE AND NOT ASKED FOR
-----------------------------------------------------
`corroborating` and `contradicting` are derived by rules in this module from figures the
engine already computed. They are not the model's opinions about the evidence; they are
facts the model is given. That distinction is what makes it safe for a drafter to reason
from them, and it is why the orchestrator can be genuinely useful while still being unable
to produce a number.

Deliberately included: findings that argue AGAINST concern. A case file that only assembles
incriminating detail is a prosecutor's brief, and an investigation tool that only ever
builds one is worse than no tool, because it manufactures the confidence a reviewer then
rubber-stamps.
"""

from __future__ import annotations

import numpy as np

from ai.contract import CaseFile, ClusterEvidence

# A cluster whose activity is spread over more days than this looks less like a
# card-testing burst and more like ordinary shared usage. A heuristic for framing, not a
# decision rule -- nothing thresholds on it.
BURST_WINDOW_DAYS = 7

# Below this share of flagged transactions, the cluster is mostly ordinary activity with a
# few hits, which argues against treating the whole cluster as coordinated.
LOW_FLAGGED_SHARE = 0.34


def _findings(
    evidence: ClusterEvidence, entities_in_other_clusters: int, flagged_share: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic statements for and against concern, derived from computed figures."""
    supporting: list[str] = []
    against: list[str] = []

    shared = [a for a in evidence.shared_attributes if a != "indirect_attribute_chain"]
    if len(shared) >= 2:
        supporting.append(
            f"all {evidence.entity_count} entities share {len(shared)} identifying "
            f"attributes ({', '.join(shared)})"
        )
    elif shared:
        supporting.append(
            f"all {evidence.entity_count} entities share {shared[0]}"
        )
    else:
        against.append(
            "no single attribute is shared by every entity; the link is an indirect chain"
        )

    if evidence.core_number >= 2:
        supporting.append(
            f"k-core number {evidence.core_number} indicates mutually reinforcing links "
            "rather than a chain of one-off coincidences"
        )

    if evidence.span_days <= BURST_WINDOW_DAYS:
        supporting.append(
            f"activity is compressed into {evidence.span_days} days, consistent with a burst"
        )
    else:
        against.append(
            f"activity spans {evidence.span_days} days, longer than a typical "
            "card-testing burst and consistent with ordinary shared usage"
        )

    if flagged_share >= 0.5:
        supporting.append(
            f"{evidence.flagged_transaction_count} of {evidence.transaction_count} "
            "transactions were independently flagged by the model"
        )
    elif flagged_share <= LOW_FLAGGED_SHARE:
        against.append(
            f"only {evidence.flagged_transaction_count} of "
            f"{evidence.transaction_count} transactions were flagged; most of the "
            "cluster's activity scored normally"
        )

    if entities_in_other_clusters > 0:
        supporting.append(
            f"{entities_in_other_clusters} of its entities also appear in other flagged "
            "clusters"
        )

    if evidence.distinct_cards == 1 and evidence.entity_count > 1:
        against.append(
            "a single card across several entities is also the ordinary signature of one "
            "household or a shared business account"
        )

    return tuple(supporting), tuple(against)


def build_case_files(evidence_items: list[ClusterEvidence]) -> list[CaseFile]:
    """Turn flagged clusters into case files, with comparative context.

    Context is computed across the whole flagged set, which is why this takes the list
    rather than one cluster: rank, percentile and the population mean only exist relative
    to the others.
    """
    if not evidence_items:
        return []

    risks = np.array([item.max_risk_score for item in evidence_items], dtype=np.float64)
    population_mean = float(np.mean([item.mean_risk_score for item in evidence_items]))
    order = np.argsort(-risks)
    rank_of = {int(position): rank + 1 for rank, position in enumerate(order)}

    # Entities are not identified across clusters in the evidence objects, so overlap is
    # approximated by shared attribute signatures. Named honestly rather than presented as
    # exact identity matching, which the evidence contract deliberately does not carry.
    signatures: dict[tuple[str, ...], int] = {}
    for item in evidence_items:
        key = tuple(sorted(item.shared_attributes))
        signatures[key] = signatures.get(key, 0) + 1

    files: list[CaseFile] = []
    for position, item in enumerate(evidence_items):
        flagged_share = (
            item.flagged_transaction_count / item.transaction_count
            if item.transaction_count
            else 0.0
        )
        overlap = signatures.get(tuple(sorted(item.shared_attributes)), 1) - 1
        supporting, against = _findings(item, overlap, flagged_share)

        files.append(
            CaseFile(
                case_id=f"CASE-{item.cluster_id:03d}",
                cluster=item,
                rank=rank_of[position],
                total_flagged_clusters=len(evidence_items),
                risk_percentile=float(
                    100.0 * (risks < item.max_risk_score).mean()
                ),
                entities_in_other_clusters=overlap,
                transactions_per_entity=(
                    item.transaction_count / item.entity_count
                    if item.entity_count
                    else 0.0
                ),
                flagged_share=flagged_share,
                population_mean_risk=population_mean,
                corroborating=supporting,
                contradicting=against,
            )
        )
    return files
