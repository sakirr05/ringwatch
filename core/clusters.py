"""Select flagged clusters and package their evidence for the narrative layer.

This module is the last thing on the deterministic side of the boundary. It decides which
clusters an analyst should look at, and it computes every number the language model will
later be allowed to quote. The model receives the output of this module and nothing else.

Note the direction of the dependency: `core.clusters` imports `ai.contract` to build the
handover object. `ai/` does not import `core/` at all. The deterministic side knows how
to talk to the narrative side; the narrative side has no way to talk back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai.contract import ClusterEvidence
from core.evaluate import USD_TO_INR
from core.graph import UID_COL

# Clusters are ranked for analyst attention by their peak risk score. Peak rather than
# mean because a ring with one blatant transaction and nine subtle ones is exactly what
# should reach a human, and averaging would bury it.
DEFAULT_TOP_N = 12


def build_cluster_evidence(
    test: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    top_n: int = DEFAULT_TOP_N,
    selected_components: list[int] | None = None,
) -> list[ClusterEvidence]:
    """Package the top flagged multi-entity clusters as evidence objects.

    Only clusters that (a) contain more than one entity and (b) contain at least one
    transaction the deterministic model scored at or above `threshold` are eligible. Both
    conditions are decided here, by code, before the model is ever consulted.
    """
    if selected_components is None:
        selected_components = []

    frame = test.copy()
    frame["_score"] = scores
    frame["_flagged"] = scores >= threshold

    eligible = frame[frame["g_component"].notna() & (frame["g_component_size"] > 1)]
    if eligible.empty:
        selected_components.clear()
        return []

    # `g_component_size` counts entities in the GRAPH, which spans the whole timeline.
    # A component can be genuinely multi-entity while only one of its entities appears in
    # the period being scored -- and presenting that to an analyst as a "cluster" of
    # coordinated accounts would be misleading, because there is only one account here to
    # look at. Filter on entities actually present in the scored data.
    entities_present = eligible.groupby("g_component", observed=True)[UID_COL].nunique()
    multi_entity = entities_present[entities_present >= 2].index
    eligible = eligible[eligible["g_component"].isin(multi_entity)]
    if eligible.empty:
        selected_components.clear()
        return []

    grouped = eligible.groupby("g_component", observed=True)
    ranked = (
        grouped["_score"]
        .max()
        .sort_values(ascending=False)
        .loc[lambda s: s >= threshold]
        .head(top_n)
    )

    # The component label each cluster came from. Returned via `selected_components` so
    # callers can pull the cluster's actual subgraph out of the entity graph for display,
    # without re-deriving this selection and risking the two drifting apart.
    #
    # Deliberately NOT added to ClusterEvidence: that object defines what the language
    # model may quote, and a component label is a large arbitrary integer that would become
    # a legitimately quotable "number" in the provenance guard. Keeping it out means the
    # model can never cite a graph-internal id as though it meant something.
    selected_components.clear()
    selected_components.extend(int(component) for component in ranked.index)

    evidence: list[ClusterEvidence] = []
    for rank, (component, _) in enumerate(ranked.items()):
        rows = eligible[eligible["g_component"] == component]
        evidence.append(_evidence_for(rank, rows))
    return evidence


def _evidence_for(cluster_id: int, rows: pd.DataFrame) -> ClusterEvidence:
    """Compute every figure the narrative layer is permitted to mention."""
    shared: list[str] = []
    if rows["card1"].nunique(dropna=True) == 1:
        shared.append("card1")
    if rows["addr1"].nunique(dropna=True) == 1:
        shared.append("addr1")
    if rows["P_emaildomain"].nunique(dropna=True) == 1:
        shared.append("P_emaildomain")
    if not shared:
        shared.append("indirect_attribute_chain")

    span_seconds = rows["TransactionDT"].max() - rows["TransactionDT"].min()

    return ClusterEvidence(
        cluster_id=int(cluster_id),
        entity_count=int(rows[UID_COL].nunique()),
        transaction_count=int(len(rows)),
        flagged_transaction_count=int(rows["_flagged"].sum()),
        component_size=int(rows["g_component_size"].iloc[0]),
        core_number=int(rows["g_core_number"].max()),
        max_degree=int(rows["g_degree"].max()),
        shared_attributes=tuple(shared),
        distinct_cards=int(rows["card1"].nunique(dropna=True)),
        distinct_addresses=int(rows["addr1"].nunique(dropna=True)),
        distinct_email_domains=int(rows["P_emaildomain"].nunique(dropna=True)),
        span_days=int(span_seconds // 86_400),
        total_amount_inr=round(float(rows["TransactionAmt"].sum() * USD_TO_INR), 2),
        max_risk_score=round(float(rows["_score"].max()), 4),
        mean_risk_score=round(float(rows["_score"].mean()), 4),
    )


# ---------------------------------------------------------------------------
# RETROSPECTIVE OUTCOMES -- DELIBERATELY NOT PART OF ClusterEvidence
#
# `ClusterEvidence` defines exactly what the language model is permitted to see and quote.
# The held-out fraud label must never enter it. A narrator handed ground truth would write
# narratives that look accurate for a reason that has nothing to do with the evidence, and
# the whole "the model wrote prose about numbers core/ computed" framing would be a lie.
#
# So outcomes live here, in a separate object, produced for the DASHBOARD only. They answer
# the question a reviewer actually has -- "was the engine right about these clusters?" --
# against labels the engine never used.
#
# WHAT `all_fraud` DOES AND DOES NOT MEAN
# ---------------------------------------
# It means: every transaction in this cluster carries a fraud label in the held-out data.
# It does NOT mean the cluster is a verified ring. Three unrelated fraudsters who happen to
# share an address form an all-fraud cluster and coordinate nothing. RingWatch has no
# ring-level ground truth and this does not create any; conflating the two is exactly the
# claim this project refuses to make.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterOutcome:
    """How one flagged cluster turned out, measured against the held-out labels."""

    cluster_id: int
    transaction_count: int
    fraud_transactions: int
    fraud_share: float
    all_fraud: bool
    caught: int         # fraud the model flagged at this threshold
    missed: int         # fraud the model did not flag
    false_alarms: int   # non-fraud the model flagged


def cluster_outcomes(
    test: pd.DataFrame,
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    selected_components: list[int],
) -> list[ClusterOutcome]:
    """Ground-truth outcome per flagged cluster, in the order `build_cluster_evidence` chose.

    Takes `selected_components` rather than re-running the selection, for the same reason
    that list exists at all: two independent derivations of "which clusters" would
    eventually disagree, and the disagreement would surface as a dashboard whose evidence
    panel and summary card describe different clusters.
    """
    frame = test.copy()
    frame["_flagged"] = np.asarray(scores) >= threshold
    frame["_fraud"] = np.asarray(y_true).astype(bool)

    outcomes: list[ClusterOutcome] = []
    for cluster_id, component in enumerate(selected_components):
        rows = frame[frame["g_component"] == component]
        fraud = rows["_fraud"]
        flagged = rows["_flagged"]
        n = int(len(rows))
        n_fraud = int(fraud.sum())

        outcomes.append(
            ClusterOutcome(
                cluster_id=cluster_id,
                transaction_count=n,
                fraud_transactions=n_fraud,
                fraud_share=(n_fraud / n) if n else 0.0,
                # `n > 0` guards the vacuous truth: an empty cluster is not "all fraud".
                all_fraud=bool(n > 0 and n_fraud == n),
                caught=int((fraud & flagged).sum()),
                missed=int((fraud & ~flagged).sum()),
                false_alarms=int((~fraud & flagged).sum()),
            )
        )
    return outcomes
