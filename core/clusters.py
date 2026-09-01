"""Select flagged clusters and package their evidence for the narrative layer.

This module is the last thing on the deterministic side of the boundary. It decides which
clusters an analyst should look at, and it computes every number the language model will
later be allowed to quote. The model receives the output of this module and nothing else.

Note the direction of the dependency: `core.clusters` imports `ai.contract` to build the
handover object. `ai/` does not import `core/` at all. The deterministic side knows how
to talk to the narrative side; the narrative side has no way to talk back.
"""

from __future__ import annotations

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
) -> list[ClusterEvidence]:
    """Package the top flagged multi-entity clusters as evidence objects.

    Only clusters that (a) contain more than one entity and (b) contain at least one
    transaction the deterministic model scored at or above `threshold` are eligible. Both
    conditions are decided here, by code, before the model is ever consulted.
    """
    frame = test.copy()
    frame["_score"] = scores
    frame["_flagged"] = scores >= threshold

    eligible = frame[frame["g_component"].notna() & (frame["g_component_size"] > 1)]
    if eligible.empty:
        return []

    grouped = eligible.groupby("g_component", observed=True)
    ranked = (
        grouped["_score"]
        .max()
        .sort_values(ascending=False)
        .loc[lambda s: s >= threshold]
        .head(top_n)
    )

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
