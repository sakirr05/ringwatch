"""Elliptic Bitcoin dataset — structural replication on an OBSERVED graph.

WHY THIS DATASET, AND WHAT IT DOES NOT DO
------------------------------------------
It does **not** close RingWatch's ring-level ground-truth gap. Elliptic's labels are
attached to transactions (`illicit` / `licit` / `unknown`), exactly as `isFraud` is in
IEEE-CIS. There is no ring, cluster or actor identifier anywhere in it. That limitation
stands, and `PLAN_ELLIPTIC.md` records the check that established it before any code was
written here.

What it provides instead is a **real, observed graph**. IEEE-CIS has no edges at all;
RingWatch had to infer an entity graph from a `card1 + addr1 + (day − D1)` fingerprint and
then suppress hubs to stop it percolating. Every structural claim in the project therefore
rests on that heuristic being reasonable — something asserted rather than measured. Running
the same concentration test on a graph nobody had to invent removes entity resolution as a
confound.

THE SEMANTIC CAVEAT, WHICH LIMITS WHAT A REPLICATION MEANS
-----------------------------------------------------------
These are not the same kind of graph:

  IEEE-CIS (inferred)  nodes = entities,     edges = shared identity attributes
  Elliptic (observed)  nodes = transactions, edges = money flowed between them

"Does illicit activity concentrate in components?" is coherent for both, but it is not the
same question. A positive result here supports the general claim that illicit activity is
structurally clustered. It does **not** validate RingWatch's entity graph.

HANDLING THE UNLABELLED MAJORITY
--------------------------------
Roughly 77% of Elliptic nodes are `unknown`. Two tempting approaches are wrong: treating
unknown as licit invents ~157k negative labels and would inflate any concentration result,
and dropping unknown nodes before building the graph deletes observed edges — the very
property this dataset was brought in for.

So the graph is built on **all** observed edges and the statistic is evaluated over
**labelled nodes only**. "All-illicit component" therefore means all-illicit among its
labelled members, which is what gets reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.graph import connected_components, k_core_numbers

REPO_ROOT = Path(__file__).resolve().parent.parent
ELLIPTIC_DIR = REPO_ROOT / "data" / "elliptic"
CLASSES_CSV = ELLIPTIC_DIR / "elliptic_txs_classes.csv"
EDGELIST_CSV = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"

EXPECTED_NODES = 203_769
EXPECTED_EDGES = 234_355

ILLICIT, LICIT = "1", "2"


@dataclass
class EllipticGraph:
    """The observed transaction-flow graph, with labels aligned to node indices."""

    adjacency: list[list[int]]
    labels: np.ndarray          # "1" illicit, "2" licit, "unknown"
    tx_ids: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.adjacency)

    @property
    def n_edges(self) -> int:
        return sum(len(n) for n in self.adjacency) // 2

    @property
    def is_labelled(self) -> np.ndarray:
        return np.isin(self.labels, [ILLICIT, LICIT])

    @property
    def is_illicit(self) -> np.ndarray:
        return self.labels == ILLICIT


def load_elliptic(strict: bool = True) -> EllipticGraph:
    """Build the observed graph. Node ids are remapped to contiguous indices.

    Integrity is asserted against the published node and edge counts so a truncated
    download fails here rather than surfacing later as an odd graph statistic.
    """
    if not CLASSES_CSV.exists() or not EDGELIST_CSV.exists():
        raise FileNotFoundError(
            f"Elliptic files not found in {ELLIPTIC_DIR}. "
            "Run: python scripts/fetch_elliptic.py"
        )

    classes = pd.read_csv(CLASSES_CSV, dtype={"txId": np.int64, "class": str})
    edges = pd.read_csv(EDGELIST_CSV, dtype=np.int64)

    if strict:
        if len(classes) != EXPECTED_NODES:
            raise ValueError(f"expected {EXPECTED_NODES} nodes, got {len(classes)}")
        if len(edges) != EXPECTED_EDGES:
            raise ValueError(f"expected {EXPECTED_EDGES} edges, got {len(edges)}")

    tx_ids = classes["txId"].to_numpy()
    index = {tx: i for i, tx in enumerate(tx_ids)}

    adjacency: list[list[int]] = [[] for _ in range(len(tx_ids))]
    for source, target in zip(edges["txId1"].to_numpy(), edges["txId2"].to_numpy()):
        a, b = index.get(source), index.get(target)
        if a is None or b is None or a == b:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)

    return EllipticGraph(
        adjacency=adjacency, labels=classes["class"].to_numpy(), tx_ids=tx_ids
    )


def project_bipartite(
    adjacency: list[list[int]], n_left: int, max_group: int = 64
) -> list[list[int]]:
    """Project a bipartite graph onto its left-hand nodes.

    WHY THIS IS NEEDED FOR THE COMPARISON
    -------------------------------------
    RingWatch's entity graph is bipartite: entities connect to attribute-value nodes and
    never directly to each other. Elliptic's is unipartite -- transactions link to
    transactions. An edge-level statistic run on the bipartite form finds **zero**
    entity-to-entity edges and returns nan, which is a safe failure but not a comparison.

    Projecting connects two entities whenever they share an attribute node, which is what
    an edge in that graph already means. `max_group` guards the quadratic blow-up: a shared
    value held by k entities contributes k(k-1)/2 edges, and the batch graph's hub
    suppression already caps k at 5, so the guard only matters if this is ever pointed at
    an unsuppressed graph.
    """
    projected: list[set[int]] = [set() for _ in range(n_left)]

    for right in range(n_left, len(adjacency)):
        members = [node for node in adjacency[right] if node < n_left]
        if len(members) < 2 or len(members) > max_group:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                projected[a].add(b)
                projected[b].add(a)

    return [sorted(neighbours) for neighbours in projected]


@dataclass
class GraphShape:
    """Structural description of one graph, for comparing inferred against observed."""

    name: str
    n_nodes: int
    n_edges: int
    mean_degree: float
    isolated_fraction: float
    n_components: int
    largest_component: int
    mean_component_size: float
    max_core_number: int

    def summary_lines(self) -> list[str]:
        return [
            f"  {'nodes':<26} {self.n_nodes:,}",
            f"  {'edges':<26} {self.n_edges:,}",
            f"  {'mean degree':<26} {self.mean_degree:.3f}",
            f"  {'isolated nodes':<26} {100 * self.isolated_fraction:.1f}%",
            f"  {'components (size >= 2)':<26} {self.n_components:,}",
            f"  {'largest component':<26} {self.largest_component:,} nodes",
            f"  {'mean component size':<26} {self.mean_component_size:.1f}",
            f"  {'max k-core number':<26} {self.max_core_number}",
        ]


def describe_graph(name: str, adjacency: list[list[int]]) -> GraphShape:
    """Measure a graph's shape using the project's own components and k-core code.

    Deliberately the same `connected_components` and `k_core_numbers` used everywhere
    else — if they needed changing to run on a different graph, that would itself be a
    finding about their generality.
    """
    degrees = np.array([len(n) for n in adjacency], dtype=np.int64)
    labels = connected_components(adjacency)
    cores = k_core_numbers(adjacency)

    sizes = pd.Series(labels).value_counts()
    multi = sizes[sizes >= 2]

    return GraphShape(
        name=name,
        n_nodes=len(adjacency),
        n_edges=int(degrees.sum() // 2),
        mean_degree=float(degrees.mean()),
        isolated_fraction=float((degrees == 0).mean()),
        n_components=int(len(multi)),
        largest_component=int(sizes.max()) if len(sizes) else 0,
        mean_component_size=float(multi.mean()) if len(multi) else 0.0,
        max_core_number=int(cores.max()) if len(cores) else 0,
    )
