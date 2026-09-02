"""PageRank and betweenness centrality over the entity graph.

WHY THESE TWO
-------------
PayPal's engineering blog names them directly for ring detection: "we can leverage graph
algorithms like page rank to identify the most influential sellers or high frequency paths;
or we can use connected components, clustering, and centrality algorithms to detect network
or community to determine whether the connected network is a group of friends or a fraud
ring."

The intuition is that a coordinated ring has structure a benign cluster does not — a
coordinator sits at the centre, and the paths between members run through them.

WHY IT PROBABLY WILL NOT WORK HERE, PREDICTED IN ADVANCE
---------------------------------------------------------
PayPal runs this over a graph with 400M+ actors, where centrality genuinely varies. This
project's entity graph has components averaging **4.3 entities**, maxing out at 39. Across
four nodes PageRank is roughly 0.25 each and betweenness is 0 for most of them, so these
features are expected to be close to constant and carry almost no information.

`PLAN_VALUE_WEIGHTED.md` records that prediction before measurement, along with the specific
falsifiable thresholds. `component_centrality_variance` below measures it directly, so the
claim is settled by data rather than by argument — whichever way the ablation lands.

BOTH ARE HAND-IMPLEMENTED, and validated against networkx in `tests/test_centrality.py`, the
same oracle pattern used for `k_core_numbers`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

DAMPING = 0.85
MAX_ITERATIONS = 100
TOLERANCE = 1e-10


def pagerank(adjacency: list[list[int]], damping: float = DAMPING) -> np.ndarray:
    """PageRank by power iteration on an undirected graph.

    Iterates the random-surfer recurrence to a fixed point:

        PR(v) = (1 - d) / N  +  d * ( sum over neighbours u of PR(u) / deg(u)
                                      + dangling_mass / N )

    Dangling nodes (degree 0) would otherwise leak probability mass out of the system, so
    their rank is redistributed uniformly each iteration — this is what keeps the vector
    summing to 1 and is the detail most naive implementations get wrong.

    Deterministic: fixed iteration cap and tolerance, no randomness anywhere.
    """
    n = len(adjacency)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.float64)
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    dangling = degree == 0

    for _ in range(MAX_ITERATIONS):
        contribution = np.zeros(n, dtype=np.float64)
        # Mass sitting on degree-0 nodes has nowhere to flow; spread it over everyone.
        leaked = rank[dangling].sum() / n

        for node, neighbours in enumerate(adjacency):
            if not neighbours:
                continue
            share = rank[node] / degree[node]
            for neighbour in neighbours:
                contribution[neighbour] += share

        updated = (1.0 - damping) / n + damping * (contribution + leaked)

        if np.abs(updated - rank).sum() < TOLERANCE:
            return updated
        rank = updated

    return rank


def _components(adjacency: list[list[int]]) -> list[list[int]]:
    """Vertex sets of each connected component, via BFS."""
    n = len(adjacency)
    seen = np.zeros(n, dtype=bool)
    groups: list[list[int]] = []

    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        group = [start]
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    group.append(neighbour)
                    queue.append(neighbour)
        groups.append(group)
    return groups


def betweenness(adjacency: list[list[int]]) -> np.ndarray:
    """Betweenness centrality via Brandes' algorithm, normalised.

    Brandes computes, for every source s, the shortest-path DAG by BFS, counts the number
    of shortest paths sigma[v] reaching each vertex, then accumulates dependencies in
    reverse BFS order. That accumulation is what makes it O(V*E) instead of the O(V^3) a
    naive all-pairs implementation would cost.

    RUN PER COMPONENT, WHICH IS WHY THIS FINISHES
    ----------------------------------------------
    O(V*E) is still hopeless at this project's scale: the entity graph has 208,914 nodes
    and 29,285 edges, so sourcing a BFS from every vertex is ~6e9 Python-level operations.
    The first version of this function did exactly that and had to be killed after twelve
    minutes.

    The fix is exact rather than approximate. Shortest paths never cross between connected
    components, so a vertex can only lie between vertices in its own component. Running
    Brandes separately per component therefore gives identical results, and the entity
    graph's components max out at 39 vertices. Components smaller than 3 are skipped
    outright, since nothing can lie between fewer than two other vertices.

    Normalisation matches networkx's default for undirected graphs and uses the FULL graph
    size, not the component size: divide by (n-1)(n-2)/2. That convention is what
    `test_betweenness_matches_networkx_on_a_disconnected_graph` pins down.
    """
    n = len(adjacency)
    score = np.zeros(n, dtype=np.float64)
    if n < 3:
        return score  # no vertex can lie between two others

    for group in _components(adjacency):
        if len(group) >= 3:
            _brandes_within(adjacency, group, score)

    # Undirected: each pair counted twice above.
    score /= 2.0
    scale = 2.0 / ((n - 1) * (n - 2))
    return score * scale


def _brandes_within(
    adjacency: list[list[int]], group: list[int], score: np.ndarray
) -> None:
    """Accumulate unnormalised betweenness for one connected component, in place.

    State is held in dicts scoped to the component rather than arrays sized to the whole
    graph. That distinction matters enormously here: allocating three 208,914-element
    arrays per source, for thousands of sources, dominates the runtime completely even
    though each component only has a handful of vertices in it.
    """
    for source in group:
        stack: list[int] = []
        predecessors: dict[int, list[int]] = {}
        sigma: dict[int, float] = {source: 1.0}
        distance: dict[int, int] = {source: 0}
        queue = deque([source])

        while queue:
            node = queue.popleft()
            stack.append(node)
            node_distance = distance[node]
            for neighbour in adjacency[node]:
                if neighbour not in distance:  # first time seen
                    distance[neighbour] = node_distance + 1
                    queue.append(neighbour)
                if distance[neighbour] == node_distance + 1:  # on a shortest path
                    sigma[neighbour] = sigma.get(neighbour, 0.0) + sigma[node]
                    predecessors.setdefault(neighbour, []).append(node)

        dependency: dict[int, float] = {}
        while stack:  # reverse BFS order
            node = stack.pop()
            coefficient = (1.0 + dependency.get(node, 0.0)) / sigma[node]
            for predecessor in predecessors.get(node, ()):
                dependency[predecessor] = (
                    dependency.get(predecessor, 0.0) + sigma[predecessor] * coefficient
                )
            if node != source:
                score[node] += dependency.get(node, 0.0)


CENTRALITY_FEATURE_COLUMNS = ["g_pagerank", "g_betweenness"]


def centrality_features(graph) -> "pd.DataFrame":  # noqa: F821
    """Per-entity PageRank and betweenness for a built EntityGraph.

    Structural only, like every other graph feature here — neither quantity touches the
    fraud label, so neither can leak it.
    """
    import pandas as pd

    ranks = pagerank(graph.adjacency)
    between = betweenness(graph.adjacency)
    return pd.DataFrame(
        {
            "uid": graph.uids,
            "g_pagerank": ranks[: graph.n_uids],
            "g_betweenness": between[: graph.n_uids],
        }
    )


def build_centrality_for_split(train, test):
    """Centrality features for train and test, mirroring the batch feature split exactly.

    Deliberately rebuilds the two graphs rather than reusing `build_features_for_split`'s
    internals: `core/graph.py` must stay byte-identical, because the cached model scores
    behind every published number in this project were produced by it. Rebuilding costs
    about two seconds and buys that guarantee.

    Same leakage discipline as the batch features: the training graph is built from
    training transactions only, so a training row's centrality cannot depend on
    transactions that had not happened yet.
    """
    import pandas as pd

    from core.graph import UID_COL, build_graph, entity_frame

    train_graph = build_graph(entity_frame(train))
    full_graph = build_graph(entity_frame(pd.concat([train, test], axis=0)))

    def attach(frame, features):
        merged = frame.merge(features, on=UID_COL, how="left")
        merged.index = frame.index
        return merged

    return (
        attach(train, centrality_features(train_graph)),
        attach(test, centrality_features(full_graph)),
        full_graph,
    )


@dataclass
class CentralityVariance:
    """How much do centrality scores actually vary inside a component?

    If they do not vary, the feature is constant where it applies and cannot help a model,
    regardless of how good the algorithm is. This is the mechanism behind the prediction in
    PLAN_VALUE_WEIGHTED.md.
    """

    n_components: int
    median_pagerank_cv: float
    mean_pagerank_cv: float
    zero_betweenness_fraction: float
    mean_component_size: float

    def summary_lines(self) -> list[str]:
        return [
            f"  components (size >= 2)          {self.n_components:,}",
            f"  mean component size             {self.mean_component_size:.1f} entities",
            f"  median within-component CV      {self.median_pagerank_cv:.4f}  (PageRank)",
            f"  mean within-component CV        {self.mean_pagerank_cv:.4f}  (PageRank)",
            f"  entities with betweenness == 0  "
            f"{100 * self.zero_betweenness_fraction:.1f}%",
        ]


def component_centrality_variance(
    labels: np.ndarray,
    pagerank_scores: np.ndarray,
    betweenness_scores: np.ndarray,
    linked_mask: np.ndarray,
) -> CentralityVariance:
    """Measure whether centrality carries information at this graph's component sizes."""
    labels = np.asarray(labels)
    linked = np.asarray(linked_mask, dtype=bool)

    coefficients: list[float] = []
    sizes: list[int] = []
    for label in np.unique(labels[linked]):
        members = (labels == label) & linked
        size = int(members.sum())
        if size < 2:
            continue
        sizes.append(size)
        values = pagerank_scores[members]
        mean = values.mean()
        coefficients.append(float(values.std() / mean) if mean > 0 else 0.0)

    linked_betweenness = betweenness_scores[linked]
    return CentralityVariance(
        n_components=len(coefficients),
        median_pagerank_cv=float(np.median(coefficients)) if coefficients else float("nan"),
        mean_pagerank_cv=float(np.mean(coefficients)) if coefficients else float("nan"),
        zero_betweenness_fraction=(
            float((linked_betweenness == 0).mean()) if linked_betweenness.size else float("nan")
        ),
        mean_component_size=float(np.mean(sizes)) if sizes else float("nan"),
    )
