"""Incremental k-core maintenance under edge insertion.

`core/graph.py` is untouched and is the correctness oracle for everything here. This module
maintains the same core numbers that a full Batagelj-Zaversnik rebuild would produce, but
repairs them locally after each inserted edge instead of recomputing from scratch.

THE RESULT THAT MAKES LOCAL REPAIR POSSIBLE
--------------------------------------------
Inserting a single edge (u, v) raises the core number of any vertex by **at most 1**, and
only vertices whose core number equals K = min(core(u), core(v)) can change at all.

Both halves matter. The first bounds how much repair is needed (one increment, never more).
The second bounds where: everything with core > K is already too deep to be affected, and
everything with core < K cannot be lifted past its own bottleneck by one new edge.

THE REPAIR
----------
After inserting the edge:

  1. K = min(core(u), core(v)); root is the endpoint attaining it.
  2. Candidate set C = every vertex with core == K reachable from root through vertices of
     core == K. Nothing outside C can change.
  3. Local peel. A candidate is promoted only if enough of its neighbours also end up at
     K+1, so its support counts neighbours with core > K (already above) plus neighbours
     still in C (promoted alongside it). Evict any candidate whose support is <= K,
     decrement its neighbours' support, repeat to a fixed point.
  4. Survivors get core += 1.

Step 3 is the subtle one. Counting *all* core == K neighbours would be wrong: a neighbour
in a different K-subcore component stays at K and cannot help lift this vertex to K+1.

WHY TRAVERSAL AND NOT THE ORDER-BASED ALGORITHM
------------------------------------------------
Order-based core maintenance (Zhang et al.) has better worst-case behaviour but maintains a
global vertex order as auxiliary state. The traversal approach (the Sariyuce et al. subcore
family) needs no state beyond the core numbers themselves and can be checked line by line
against a batch rebuild. Given that correctness here is non-negotiable while the benchmark
is explicitly allowed to come out negative, verifiability was worth more than constant
factors. See PLAN_INCREMENTAL.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.graph import k_core_numbers


@dataclass
class IncrementalCore:
    """A graph whose core numbers are maintained as edges arrive.

    `adjacency` and `core` are always consistent with what a full rebuild would produce —
    that invariant is what `tests/test_graph_incremental.py` asserts after every insertion.
    """

    adjacency: list[list[int]] = field(default_factory=list)
    core: list[int] = field(default_factory=list)

    # Instrumentation, for the benchmark rather than the algorithm.
    insertions: int = 0
    candidates_examined: int = 0
    promotions: int = 0

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_adjacency(cls, adjacency: list[list[int]]) -> "IncrementalCore":
        """Seed from an existing graph, computing core numbers once via the batch oracle."""
        copied = [list(neighbours) for neighbours in adjacency]
        return cls(adjacency=copied, core=list(k_core_numbers(copied)))

    def add_node(self) -> int:
        """Append an isolated vertex. Core number 0; nothing else can change."""
        self.adjacency.append([])
        self.core.append(0)
        return len(self.adjacency) - 1

    def ensure_node(self, node: int) -> None:
        while len(self.adjacency) <= node:
            self.add_node()

    # ------------------------------------------------------------------
    # insertion
    # ------------------------------------------------------------------

    def insert_edge(self, u: int, v: int) -> bool:
        """Insert an undirected edge and repair core numbers locally.

        Returns True if any core number changed. Self-loops and duplicate edges are
        ignored, since neither changes the core structure and both would corrupt degree
        counts if admitted.
        """
        if u == v:
            return False
        self.ensure_node(max(u, v))
        if v in self.adjacency[u]:
            return False

        self.adjacency[u].append(v)
        self.adjacency[v].append(u)
        self.insertions += 1

        k = min(self.core[u], self.core[v])
        root = u if self.core[u] <= self.core[v] else v

        candidates = self._collect_candidates(root, k)
        self.candidates_examined += len(candidates)

        promoted = self._peel(candidates, k)
        for node in promoted:
            self.core[node] += 1
        self.promotions += len(promoted)
        return bool(promoted)

    def _collect_candidates(self, root: int, k: int) -> set[int]:
        """Vertices with core == k reachable from root through core == k vertices.

        This is the "subcore" containing root. Traversal stops at any vertex with a
        different core number, which is what keeps the work local.
        """
        if self.core[root] != k:
            return set()

        seen = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbour in self.adjacency[node]:
                if neighbour not in seen and self.core[neighbour] == k:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen

    def _peel(self, candidates: set[int], k: int) -> set[int]:
        """Evict candidates that cannot reach core k+1, return those that can.

        Support counts neighbours that will sit at k+1 or above: those already deeper than
        k, plus those still surviving in the candidate set. A vertex needs more than k such
        neighbours to belong to the (k+1)-core.
        """
        if not candidates:
            return set()

        remaining = set(candidates)
        support = {
            node: sum(
                1
                for neighbour in self.adjacency[node]
                if self.core[neighbour] > k or neighbour in remaining
            )
            for node in remaining
        }

        # Cascade: evicting one vertex can drop a neighbour below the threshold.
        queue = [node for node in remaining if support[node] <= k]
        while queue:
            node = queue.pop()
            if node not in remaining:
                continue
            remaining.discard(node)
            for neighbour in self.adjacency[node]:
                if neighbour in remaining:
                    support[neighbour] -= 1
                    if support[neighbour] <= k:
                        queue.append(neighbour)

        return remaining

    # ------------------------------------------------------------------
    # verification helper
    # ------------------------------------------------------------------

    def batch_core(self) -> list[int]:
        """Core numbers as a full rebuild would compute them. The oracle."""
        return list(k_core_numbers(self.adjacency))

    def matches_batch(self) -> bool:
        return self.core == self.batch_core()


def build_incrementally(
    n_nodes: int, edges: list[tuple[int, int]]
) -> IncrementalCore:
    """Insert every edge one at a time into an empty graph."""
    graph = IncrementalCore(adjacency=[[] for _ in range(n_nodes)], core=[0] * n_nodes)
    for u, v in edges:
        graph.insert_edge(u, v)
    return graph
