"""The entity graph: construction, connected components, and k-core decomposition.

No AI, no network, no randomness. Every function here is a deterministic pure function of
its input, and the tests assert that.

ENTITY RESOLUTION
-----------------
IEEE-CIS has no customer ID. The competition-validated proxy is a "uid" fingerprint:

    uid = card1 + addr1 + (transaction_day - D1)

`D1` is days-since-the-card-was-first-seen, so `transaction_day - D1` is that card's
constant first-seen day, and the triple identifies one underlying account across its
transactions. It is a heuristic and it is wrong at the margins -- see README limitations.

Deliberately NOT built on `DeviceInfo` or `id_30`-`id_33`: the identity table covers only
a minority of rows, so those columns would produce a mostly-disconnected graph of noise.

WHY BIPARTITE, AND WHY HUB SUPPRESSION
--------------------------------------
Nodes are uids AND the attribute values that link them; an edge means "this uid has this
attribute value." The alternative -- projecting straight to a uid-uid graph by connecting
every pair sharing a value -- is quadratic in group size, and measurement showed the
largest shared-attribute group holds 79,048 uids. Expanding that one group alone would be
~3 billion edges of noise. The bipartite form costs k edges for a group of size k instead
of k(k-1)/2, and connected components are identical under projection either way.

Hub suppression is then not optional but load-bearing. An attribute value shared by
thousands of entities is a *generic* value (a common email provider, a populous
postcode), not evidence of a relationship. Values with more than MAX_GROUP_SIZE members
are dropped entirely. The threshold was chosen by measuring where the graph percolates:

    cap  uids linked   components>1   LARGEST COMPONENT
      3        8,988          3,694          10
      5       15,888          4,934          39
     10       28,077          5,627       1,752   <-- percolation
     20       44,186          5,506       9,867
     50       68,616          4,398      43,274   <-- one giant hairball

Between 5 and 10 the graph undergoes a phase transition and collapses into a giant
component, at which point "connected component" stops meaning "candidate ring" and starts
meaning "most of the dataset." The cap is set at 5, just below the transition. This buys
component *quality* at the cost of *coverage* (only ~8% of uids get any edge), which is a
real and stated limitation, not a hidden one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Below the measured percolation threshold. See module docstring.
MAX_GROUP_SIZE = 5

# Attribute combinations used to link entities. Each is a composite chosen to be more
# specific than any single column: sharing an email domain means nothing, but sharing an
# email domain AND a postcode is a coincidence worth a graph edge.
LINK_KEYS: dict[str, tuple[str, ...]] = {
    "card1": ("card1",),
    "addr1_pemail": ("addr1", "P_emaildomain"),
    "card3_card5_addr1": ("card3", "card5", "addr1"),
    "card1_card3_card5": ("card1", "card3", "card5"),
}

UID_COL = "uid"


def build_uid(df: pd.DataFrame) -> pd.Series:
    """Entity fingerprint per transaction; NA where any component is missing.

    Rows with a missing component get no entity and therefore no graph features, rather
    than being lumped into a spurious shared 'unknown' entity -- which would create an
    enormous fake ring out of every row with a null addr1.
    """
    day = df["TransactionDT"] // 86_400
    start_day = day - df["D1"]

    def part(series: pd.Series) -> pd.Series:
        return series.astype("float64").astype("Int64").astype(str)

    uid = part(df["card1"]) + "_" + part(df["addr1"]) + "_" + part(start_day)
    missing = df["card1"].isna() | df["addr1"].isna() | df["D1"].isna()
    return uid.where(~missing, other=pd.NA)


@dataclass
class EntityGraph:
    """A bipartite uid <-> attribute-value graph in CSR-ish adjacency-list form.

    Nodes 0 .. n_uids-1 are entities; nodes n_uids .. n_nodes-1 are attribute values.
    """

    uids: np.ndarray  # uid string per entity node, index-aligned
    adjacency: list[list[int]]
    n_uids: int
    n_nodes: int
    n_edges: int
    dropped_hub_values: int
    dropped_hub_members: int

    def uid_index(self) -> dict[str, int]:
        return {uid: i for i, uid in enumerate(self.uids)}


def build_graph(
    entity_frame: pd.DataFrame,
    link_keys: dict[str, tuple[str, ...]] | None = None,
    max_group_size: int = MAX_GROUP_SIZE,
) -> EntityGraph:
    """Build the bipartite entity graph from one row per uid.

    `entity_frame` must be one row per uid, carrying the columns named in `link_keys`.
    Ordering is made explicit (uids sorted) so the node numbering -- and therefore every
    downstream metric -- is identical across runs and independent of input row order.
    """
    if link_keys is None:
        link_keys = LINK_KEYS

    frame = entity_frame.sort_values(UID_COL, kind="mergesort").reset_index(drop=True)
    uids = frame[UID_COL].to_numpy()
    n_uids = len(uids)

    adjacency: list[list[int]] = [[] for _ in range(n_uids)]
    next_node = n_uids
    n_edges = 0
    dropped_values = 0
    dropped_members = 0

    for key_name, columns in sorted(link_keys.items()):
        missing = np.zeros(n_uids, dtype=bool)
        for column in columns:
            missing |= frame[column].isna().to_numpy()

        # Built column-by-column with str.cat rather than a row-wise agg: row-wise
        # joining breaks on mixed category/float dtypes and is orders of magnitude
        # slower over 199k entities.
        parts = [frame[column].astype(str) for column in columns]
        composite = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep="|")
        composite = composite.where(~pd.Series(missing, index=composite.index))

        # Deterministic group iteration: sort_index() fixes the order regardless of the
        # hash-table ordering pandas happens to produce.
        for value, positions in sorted(
            composite.dropna().groupby(composite.dropna()).groups.items()
        ):
            members = np.sort(np.asarray(positions, dtype=np.int64))
            size = len(members)

            if size < 2:
                # A value held by a single entity links nothing.
                continue
            if size > max_group_size:
                # Hub: generic value, not evidence of a relationship. Dropped.
                dropped_values += 1
                dropped_members += size
                continue

            attribute_node = next_node
            next_node += 1
            adjacency.append([])
            for member in members:
                adjacency[member].append(attribute_node)
                adjacency[attribute_node].append(int(member))
                n_edges += 1

    return EntityGraph(
        uids=uids,
        adjacency=adjacency,
        n_uids=n_uids,
        n_nodes=next_node,
        n_edges=n_edges,
        dropped_hub_values=dropped_values,
        dropped_hub_members=dropped_members,
    )


def connected_components(adjacency: list[list[int]]) -> np.ndarray:
    """Component label per node, via union-find with path compression + union by size.

    Labels are canonicalised to be order-independent: the label of a component is the
    smallest node id it contains, so two runs -- or two different internal union orders --
    produce identical labels.
    """
    n = len(adjacency)
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for node, neighbours in enumerate(adjacency):
        for neighbour in neighbours:
            a, b = find(node), find(neighbour)
            if a == b:
                continue
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]

    representative = np.array([find(i) for i in range(n)], dtype=np.int64)

    # Canonicalise: component label := min node id in that component.
    labels = np.full(n, -1, dtype=np.int64)
    smallest: dict[int, int] = {}
    for node in range(n):
        root = int(representative[node])
        if root not in smallest:
            smallest[root] = node
    for node in range(n):
        labels[node] = smallest[int(representative[node])]
    return labels


def k_core_numbers(adjacency: list[list[int]]) -> np.ndarray:
    """Core number per node — Batagelj-Zaversnik O(V + E) bucket peeling.

    A node's core number k means it belongs to the maximal subgraph in which every node
    has degree >= k. It is the standard measure of how deeply embedded a node is in a
    dense region, and it is exactly the collusion signal we want: an entity linked through
    several independent attributes to other entities that are themselves multiply linked
    sits in a high core, whereas a chain of one-off coincidences peels away at k=1.

    Implemented directly rather than called from networkx. The algorithm sorts vertices
    into degree buckets once, then repeatedly removes a minimum-degree vertex and
    decrements its neighbours, maintaining the bucket order in O(1) per operation by
    swapping vertices within the bucket array. Validated against networkx.core_number in
    the tests, including on pathological graphs.
    """
    n = len(adjacency)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)
    max_degree = int(degree.max()) if n else 0

    # bin[d] will become the starting index of degree-d vertices inside `vert`.
    bin_start = np.zeros(max_degree + 2, dtype=np.int64)
    for d in degree:
        bin_start[d] += 1

    start = 0
    for d in range(max_degree + 1):
        count = bin_start[d]
        bin_start[d] = start
        start += count

    # `vert` holds vertices ordered by degree; `pos` is the inverse permutation.
    position = np.zeros(n, dtype=np.int64)
    vert = np.zeros(n, dtype=np.int64)
    for v in range(n):
        position[v] = bin_start[degree[v]]
        vert[position[v]] = v
        bin_start[degree[v]] += 1

    # Restore bin starts, shifted right by the loop above.
    for d in range(max_degree, 0, -1):
        bin_start[d] = bin_start[d - 1]
    bin_start[0] = 0

    core = degree.copy()
    for i in range(n):
        v = int(vert[i])
        for u in adjacency[v]:
            if core[u] > core[v]:
                du, pu = core[u], position[u]
                pw = bin_start[du]
                w = int(vert[pw])
                if u != w:
                    # Swap u into the front slot of its degree bucket.
                    position[u], vert[pu] = pw, w
                    position[w], vert[pw] = pu, u
                bin_start[du] += 1
                core[u] -= 1

    return core


def graph_features(graph: EntityGraph) -> pd.DataFrame:
    """Per-uid structural features.

    STRICTLY STRUCTURAL -- no feature here touches the fraud label. A feature like
    "fraud rate of this entity's component" would be far more predictive and would also
    leak test labels into training through the graph, which is precisely the trap this
    layer exists to avoid falling into. Everything below is computable at inference time
    for an entity whose label is unknown.
    """
    labels = connected_components(graph.adjacency)
    cores = k_core_numbers(graph.adjacency)

    component_sizes = pd.Series(labels).value_counts()
    uid_labels = labels[: graph.n_uids]

    # Component size counted in ENTITIES, excluding the attribute-value nodes, which are
    # scaffolding rather than members of the ring.
    entity_component_size = (
        pd.Series(uid_labels).map(pd.Series(uid_labels).value_counts()).to_numpy()
    )

    degrees = np.array(
        [len(graph.adjacency[i]) for i in range(graph.n_uids)], dtype=np.int64
    )

    return pd.DataFrame(
        {
            UID_COL: graph.uids,
            "g_component_size": entity_component_size,
            "g_component_size_total": pd.Series(uid_labels)
            .map(component_sizes)
            .to_numpy(),
            "g_core_number": cores[: graph.n_uids],
            "g_degree": degrees,
            "g_is_linked": (degrees > 0).astype(np.int64),
        }
    )


def entity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse transactions to one row per entity, carrying the link attributes.

    Where an entity's transactions disagree on an attribute (possible for card3/card5/
    email, though not for card1/addr1 which are baked into the uid itself), the earliest
    observed value wins. Sorting first makes that choice deterministic instead of
    dependent on groupby internals.
    """
    columns = sorted({column for cols in LINK_KEYS.values() for column in cols})
    subset = df.dropna(subset=[UID_COL]).sort_values("TransactionDT", kind="mergesort")
    return subset.groupby(UID_COL, observed=True, as_index=False)[columns].first()


GRAPH_FEATURE_COLUMNS = [
    "g_component_size",
    "g_component_size_total",
    "g_core_number",
    "g_degree",
    "g_is_linked",
]


def attach_graph_features(
    df: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Left-join per-entity graph features onto transactions.

    Transactions with no resolvable uid (11.3% of rows -- a null addr1 or D1) get NaN
    rather than 0. The distinction matters: 0 means "this entity exists and is connected
    to nothing", NaN means "we could not identify an entity at all", and LightGBM can
    learn a different split for each. Collapsing them would assert something false.
    """
    merged = df.merge(features, on=UID_COL, how="left")
    merged.index = df.index
    return merged


def build_features_for_split(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build graph features for both sides WITHOUT leaking the future into training.

    This is the subtle part of the whole layer. A single graph spanning the full dataset
    would connect training entities to transactions that had not happened yet at training
    time, letting a train row's features depend on the future. That is exactly the
    leakage the temporal split exists to prevent, smuggled back in through the graph.

    So two graphs are built:
      * train features come from a graph over TRAIN transactions only;
      * test features come from a graph over train + test transactions, which is what a
        deployed system would legitimately have -- all history up to the moment of
        scoring.

    An entity therefore may have different features in the two frames, which is correct:
    its neighbourhood genuinely grew over time.
    """
    train_graph = build_graph(entity_frame(train))
    train_features = graph_features(train_graph)

    combined = pd.concat([train, test], axis=0)
    full_graph = build_graph(entity_frame(combined))
    full_features = graph_features(full_graph)

    return (
        attach_graph_features(train, train_features),
        attach_graph_features(test, full_features),
        {
            "train_graph": graph_summary(train_graph, train_features),
            "full_graph": graph_summary(full_graph, full_features),
        },
    )


def graph_summary(graph: EntityGraph, features: pd.DataFrame) -> dict:
    linked = features[features["g_is_linked"] == 1]
    sizes = linked["g_component_size"]
    return {
        "uid_nodes": graph.n_uids,
        "attribute_nodes": graph.n_nodes - graph.n_uids,
        "edges": graph.n_edges,
        "hub_values_dropped": graph.dropped_hub_values,
        "hub_memberships_dropped": graph.dropped_hub_members,
        "linked_uids": int(len(linked)),
        "linked_fraction": float(len(linked) / graph.n_uids) if graph.n_uids else 0.0,
        "largest_component_entities": int(sizes.max()) if len(sizes) else 0,
        "mean_component_size": float(sizes.mean()) if len(sizes) else 0.0,
        "max_core_number": int(features["g_core_number"].max()),
    }
