"""Evidence that fraud clusters in the entity graph — tested properly.

WHY THIS MODULE EXISTS
----------------------
The obvious way to ask "does the graph see rings?" is to tabulate fraud rate by core
number or by component size. I did that first and it showed essentially nothing: rates of
2.46% / 4.09% / 2.10% / 6.76% across cores 0-3, non-monotonic, with linked entities
sitting *below* the 3.50% base rate. On that evidence the honest conclusion would have
been "the graph layer finds nothing."

That test was simply the wrong one. Rings are **rare**: a handful of fully-fraudulent
components among thousands of benign ones. Averaging fraud rate over all components
drowns twelve real rings in five thousand ordinary ones, so a null result there is
uninformative rather than negative.

The right question is about CONCENTRATION, not rate: given the number of fraudulent
entities that exist, are they distributed across components the way random chance would
scatter them, or are they bunched together into components that are *entirely*
fraudulent? That has a clean null hypothesis and a clean test.

THE TEST
--------
Statistic: the number of components (size >= 2) in which EVERY entity is fraudulent.
Null: entity fraud labels are shuffled uniformly at random among linked entities, holding
the graph topology and the number of fraudulent entities fixed. Repeat, build the null
distribution of the statistic, and report the observed value as a z-score against it.

This isolates topology from prevalence. If the graph carried no ring information, wiring
would be independent of labels and the observed count would sit inside the null.

RESULT (cap 5, full dataset): 12 all-fraud components observed against a null of
1.4 +/- 1.2, i.e. **z = +8.8**. Stable across hub-suppression caps (+8.8, +8.6, +9.1 at
caps 5, 10, 20), so it is not an artifact of one threshold choice.

WHAT THIS DOES AND DOES NOT LICENSE
-----------------------------------
It licenses: "fraud-labeled transactions cluster more densely in the entity graph than
chance allows." It does NOT license "RingWatch caught 12 fraud rings" -- IEEE-CIS has no
ring-level ground truth, so those 12 components are *statistically anomalous clusters*,
not verified rings. See the README limitations section.

It also sets an honest expectation for the ablation: rings this rare can only move a
dataset-wide PR-AUC a little, because the graph features are informative on a very small
slice of the data. A large lift here would be evidence of a bug, not of success.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_PERMUTATIONS = 200
PERMUTATION_SEED = 0


@dataclass
class RingEvidence:
    linked_entities: int
    components: int
    entity_fraud_rate: float
    all_fraud_components: int
    mixed_components: int
    clean_components: int
    null_mean: float
    null_std: float
    z_score: float

    def summary_lines(self) -> list[str]:
        return [
            f"  linked entities            {self.linked_entities:,}",
            f"  components (size >= 2)     {self.components:,}",
            f"  entity-level fraud rate    {self.entity_fraud_rate:.4f}",
            f"  all-fraud components       {self.all_fraud_components}",
            f"  mixed components           {self.mixed_components}",
            f"  all-clean components       {self.clean_components}",
            f"  permutation null           {self.null_mean:.1f} +/- {self.null_std:.1f}"
            f"  (n={N_PERMUTATIONS})",
            f"  z-score                    {self.z_score:+.1f}",
        ]


@dataclass
class HomophilyEvidence:
    """Edge-level clustering: do illicit nodes neighbour illicit nodes beyond chance?"""

    labelled_edges: int
    illicit_illicit_edges: int
    observed_rate: float
    null_mean: float
    null_std: float
    z_score: float

    def summary_lines(self) -> list[str]:
        return [
            f"  edges between labelled nodes  {self.labelled_edges:,}",
            f"  illicit-illicit edges         {self.illicit_illicit_edges:,}",
            f"  observed rate                 {self.observed_rate:.4f}",
            f"  permutation null              {self.null_mean:.4f} +/- {self.null_std:.4f}"
            f"  (n={N_PERMUTATIONS})",
            f"  z-score                       {self.z_score:+.1f}",
        ]


def label_homophily_test(
    adjacency: list[list[int]],
    labels: np.ndarray,
    testable: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> HomophilyEvidence:
    """Do illicit nodes sit next to illicit nodes more than chance allows?

    WHY THIS EXISTS ALONGSIDE `ring_concentration_test`
    ----------------------------------------------------
    The component statistic counts components in which EVERY labelled member is illicit.
    That is a good statistic when components are small -- IEEE-CIS averages about five
    entities, so "all of them" is an achievable and meaningful event.

    It is **degenerate on a percolated graph**. Elliptic's observed transaction-flow graph
    has 49 components averaging 4,158 labelled members; no component there could be
    all-illicit, and the permutation null predicts none either. The result is 0 against
    0 +/- 0, which reads as z = 0 and looks like a clean null while actually meaning the
    test had no power at all. Reporting that as "no concentration" would have been a false
    negative dressed as a finding.

    This statistic works at any component size because its unit is the EDGE, not the
    component: the share of labelled-to-labelled edges that join two illicit nodes,
    compared against shuffling the labels across testable nodes with the topology held
    fixed. On a graph where clustering is real, illicit nodes adjoin each other more often
    than a shuffle produces.
    """
    labels = np.asarray(labels).astype(int)
    testable = np.asarray(testable, dtype=bool)

    # Every edge with both endpoints labelled, deduplicated.
    pairs = [
        (node, neighbour)
        for node, neighbours in enumerate(adjacency)
        if testable[node]
        for neighbour in neighbours
        if neighbour > node and testable[neighbour]
    ]
    if not pairs:
        return HomophilyEvidence(0, 0, float("nan"), float("nan"), float("nan"), float("nan"))

    left = np.array([a for a, _ in pairs], dtype=np.int64)
    right = np.array([b for _, b in pairs], dtype=np.int64)

    def illicit_pairs(values: np.ndarray) -> int:
        return int(np.sum(values[left] & values[right]))

    marks = (labels == 1) & testable
    observed = illicit_pairs(marks)

    # Shuffle labels across testable nodes, holding the graph fixed. Topology is held
    # constant so the null isolates label placement rather than graph structure.
    testable_index = np.flatnonzero(testable)
    testable_marks = marks[testable_index]

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    shuffled = np.zeros(len(labels), dtype=bool)
    for i in range(n_permutations):
        shuffled[:] = False
        shuffled[testable_index] = rng.permutation(testable_marks)
        null[i] = illicit_pairs(shuffled)

    std = float(null.std())
    return HomophilyEvidence(
        labelled_edges=len(pairs),
        illicit_illicit_edges=observed,
        observed_rate=observed / len(pairs),
        null_mean=float(null.mean()) / len(pairs),
        null_std=std / len(pairs),
        z_score=float((observed - null.mean()) / (std + 1e-9)),
    )


def ring_concentration_test(
    component_labels: np.ndarray,
    entity_fraud: np.ndarray,
    linked_mask: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> RingEvidence:
    """Permutation test for fraud concentration within connected components.

    Deterministic given `seed`, so the reported z-score is reproducible.
    """
    labels = component_labels[linked_mask]
    fraud = entity_fraud[linked_mask].astype(np.float64)

    codes, _ = pd.factorize(labels)
    sizes = np.bincount(codes)
    keep = sizes >= 2

    def count_all_fraud(values: np.ndarray) -> int:
        sums = np.bincount(codes, weights=values)
        return int(np.sum(sums[keep] == sizes[keep]))

    observed = count_all_fraud(fraud)

    rng = np.random.default_rng(seed)
    null = np.array(
        [count_all_fraud(rng.permutation(fraud)) for _ in range(n_permutations)],
        dtype=np.float64,
    )

    fraud_sums = np.bincount(codes, weights=fraud)
    kept_sums, kept_sizes = fraud_sums[keep], sizes[keep]

    null_std = float(null.std())
    return RingEvidence(
        linked_entities=int(linked_mask.sum()),
        components=int(keep.sum()),
        entity_fraud_rate=float(fraud.mean()),
        all_fraud_components=observed,
        mixed_components=int(np.sum((kept_sums > 0) & (kept_sums < kept_sizes))),
        clean_components=int(np.sum(kept_sums == 0)),
        null_mean=float(null.mean()),
        null_std=null_std,
        z_score=float((observed - null.mean()) / (null_std + 1e-9)),
    )
