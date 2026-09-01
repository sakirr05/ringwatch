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
