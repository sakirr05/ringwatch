# Elliptic — plan, label finding, and pre-registered predictions

Written **before** any integration work, so the conclusion cannot be retrofitted to
whatever the data happens to show. Same discipline as `PLAN_INCREMENTAL.md`, where the
recorded prediction turned out to be wrong by ~100× and stayed on the record.

---

## Why this was proposed, and why that reason does not survive contact with the data

RingWatch's largest stated limitation is that IEEE-CIS carries **transaction-level fraud
labels, not ring labels**. The project therefore never claims "N rings caught" — that
number cannot be validated. Elliptic was proposed to close that gap.

**It does not close it.** I checked the actual files before writing this, rather than
assuming from the dataset's reputation:

```
elliptic_txs_classes.csv    txId,class            1 = illicit, 2 = licit, unknown
                            230425980,unknown
                            232438397,2

elliptic_txs_edgelist.csv   txId1,txId2
                            230425980,5530458
```

`class` is attached to a **transaction**, exactly as `isFraud` is in IEEE-CIS. There is no
ring identifier, no cluster identifier, no actor identifier. **The ring-level ground-truth
limitation stands unchanged and must remain in the README.**

If the phase were justified only by that claim, the honest move would be to stop here —
and the brief explicitly permits it.

## What Elliptic does provide, which is worth having

A **real, observed graph**. IEEE-CIS has no edges at all; RingWatch had to *infer* an entity
graph from a `card1 + addr1 + (day − D1)` fingerprint heuristic, then suppress hubs to stop
it percolating. Every structural claim the project makes therefore rests on that heuristic
being reasonable — something the README currently asserts rather than measures.

Elliptic's edges are observed bitcoin transaction flows. Running the same concentration test
on a graph nobody had to invent **removes entity resolution as a confound**.

So the phase proceeds with a corrected claim:

| what it tests | what it does NOT test |
|---|---|
| Does the concentration finding replicate on a graph whose edges are observed? | Whether the clusters are rings — still unverifiable |
| How much does the fingerprint heuristic distort structure? | Whether RingWatch's specific uid heuristic is *correct* |

## The semantic caveat that limits the replication

**These are not the same kind of graph, and a replication is weaker evidence than it looks.**

- **IEEE-CIS (inferred):** nodes are entities, edges mean *shared identity attributes*. A
  component is a set of accounts that look like the same actor.
- **Elliptic (observed):** nodes are transactions, edges mean *money flowed between them*. A
  component is a connected payment flow.

"Does illicit activity concentrate in components?" is a coherent question for both, but it
is not the *same* question. A positive result on Elliptic supports the general claim that
illicit activity is structurally clustered; it does not validate RingWatch's entity graph.
That distinction has to survive into the README.

## A methodological wrinkle to handle explicitly

Elliptic is mostly **unlabelled** — roughly 2% illicit, 21% licit, and **77% unknown**. The
permutation test needs labels, and there are three ways to handle that, two of them wrong:

- ~~Treat unknown as licit~~ — invents 157k negative labels and would inflate any
  concentration result.
- ~~Drop unknown nodes before building the graph~~ — deletes observed edges, which is
  precisely the property Elliptic was brought in for.
- **Build the graph on all observed edges, then run the concentration test over labelled
  nodes only.** Components are computed from real structure; the statistic is evaluated
  only where ground truth exists. "All-illicit component" then means all-illicit *among its
  labelled members*, which is what will be reported.

---

## Pre-registered predictions

1. **Concentration will replicate, and strongly** — a larger z than IEEE-CIS's +8.8. Money
   flow physically connects one actor's transactions, which is a more direct link than two
   accounts happening to share a postcode and an email domain.
2. **Elliptic's components will be far larger.** Transaction-flow graphs percolate; I expect
   a giant component holding a large share of nodes, where RingWatch's hub-suppressed graph
   maxes out at 39 entities. If so, "connected component" is a **weaker** unit of analysis
   on Elliptic than on IEEE-CIS, not a stronger one — and that would be an argument *for*
   hub suppression, arrived at from the opposite direction.
3. **The graph-quality comparison will show the inferred graph is much sparser** — average
   degree well under 1 versus something near 2 for Elliptic, and a far larger isolated
   fraction (IEEE-CIS: ~92% of entities isolated).
4. **k-core will reach deeper on Elliptic.** RingWatch's max core number is 3; I expect
   Elliptic's to exceed it.

### What would falsify these

If illicit activity turns out **not** to concentrate on the observed graph (z inside ±3),
that is a genuinely awkward result for the project's central structural claim and gets
reported as prominently as the graph layer's failure to lift. It would suggest the IEEE-CIS
concentration finding is an artifact of how the entity graph was built rather than a
property of fraud.

## Scope

- Reuse `connected_components`, `k_core_numbers` (`core/graph.py`) and
  `ring_concentration_test` (`core/ring_evidence.py`) **unchanged**. If they need modifying
  to run on Elliptic, that is a finding about their generality and gets reported.
- No model is trained on Elliptic. No ablation row. This is a structural replication, not a
  second detector.
- `scripts/fetch_elliptic.py` mirrors `scripts/fetch_data.py`, with integrity assertions so
  a truncated download fails loudly.
