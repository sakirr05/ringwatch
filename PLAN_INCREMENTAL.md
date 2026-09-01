# Incremental k-core maintenance — plan and pre-registered prediction

Written **before** implementing or benchmarking, so that the result cannot be retrofitted
to whatever the measurement happens to show. That is the same discipline the project's
central negative ablation was held to.

---

## The problem

RingWatch rebuilds its entity graph in batch. Production fraud graphs do not arrive in
batch — transactions stream in, and each one can add entities and edges. Recomputing every
core number from scratch on each arrival is O(V + E) per update; maintaining them
incrementally should cost far less, because a single edge insertion can only perturb a
small neighbourhood.

This is the one enhancement that exercises what this project is actually differentiated on:
hand-implemented, reference-validated classical graph algorithms.

## Background

Core maintenance under edge updates is a studied problem. The relevant line of work:

- **Sarıyüce et al.**, streaming k-core maintenance — the traversal/subcore family, which
  identifies a restricted candidate set around the inserted edge and repairs only that.
- **Zhang et al.**, order-based core maintenance — maintains a k-order over vertices,
  achieving better worst-case behaviour at the cost of extra maintained state.
- Related work on distributed and parallel core maintenance, out of scope here.

The theoretical result everything rests on:

> **Inserting a single edge (u, v) increases the core number of any vertex by at most 1,
> and only vertices whose core number equals K = min(core(u), core(v)) can change at all.**

That is what makes a local repair possible: the blast radius of one insertion is bounded.

## Chosen approach: traversal / subcore repair

The order-based algorithm has better asymptotics but requires maintaining a global vertex
order as auxiliary state. The traversal approach is simpler, needs no extra state beyond the
core numbers themselves, and is far easier to verify line-by-line against a batch oracle —
which matters more here than constant factors, because **correctness is non-negotiable and
the benchmark is allowed to come out negative**.

Algorithm for inserting edge (u, v):

1. Insert the edge into the adjacency structure.
2. Let `K = min(core(u), core(v))`, and let `root` be the endpoint attaining it.
3. Build the **candidate set** `C`: all vertices with `core == K` reachable from `root`
   through vertices of `core == K`. Nothing outside this set can change.
4. Repair by local peeling. For `w ∈ C`, its support is the number of neighbours that will
   end up with core ≥ K+1:
   - neighbours with `core > K` (already above), plus
   - neighbours still remaining in `C` (which would be promoted alongside `w`).

   Iteratively evict any `w` with `support(w) ≤ K`, decrementing its neighbours' support,
   until a fixed point.
5. Every survivor of `C` has `core += 1`.

**Complexity.** O(|affected subcore| + edges incident to it) per insertion, versus
O(V + E) for a rebuild. The win depends entirely on the affected subcore being small
relative to the graph, which is an empirical property of the data, not a guarantee.

## An assumption that does not hold cleanly, stated up front

The batch graph applies **hub suppression at cap 5**: an attribute value shared by more
than five entities is treated as generic and contributes no edges. In a streaming setting a
group can *cross* that cap — the sixth entity arrives and the group's five existing edges
must be **removed**.

Edge deletion is a different and harder problem than insertion, and pure insertion
maintenance cannot express it. Rather than pretend the issue away or silently diverge from
batch semantics, the implementation will **fall back to a full rebuild whenever a group
crosses the cap**, and count how often that happens. That frequency is part of the result:
if cap-crossings are common, incremental maintenance is structurally a poor fit for this
graph regardless of its per-insertion speed, and that is worth reporting.

## Correctness plan

The existing `core/graph.py` is untouched and becomes the oracle. Incremental core numbers
must equal a full batch rebuild **exactly** — not approximately — at three tiers:

1. **Random graphs**, asserted after every single-edge insertion, not just at the end.
2. **Pathological structures**, reusing the fixtures already in `tests/test_graph.py`:
   barbell, wheel, lollipop, karate club, complete bipartite.
3. **The real IEEE-CIS entity graph**, replayed in `TransactionDT` order, with the final
   state asserted identical to the batch build.

Any disagreement is a bug in the incremental implementation. The assertion does not get
loosened.

---

## Pre-registered prediction

Recorded now, before any measurement.

**I expect incremental maintenance to lose on this graph, and to be reported as a negative
result.**

The reasoning: the batch graph builds in **1.7 seconds** with 199,070 entity nodes and
29,285 edges after hub suppression. That is already fast in absolute terms, and the graph
is extremely sparse — average degree well under 1, with 92% of entities isolated. A sparse
graph with tiny components means the affected subcore per insertion is small, which favours
incremental in *relative* terms, but the absolute batch cost is so low that Python-level
per-edge bookkeeping may well exceed it.

Concretely, I predict:

- **Per-insertion latency** in the tens of microseconds; the affected subcore will usually
  be a handful of vertices, because components at cap 5 max out around 39 entities.
- **Crossover**: incremental wins only when updates arrive in small batches — roughly when
  a batch is under **1–5%** of total edges. Above that, rebuilding is cheaper.
- **Bulk replay of all 29,285 edges will be slower incrementally than a single batch
  build**, probably by a large multiple, because a rebuild amortises across all edges while
  incremental pays per-edge overhead 29,285 times.
- **Cap-crossing fallbacks will be frequent** — 8,852 attribute values are hub-suppressed in
  the batch graph, so a faithful streaming replay must cross the cap at least that many
  times. This may dominate everything else and is the result I am least certain about.

If the measurement contradicts any of this, the measurement wins and the prediction stays
in this file as a record of what I expected. The benchmark will not be tuned until
incremental looks good.

---

## Outcome: the prediction was wrong

Appended after measuring. The prediction above is left exactly as written.

**I predicted incremental maintenance would lose on this graph. It wins, by 2.8×.**

| what I predicted | what was measured | verdict |
|---|---|---|
| per-insertion latency in the tens of µs | **2.0 µs** on the real graph | better than predicted |
| crossover at 1–5% of total edges | **282%** of total edges | badly wrong — I was off by ~100× |
| bulk replay slower than one batch build "by a large multiple" | **2.8× faster** (0.058 s vs 164 ms) | wrong in direction |
| cap-crossing fallbacks would dominate | not exercised by this benchmark at all | see the caveat below |

**Why I was wrong.** I reasoned that the batch build is already fast in absolute terms
(1.7 s including graph construction; the k-core computation alone is only 164 ms), so
Python-level per-edge bookkeeping would swamp it. What I failed to carry through was the
consequence of my own sparsity observation. The affected subcore per insertion on the real
graph is **2.8 vertices**. Repairing three vertices is so much cheaper than peeling 208,914
that even slow per-edge Python wins comfortably.

**What the benchmark revealed that the prediction did not consider at all:** the result is
entirely determined by graph density, and it inverts.

| graph | candidates examined per insert | per-insert | verdict |
|---|---|---|---|
| dense random (20k nodes, 60k edges) | 9,543 | 21,887 µs | crossover at **3 edges** — useless |
| dense random (5k nodes, 15k edges) | 2,109 | 3,485 µs | bulk replay **667× slower** |
| sparse, entity-graph-shaped | 2.8 | 4.4 µs | crossover at 67% of the graph |
| **real IEEE-CIS entity graph** | **2.8** | **2.0 µs** | **2.8× faster than rebuilding** |

On dense graphs the "local" repair is not local: most vertices share a core number, so the
candidate set is a large fraction of the graph and each insertion re-peels it. The win here
is a property of **this data's sparsity**, not of the algorithm. Claiming "incremental
k-core is faster" without that qualifier would be the kind of unfalsifiable statement this
project exists to avoid.

### A caveat that limits the claim

The replay inserts the **post-suppression** edge set — the 29,285 edges that survive hub
suppression. It therefore never exercises a cap-crossing, even though the batch graph
suppresses **8,852 hub values**, each of which a faithful stream would hit as an edge
*deletion*. So the measured 2.8× is an upper bound on what a real streaming deployment
would see, and the deletion path remains unimplemented and unmeasured. That is stated
rather than papered over, and it is the honest limit of what this benchmark supports.
