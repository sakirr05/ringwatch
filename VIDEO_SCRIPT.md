# RingWatch — 5-minute pitch video script

Unlisted YouTube/Loom. Target 4:45–5:00. Screen recording with voiceover; no slides
except the architecture diagram. Record the terminal at a readable font size.

**Before recording:** run `python run.py --stage ablation` once so model artifacts are
cached — otherwise there is a 3-minute silence per variant on camera. The LLM cache
should also be warm so the narrate stage returns instantly.

---

## 0:00–0:40 · The problem (why this matters)

> "Card-testing rings and coordinated chargeback fraud don't look like fraud one
> transaction at a time. Each payment sits inside a perfectly normal distribution —
> ordinary amount, ordinary merchant, ordinary hour. The only real signal is the
> *coordination between* them. A row-by-row classifier is structurally blind to that:
> it scores each transaction independently and never sees that forty of them share a
> fingerprint."

Show: the repo README problem statement. Keep this tight — 40 seconds, no more.

> "RingWatch adds a classical graph layer to surface that coordination — and, just as
> importantly, measures honestly whether doing so actually helps."

## 0:40–1:30 · The architecture boundary (the AI-judgment story)

Show: the Mermaid architecture diagram in the README, full screen.

> "The design rule this whole project is built around: one box computes numbers, a
> completely separate box writes sentences, and code — not convention — stops the second
> from touching the first."

Point at the green box:

> "Everything green is deterministic. The temporal split, the LightGBM classifier, the
> connected components, the k-core decomposition, every metric. Every number in this
> system is produced here."

Point at the blue box:

> "The language model lives over here. It receives clusters the deterministic engine has
> *already flagged*, plus the evidence it already computed, and it writes prose. It never
> decides a match, never computes an amount, never sets a flag."

**Then prove it on camera** — this is the moment that separates this from a claim:

```
pytest tests/test_ai_boundary.py -v
```

> "This isn't a promise in a README. This test parses the syntax tree of every module in
> the AI package and fails if any of them can even *import* the scoring engine. The model
> has no code path to a number."

## 1:30–2:15 · The graph layer (the differentiating skill)

Show: `core/graph.py`, scroll to `k_core_numbers`.

> "The k-core decomposition is Batagelj–Zaversnik bucket peeling, written directly rather
> than pulled from a library — I shipped this algorithm in the Boost Graph Library inside
> pgRouting during Google Summer of Code, and I wanted it to be load-bearing here."

Show: `pytest tests/test_graph.py -q` (41 tests, <1s).

> "It's validated against networkx as an independent oracle on twelve random graphs and
> six pathological ones — barbell, karate club, complete bipartite, wheel, lollipop,
> ladder."

Show the percolation table in the module docstring.

> "Hub suppression isn't a taste call. I swept the cap and found a phase transition: at
> cap 5 the largest component is 39 entities; at cap 10 it's 1,752; at 50 it's 43,000 —
> one giant hairball where 'component' stops meaning 'candidate ring.' The cap sits just
> below the transition."

## 2:15–3:15 · Live run and the measured metrics

Run on camera:

```
python run.py --stage graph
```

> "The entity graph builds in under two seconds."

Point at the ring-concentration test output.

> "Here's the honest part. My first instinct was to tabulate fraud rate by core number —
> and it showed nothing. Flat, non-monotonic. On that evidence the conclusion would have
> been 'the graph finds nothing.'
>
> That was the wrong statistic. Rings are *rare* — a handful of real ones among thousands
> of benign components — so averaging drowns them. The right question is whether fraud
> *concentrates*: given the number of fraudulent entities, are they scattered the way
> chance would scatter them, or bunched into components that are entirely fraudulent?
>
> Against a label-permutation null: twelve all-fraud components observed, 1.4 expected.
> That's **z = +8.8**, and it's stable across every hub cap I tried."

Then the ablation table.

> "And here's what that buys, honestly reported: [state the actual measured lift]. Small —
> because rings this rare can only move a dataset-wide metric a little. A large lift here
> would have been evidence of a bug, not of success."

## 3:15–4:05 · The edge case, handled honestly

**This is the section judges remember. Do not skip it or soften it.**

Show the `HONEST NEGATIVE CASE` block from the ablation output.

> "The track asks for honest metrics, so here's where my own system makes things worse.
> These are *legitimate* transactions that the graph layer pushed toward being declined —
> real customers who'd be insulted because they happen to sit in a dense component. A
> high-volume account that looks ring-like from the topology alone."

Then the two operating points:

> "Same honesty applies to the threshold. The cost-minimising threshold declines 5% of
> all legitimate traffic. Arithmetically correct, operationally unshippable — and the
> reason is that my cost model deliberately doesn't monetise the churn from insulting a
> good customer, because putting a number on that would be inventing data. An unpriced
> cost reads to an optimiser as a free one. So I report both points, and the gap between
> them *is* the honest statement of what's missing."

Show `FAILURE_LOG.md`, scroll through the pivot entry.

> "I also abandoned a working project to build this one. Day 1 was a reconciliation
> agent — it ran, tests passed. But I'd authored the synthetic data *and* graded my own
> matcher against it. A closed loop that proves nothing. That's entry one in the failure
> log, and the code is still in the repo under legacy/."

## 4:05–4:45 · Ground-truth honesty and limitations

> "The claim I am **not** making: RingWatch does not report 'N rings caught.' IEEE-CIS has
> transaction-level fraud labels, not ring labels. There is no ground truth to validate a
> ring count against, so any such number would be fabricated. The two claims I can defend
> are that fraud clusters more densely than chance allows, and that graph features produce
> a measured PR-AUC lift on a temporally held-out test set."

> "Also stated plainly: this is **detection-only**. It never generates, simulates, or
> optimizes evasive transactions, and contains no capability to do so."

Show the limitations section: entity resolution is a heuristic, `addr1` has only 332
distinct values, ~8% graph coverage, single dataset.

## 4:45–5:00 · Close

> "Everything here is reproducible from a clean clone: fetch the data, run the pipeline,
> get these exact numbers. Deterministic seeds throughout, and the tests cover the
> temporal split, the graph algorithms, and the AI boundary. Thanks for watching."

---

## Recording checklist

- [ ] Model artifacts cached (`data/cache/scores_*.npy`) so no dead air
- [ ] LLM response cache warm
- [ ] Terminal font large enough to read at 720p
- [ ] Architecture diagram rendered (GitHub renders the Mermaid natively)
- [ ] Video set to **unlisted**, link tested in a private window
- [ ] Total runtime under 5:00
