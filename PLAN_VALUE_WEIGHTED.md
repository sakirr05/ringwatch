# Value-weighted evaluation and centrality — plan and pre-registered predictions

Written **before** implementing the metrics or running the ablation. Predictions are
recorded here first so the result cannot be retrofitted, which is the same discipline
applied to the incremental k-core work — where the prediction turned out to be wrong and
stayed on record.

---

## Why this test exists

RingWatch's headline finding is negative: graph features produce no significant AUC-PR lift,
and k-core is measurably harmful. **AUC-PR is count-uniform** — it treats a small fraud and a
large one as equally important.

That invites a fair objection. PayPal's engineering blog states they "optimize for higher
accuracy of dollar-weighted fraud detection" (Nitin Sharma, PayPal Technology Blog). If the
economically relevant metric is value-weighted, perhaps the graph layer looks useless only
because the metric ignores money. Fraud rings might plausibly go after larger amounts.

That is a real objection to a real conclusion, and it deserves a measurement rather than an
argument.

*(Terminology note: "dollar-weighted fraud detection" is PayPal's own phrase. The formalised
metric name **Value Detection Rate (VDR)** comes from a separate paper — Dervovic, Amiri and
Cashmore, JP Morgan AI Research, FinPlan 2023 — and is not PayPal's term. Do not attribute
the acronym to PayPal.)*

## Two methodological risks, stated before the result

**1. This is a second metric run after the first returned null.** That is structurally
"changed the outcome measure after not liking the answer" — precisely what this project
refused to do when it declined to hunt link keys and hub caps until one beat baseline.

Two things make it defensible, and both are required:

- The predicted null is recorded here **before** measuring.
- **Value-awareness is not a pivot.** `core/evaluate.py` already weights by `TransactionAmt`
  to cost false positives. Extending the same weighting to the ablation metric continues an
  existing commitment rather than inventing a new one to escape a bad result.

If the value-weighted result comes back *positive*, it must be reported as
**hypothesis-generating, not confirmatory** — one of two metrics examined, where the first
was null.

**2. This prediction is informed, not blind.** The value-concentration check below was run
*before* these predictions were written, on the research's own advice to check first. So
this is not a clean pre-registration in the strict sense, and saying otherwise would be
dishonest. What it still does is fix the interpretation in advance, so the write-up cannot
drift toward whichever story the numbers happen to flatter.

## The check that was run first

Measured on the held-out test set:

| | share of fraud **count** | share of fraud **value** |
|---|---|---|
| graph-linked rows | 3.35% | **3.57%** |

Mean fraud amount: **160.18** linked vs **149.73** isolated. Median 107.95 vs 64.80, on 136
linked fraud cases.

A 6.6% relative value enrichment. **Graph-linked fraud is not meaningfully more valuable
than isolated fraud.** The mechanism required for value-weighting to rescue the graph layer
is absent.

A further complication for the optimistic reading: the **top 1% of frauds carry 10.9% of all
fraud value**. That heavy tail means value-weighted estimates resample far more violently
than count-weighted ones, so the intervals get *wider* — the metric hoped to reveal a hidden
effect is less sensitive, not more, and it has only 136 linked fraud cases to work with.

---

## Pre-registered predictions

### Experiment A — value-weighted ablation

1. **No graph variant shows a significant value-weighted lift.** Every 95% CI spans zero, or
   is negative. Same conclusion as the count-weighted ablation.
2. **Value-weighted CIs will be wider than their count-weighted counterparts** — I predict at
   least **1.5×** wider for `+ full graph`, whose count-weighted CI is [−0.0056, +0.0013]
   (width 0.0069). This is the sharpest falsifiable claim here.
3. **VDR at the ≤1% insult cap will be near-identical across variants**, differing by less
   than the spread between the two operating points already reported.
4. k-core stays the worst variant under value weighting too.

### Experiment B — centrality as a fifth variant

5. **PageRank and betweenness will be near-constant within components**, because components
   average 4.3 entities and max at 39. In a 4-node component PageRank is ≈0.25 per node and
   betweenness is 0 for most. I predict a **median within-component coefficient of variation
   below 0.3 for PageRank**, and **betweenness exactly 0 for more than 70% of linked
   entities**.
6. **The centrality variant produces no significant lift** on either metric — null, or
   negative like k-core.

### What would falsify these

A bootstrap CI strictly above zero on any graph variant under value weighting. If that
happens it is a genuine finding, it gets reported as the headline, and prediction 1 is
recorded as wrong — with the multiplicity caveat from risk 1 attached, because two metrics
were examined.

## What this cannot claim regardless of outcome

- **`TransactionAmt` is anonymised and possibly transformed.** Results are evidence about
  **value structure in the dataset**, never literal money saved.
- VDR is a **ratio**, so units cancel and no currency conversion is applied. The
  value-weighted result therefore needs **no FX assumption at all** — strictly fewer premises
  than the project's existing rupee costing.
- Nothing here says anything about Razorpay's internal capability. Value-weighted evaluation
  and centrality-based ring scoring are **not publicly documented** by Razorpay, which is not
  the same as absent.

## Deliberately out of scope, and why

Named rather than quietly skipped, since they are the most impressive-sounding things PayPal
discloses:

- **Production Graph Neural Networks** (GraphSAGE, temporal GCNs). Real at PayPal, and not
  reproducible solo on CPU without GPUs or labelled cross-account data. A GNN would also face
  the same rarity problem that produced this project's negative result.
- **A home-grown real-time graph database at million-QPS.** Infrastructure, not an algorithm.
- **Consortium / cross-merchant identity linking.** Requires data this project cannot access;
  a toy version would be an unbackable claim.

---

# Outcome

Appended after measuring. Every prediction above is left exactly as written.

## Scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | no significant value-weighted lift on any variant | **wrong at face value, right after correction** — see below |
| 2 | value CIs ≥1.5× wider than count CIs | **correct** — 2.32× to 2.64× |
| 3 | VDR near-identical across variants | **correct** — 0.3228 to 0.3329 |
| 4 | k-core worst under value weighting too | **wrong** — k-core is the *worst* on counts but middling on value |
| 5a | PageRank near-constant within components (CV < 0.3) | **correct** — median CV **0.0000**, mean 0.0189 |
| 5b | betweenness zero for >70% of linked entities | **wrong** — only **19.2%** are zero |
| 6 | centrality variant null or negative | **correct on counts** (significantly worse); value result does not survive correction |

## The result that nearly became a false headline

`+ centrality` came back **significantly better under value weighting** — delta +0.0125,
95% CI [+0.0024, +0.0220] — while being **significantly worse** on counts. That is exactly
the "second connection" story the research hoped for, and it would have made a compelling
video beat.

It does not survive multiplicity correction.

Four variants were tested under two weightings, so **eight comparisons** were made. At 95%
confidence each, the probability of at least one false positive across eight is roughly
34%. Correcting for that family:

| comparison | delta | 95% CI | Bonferroni (k=8) | survives |
|---|---|---|---|---|
| + components, count | −0.0011 | [−0.0045, +0.0024] | [−0.0061, +0.0034] | no |
| + components, value | +0.0028 | [−0.0057, +0.0117] | [−0.0095, +0.0146] | no |
| + k-core, count | −0.0064 | [−0.0098, −0.0031] | [−0.0110, −0.0019] | **YES** |
| + k-core, value | +0.0041 | [−0.0041, +0.0124] | [−0.0082, +0.0157] | no |
| + full graph, count | −0.0019 | [−0.0056, +0.0016] | [−0.0070, +0.0025] | no |
| + full graph, value | +0.0042 | [−0.0042, +0.0135] | [−0.0077, +0.0170] | no |
| + centrality, count | −0.0053 | [−0.0093, −0.0014] | [−0.0116, −0.0001] | **YES** |
| **+ centrality, value** | **+0.0125** | **[+0.0024, +0.0220]** | **[−0.0022, +0.0257]** | **no** |

Two-sided bootstrap p for the centrality value result is ≈0.012, against a corrected
threshold of 0.05/8 = 0.00625.

**So prediction 1 holds after correction: no graph variant produces a value-weighted lift
that survives the number of tests actually run.** The pre-registration committed in advance
to treating a positive as hypothesis-generating rather than confirmatory, and that is
exactly what this is. Reporting it as a reveal would have meant announcing the one hit out
of eight and not mentioning the eight.

What survives correction is unflattering in the other direction: **both k-core and
centrality are significantly harmful on the count metric.**

## Where I was wrong, and why

**Prediction 5b (betweenness mostly zero) was wrong**, and the reason is instructive. I
reasoned about a projected entity-to-entity graph, where a 4-node component leaves most
vertices with nothing to sit between. The actual graph is **bipartite** — entities connect
through attribute-value nodes — so a chain like `uid → attr → uid → attr → uid` makes the
middle entities genuine intermediaries. 80.8% of linked entities have non-zero betweenness
because the topology is not the one I pictured.

**Prediction 4 (k-core worst everywhere) was wrong** in a way that is consistent with the
larger finding: the count and value metrics disagree about ordering. That disagreement is
real, and it is precisely why value weighting was worth measuring even though it changed no
conclusion.

## What this licenses

- "The graph layer produces no lift, and that conclusion survives being re-tested under the
  economically relevant metric." **Supported.**
- "Value weighting reveals hidden ring value." **Not supported** — graph-linked fraud carries
  3.57% of value against 3.35% of count.
- "Centrality helps when you weight by money." **Not supported at the level of evidence
  available** — significant uncorrected, not significant across the family of tests run.
  Reportable only as a hypothesis worth a dedicated, pre-registered test on fresh data.
