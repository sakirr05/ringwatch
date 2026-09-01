# Failure Log — RingWatch

Real entries, written as things broke. Not sanitised: the dead ends and the
wrong assumptions are the point. This log feeds the buildathon application's
failure-analysis field.

Format: `## [timestamp] — Symptom / Diagnosis / Fix`

---

## [2026-09-01 14:52] — Abandoned an entire working Day-1 project (LedgerLoop)

- **Symptom:** Day 1 of a Track 04 reconciliation agent (LedgerLoop) was built and its
  tests passed — integer-paise money module, 14 tests green. Nothing was technically
  broken. But reviewing the plan for Day 2 (the grading/verifier layer), the whole
  design collapsed under one question: *what is actually being proven here?*
- **Diagnosis:** The project was a closed loop. I authored the synthetic settlement and
  bank data, injected the breaks myself, and then graded my own matcher against ground
  truth I had also authored. A high match rate would demonstrate only that my generator
  and my matcher shared assumptions — not that the system works on real data. Worse, it
  made no use of my one genuinely rare credential (production graph-algorithm
  engineering: Boyer–Myrvold planarity, Batagelj–Zaversnik k-core in the Boost Graph
  Library, shipped in pgRouting during GSoC). The submission would have been competent
  and unmemorable.
- **Fix:** Deliberate pivot to Track 02 (AI Risk Manager) and RingWatch, built on the
  **real, public, externally-labeled** IEEE-CIS Fraud Detection dataset — data I did not
  author and cannot tune to flatter myself. LedgerLoop retired to `legacy/ledgerloop/`
  via `git mv` with history preserved, rather than deleted: the decision to abandon
  working code is itself the failure-recovery story, and deleting the evidence would
  erase it. **Cost of this decision: roughly half a day of build time, knowingly spent.**

## [2026-09-01 15:05] — Data access blocked: no Kaggle credentials, and IEEE-CIS is gated

- **Symptom:** The entire project gates on the IEEE-CIS dataset. `~/.kaggle/` does not
  exist, `KAGGLE_USERNAME`/`KAGGLE_KEY` are unset, and the `kaggle` CLI is not installed.
  IEEE-CIS additionally requires accepting competition rules on an authenticated account,
  so simply installing the CLI would not have been sufficient either.
- **Diagnosis:** Treating this as "ask the user for a Kaggle token and wait" would have
  stalled every downstream phase on a human round-trip, on a 2-day budget. The real
  question was whether an *ungated* mirror of the raw files existed.
- **Fix:** Searched the HuggingFace datasets API and found `aliceczr/ieee-fraud-detection`
  hosting the raw competition CSVs ungated. Before committing to a 683 MB download, I
  verified authenticity cheaply with an HTTP **ranged request** for the first 3 KB and
  parsed the header: 394 columns, with `TransactionID`, `isFraud`, `TransactionDT`,
  `card1`, `addr1`, `D1` all present — the exact IEEE-CIS schema. Post-download
  verification confirmed it: **590,540 rows, 3.4990% fraud rate, 182-day span**, matching
  the published dataset characteristics. A second candidate mirror
  (`Kshitijbhatt1998/…-pipeline-features`) was **rejected** — it ships someone else's 16
  engineered features and a reduced column set, which would both contaminate the ablation
  and mean I hadn't built the feature layer myself.

## [2026-09-01 17:40] — The central hypothesis failed: the graph layer produces no lift

**The most important entry in this log.** The project was built to show that graph
features improve fraud detection. Measured honestly, they do not.

- **Symptom:** The four-variant ablation on the temporally held-out test set:

  | variant | AUC-PR | vs baseline |
  |---|---|---|
  | tabular only | **0.5188** | — |
  | + components | 0.5176 | −0.0011 |
  | + k-core | 0.5123 | −0.0064 |
  | + full graph | 0.5168 | −0.0019 |

  Every graph-augmented variant scored *below* the tabular baseline. This directly
  contradicts the premise the project was designed around.

- **Diagnosis:** My first instinct was that a −0.002 delta is noise and I could describe
  the layer as "roughly neutral." That instinct was itself the thing to catch: a raw
  delta from a single run is not a result. I ran a **paired bootstrap** (400 resamples of
  the test set, both models scored on identical resampled indices):

  - `+ components`: −0.0011, 95% CI [−0.0042, +0.0024] → not significant
  - `+ k-core`: −0.0064, 95% CI [−0.0102, −0.0031] → **significantly WORSE**
  - `+ full graph`: −0.0020, 95% CI [−0.0056, +0.0013] → not significant

  So "roughly neutral" was wrong in both directions: two variants are genuinely
  indistinguishable from baseline, and k-core is genuinely harmful, not noise.

  The cause is coverage, not correctness. The graph touches only **5.38% of test rows**,
  and the ring structure it finds is real but *rare* — 12 all-fraud components. Twelve
  rings cannot move a metric computed over 118,108 transactions, while the graph features
  add a small amount of noise to the other 94.6% of rows. Meanwhile LightGBM already has
  433 features that capture most of what little signal exists. Restricting to the linked
  subgroup, the point estimate does flip positive (0.4118 → 0.4298, **+0.017**) but its CI
  spans zero: 6,354 rows containing 136 fraud cases cannot resolve an effect that size.

- **Fix:** Not a code fix — a **claim fix**. The honest finding is that fraud
  *concentrates* in the entity graph far beyond chance (z = +8.8 against a
  label-permutation null, stable across every hub cap) while that concentration
  *does not convert into dataset-wide predictive lift*. Those two statements are both
  true and are not in tension: rare structure is real structure and still cannot move an
  average.

  So the graph layer was retargeted to what it demonstrably does — surfacing
  statistically anomalous clusters for analyst review, which is what feeds the narrative
  layer — rather than to feature-level lift, which it demonstrably does not do. k-core is
  kept and **reported as the measured harm case** (its CI excludes zero, which makes it
  far better evidence than a cherry-picked anecdote would have been) but is dropped from
  the recommended model configuration, because shipping a config I have measured as worse
  would be indefensible.

  The tempting alternative was to keep engineering until something looked positive: more
  link keys, ring-specific aggregates, a tuned cap. Given enough configurations, one of
  them scores above baseline by chance. That is p-hacking, it would not survive an
  interview question about how many variants I tried, and I would not be able to honestly
  answer that question. Rejected deliberately.

  **One follow-up was worth running**, because it is the first question any reviewer asks
  and because it has a pre-registered answer rather than a hunted one: *is the problem
  simply that coverage is too low?* Re-running at higher hub caps — cap 20 (15.41%
  coverage) gives Δ = **+0.0003**, cap 50 (26.18% coverage) gives Δ = **−0.0036**.
  Tripling coverage moves the metric by three ten-thousandths, an order of magnitude
  inside the ±0.004 noise band, and the cap-20 run never early-stopped (it exhausted all
  2,000 boosting rounds), so it is the least trustworthy of the three. Coverage is not the
  binding constraint, and the conclusion holds. I am reporting the +0.0003 *because* it is
  positive and meaningless — picking it out as "the graph helps at cap 20" is precisely
  the move this entry exists to refuse.

## [2026-09-02 12:40] — The webhook feature I planned turned out to be dishonest

- **Symptom:** Not a crash. The plan said "on a `payment.*` event, extract the fields the
  model needs, run the existing deterministic scorer." Checking what the booster actually
  requires before writing the mapping: **433 features, of which a Razorpay payload can
  supply 3.**
- **Diagnosis:** `card1`, `card2` and `card5` are Vesta-internal identifiers, not card
  network or type. `addr1/2` and `dist1/2` likewise. `C1–C14`, `D1–D15`, `M1–M9` and
  `V1–V339` — roughly 400 columns — are Vesta's proprietary engineered features with no
  counterpart in any processor's webhook. LightGBM accepts 430 NaNs and returns a number
  quite happily, and that number would have gone on a public dashboard labelled as a fraud
  score. For a project that abandoned LedgerLoop for being a closed loop and refuses to
  calibrate LLM confidence on 12 samples, shipping that would have discredited the honest
  parts retroactively — and a payments or ML reviewer spots it immediately.
  I also nearly mapped `card4`/`card6`/`P_emaildomain`, which *look* like clean matches;
  LightGBM encodes categoricals as integer codes fixed at training time, so feeding a
  category with a different code ordering silently maps "visa" onto whatever occupied that
  code in training. A wrong number that looks plausible is worse than an honest missing one.
- **Fix:** Two tracks that the interface never blurs. **Track 1** runs the existing
  networkx-validated connected-components and k-core over an entity graph built from
  Razorpay-native identifiers — a real computation, because topology assumes no
  distribution and was fitted to nothing. **Track 2** runs the model and displays a
  *measured* coverage figure ("3 of 433 features present") rather than a vague disclaimer,
  labelled as demonstrating the ingestion path and not assessing the transaction.
  The contrast turned out to be a better result than the feature would have been: **the
  graph algorithms transfer across payment ecosystems; the trained model does not.**

## [2026-09-02 14:15] — I pre-registered a prediction about incremental k-core and it was wrong

- **Symptom:** `PLAN_INCREMENTAL.md` recorded, before any measurement, that incremental
  k-core maintenance would **lose** on this graph — predicted crossover at 1–5% of edges,
  bulk replay "slower by a large multiple". Measured: crossover at **282%** of edges, and
  full replay **2.8× faster** than a rebuild (0.058 s vs 164 ms). Wrong in direction and
  off by roughly 100× in magnitude.
- **Diagnosis:** I reasoned that the batch build is already fast in absolute terms, so
  per-edge Python bookkeeping would swamp it. I had already observed that the graph is
  extremely sparse — 92% of entities isolated, components maxing out at 39 — and then
  failed to carry that observation through. The affected subcore per insertion is **2.8
  vertices**. Repairing three vertices beats peeling 208,914 by so much that slow Python
  wins comfortably. I anchored on absolute batch speed and never estimated the candidate
  set size, which is the only quantity that actually matters.
- **Fix:** Nothing to fix in the code — it was correct throughout, matching the batch
  oracle exactly at every step. The prediction stays in `PLAN_INCREMENTAL.md` verbatim with
  the outcome appended beneath it, because a pre-registration you quietly amend afterwards
  is worth nothing. The benchmark also surfaced what the prediction never considered: the
  answer **inverts with density**. On dense random graphs incremental is 667× *slower* and
  the crossover falls to 3 edges. So the honest claim is not "incremental k-core is faster"
  but "it is faster on sparse entity graphs, which is what payment fingerprint graphs
  happen to be" — and the replay never exercises the 8,852 hub cap-crossings a real stream
  would hit as deletions, so even that is an upper bound.

## [2026-09-01 18:05] — Hardcoded a model name that Google had already retired

- **Symptom:** With a valid API key finally configured, the first live call returned
  `404 NOT_FOUND: This model models/gemini-2.0-flash is no longer available. Please update
  your code to use models/gemini-3.6-flash`.
- **Diagnosis:** `ai/provider.py` pinned `gemini-2.0-flash`, which was current when I wrote
  it and had since been retired. Two things made this cheap to find instead of expensive.
  First, the failure was a **404, not a 401** — which immediately separated "the key is
  bad" from "the request is bad", and told me the credential was fine. Second, I tested
  the key with a single `curl` before running the pipeline, so the error surfaced in two
  seconds rather than inside a batch of twelve clusters where twelve identical
  NARRATIVE_UNAVAILABLE fallbacks would have looked like a credential problem.
- **Fix:** One-line model bump to `gemini-3.6-flash`, verified with a live call before
  re-running. Result: **11 of 12 clusters produced validated narratives.** The general
  lesson is that a pinned third-party model name is a dependency with an expiry date and
  no compiler to catch it, which is an argument for the fallback provider existing at all.

## [2026-09-01 18:12] — The retry-and-fallback path fired for real, unprompted

- **Symptom:** Cluster 7 of 12 returned `NARRATIVE_UNAVAILABLE` while the other 11
  succeeded: `gemini attempt 1: Read timed out (45s); gemini attempt 2: Read timed out`.
- **Diagnosis:** A genuine transient network timeout against the Gemini endpoint, not a
  bug. Both attempts were consumed, no Groq key was configured to fall back to, so the
  cluster degraded to the honest failure value exactly as designed.
- **Fix:** None required — this is the system working. Worth logging because it is
  unplanned evidence that the degradation path is real rather than a code path that has
  only ever been exercised by a unit test. An analyst reading that report sees eleven
  narratives and one explicit "unavailable, here is why", instead of twelve narratives one
  of which is quietly invented. Configuring `GROQ_API_KEY` would have covered this;
  keeping it uncovered demonstrates the fallback more honestly.

## [2026-09-01 16:20] — LightGBM rejected 31 columns: a pandas 3.0 dtype change

- **Symptom:** First baseline training run died immediately:
  `ValueError: pandas dtypes must be int, float or bool. Fields with bad pandas dtypes:
  ProductCD: str, card4: str, ... DeviceInfo: str` — 31 columns.
- **Diagnosis:** My `_downcast()` converted text columns to `category` with a
  `dtype == object` test. That is the correct idiom for pandas 1.x/2.x, but **pandas 3.0
  stores text in a dedicated `str` dtype, not `object`**, so the branch never fired. Every
  categorical column stayed as strings, survived the Parquet round-trip as strings, and
  LightGBM — which accepts `category` natively but not raw strings — refused them. The
  failure was loud and immediate, but the *cause* was three layers away from the error
  message, in an ingest function that looked obviously correct.
- **Fix:** Widened the branch to `dtype == object or pd.api.types.is_string_dtype(dtype)`,
  deleted the poisoned Parquet cache and rebuilt. 31 category columns now convert, and
  memory dropped from 1,126 MB to 969 MB as a side benefit. Worth noting the categorical
  conversion happens **before** the temporal split on purpose: converting train and test
  separately would give them different category codes for the same string, which LightGBM
  would silently treat as different values — a much quieter bug than this one.

## [2026-09-01 16:35] — The cost-optimal threshold was operationally unshippable

- **Symptom:** Not a crash. The threshold that minimises expected rupee cost declined
  **5.047% of all legitimate traffic** (5,756 customers) to catch 63% of fraud. The
  arithmetic was right and the result was useless.
- **Diagnosis:** My cost model prices a missed fraud at the full transaction value plus a
  ₹1,200 chargeback fee, but prices a false positive at only 12% gross margin. Under those
  weights the optimiser correctly concludes it should decline aggressively. The flaw is
  that the model deliberately does *not* monetise the lifetime-value damage of insulting a
  good customer — I left it out because putting a number on churn would be inventing data —
  and an unpriced cost is treated by an optimiser as a zero cost.
- **Fix:** Report **two** operating points rather than quietly picking one: [A] the
  cost-minimising threshold, and [B] an insult-constrained threshold capped at 1% of
  legitimate traffic, which is what a payments team could actually deploy. The gap between
  them is the honest statement of what the unpriced churn cost is doing. Keeping only [A]
  would have looked better on a slide and been wrong.

## [2026-09-01 15:10] — Python 3.14 could have had no ML wheels

- **Symptom:** Environment is Python 3.14.6, which is very new. LightGBM/XGBoost wheel
  availability lags new CPython releases, and rule 9 of the design requires
  gradient-boosted trees on CPU.
- **Diagnosis:** If no wheels existed, the options were building from source (slow,
  fragile) or provisioning a second Python — either one a meaningful schedule hit.
  Cheaper to check immediately than to discover it in Phase 3.
- **Fix:** Ran `pip install --dry-run` on the full stack during Phase 0 recon, before
  writing any code. All resolve: pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0,
  **lightgbm 4.7.0**, networkx 3.6.1. No source builds needed, XGBoost fallback not
  required. Risk retired at a cost of about two minutes.
