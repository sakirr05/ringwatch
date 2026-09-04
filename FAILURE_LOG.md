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

## [2026-09-05 08:30] — Four hand-typed confidence bounds on the live dashboard, all stale

- **Symptom:** Found by grepping the rendered page for numeric literals and diffing them
  against `docs/results.json`. The section 2b callout -- "The result that nearly became a
  false headline", the one this project is most proud of -- carried four confidence-interval
  bounds typed into the HTML by hand. Every one was wrong in the fourth decimal:
  uncorrected `[+0.0024, +0.0220]` against the artifact's `[+0.0023, +0.0217]`, corrected
  `[-0.0022, +0.0257]` against `[-0.0013, +0.0253]`. The delta beside them, `+0.0125`, was
  right, which is what made it look fine.
- **Diagnosis:** Nothing could have caught it. The clean-cache re-run reproduced the
  artifact bit-identically. The README matched the artifact on 28 of 28 figures. The page
  rendered without error and 525 tests passed. A hand-typed number is invisible to all of
  it precisely because it is not derived from anything -- it does not participate in any
  computation, so no consistency check touches it. The figures were presumably correct when
  typed and went stale under some later change to the bootstrap family.
- **The bad part:** these were on the LIVE page, in the callout that argues this project
  corrects itself. Numbers that disagree with the artifact, inside the paragraph explaining
  how carefully the artifact was checked.
- **Fix:** Rendered from the artifact through Jinja rather than corrected in place, so the
  same drift cannot recur. `tests/test_dashboard_freshness.py` asserts both the values and
  the *structure* -- the callout must reference `cen.value_ci_corrected[0]` and friends, and
  a four-decimal literal left beside them fails the test. A second test forbids hardcoding
  any headline metric anywhere in the template.
- **Worth naming:** every verification in this project so far compared two DERIVED things --
  artifact against a fresh run, README against artifact. Nothing compared a *typed* thing
  against a derived one, and that was the gap. Text is where numbers go to stop being
  checked.

## [2026-09-05 07:45] — One published table had no reproducible path, and the audit is what found it

- **Symptom:** The clean-cache re-run regenerated six of eight cached score files
  bit-identically. Two -- `scores_graph_cap20.npy` and `scores_graph_cap50.npy` -- were
  never regenerated at all, because **nothing in `run.py` or `scripts/` references them**.
- **Diagnosis:** The README publishes a hub-suppression coverage sweep at caps 5 / 20 / 50.
  It is the table that answers the first objection anyone raises about the graph layer, so
  it is load-bearing. And it had been produced by editing `MAX_GROUP_SIZE` in
  `core/graph.py` by hand, running the ablation, and copying the numbers down. Every other
  figure in this project regenerates from a committed entry point; this one did not. In a
  project whose whole argument is "do not take my word for it", exactly one table required
  taking my word for it -- and I would not have noticed without wiping the cache, because a
  stale artifact reproduces a stale claim perfectly.
- **Fix:** `build_features_for_split` now takes `max_group_size`, defaulting to
  `MAX_GROUP_SIZE` so every existing caller is byte-for-byte unchanged (asserted by
  `test_omitting_the_cap_matches_passing_the_shipped_one`). `scripts/coverage_sweep.py`
  drives it, reusing the shipped `scores_graph_full.npy` for cap 5 rather than training a
  second model for a configuration that already has one -- if that retrain disagreed even
  slightly, the table would contradict the ablation above it.
- **Two fixture bugs of my own on the way there**, both mine and not the code's: the test
  frame omitted `card3`/`card5`, which `entity_frame` requires; and the hub shared only
  `addr1`, which links nothing, because `LINK_KEYS` joins on the composite
  `(addr1, P_emaildomain)` and never on an address alone. The second is a nice demonstration
  that the graph does not link entities on a shared postcode by itself.
- **The general lesson:** reproducibility that is never exercised is indistinguishable from
  reproducibility that does not exist. Re-running against a warm cache would have "passed".

## [2026-09-05 06:10] — Two numbers both called "the delta", differing in the last decimal

- **Symptom:** Found during the Phase 11 clean-cache re-run, by comparing fresh stdout
  against the published table. `run.py`'s ablation table prints `+ full graph  -0.0019`.
  `docs/results.json`, the README and the dashboard all say `-0.0020`.
- **Diagnosis:** Not a reproducibility failure -- every score file came back
  **bit-identical** from an empty cache, so nothing had drifted. They are two different
  estimators wearing the same label. The table's "vs base" is the POINT delta, one model's
  AUC-PR minus the other's on the test set as it happens to be: 0.5168323 - 0.5187613 =
  -0.0019290. The published figure is the MEAN of 400 paired resampled deltas, -0.0020259,
  and that is the quantity the confidence interval belongs to. For `+ components` and
  `+ k-core` the two agree at four decimals, which is exactly why this survived: it only
  becomes visible on one of three rows.
- **Fix:** Labelled, not changed. Per this project's own rule -- never move a reported
  metric to make two displays agree -- both numbers stay as they are, because both are
  correct for what they measure. The stdout column is now headed `(point)` with a comment
  explaining the distinction, and the bootstrap section states that its deltas are resampled
  means and are the published ones because they carry the CI. A reviewer running the
  pipeline and diffing against the README now finds an explanation instead of a discrepancy.
- **Worth noting:** this is the kind of thing only a full re-run surfaces. No test compares
  stdout to the README, and writing one would be over-fitting to this instance -- the real
  defence was regenerating everything from scratch and actually reading the output.

## [2026-09-05 03:20] — Shipping a feature quietly falsified a sentence written four phases earlier

- **Symptom:** Found during the Phase 11 audit, not by a failing test. The dashboard led with
  "**This server renders numbers; it never produces them.**" and the README with "It never
  retrains, **rescores**, or recomputes a metric." Both were true when written. Neither was
  true any more.
- **Diagnosis:** Phase 10 added `POST /api/score`, which runs the committed booster and
  returns a number. The webhook's background task already did the same, but it was buried in
  a clearly-labelled section, so the absolute sentence at the top of the page had survived.
  Adding a route at the top level made the summary claim false without touching the file the
  claim lives in — no test could have caught it, because nothing was wrong with the code.
  The scoped clause immediately after ("no *page load* trains a model, scores a transaction,
  or recomputes a metric") was and remains exactly true; it was the broader sentence in front
  of it that had quietly stopped being.
- **Fix:** Narrowed both to what is actually true, and said in each place what changed and
  why. The dashboard now leads with the scoped claim and names the two routes that *do* run
  a model, along with their 0.69% feature coverage and the fact that their output appears in
  no reported figure. The README does the same.
- **Why this is the third time:** this project has now corrected an overclaim of exactly
  this shape three times -- a test named `test_app_layer_computes_nothing` that only checked
  direct imports, a provenance guard described as catching more than it does, and now this.
  The pattern is always the same: a true, narrow property gets restated as a broad one, and
  the broad version is the one that ends up in the summary a reader actually reads. Worth
  naming as a recurring failure mode rather than three unrelated slips.

## [2026-09-05 01:40] — A Dockerfile that built, ran, served traffic, and could not be stopped

- **Symptom:** The image built clean and served every route. Docker printed one warning I
  was inclined to treat as style advice: `JSONArgsRecommended: JSON arguments recommended
  for CMD to prevent unintended behavior related to OS signals`.
- **Diagnosis:** It is not style advice. `CMD uvicorn app.main:app ...` in shell form runs
  under `/bin/sh -c`, so the shell is PID 1 and uvicorn is its child. SIGTERM goes to PID 1,
  `sh` does not forward it, uvicorn never learns it should stop, and the container sits
  there until Docker's grace period expires and SIGKILL takes it -- cutting any in-flight
  request off mid-response and exiting 137. On Render, every deploy and every free-tier
  spin-down goes through exactly that path. The container would have "worked" in every test
  that only checks whether it answers.
- **Fix:** `CMD ["sh", "-c", "exec uvicorn ..."]`. The shell is still there to expand
  `${PORT}`, which platforms that inject a port require, but `exec` replaces it with uvicorn
  so uvicorn becomes PID 1 and receives signals directly. Measured before and after by
  timing `docker stop`: **608 ms, exit code 0**, with `Application shutdown complete` in the
  log. A test asserts `exec uvicorn` is in the CMD, because the shell form is the one
  anybody would write from memory.
- **Worth stating:** I only saw this because I actually built and ran the image. A Dockerfile
  that is committed and never executed passes code review, passes any test that curls it,
  and is broken in the one way that only shows up in production.

## [2026-09-05 00:15] — Two different 12s, and I nearly badged the wrong one

- **Symptom:** The plan for the cluster grid asked for a "fully-fraudulent flag" per card.
  The ring-concentration test reports **12 all-fraud components**, and the dashboard shows
  **12 flagged clusters**. The obvious reading is that these are the same twelve, which
  would make the flag trivial: badge every card.
- **Diagnosis:** They are not the same twelve, and the matching count is a coincidence. The
  concentration test's 12 are components of the *entity graph* — 12 out of 4,935 — where
  every labelled member is fraud. The dashboard's 12 are the top clusters by peak risk
  score, selected at the insult-constrained threshold. Computing the actual outcome per
  flagged cluster gives **2 of 12** all-fraud, one cluster with **no labelled fraud at
  all**, and one missed fraud. Badging all twelve "fully fraudulent" would have been a
  straightforwardly false claim on the most prominent new element on the page, and the
  coincidence of the two numbers is exactly what would have stopped anyone noticing.
- **Second problem, found while fixing the first:** the label was not in the exported data
  at all, and the tempting place to put it was `ClusterEvidence` — which is what the
  narrative and orchestrator layers read. Handing a narrator the ground-truth label would
  produce narratives that look accurate for a reason having nothing to do with the evidence,
  and the project's central claim would be hollow. Outcomes therefore live in a separate
  `ClusterOutcome` object, computed after the evidence, exported as a sibling block, with a
  test asserting no ground-truth field appears in the evidence or the case file.
- **Fix:** `core.clusters.cluster_outcomes`, tested first this time. The grid shows what is
  true — 2 all-fraud, 1 zero-fraud, 48.8% fraud share against a 3.44% base rate (14.2x
  enrichment) — with the zero-fraud cluster displayed rather than dropped, and the page
  states in as many words that "all N fraud" is not a ring claim.
- **Also found:** the README's "Ground-truth honesty" section still listed as a defensible
  claim that graph features "produce a **measured PR-AUC lift**". That has been false since
  the headline result was found — the ablation says no lift, and k-core is significantly
  worse. Stale prose contradicting the project's own headline, three sections above it.
  Corrected the prose, not the metric.

## [2026-09-04 22:30] — Rounding for file size made an exactness claim false by 5e-9

- **Symptom:** The threshold explorer's curve is exported with rounded fields to keep
  `results.json` small, and thresholds were rounded to 8 decimals. Everything looked right:
  the page displays 4 decimals, both operating-point marks landed on the correct slider
  positions, and every derived figure — precision, recall, tp/fp/fn, insult rate, rupee
  cost — matched the published panels exactly. My own first verification, written with a
  `< 1e-8` tolerance, passed.
- **Diagnosis:** The test I then wrote for the repo used `< 1e-9`, and failed. The published
  `cost_minimising` threshold is `0.04384046535165725`; the curve carried `0.04384047`. The
  gap is ~4.6e-9 — numerically irrelevant, since no score falls inside it and the confusion
  matrix is identical either way. But the page and the README both say the marks *reproduce
  the panel figures exactly rather than approximately*, and the threshold field is the one
  place a reviewer can check that claim directly. Rounding had quietly turned "exactly" into
  "to eight decimal places," which is a different sentence.
- **Fix:** Stopped rounding the threshold field. Every other field stays rounded; the extra
  cost is ~1.5 KB raw and about 0.3 KB gzipped. The tolerance in the test stays at 1e-9 so
  the claim cannot drift back.
- **Worth noting:** the tolerance I picked while checking my own work by hand was looser
  than the one I picked while writing a test for someone else to run, and only the second
  caught it. That is an argument for writing the test first, which I did not do here.

## [2026-09-04 21:05] — The cold-start optimisation I was about to do would have been theatre

- **Symptom:** The plan for this phase said to make `/health` "genuinely cheap" because it
  reads `docs/results.json`, and to add a skeleton loading state for the free tier's 30-60
  second cold start. Both sound obviously right. Neither survived measurement.
- **Diagnosis:** Reading the whole 112 KB artifact costs **0.34 ms** — 18% of a 1.86 ms
  `/health` request. Rendering the entire 250 KB dashboard costs **5.2 ms**. Importing the
  app costs **178 ms**, and 154 ms of that is FastAPI itself, which I do not control. The
  cold start is 30-60 *seconds* of container scheduling, during which our process is not
  running at all: no markup we serve can appear, because none of it has been served. So a
  dashboard skeleton would have looked like a fix, measured as an improvement to nothing,
  and cost a round trip on every warm load. The `load_results` docstring also still claimed
  the file was 24 KB; it is 112 KB, and the stale figure was what made "cheap /health" sound
  urgent in the first place.
- **Fix:** Reported the measurements instead of acting on the premise, then did the three
  things the measurements actually justify. (1) `/health` had a real bug that had nothing to
  do with speed: `render.yaml` points `healthCheckPath` at it, and it returned **503 when
  the results artifact could not be read** — reporting a data problem to the platform as
  "restart me", which cannot fix a missing committed file. That is a restart loop, so
  liveness and readiness are now separate endpoints. (2) The loading state went where a
  measurement pointed: both plots are ~123 KB, below the fold, and had **no declared
  height**, so each occupied zero height until decode and then snapped — a double layout
  shift on exactly the slow connection in question. Intrinsic dimensions, pinned aspect
  ratio, lazy loading: first-paint bytes 498 KB -> 252 KB, 49% deferred. (3) The external
  ping, which is the only change that touches the actual cause, shipped with its real
  caveats (GitHub cron is best-effort, stops after 60 days of repo inactivity, and staying
  warm eats ~730 of 750 free instance-hours).
- **Worth stating plainly:** the honest output of this phase is that most of the cold start
  is unfixable from inside the application, and the README says so with the numbers attached
  rather than implying it was optimised away.

## [2026-09-04 19:20] — I widened a safety guard and nearly shipped it without checking it still caught anything

- **Symptom:** No error. The orchestrator's `CaseFile.allowed_numbers()` extends the
  narrative layer's provenance allow-set with rank, percentile, cross-cluster overlap, and
  every figure quoted inside the derived findings. All 12 drafts validated on the first
  attempt, zero rejections, zero correction rounds. I wrote that down as a good result.
- **Diagnosis:** A 100% pass rate is exactly what a guard that no longer rejects anything
  looks like. I had made the allow-set wider and had no measurement of whether it was still
  selective — the honest reading of "12/12 passed" is ambiguous between "the model behaved"
  and "the check stopped working," and I could not tell which. So I measured it: 20,000
  sampled plausible figures against the 12 real case files. **0.12% accepted** (23 of
  20,000), so fabrication is genuinely caught. But the same measurement surfaced something
  I had not thought about — a figure borrowed from a *different* cluster passes 29% of the
  time above 10, and **100% of the time at 0–10**, because the allow-set is small (median
  27 tokens, 11 of them the unconditional small-integer allowance) and small clusters
  honestly share figures.
- **Fix:** Pinned both numbers as tests and narrowed the README claim to what is actually
  true: the guard prevents **fabrication**, not **misattribution**. Writing the second test
  caught me a third time — I first asserted the guard would reject `9` borrowed from a
  sibling cluster, and it did not, because 0–10 are allowed unconditionally by design. My
  test was wrong, not the code, and fixing the test rather than loosening the guard is the
  whole point. The gap is now documented in three places instead of being an unexamined
  assumption in one.

## [2026-09-04 15:40] — A z-score of exactly zero that meant "no power", not "no effect"

- **Symptom:** Running the project's own `ring_concentration_test` on Elliptic's observed
  graph returned **z = +0.0**: 0 all-illicit components observed against a null of
  0.0 ± 0.0. Read at face value, illicit activity does not cluster on a real transaction
  graph — which would have been a serious result against the project's central structural
  claim, and I very nearly wrote it up as one.
- **Diagnosis:** The statistic counts components in which *every* labelled member is
  illicit. That works on IEEE-CIS, where hub suppression keeps components around five
  entities and "all of them" is an achievable event. Elliptic's transaction-flow graph
  percolates: **49 components averaging 950 labelled members**, the largest 7,880. No
  component there *could* be all-illicit, so the observed count is zero, the null is zero,
  and the z-score is 0/0. The test had no power at all. A null result and an untestable
  hypothesis look identical in that number, and only the component-size distribution
  distinguishes them.
- **Fix:** A statistic whose unit is the **edge** rather than the component — the share of
  labelled-labelled edges joining two illicit nodes, against a null that shuffles labels
  with topology held fixed. It works at any component size. Run on both graphs:
  **Elliptic z = +24.9, IEEE-CIS z = +17.0.** The clustering is real on both, and the
  original z = 0 was an artifact of the wrong measuring instrument.

  A second, smaller trap surfaced immediately after: the same edge statistic on RingWatch's
  graph returned `nan`, because that graph is **bipartite** — entities touch attribute
  nodes and never each other, so there are no entity-entity edges to count. It had to be
  projected onto entities first. That one at least failed loudly rather than returning a
  plausible wrong number.

  Both are pinned by tests, including one that asserts the component statistic *is*
  degenerate on a percolated graph, so the reason for having two statistics cannot be
  forgotten later and quietly reverted.

## [2026-09-04 13:15] — I found a confound, "corrected" it wrongly, and had to catch that too

- **Symptom:** Windowed drift analysis showed AUC-PR rising **+23.4%** across the held-out
  period, with the first and last bootstrap intervals not overlapping. Read at face value:
  the model improved by a quarter over 42 days, which is not a thing that happens to a
  model that never retrains.
- **Diagnosis, in two wrong steps and one right one.**
  *First:* AUC-PR's floor is prevalence, and fraud prevalence rose **+22.6%** over the same
  window (3.43% → 4.21%). The two numbers track almost exactly, so the "improvement" was
  the base rate moving.
  *Second, and this is the part I nearly shipped:* I "fixed" it by dividing AP by
  prevalence, got a satisfyingly flat 13.2× → 13.3×, and wrote it up as the corrected
  finding. Then a test I had written for the fixture failed — and it was right to. AP
  scales with prevalence only when AP is **small**. I measured it on synthetic data with a
  fixed-quality ranker as prevalence went 2% → 6%: a weak ranker's lift stayed flat
  (1.3× → 1.2×), but a strong one's collapsed (29.9× → 12.5×). This project runs at AP ≈
  0.5, in the regime where the ratio over-corrects badly. My correction was as wrong as the
  raw number, just wrong in a more sophisticated way.
- **Fix:** Use a metric that is prevalence-invariant **by construction rather than by
  approximation** — AUC-ROC. It is flat across every window (0.8876 to 0.9055, intervals
  overlapping), so ranking quality genuinely did not change. The ratio-based
  `prevalence_adjusted_trend` and its `lift_ci` were deleted rather than patched;
  `lift_over_prevalence` survives as a reported number with a docstring stating the regime
  where it is meaningless, and the trend verdict never consults it.

  The honest result is more interesting than either wrong version: **no model drift, no
  feature drift (max PSI 0.0531), but real label drift** — the fraud rate itself moves
  22.6% inside a 42-day window. That is what a deployed system would actually have to
  respond to.

  Worth recording that the thing which caught the bad correction was a test written to
  check a fixture, not a review of the reasoning. I had already written the wrong version
  into a docstring as though it were established.

## [2026-09-04 10:30] — Cost-sensitive training backfired, and the diagnostic said why

- **Symptom:** Weighting each training row by its misclassification cost made the model
  **significantly worse on both axes**. AUC-PR fell 0.5188 -> 0.4782 (delta −0.0408, 95% CI
  [−0.0472, −0.0341]) — an effect roughly six times larger than any graph variant. And it
  lost on the metric it exists to optimise: total expected cost at the insult cap rose from
  ₹4,07,47,202 to ₹4,47,04,444, **+9.71%**.
- **Diagnosis:** My first instinct was that the cost signal must be too weak to help. The
  actual cause is the opposite kind of problem, and it is measurable. Weighting does not
  add information, it redistributes attention — and these weights redistribute it into a
  far smaller effective dataset. Kish effective sample size: **55,864 of 472,432 rows,
  11.8% of nominal.** A legitimate row is weighted by amount × 0.12 while a fraud row is
  amount + a flat ₹1,200, producing a **172,000× spread** in which the heaviest 1% of rows
  hold 20% of the total weight. Early stopping corroborates it independently — the model
  halts at 161 boosting rounds against the baseline's 633, which is what starved gradients
  look like. The cost signal was real but cheap; the information destroyed to encode it was
  expensive.
- **Fix:** None to the code — the implementation is correct, and the tests confirm the
  weights genuinely reach LightGBM and genuinely change the model. What changed is the
  claim. `effective_sample_size()` is now a first-class function in `core/costs.py` with
  tests, so the mechanism is reproducible rather than a number I once printed in a shell.

  Worth being careful about the scope of the conclusion: this measured **one weighting
  scheme on one dataset**, not cost-sensitive learning in general. Capping the weight
  ratio, or expressing cost through a custom objective instead of row weights, could
  behave differently. Saying "cost-sensitive learning doesn't work" would be exactly the
  overreach this project keeps having to catch itself on.

## [2026-09-02 17:25] — Betweenness centrality hung for 12 minutes and had to be killed

- **Symptom:** `run.py --stage value` sat at 117% CPU (single-threaded) producing nothing
  for twelve minutes, and was killed. Everything before it — training, bootstrap — had
  completed fine.
- **Diagnosis:** Brandes' algorithm is O(V·E), which is the *good* complexity for
  betweenness and is still hopeless here. The entity graph has **208,914 nodes and 29,285
  edges**, so sourcing a BFS from every vertex is roughly 6×10⁹ Python-level operations. I
  had validated the implementation against networkx on graphs of 40–200 nodes, where the
  cost is invisible, and never estimated it at the real scale. A second, quieter problem
  compounded it: each source allocated three arrays sized to the *whole graph*, so a
  4-vertex component was doing 208,914-element allocations thousands of times over.
- **Fix:** Both are exact, not approximations. Shortest paths never cross between connected
  components, so Brandes can run **per component** and give identical results — and this
  graph's components max out at 39 vertices. Components smaller than 3 are skipped entirely,
  since nothing can lie between fewer than two vertices. Per-source state moved from
  full-graph arrays to dicts scoped to the component. **Runtime went from >12 minutes
  (killed, never finished) to 0.3 seconds**, with all 38 correctness tests against networkx
  still passing — including the disconnected-graph case that pins the normalisation
  convention. The lesson is that validating an algorithm's *correctness* at toy scale says
  nothing about its *cost* at real scale.

## [2026-09-02 18:40] — I nearly published a false headline, and the multiplicity check caught it

**The most important entry in this log since the pivot.**

- **Symptom:** The value-weighted ablation returned `+ centrality` as **significantly
  better** — delta +0.0125, 95% CI [+0.0024, +0.0220] — while being significantly *worse*
  on the count metric. That is exactly the dramatic result the whole exercise was hoping
  for: "the graph layer looked useless only because the metric ignored money." It would
  have been the headline of the video.
- **Diagnosis:** I ran **four variants under two weightings — eight comparisons** — and
  reported the one that came back positive. At 95% confidence per test, the probability of
  at least one false positive across eight is about 34%. Announcing that hit without
  mentioning the other seven would have been textbook selective reporting, and it is the
  identical failure this project already refused twice: once when declining to hunt link
  keys and hub caps until one beat baseline, and once when refusing to calibrate LLM
  confidence on 12 clusters.

  Correcting for the family: the centrality value interval becomes **[−0.0022, +0.0257] —
  it spans zero.** Two-sided bootstrap p ≈ 0.012 against a Bonferroni threshold of
  0.05/8 = 0.00625. The effect is suggestive and it is not established.
- **Fix:** Bonferroni correction is now part of the tooling rather than an ad-hoc script —
  `bootstrap_auc_pr_delta` takes `n_comparisons`, and `--stage value` prints corrected and
  uncorrected intervals side by side for every comparison so a reader can see the whole
  family rather than the winner. The result is reported as **hypothesis-generating, not
  confirmatory**, which is what `PLAN_VALUE_WEIGHTED.md` committed to in advance for exactly
  this scenario.

  What survives correction points the other way: **both k-core and centrality are
  significantly harmful on the count metric.**

  Worth stating plainly: the pre-registration is what made this recoverable. Without a
  written commitment to treat a positive as hypothesis-generating, I would have had every
  incentive to find the correction pedantic after seeing the number.

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
