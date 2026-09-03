# RingWatch — Build Plan

**Track 02 (AI Risk Manager) · Razorpay AI Buildathon 2026**

Budget: **2 focused days is the plan; a 3rd is a stretch, not assumed.** Rough hour
estimates below are against ~16–18h of real build time.

---

## Core claim — REVISED after Phase 5 measured it

> **Revision, 2026-09-01 17:40.** The original claim below assumed the graph layer would
> produce a measured PR-AUC lift. It does not. The ablation found no significant lift,
> and k-core significantly *hurts* (95% CI excludes zero). The claim has been retargeted
> to what the evidence supports rather than the code tuned until the evidence agreed.
> Full reasoning in `FAILURE_LOG.md`, entry 17:40.

**Original claim (superseded):** ~~a classical graph layer that surfaces collusion
structure, measured through an ablation showing PR-AUC lift, plus one case where it
hurts.~~

**Revised claim (what the measurements actually support):**

A deterministic fraud classifier, honestly evaluated with AUC-PR and false-positive
"insult-rate" costing, plus a **classical graph layer** whose contribution is stated
honestly in two parts:

1. **What it does:** fraud concentrates in the entity graph far beyond chance
   (z = +8.8 vs a label-permutation null, stable across hub caps), and the graph surfaces
   statistically anomalous clusters for analyst review, which is what feeds the narrative
   layer.
2. **What it does NOT do:** improve dataset-wide prediction. No variant beats the tabular
   baseline; k-core is significantly worse. Reported with bootstrap CIs, up front.

Rare structure is real structure and still cannot move an average. Both statements are
true, and reporting only the first would have been the dishonest submission.

An LLM writes analyst-facing narratives for already-flagged clusters and is never allowed
near the score, the flag, or any number it wasn't handed.

## Guiding constraints (locked, not revisited)

- **Detection-only.** Nothing in this repo generates, simulates, or optimizes evasive
  transactions. Track rule, stated in README.
- **Temporal split**, never random — split at the 80th percentile of `TransactionDT`.
- **AUC-PR, never accuracy** — 3.5% positive class makes accuracy meaningless.
- **Classical graph algorithms, not a GNN** — deliberate, given a zero-compute budget.
- **No GPU code paths.** LightGBM on CPU (32 cores available).
- **The AI/determinism boundary is enforced by code**, not by convention.

---

## Phase 0 — Recon & de-risk ✅ DONE (~0.5h)

Front-loaded because two things could have invalidated the whole plan.

| Risk | Finding |
|---|---|
| Kaggle credentials absent | **Resolved.** No creds, no CLI. Found ungated HF mirror `aliceczr/ieee-fraud-detection` with the raw competition files (`train_transaction.csv` 683 MB, `train_identity.csv` 26.5 MB). Verified via ranged HTTP request: **394 columns**, exact IEEE-CIS schema, `TransactionDT`/`card1`/`addr1`/`D1`/`isFraud` all present. |
| Python 3.14.6 too new for ML wheels | **Resolved.** pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, lightgbm 4.7.0, networkx 3.6.1 all install and import cleanly. |
| Compute | 32 CPUs, 11 GB RAM, 870 GB disk. Fine for LightGBM on 590k×394. RAM is the one real constraint → dtype downcasting + parquet cache. |

## Phase 1 — Restructure & scaffold (~0.5h)

- Retire LedgerLoop into `legacy/ledgerloop/` via `git mv` (history preserved). **Never touched again.**
- Root scaffolding: `requirements.txt`, `pytest.ini`, `.gitignore` (raw data + parquet cache excluded), `.env.example`.
- `FAILURE_LOG.md` seeded with the **pivot itself** as entry #1 — the closed-loop
  self-graded-synthetic-data problem is a genuine failure-recovery story and belongs in the log.
- README skeleton with the detection-only statement and limitations section stubbed
  **now**, so honest caveats get written while they're fresh rather than reconstructed at the end.

## Phase 2 — Data pipeline & temporal split (~1.5h)

- `core/data.py` — load with explicit dtype downcasting (float64→float32, ints→smallest
  safe), cache to parquet so every later run is fast and reproducible.
- `core/split.py` — temporal split at the 80th percentile of `TransactionDT`.
- **Tests:** every train timestamp strictly precedes every test timestamp; split ratio;
  both sides non-empty and contain both classes.
- **Output:** row counts, overall/train/test fraud rates, split boundary in days.

## Phase 3 — Tabular baseline (~2h)

- `core/features.py` — baseline (non-graph) prep: categorical handling, NaN policy.
- `core/model.py` — LightGBM, CPU, fixed seed.
- `core/evaluate.py` — **AUC-PR primary**; PR curve; threshold selection defended in
  code comments; **insult-rate costing** with a named, documented assumed order value
  (no silent magic numbers).
- **Milestone check-in:** baseline AUC-PR + PR curve + insult-rate table.

## Phase 4 — Graph layer (~4h) — *the differentiator*

This is where the GSoC/pgRouting graph-algorithms background is load-bearing rather than
decorative, so it gets the largest single block.

- `core/graph.py`:
  - **Entity fingerprint (uid):** `card1` + `addr1` + (`TransactionDT`-derived day − `D1`).
    `D1` is days-since-card-opened, so `day − D1` is a card's constant start-day —
    the competition-validated way to link transactions to one underlying account when
    the dataset has no customer ID. **Deliberately NOT built on `DeviceInfo` / `id_30`–`id_33`** —
    too sparse, would yield a mostly-disconnected noise graph.
  - **Graph:** nodes = uid entities; edges = shared identifying attributes between
    distinct uids. Linking on high-cardinality-but-common attributes creates hairball
    components, so **hub suppression** (degree/value-frequency caps) is applied and its
    threshold documented. I expect this to be a real source of failure-log entries.
  - **Connected components** — candidate rings.
  - **k-core decomposition** — Batagelj–Zaversnik O(m) bucket-peeling, **implemented
    directly**, not called from a library. Validated against `networkx.core_number` in tests.
- **Leakage guard:** graph features are **purely structural** (component size, core
  number, degree, velocity). No label-derived features (e.g. "component fraud rate"),
  which would leak test labels through the graph. Stated in code and README.
- **Tests:** k-core matches networkx on random + pathological graphs; graph construction
  is deterministic across runs; feature computation is order-independent.
- **Milestone check-in:** component-size distribution, fraud density by core number —
  i.e. evidence for the honest claim that fraud clusters more densely, *before* any
  claim about lift.

## Phase 5 — Graph-augmented model & honest ablation (~2h)

- Retrain with graph features appended.
- **Ablation table:** tabular-only → +components → +k-core → full.
- **Find and report where the graph layer hurts.** Segment-level analysis to surface a
  legitimate high-volume entity that looks ring-like and gets pushed toward a false
  positive. This is a required deliverable, not an optional nicety — reporting only
  favorable results is the failure mode this phase exists to avoid.
- **Milestone check-in:** ablation + the negative case.

## Phase 6 — LLM narrative layer (~2h)

- `ai/provider.py` — Gemini primary, Groq fallback, **retry once**. Deliberately *no*
  circuit breaker / backoff-with-jitter: this runs offline in a batch over already-flagged
  clusters, not in a live latency-critical loop, so that machinery wouldn't be earning its
  place. Noted in README as an explicit AI-judgment call.
- `ai/schema.py` + `ai/narrate.py` — strict schema-validated JSON out.
- **Number-provenance guard:** any numeric token in the LLM output that isn't present in
  the evidence it was given → reject, retry once, then `NARRATIVE_UNAVAILABLE`. Never guess.
- **Structural boundary:** `ai/` receives a frozen evidence object and has no import path
  to scoring or model code. One box computes numbers; a separate box writes sentences;
  code prevents the second from touching the first.
- **Tests:** schema validation, the number-guard (including a hallucinated-figure case),
  the `NARRATIVE_UNAVAILABLE` fallback path.
- **Milestone check-in:** real narratives for real flagged clusters.

## Phase 7 — README, architecture diagram, video script (~2h)

- Mermaid architecture diagram with the deterministic/AI boundary drawn explicitly.
- README: problem statement · architecture · deployment instructions · metrics ·
  **ground-truth honesty caveat** · **detection-only statement** · honest limitations.
- `VIDEO_SCRIPT.md` — 5-minute running order: problem → architecture boundary → live run
  → the one edge case handled honestly → measured metrics.
- Final `FAILURE_LOG.md` pass.

## Phase 8 — STRETCH ONLY: Elliptic dataset (~3h, day 3)

A second dataset with real ring-level ground truth for a cleaner ring-topology demo.
**Only after everything above is genuinely done.** First thing cut.

---

## Ground-truth honesty (stated up front, not retrofitted)

IEEE-CIS has **transaction-level fraud labels, not ring labels.** RingWatch will never
claim "N rings caught" — that number cannot be validated against this dataset and would
be fabricated. The two honest, defensible claims are:

1. Fraud-labeled transactions **cluster more densely** in the entity graph than legitimate
   ones — shown via graph statistics.
2. Graph-derived features produce a **measured PR-AUC lift** over the tabular baseline —
   shown via ablation on a temporally held-out test set.

## Cut order under time pressure

1. Phase 8 (Elliptic stretch) — cut first.
2. k-core → components-only, if Phase 4 overruns.
3. **Never cut:** temporal split correctness, AUC-PR reporting, insult-rate costing, the
   negative ablation case. The honest metrics and the ablation are the part of this
   submission worth defending in an interview — more than any individual feature.

## Deliverables checklist

- [x] Reproducible pipeline: data → temporal split → baseline → graph features → augmented model → evaluation → LLM narratives
- [x] `core/evaluate.py` output: baseline AUC-PR, augmented AUC-PR, PR curve (`docs/pr_curve.png`), insult-rate costing at two operating points, honest negative case with harm broken down by group
- [x] Architecture diagram showing the deterministic/AI boundary (Mermaid, in README)
- [x] README: problem · architecture · deployment · metrics · ground-truth caveat · detection-only · limitations
- [x] `FAILURE_LOG.md` with real entries (pivot, data access, pandas 3.0 dtype, unshippable threshold, the failed hypothesis, cluster-scope bug)
- [x] `VIDEO_SCRIPT.md` for the 5-minute pitch
- [x] Tests: temporal-split correctness, graph metrics vs networkx, ring-concentration statistic, LLM JSON schema, number-provenance guard, AI/determinism import boundary — **100 tests at this phase** (312 as of the final build)

- [x] **Real LLM narratives** — 11/12 validated against live Gemini, 1
      `NARRATIVE_UNAVAILABLE` from a genuine timeout. Number-provenance guard held on all
      11.

### Outstanding — submission, not code

The build is complete and verified. What remains is publication:

- [ ] **Push to a public GitHub repo.** No git remote exists; all 12 commits are local
      only. The buildathon requires a public repo URL on the form.
- [ ] **Record the 5-minute video.** Script with timings and exact commands is in
      `VIDEO_SCRIPT.md`. Needs an unlisted URL for the form.
- [ ] **Fill the application form** — track, project name, problem statement, repo URL,
      video URL, and the failure-analysis field (draft it from `FAILURE_LOG.md`; the
      17:40 entry on the failed hypothesis is the strongest material).
- [ ] **Rotate the Gemini API key** after submission — it was pasted in plaintext during
      the build conversation, so treat it as compromised even though it never entered git.
- [ ] Phase 8 (Elliptic stretch dataset) — not started, first thing cut as planned.
