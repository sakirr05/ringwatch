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
