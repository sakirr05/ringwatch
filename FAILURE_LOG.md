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
