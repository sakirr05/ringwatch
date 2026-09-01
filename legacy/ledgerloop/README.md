# LedgerLoop

Reconciliation agent built for the Razorpay AI Buildathon 2026, Track 04 (AI Finance
Controller).

## What it does

Matches a Razorpay settlement-recon payload against an ISO 20022 `camt.053` bank
statement across 50+ synthetic records, reports a match rate, and produces an honest
exception list. Exceptions the deterministic engine can't resolve get a human-readable
diagnosis from an LLM.

## Architectural rule

Every number in this system is computed deterministically. The LLM never decides a
match, never computes an amount, and never resolves an exception — it only writes a
diagnosis for records the deterministic engine has already given up on. See
`core/matcher.py` for the matching logic and `ai/diagnose.py` for the diagnosis layer.

## Layout

```
data/
  generate.py          # synthetic data generator
  out/                 # gitignored: generated artifacts
core/
  money.py             # integer-paise arithmetic
  settlement.py        # parse Razorpay recon payload
  camt053.py           # parse ISO 20022 camt.053 XML
  matcher.py           # the deterministic matching engine
ai/
  diagnose.py          # LLM exception diagnosis
  provider.py          # Gemini primary, Groq fallback, circuit breaker
tests/
run.py                 # entrypoint
FAILURE_LOG.md
```

## Hard rules

1. All money is integer paise. Never a float, anywhere — `core/money.py` is the only
   place amounts are parsed or formatted.
2. Synthetic data generation is deterministic: same `--seed` → byte-identical output.
3. `ground_truth.json` (written by the generator) is quarantined — `core/` must never
   import or read it.
4. The LLM sees only unmatched records: no ground truth, no matched records, no amounts
   it could be tempted to "correct."
5. LLM output is strict JSON validated by schema. Parse failure → retry once, then mark
   `DIAGNOSIS_UNAVAILABLE`. Never fall back to guessing.
6. No live Razorpay API calls — test mode only, schemas modelled from public docs.

## Status

Phase 1 (scaffold) in progress. See `FAILURE_LOG.md` for what's broken and been fixed
along the way.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY / GROQ_API_KEY
pytest
```
