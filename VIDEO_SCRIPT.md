# RingWatch — 5-minute pitch video script

Unlisted YouTube/Loom. Target 4:45–5:00. Screen recording with voiceover. Two windows only:
a browser on the live demo, and a terminal. No slides.

**The through-line:** Razorpay's own thesis is that verification capacity, not generation
speed, is the bottleneck. This project takes that literally — so the evaluation harness is
the deliverable, and the proof that it works is that it disconfirmed my own hypothesis
twice.

---

## Before recording

```bash
python run.py --stage ablation          # warm the model/score caches
python run.py --stage narrate           # warm the LLM cache (no dead air on camera)
python scripts/export_results.py        # refresh docs/results.json
uvicorn app.main:app --port 8000        # leave running
```

- [ ] Terminal font large enough to read at 720p
- [ ] Browser on the live Render URL (or localhost:8000 if not yet deployed)
- [ ] A signed webhook request ready to paste (see §6)
- [ ] Video set to **unlisted**, link tested in a private window

---

## 0:00–0:35 · The problem, and the thesis

> "Card-testing rings and coordinated chargeback fraud don't look like fraud one
> transaction at a time. Each payment sits inside a perfectly normal distribution. The only
> signal is the *coordination between* them — and a row-by-row classifier is structurally
> blind to that."

> "But the harder problem is the one Razorpay's own brief names: in AI-native financial
> systems, the bottleneck isn't generation speed, it's **verification capacity**. So I built
> RingWatch around that claim. The fraud detector took an afternoon. The evaluation harness
> took the rest of the project — and it's the reason I can tell you, with confidence
> intervals, that my central idea didn't work."

## 0:35–1:20 · The live demo, leading with the failure

Open the deployed URL on camera. Scroll to section 1.

> "This is the deployed dashboard, and the first thing it shows is the negative result.
> That's deliberate — it's the thesis, not the embarrassment."

Point at the ablation table.

| variant | AUC-PR | Δ | 95% CI |
|---|---|---|---|
| tabular only | 0.5188 | — | — |
| + components | 0.5176 | −0.0011 | not significant |
| + k-core | 0.5123 | −0.0064 | **significantly worse** |
| + full graph | 0.5168 | −0.0020 | not significant |

> "I built a classical graph layer to improve fraud detection. Measured honestly, it does
> not. My first instinct was that two thousandths is noise and I could call it 'roughly
> neutral' — catching that instinct is the most important thing I did here. A single-run
> delta isn't a result, so I ran a paired bootstrap. It showed 'roughly neutral' was wrong
> in *both* directions: two variants are genuinely indistinguishable, and k-core's interval
> excludes zero. It's significantly **worse**."

> "One more thing about this page: it computes nothing. Every number is a committed
> artifact produced offline, and the module that feeds this page imports nothing from the
> analysis code at all — there's a test that walks the import graph and fails if it ever
> does. The webhook you'll see in a minute *does* compute; this page provably can't."

## 1:20–2:00 · The reconciliation

Scroll to section 2 — the two panels side by side.

> "Here's what makes this interesting rather than just a failure. Fraud *does* cluster in
> the entity graph: twelve entirely-fraudulent components against 1.4 expected under a
> label-permutation null. That's **z = +8.8**, stable across every hub cap I tried."

> "And right beside it: the graph reaches **5.38%** of test rows. Those two numbers together
> are the whole argument. The ring structure is real, and twelve rings cannot move a metric
> averaged over 118,000 transactions. Rare structure is real structure and still can't shift
> an average."

> "What I don't claim: IEEE-CIS has transaction-level labels, not ring labels. So those
> twelve are statistically anomalous clusters, not verified rings. There is no 'N rings
> caught' number anywhere in this project, because it couldn't be validated."

## 2:00–2:45 · The determinism boundary, proven not asserted

Scroll to section 5, expand a cluster.

> "Every field here is tagged. Green was computed by deterministic code. Blue was written by
> the language model. The model receives those numbers as frozen evidence and writes prose
> about them — it never computes a score, sets a flag, or picks a cluster. Any figure in its
> output that wasn't in the evidence it was handed is rejected."

Switch to terminal:

```bash
pytest tests/test_ai_boundary.py tests/test_app.py -q
```

> "That isn't a promise in a README. These tests parse the syntax tree of every module in
> the AI layer and the web layer and fail if either can reach the scoring engine. The
> boundary is enforced by code."

Optionally show `core/graph.py` → `k_core_numbers`:

> "The k-core decomposition is Batagelj–Zaversnik peeling, written directly rather than
> imported — I shipped this algorithm in the Boost Graph Library inside pgRouting during
> Google Summer of Code. It's validated against networkx on twelve random graphs and six
> pathological ones."

## 2:45–3:40 · The live Razorpay webhook, and what it revealed

Send a signed test event on camera:

```bash
SIG=$(python3 -c "import hmac,hashlib;print(hmac.new(b'$SECRET',open('payload.json','rb').read(),hashlib.sha256).hexdigest())")
curl -X POST $URL/webhooks/razorpay \
  -H "X-Razorpay-Signature: $SIG" -H "X-Razorpay-Event-Id: evt_demo_001" \
  --data-binary @payload.json
```

Refresh the dashboard, section 6.

> "Real test-mode webhook. Signature is HMAC over the **raw request bytes** — not a parsed
> and re-serialised body, which changes the bytes and breaks verification on payloads nobody
> tampered with. There's a test that reproduces exactly that bug. Idempotent on the event
> ID, because Razorpay delivers at-least-once. And it returns 200 before any analysis,
> because Razorpay disables endpoints slower than five seconds."

**Then the finding — this is the part worth the airtime:**

> "Building this surfaced something better than the feature. The obvious version is 'score
> incoming payments with the model.' That isn't honestly possible. The classifier expects
> **433 features**. A Razorpay payload supplies **three**. Everything else is Vesta's
> proprietary engineered columns — they don't exist in any processor's webhook."

> "LightGBM will happily accept 430 missing values and hand me a number. On a project that
> abandoned its predecessor for being a closed loop, putting that on screen as a fraud score
> would have discredited everything else. So there are two tracks. The graph analysis is
> real — topology assumes no distribution, so those algorithms transfer to a new payment
> ecosystem intact. The model score is shown with a *measured* coverage figure, three of
> 433, and labelled as demonstrating the ingestion path, not assessing the transaction."

> "The contrast is the actual result: **the graph transfers, the trained model doesn't.**"

## 3:40–4:20 · Incremental k-core, and a prediction I got wrong

Terminal:

```bash
python scripts/benchmark_incremental.py
```

> "Last piece. Production fraud graphs stream; this one rebuilds in batch. So I implemented
> incremental k-core maintenance under edge insertion, and before benchmarking I wrote down
> a prediction — that it would lose on this graph."

> "It won. By 2.8×. Full replay in 58 milliseconds against 164 for a rebuild, and a
> crossover at 282% of the graph's edges rather than the 1–5% I predicted."

> "I anchored on the batch build being fast and never estimated the candidate set size,
> which is the only quantity that matters — it's 2.8 vertices. The prediction stays in the
> repo verbatim with the outcome underneath it, because a pre-registration you quietly amend
> afterwards is worth nothing."

> "And the benchmark found what my prediction missed entirely: the answer **inverts with
> density**. On dense graphs incremental is 667 times *slower*. So the honest claim isn't
> 'incremental k-core is faster' — it's 'faster on sparse entity graphs, which is what
> payment fingerprint graphs happen to be.'"

> "Correctness is exact against the batch implementation as an oracle — asserted after every
> single insertion, not just at the end."

## 4:20–5:00 · Limitations, and close

Scroll the README limitations section.

> "What I'd want a reviewer to know. The graph layer doesn't improve prediction — that's
> stated at the top of the README, not buried. The classifier's probabilities are
> systematically under-confident, which is part of why the cost-optimal threshold would
> decline 5% of legitimate traffic and is unshippable. The LLM's confidence field is
> deliberately unvalidated, because twelve clusters and no ring labels can't support a
> calibration claim — and faking one would repeat the exact mistake that got my first
> project abandoned."

> "This is detection-only. It never generates or optimises evasive transactions and has no
> capability to."

> "Everything reproduces from a clean clone: fetch the data, run the pipeline, get these
> numbers. 213 tests covering the temporal split, the graph algorithms against networkx, the
> LLM schema and its number-provenance guard, the webhook's signature and idempotency, and
> the import boundaries. Thanks for watching."

---

## If you overrun

Cut in this order — never cut the negative result or the reconciliation, they're the spine:

1. The pgRouting/networkx aside in §2:00 (30s)
2. The calibration line in §4:20 (10s)
3. The incremental-k-core section down to just "I predicted it would lose; it won by 2.8×,
   and the prediction is still in the repo" (25s)
