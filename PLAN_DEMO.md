# RingWatch — demo and webhook design

Plan for the live demo layer. Written before implementation.

The whole of this document is constrained by one rule that the rest of the project already
lives under: **the demo displays results, it never produces them.** Nothing in `app/`
computes a score, recomputes an ablation, or alters a metric. If the demo layer were
deleted, every number RingWatch reports would be identical — which is the claim the project
already makes about its LLM layer, extended to its web layer.

---

## 1. What the dashboard surfaces, and where it comes from

### The problem: nothing is currently persisted

Every number this project reports exists only as stdout from `python run.py`. `data/cache/`
holds `.npy` score arrays and `.txt` boosters — no structured results. And `data/cache/` is
gitignored, as is `data/raw/`, so a deployed instance has neither the 683 MB dataset nor the
84 MB parquet cache.

That constraint is convenient rather than annoying. It forces the honest architecture:

```
   LOCAL (has the dataset)                    DEPLOYED (has neither)
   ─────────────────────────                  ──────────────────────
   python run.py --stage ablation
   python scripts/export_results.py  ──────►  docs/results.json  ──►  FastAPI renders it
        (computes everything)                  (committed, small)      (computes nothing)
```

`scripts/export_results.py` runs where the data is, orchestrates **only functions that
already exist**, and serialises their output. The web app reads that file. There is no code
path by which a page load can trigger a computation.

### Functions the export step reuses (no new analysis code)

| Artifact | Existing function | Module |
|---|---|---|
| Split sizes, fraud rate per split | `temporal_split`, `split_summary` | `core/split.py` |
| AUC-PR, AUC-ROC, both operating points | `evaluate` | `core/evaluate.py` |
| Ablation CIs | `bootstrap_auc_pr_delta` | `core/evaluate.py` |
| Ring concentration (z = +8.8) | `ring_concentration_test` | `core/ring_evidence.py` |
| Graph coverage, component sizes | `build_graph`, `graph_summary` | `core/graph.py` |
| Brier, ECE, reliability curve | `calibration_report` | `core/calibration.py` |
| The 12 flagged clusters | `build_cluster_evidence` | `core/clusters.py` |
| Narratives for those clusters | `narrate_all` | `ai/narrate.py` |

Output: `docs/results.json`, stamped with the git commit and generation time so a reader can
tell exactly which run produced the numbers on screen. Expected well under 100 KB.

### Page sections, in order

The ordering is the argument. A visitor sees the negative result first, because that is the
project's thesis, not its embarrassment.

1. **Headline — the graph layer does not work.** The four-variant table with 95% bootstrap
   CIs. `+ k-core` marked as significantly *worse*. Explained in a sentence, not buried
   beneath the things that did work.
2. **The reconciliation.** The permutation test (12 all-fraud components vs 1.4 ± 1.2,
   z = +8.8) placed directly beside graph coverage (5.38% of test rows). Those two numbers
   next to each other *are* the argument: the ring structure is real, and it is far too rare
   to move a metric averaged over 118,108 transactions.
3. **Operating points.** [A] cost-minimising and [B] insult-constrained side by side, with
   the insult-rate figures and a plain note on why [A] — declining 5.047% of legitimate
   traffic — is arithmetically correct and operationally unshippable.
4. **Calibration.** The reliability diagram, Brier and ECE, and the caveat that Brier
   conflates calibration with discrimination/refinement so it cannot be read as a pure
   calibration metric alone.
5. **Flagged clusters.** All 12, with every field tagged for provenance — `computed by
   core/` versus `written by the LLM`. The determinism boundary is the project's central
   design claim, so it should be legible on screen, not just in a test.
6. **Live webhook feed.** Section 2 of this document.

### Aesthetic

Plain and typographic. System font stack, real tables, generous whitespace, no gradients,
no cards-with-shadows, no hero section. Deliberately **not** a fraud-detection SaaS landing
page: this is a lab report that happens to be served over HTTP. No animated counters, no
invented metrics, no chart that isn't backed by a computation in `results.json`.

---

## 2. Webhook endpoint design

`POST /webhooks/razorpay`

### Verification and delivery semantics

These are the things a payments engineer checks first, so they are implemented properly
rather than sketched.

**Raw-body HMAC.** The handler takes `Request` and reads `await request.body()` — the raw
bytes — and computes `hmac.new(secret, raw_body, sha256).hexdigest()`, compared against the
`X-Razorpay-Signature` header with `hmac.compare_digest`.

There is deliberately **no Pydantic body model on this route**. Binding one would make
FastAPI parse the JSON, and any subsequent re-serialisation to obtain "the body" produces
different bytes — different key order, different whitespace, `1.0` where the sender wrote
`1.00`. The signature then fails for a payload that was never tampered with, and the bug
presents as intermittent verification failures that look like a Razorpay problem. A test
(`test_reserialized_body_fails_verification`) reproduces exactly this and asserts it fails,
so the constraint is documented in executable form rather than in a comment.

**Idempotency.** Razorpay delivers at-least-once; the same event can arrive more than once,
and does. `x-razorpay-event-id` is stored as a PRIMARY KEY in SQLite. A replay is detected
on insert, short-circuits, and returns 200 without re-running analysis — a duplicate must
never double-process.

**Fast 2xx.** Razorpay retries with exponential backoff for 24 hours and disables endpoints
that do not answer within roughly five seconds. So the handler verifies, records the event
id, and returns 200 — and only then does any analysis run, via FastAPI `BackgroundTasks`,
after the response has been sent. Nothing slow happens on the request path.

**Status codes.** Bad signature, missing header, or malformed JSON → **4XX**: the request
is defective and retrying it unchanged cannot help, so Razorpay should stop. A transient
internal failure (storage unavailable) → **5XX**, so Razorpay *does* retry. The stored
payload is never mutated between attempts.

**Credentials.** `RAZORPAY_WEBHOOK_SECRET` from the environment, documented in
`.env.example`, test mode (`rzp_test_…`) only. `.env` is gitignored and stays that way.

### What the handler does with an event — two separate tracks

This is where the demo has to be careful, because the obvious implementation would be
dishonest.

**The finding that forced this design:** the trained classifier expects **433 features**.
A Razorpay payment payload can supply roughly **6** of them. `card1`, `card2` and `card5`
are Vesta-internal identifiers, not card network or type; `addr1`/`addr2` and `dist1`/`dist2`
likewise; `C1–C14`, `D1–D15`, `M1–M9` and `V1–V339` are Vesta's proprietary engineered
features with no counterpart in any Razorpay payload; and even `tx_hour` is derived from
`TransactionDT`, which counts seconds from an unpublished origin. LightGBM will accept 427
NaNs and return a number, but that number comes from a model operating almost entirely
outside its training distribution.

So the handler runs two tracks and the interface never lets them blur.

**Track 1 — graph structure. A real computation.**

Classical graph algorithms do not care what domain the identifiers came from. An entity
fingerprint is built from Razorpay-native fields:

- card fingerprint: `card.last4` + `card.network` + `card.issuer`
- `email` domain
- `contact`
- `vpa` (UPI), when present

and linked entities are found with the **existing** `connected_components` and
`k_core_numbers` from `core/graph.py`, under the same hub-suppression discipline. What is
reported is structural and factual: how many other live events share a fingerprint, the
component size, the core number, and which attribute did the linking. No probability, no
fraud claim, no distribution assumption. This is the part of RingWatch that genuinely
transfers to a new domain, and it is the project's actual differentiator.

**Track 2 — the model score. Explicitly a demonstration.**

The baseline booster is run on the mapped fields with the rest as NaN, and the result is
shown with a **computed** coverage figure beside it, e.g. *"6 of 433 features present; 427
imputed as missing."* That number is measured per event, not a fixed disclaimer, so the
caveat is falsifiable in the same way everything else here is. The accompanying text states
plainly that the model was trained on IEEE-CIS US e-commerce data, has never seen a Razorpay
payload, and that this shows the ingestion → scoring path working end to end rather than an
assessment of the transaction. If `artifacts/model_baseline.txt` is absent, this track
degrades to "unavailable" rather than failing the request.

Presenting both is more informative than presenting either. The contrast — structure
transfers, the trained model does not — is a real result about transfer learning across
payment ecosystems, and stating it is more useful to a reviewer than quietly showing a
number that means nothing.

### Other routes

| Route | Purpose |
|---|---|
| `GET /` | The dashboard |
| `GET /health` | Liveness, for the platform health check |
| `GET /api/events` | Recent webhook events as JSON, for the live feed |
| `GET /docs` | FastAPI's automatic OpenAPI page |

---

## 3. Stack and deployment

### FastAPI + Jinja2, not Streamlit

Argued rather than assumed, since the brief invited the alternative:

- **The webhook needs a real HTTP server.** Signature verification requires access to raw
  request bytes and control over status codes. Streamlit is a dashboard runtime, not an HTTP
  framework, and bolting a webhook onto it means running a second service. One process is
  simpler to deploy, reason about, and keep alive on a free tier.
- **No build step.** Server-rendered Jinja2 with a little vanilla JS adds `fastapi`,
  `uvicorn` and `jinja2` to a dependency list that is currently nine lines. No npm, no
  bundler, no lockfile churn.
- **The page is static in nature.** It renders one JSON file. Streamlit's re-execution model
  buys nothing when there is no interactive computation to drive.
- **Presentation control.** The required aesthetic is a plain lab report; Streamlit's default
  chrome fights that.

### Render, free tier

- Genuinely free web service tier, no card required, direct deploy from the GitHub repo.
- Honest caveats, to be stated in the README rather than discovered by a confused visitor:
  the free instance **spins down after ~15 minutes idle**, so a first request can take
  **30–60 seconds**; and its disk is **ephemeral**, so the SQLite webhook log does not
  survive a restart or redeploy. Both are acceptable for a demonstration and neither affects
  any reported metric, because the metrics are a committed artifact.
- `artifacts/model_baseline.txt` (6.8 MB) is committed, since `data/cache/` is gitignored and
  Track 2 needs the booster. Well inside GitHub's limits.

Local webhook testing: `cloudflared tunnel --url http://localhost:8000` (free, no account),
then point a Razorpay test-mode webhook at the resulting URL.

### What is live versus what is local

To be stated explicitly in the README, because implying the deployed app retrains anything
would be exactly the sort of overstatement this project has already had to correct once:

- **Live:** the dashboard (rendering committed results) and the webhook path (real signature
  verification, real idempotency, real graph structure computed on live events).
- **Local only:** the full pipeline — data download, temporal split, model training, the
  four-variant ablation, the bootstrap. These need the 683 MB dataset and several minutes of
  CPU. The deployed app never retrains and never recomputes a metric.

---

## 4. Build order

1. `scripts/export_results.py` → `docs/results.json`, with a test asserting the known values
   survive the round trip (AUC-PR 0.5188, z = +8.8, 12 clusters).
2. `app/main.py` + `app/templates/dashboard.html` — sections 1–5. Verify locally.
3. `app/webhook.py`, `app/store.py`, `app/live_graph.py`, `app/scoring.py` + section 6, with
   the five required tests.
4. `render.yaml`, deploy, README updates.
