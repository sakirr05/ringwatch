# RingWatch — 5-minute video script

What I built, and what building it was actually like.

Unlisted YouTube/Loom. Target **4:45–5:00**. Screen recording with voiceover. Two windows
only: a browser on the live demo, and a terminal. No slides.

**Total narration: 780 words.** That is 5:12 at 150 wpm and **4:43 at 165 wpm** — so this
script needs a brisk, confident pace, not a leisurely one. If you naturally read slowly, cut
one item from the overrun list below *before* you record rather than rushing the ending.
Word counts are marked per section so you can tell mid-recording whether you are running long.

**The through-line:** I set out to prove a graph layer improves fraud detection. It doesn't.
The project is the harness that proved it — and the reason that harness is worth showing is
that it disconfirmed my own idea three separate times and I kept every one of those results.

---

## Before recording

```bash
python run.py --stage ablation          # warm the model + score caches
python run.py --stage narrate           # warm the LLM cache — no dead air on camera
python run.py --stage investigate       # warm the orchestrator cache too
python scripts/export_results.py        # refresh docs/results.json
uvicorn app.main:app --port 8000        # leave running
```

- [ ] Terminal font large enough to read at 720p
- [ ] Browser on the live Render URL — **hit it once first** so the free tier is warm
- [ ] A signed webhook request ready to paste (for the 1:10 section)
- [ ] `FAILURE_LOG.md` open in a second tab (for the 3:20 section)
- [ ] Video set to **unlisted**, link tested in a private window

---

## 0:00–0:30 · Open with the failure · 85 words

*Screen: the live dashboard, section 1, ablation table visible.*

> "This is RingWatch. It finds fraud rings in payment data using a graph — and I'm going to
> start by telling you it doesn't work."

> "Here's my own ablation table. The graph layer I built the whole project around adds
> nothing. This row — plus k-core — is *significantly worse* than the baseline. Minus
> 0.0064, confidence interval excludes zero."

> "That's the first thing the README says too. Let me show you why I kept it, and why I
> think it's the most useful thing here."

---

## 0:30–1:10 · What I actually built · 106 words

*Screen: scroll to the flagged-clusters grid; hover one card.*

> "The idea is sound. A card-testing ring doesn't look like fraud one transaction at a time
> — each payment sits inside a perfectly normal distribution. The only signal is the
> coordination *between* them, and a row-by-row classifier is blind to that."

> "So I build an entity graph — 590,540 transactions collapsed into entities that share a
> card, an address, an email domain — and run classical graph algorithms over it. K-core
> decomposition, connected components, betweenness, PageRank. All hand-written, all
> validated against networkx. No GNN, no GPU."

> "And fraud genuinely does cluster. Twelve fully-fraudulent components against 1.4 expected
> by chance. That's 8.8 sigma. The structure is real."

---

## 1:10–1:55 · The Razorpay integration, live · 116 words

*Screen: terminal. Paste the signed webhook request. Then the dashboard's live feed.*

> "This is a real Razorpay webhook receiver, not a mock. Watch the ordering, because the
> ordering *is* the design."

> "Signature is verified against the exact bytes Razorpay sent — if you re-serialise the
> JSON first, the bytes change and verification fails on payloads nobody tampered with. The
> event ID is a primary key, so a replay collides and becomes a no-op. And the 200 goes back
> *before* any analysis runs, because an endpoint that thinks while Razorpay waits gets
> retried and eventually disabled."

*Screen: `POST /api/score`, point at `coverage_pct`.*

> "There's a scoring endpoint too — and it reports that the model needs 433 features and a
> Razorpay payload supplies 3. That's 0.69%. I'd rather publish how little transfers than
> publish the score."

---

## 1:55–2:40 · Both results are true at once · 120 words

*Screen: section 2, the reconciliation.*

> "So fraud clusters at 8.8 sigma, and the graph adds no predictive lift. Those aren't in
> tension — the rings are real but *rare*. Twelve of them can't move a metric averaged over
> 118,108 test transactions, and the graph only reaches 5.38% of test rows."

> "The obvious objection is: give it more coverage. I tested that instead of arguing about
> it. Tripling coverage moves AUC-PR by +0.0003 — an order of magnitude inside the noise
> band, confidence interval spanning zero. Quintupling it makes things worse."

*Screen: scroll to the cluster grid, point at the enrichment callout.*

> "What the graph *is* good for is surfacing. Inside those flagged clusters, 48.8% of
> transactions are fraud against a 3.44% base rate. That's 14 times enrichment — for an
> analyst's queue, not for the score."

---

## 2:40–3:20 · The boundary, and the one agent · 116 words

*Screen: hover a phrase in a narrative so the graph element lights up.*

> "There's an LLM here, and it writes prose about clusters the engine already flagged. Hover
> any number and it lights up the graph element it came from."

> "It cannot compute a score — the entire import closure of my `ai/` package is the standard
> library plus an HTTP client. There is no path from the model to the scorer, and a test
> walks the import graph to prove it."

*Screen: scroll to a disposition and its approval gate.*

> "It also drafts a recommendation — confirm, dismiss, or escalate. Twelve of twelve
> validated. Nine said escalate, which is the honest answer when a cluster is genuinely
> ambiguous. And approving one writes an audit row and executes nothing. No card is blocked.
> The gate is the feature."

---

## 3:20–4:15 · What building it was like · 136 words

*Screen: `FAILURE_LOG.md`, scroll it.*

> "Now the experience, because that's the honest part."

> "I threw away a working project on day one. A reconciliation agent, tests green — and I
> realised I was generating the data, injecting the errors, and grading my own matcher
> against ground truth I'd also written. A closed loop. It proved nothing. So I restarted
> against a dataset nobody on this project authored."

> "This is my failure log. Twenty-four entries, written as things broke."

> "Betweenness centrality hung for twelve minutes — I fixed it to 0.3 seconds. I measured
> drift wrong, then measured my own correction wrong. I nearly published a headline that
> didn't survive multiplicity correction. And in the final audit I found four numbers typed
> into the live page by hand that had gone stale — inside the paragraph explaining how
> carefully everything is checked."

---

## 4:15–5:00 · Limits, proof, close · 101 words

*Screen: terminal — run the clean-cache verification, or show the audit output.*

> "Last thing. I wiped the model cache and retrained everything from raw data. All six score
> files came back bit-identical by SHA-256, and the full export had zero differences. That's
> reproducibility measured, not claimed."

> "What I can't tell you is whether these clusters are really rings. This dataset labels
> transactions, not rings — so I never claim a ring count anywhere, because that number
> can't be validated and reporting it would be fabrication."

> "540 tests, and I never loosened one to make new code pass. The graph layer failed. The
> harness that proved it failed is the thing I'd actually ship."

---

## If you overrun

Cut in this order — each is self-contained:

1. **1:10–1:55 — the scoring endpoint half** (keep the webhook ordering) — saves ~30s
2. **2:40–3:20 — the orchestrator half** (keep the boundary) — saves ~35s
3. **1:55–2:40 — the coverage objection** (keep the enrichment) — saves ~30s

Do **not** cut the opening failure, the failure log, or the ring-count caveat. Those three
are the argument.

## If you underrun

Add, in this order:

- Cost-sensitive training made things worse, and the Kish effective sample size explained why
- The concentration finding replicates on Elliptic, a graph nobody had to infer
- The threshold explorer: 152 precomputed points, and the browser computes none of them
