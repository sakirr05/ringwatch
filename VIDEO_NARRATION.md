# RingWatch — narration script for the final cut

Read this over the six screen recordings, in order, after the generated intro.

- **Intro clip (already made):** ~30s
- **This script:** **771 words**

| Your pace | Narration | + intro = total |
|---|---|---|
| 145 wpm (slow) | 5:19 | 5:49 — **too long** |
| 155 wpm (natural) | 4:58 | 5:28 — **slightly long** |
| 165 wpm (brisk) | 4:40 | 5:10 — **acceptable** |

**To land comfortably under 5:00, make the three cuts listed at the bottom
before you record.** That removes 103 words and brings a natural 155 wpm read
to 4:18, or 4:48 with the intro.

Read at a steady pace and pause between sections — silence over a terminal is
fine and gives the viewer time to read.

---

## CLIP 1 · Ablation terminal — 40s · 100 words

> This is the ablation table, straight from the pipeline.

> Four variants. Tabular only, then the same model with graph features added.
> The column on the right is the change versus baseline, and every one of them
> is negative.

> Plus k-core is minus zero point zero zero six four, and its confidence
> interval excludes zero. That's not noise. The graph layer I built this whole
> project around makes the model measurably worse.

> Four hundred paired bootstrap resamples produced those intervals. A single-run
> difference of zero point zero zero two is a number, not a result — so I never
> reported one without a confidence interval.

---

## CLIP 2 · Browser recording — 75s · 185 words

> Here's the dashboard. The negative result is section one, because it's the
> headline, not a footnote.

> But there's a second finding, and it isn't in tension with the first. Fraud
> genuinely does cluster in the entity graph. Twelve fully-fraudulent connected
> components, against one point four expected by chance. That's eight point
> eight sigma. The structure is real — and it replicates on Elliptic, a bitcoin
> dataset whose graph edges are observed rather than inferred.

> Both are true because the rings are real but rare. Twelve of them can't move a
> metric averaged over a hundred and eighteen thousand transactions, and the
> graph only reaches five point three eight percent of test rows.

> The obvious objection is: give it more coverage. I tested that rather than
> arguing about it. Tripling coverage moves the metric by plus zero point zero
> zero zero three — an order of magnitude inside the noise band.

> What the graph is actually good for is surfacing. Inside these flagged
> clusters, forty-nine percent of transactions are fraud, against a three point
> four percent base rate. Fourteen times enrichment. That's worth an analyst's
> queue. It is not predictive lift, and the page says so on the same screen.

---

## CLIP 3 · Webhook terminal — 35s · 85 words

> This is a real Razorpay webhook receiver. Watch the ordering, because the
> ordering is the design.

> The signature is verified against the exact bytes Razorpay sent. Re-serialise
> the JSON first and those bytes change, so verification fails on payloads
> nobody tampered with.

> The event ID is a primary key. Replay the same event and the insert collides —
> the database detects the duplicate, not a check-then-insert race.

> And when I corrupt the body after signing, it's rejected. Four oh one. The
> check isn't decorative.

---

## CLIP 4 · Score endpoint and failure log — 50s · 125 words

> There's a scoring endpoint too, and it tells you how little transfers.

> The model expects four hundred and thirty-three features. A Razorpay payload
> supplies three. Zero point six nine percent coverage, and the response says
> so in its own body — is-fraud-assessment, false. Publishing that number is
> more useful than publishing the score.

> And this is my failure log. Twenty-four entries, written as things broke.

> I threw away a working project on day one — a reconciliation agent, tests
> green — because I realised I was generating the data, injecting the errors,
> and grading my own matcher against ground truth I'd also written. A closed
> loop. It proved nothing.

> Betweenness centrality hung for twelve minutes. I measured drift wrong, then
> measured my own correction wrong. I nearly published a headline that didn't
> survive multiplicity correction.

---

## CLIP 5 · Narrate terminal — 30s · 75 words

> There's a language model here, and it writes prose about clusters the engine
> already flagged.

> Look at this line: max risk, zero point seven four six four — computed by
> core, not the LLM. That's the boundary, stated in the tool's own output. The
> number came from the engine. The sentence came from the model.

> It cannot compute a score. The entire import closure of my AI package is the
> standard library plus an HTTP client, and a test walks the import graph to
> prove it.

---

## CLIP 6 · Investigate terminal — 40s · 100 words

> It also drafts a recommendation for each cluster. Confirm, dismiss, or
> escalate.

> Twelve of twelve validated. Nine said escalate — which is the honest answer
> when a cluster is genuinely ambiguous, not a failure to decide.

> And look at the line under every single one. Applied: no. Requires human
> approval. Executes nothing.

> Approving one writes an audit row. No card is blocked, no customer is
> contacted, no external service is called. The gate is the feature — that's
> the interesting engineering here, not the autonomy.

---

## CLOSE — 25s · 60 words

> Last thing. I wiped the model cache and retrained everything from raw data.
> All six score files came back bit-identical. The full export had zero
> differences.

> What I can't tell you is whether these clusters are really rings. This dataset
> labels transactions, not rings — so I never claim a ring count anywhere.

> Five hundred and forty tests, and I never loosened one. The graph layer
> failed. The harness that proved it failed is what I'd actually ship.

---

## If you run long

Cut in this order:

1. **Clip 4** — drop "Betweenness centrality hung for twelve minutes…" (28 words)
2. **Clip 2** — drop the coverage-objection paragraph (45 words)
3. **Clip 3** — drop the primary-key sentence (30 words)

Do **not** cut: the ablation result, the failure log, or the ring-count caveat.

## Pronunciation

Numbers are spelled out above on purpose — read them as written. "AUC-PR" is
avoided entirely; say "the metric" if you need to refer to it.
