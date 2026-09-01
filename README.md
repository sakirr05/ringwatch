# RingWatch

**Graph-aware fraud ring detection with an enforced AI/determinism boundary.**

Razorpay AI Buildathon 2026 · Track 02 (AI Risk Manager)

> ## ⚠️ Detection-only
>
> RingWatch is **strictly a detection and analysis system.** It does not generate,
> simulate, optimize, or assist in producing adversarial, evasive, or fraudulent
> transactions, and contains no capability to do so. There is no attack simulation, no
> synthetic-fraud generation, and no adversarial-evasion tooling anywhere in this
> repository. This is both the Track 02 requirement and a design constraint the code
> structure honours.

---

## Problem statement

Card-testing rings, refund-abuse networks, and coordinated chargeback fraud don't look
like fraud one transaction at a time. Each individual payment can sit comfortably inside
a legitimate-looking distribution — modest amount, plausible merchant category, ordinary
hour — while the *coordination between* those payments is the only real signal. A
row-by-row classifier, no matter how well tuned, is structurally blind to it: it scores
each transaction independently and never sees that forty of them share a fingerprint.

RingWatch adds a classical graph layer over an entity graph to surface that coordination,
and — critically — measures honestly whether doing so actually helps.

## Status

**In development.** Phases 0–1 (recon, data acquisition, scaffold) complete. No metrics
are reported yet; this README will carry measured numbers only once they have actually
been produced by `core/evaluate.py`. See `PLAN.md` for the phase plan and `FAILURE_LOG.md`
for the build's real dead ends.

## Data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) —
590,540 labeled transactions, 394 columns, **3.4990% fraud rate**, spanning 182 days.
Labels are real and externally authored: I did not create this data and cannot tune it to
flatter the model.

## Architecture

The central structural rule: **one box computes numbers, a completely separate box writes
sentences, and code — not convention — prevents the second from touching the first.**

```mermaid
flowchart TB
    subgraph DET["🔒 DETERMINISTIC LAYER — core/ · computes every number"]
        direction TB
        RAW["IEEE-CIS raw CSV<br/>590,540 transactions"]
        SPLIT["core/split.py<br/><b>temporal</b> split @ 80th pct<br/>472,432 train / 118,108 test"]
        FEAT["core/features.py<br/>tabular features"]

        subgraph GRAPH["core/graph.py — classical graph algorithms, no GNN"]
            UID["entity fingerprint<br/>card1 + addr1 + (day − D1)"]
            HUB["hub suppression<br/>cap=5, below percolation"]
            CC["connected components<br/>union-find"]
            KC["<b>k-core decomposition</b><br/>Batagelj–Zaversnik peeling<br/>hand-written, O(V+E)"]
            UID --> HUB --> CC & KC
        end

        MODEL["core/model.py<br/>LightGBM · CPU · seed=42"]
        EVAL["core/evaluate.py<br/><b>AUC-PR</b> · PR curve<br/>insult-rate costing"]
        CLUST["core/clusters.py<br/>selects flagged clusters<br/>computes all evidence"]

        RAW --> SPLIT --> FEAT --> MODEL
        SPLIT --> UID
        CC & KC -->|structural features only<br/>no label-derived features| MODEL
        MODEL --> EVAL --> CLUST
    end

    CONTRACT["<b>ai/contract.py</b><br/>frozen ClusterEvidence<br/>━━━ THE BOUNDARY ━━━<br/>the only shared surface"]

    subgraph AI["✍️ NARRATIVE LAYER — ai/ · writes only sentences"]
        direction TB
        PROV["ai/provider.py<br/>Gemini → Groq fallback<br/>retry once · SHA-256 cache"]
        SCHEMA["ai/schema.py<br/>strict JSON validation<br/><b>number-provenance guard</b>"]
        OUT["validated narrative<br/>or NARRATIVE_UNAVAILABLE"]
        PROV --> SCHEMA --> OUT
    end

    CLUST -->|already-flagged clusters<br/>+ their evidence| CONTRACT
    CONTRACT --> PROV
    AI -.->|❌ no import path back<br/>enforced by test| DET

    style DET fill:#e8f4ea,stroke:#2d6a4f,stroke-width:3px
    style AI fill:#eef2fb,stroke:#3d5a99,stroke-width:3px
    style CONTRACT fill:#fff4e0,stroke:#b06500,stroke-width:3px
    style KC fill:#d8eede,stroke:#2d6a4f,stroke-width:2px
    style SCHEMA fill:#dde6f7,stroke:#3d5a99,stroke-width:2px
```

**How the boundary is enforced** — not by convention, but by three mechanisms:

1. **One-way imports.** `core/` imports `ai.contract` to build the handover object.
   `ai/` never imports `core/`. `tests/test_ai_boundary.py` parses the AST of every
   module under `ai/` and fails if any of them imports the engine or a modelling library.
   The LLM has no code path to a score.
2. **Frozen evidence.** `ClusterEvidence` is an immutable dataclass. The narrative layer
   physically cannot mutate what it was given.
3. **Number-provenance guard.** Every numeric token in the model's prose must already
   appear in the evidence. An invented rupee total, an invented count — even *correct*
   arithmetic the model performed itself — is rejected. Two failures and the cluster gets
   `NARRATIVE_UNAVAILABLE` rather than a guess.

If the entire `ai/` package were deleted, every metric RingWatch reports would be
unchanged.

## Ground-truth honesty

**IEEE-CIS provides transaction-level fraud labels, not ring-level labels.** RingWatch
therefore will never claim a metric like "N fraud rings caught" — that number cannot be
validated against this dataset, and reporting it would be fabrication. The two claims
this project makes are the two it can actually defend:

1. Fraud-labeled transactions **cluster more densely** in the entity graph than
   legitimate ones, demonstrated through graph statistics.
2. Graph-derived features produce a **measured PR-AUC lift** over an identical tabular
   baseline, demonstrated by ablation on a temporally held-out test set.

## Why AUC-PR and not accuracy

At a 3.4990% positive class, a model that predicts "never fraud" for every single
transaction achieves **96.5% accuracy** while catching zero fraud. Accuracy is not a
weak metric here, it is an actively misleading one. RingWatch optimizes and reports
**AUC-PR** (area under the precision-recall curve), reports the full precision/recall
curve, and defends a specific chosen operating threshold rather than hiding behind a
single aggregate.

False positives are costed explicitly as an **insult rate**: at the chosen threshold, how
many legitimate customers were wrongly declined, and what that costs in rupees under a
named, documented assumed order value.

## Limitations

Written as they are discovered, not reconstructed at the end.

- **No ring-level ground truth.** See above. Every ring-topology claim is a statement
  about graph structure and measured lift, never a validated ring count.
- **Entity resolution is a heuristic.** The `card1 + addr1 + (day − D1)` fingerprint is
  the competition-validated proxy for a customer ID, but it is a proxy. It will merge
  distinct customers who collide on all three, and split one customer across values when
  `D1` is null (0.21% of rows) or `addr1` is null (11.13% of rows).
- **`addr1` has only 332 distinct values** across 590k transactions, so it is a weak,
  hub-forming identifier. Graph construction applies hub suppression to prevent hairball
  components; the thresholds are documented and are a judgment call, not a derived
  optimum.
- **The identity table is sparse** — it covers a minority of transactions. The graph is
  deliberately **not** built on `DeviceInfo` or `id_30`–`id_33` for this reason; those
  columns would produce a mostly-disconnected graph of noise.
- **Single dataset.** Findings are demonstrated on IEEE-CIS only.

*(Extended after each phase.)*

## Deployment

Reproducible from a clean clone on CPU alone. **No GPU, no cloud compute, no Kaggle
account.** End-to-end cold run is roughly 20 minutes, dominated by training four model
variants for the ablation.

### 1. Environment

```bash
git clone <this-repo> && cd ringwatch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.14.6 with pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0,
LightGBM 4.7.0, networkx 3.6.1. Needs ~4 GB RAM and ~2 GB disk.

### 2. Data

```bash
python scripts/fetch_data.py
```

Downloads `train_transaction.csv` (652 MB) and `train_identity.csv` (26 MB) from an
ungated mirror of the IEEE-CIS competition data and verifies row counts. The canonical
source is Kaggle, which requires an authenticated account that has accepted the
competition rules; the mirror keeps this repo reproducible for reviewers who have
neither. Integrity is re-checked on load — row count and fraud rate are asserted against
published values, so a truncated download fails loudly rather than quietly degrading a
metric.

### 3. Run the pipeline

```bash
python run.py --stage data       # build the Parquet cache, print split statistics
python run.py --stage graph      # entity graph + ring-concentration test
python run.py --stage ablation   # the four-variant comparison and the negative case
python run.py --stage all        # everything
```

Every stage is deterministic (fixed seeds, `deterministic=True` in LightGBM) and caches
its expensive artifacts under `data/cache/`, so re-runs are fast and reproduce identical
numbers.

### 4. Narratives (optional)

The LLM layer is the only part that needs network access or credentials, and **nothing
in the metrics depends on it**:

```bash
cp .env.example .env    # add GEMINI_API_KEY and/or GROQ_API_KEY
python run.py --stage narrate
```

Without keys, this stage reports `NARRATIVE_UNAVAILABLE` for every cluster and the rest
of the pipeline is unaffected. Responses are cached by SHA-256 of the prompt, so repeat
runs are free and reproducible.

### 5. Tests

```bash
pytest -q
```

Covers temporal-split correctness (including no-leakage under tied timestamps), the
k-core implementation against networkx as an independent oracle, the ring-concentration
statistic on planted and null structure, LLM JSON schema validation, the
number-provenance guard, and the AI/determinism import boundary.

## Legacy

`legacy/ledgerloop/` holds an earlier, abandoned Track 04 project, retained deliberately
with its git history. It was working code, dropped on purpose — the reasoning is entry #1
of `FAILURE_LOG.md`.
