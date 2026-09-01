"""Turn deterministic cluster evidence into analyst-facing prose.

This module is the whole of RingWatch's AI surface. Note what it does NOT do: it does not
score, rank, threshold, flag, or compute. It receives clusters that `core/` has already
decided are suspicious, along with the evidence `core/` computed, and it writes sentences
about them. If this entire package were deleted, every metric RingWatch reports would be
unchanged.

The retry policy is: one validation failure is forgiven, with the failure reason fed back
to the model so it can correct itself. A second failure is final and the cluster gets
NARRATIVE_UNAVAILABLE. There is deliberately no third path where a partially-valid
response is patched up and used.
"""

from __future__ import annotations

from ai.contract import PROBABLE_CAUSES, ClusterEvidence, ClusterNarrative, unavailable
from ai.provider import ProviderError, complete
from ai.schema import ValidationError, validate

SYSTEM_RULES = f"""You are a fraud-analysis writing assistant for a payments risk team.

You will be given deterministically-computed evidence about ONE cluster of related
payment entities that an upstream statistical model has already flagged as suspicious.

Your ONLY job is to explain that evidence to a human fraud analyst in plain language.

Absolute rules:
1. You do NOT decide whether this cluster is fraudulent. That decision has already been
   made by the upstream model. You explain, you do not adjudicate.
2. You MUST NOT state any number that does not already appear in the evidence below.
   Do not add, average, total, convert, or estimate any figure. If you want to describe a
   quantity you were not given, describe it in words instead.
3. Respond with a single JSON object and nothing else. No markdown, no code fence, no
   commentary.

The JSON object must have exactly these four string fields:
  "probable_cause":   one of {' | '.join(PROBABLE_CAUSES)}
  "confidence":       one of high | medium | low
  "human_summary":    one paragraph for a fraud analyst explaining what this cluster
                      looks like and why it was flagged
  "suggested_action": one line recommending what the analyst should do next

Choose "BENIGN_COINCIDENCE" when the evidence looks like ordinary shared infrastructure
rather than coordination. Choose "UNEXPLAINED" when the evidence does not support any
specific story. Being honest that a cluster is unremarkable is more useful to the analyst
than inventing a narrative for it.

Set "confidence" to reflect how strongly the EVIDENCE supports your stated cause. This
field is measured against outcomes, so overstating it is penalised.
"""


def build_prompt(evidence: ClusterEvidence, correction: str | None = None) -> str:
    prompt = f"{SYSTEM_RULES}\n\nEVIDENCE\n--------\n{evidence.as_prompt_facts()}\n"
    if correction:
        prompt += (
            "\nYour previous response was REJECTED for this reason:\n"
            f"  {correction}\n"
            "Return a corrected JSON object obeying every rule above.\n"
        )
    return prompt


def narrate_cluster(evidence: ClusterEvidence, use_cache: bool = True) -> ClusterNarrative:
    """Produce one validated narrative, or an honest NARRATIVE_UNAVAILABLE."""
    correction: str | None = None

    for attempt in range(2):  # initial attempt, then one correction round
        prompt = build_prompt(evidence, correction)
        try:
            response = complete(prompt, use_cache=use_cache)
        except ProviderError as exc:
            return unavailable(evidence.cluster_id, f"provider unavailable: {exc}")

        try:
            return validate(response.text, evidence)
        except ValidationError as exc:
            correction = str(exc)
            if attempt == 1:
                return unavailable(
                    evidence.cluster_id,
                    f"validation failed twice; last error: {correction}",
                )

    return unavailable(evidence.cluster_id, "exhausted attempts")


def narrate_all(
    evidence_items: list[ClusterEvidence], use_cache: bool = True
) -> list[ClusterNarrative]:
    return [narrate_cluster(item, use_cache=use_cache) for item in evidence_items]
