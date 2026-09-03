"""Draft a recommended disposition for a flagged cluster. Advisory only.

WHAT THIS IS ALLOWED TO DO
--------------------------
Read a frozen `CaseFile` that `core/` assembled, and write a recommendation with reasoning.

WHAT IT CANNOT DO, STRUCTURALLY RATHER THAN BY POLICY
------------------------------------------------------
- **Compute or alter a score.** This module imports nothing from `core/`; there is no
  scoring code reachable from here, which `tests/test_ai_boundary.py` enforces by walking
  the import graph.
- **Decide a match.** Cluster selection happened in `core/clusters.py` before this ran.
- **Execute anything.** There is no action surface in the contract — no client, no
  callback, no write path. It returns a dataclass.
- **State a number it was not given.** Every numeric token in its output must already
  appear in the case file, enforced by the same provenance guard the narrative layer uses.

THE APPROVAL GATE IS THE FEATURE
--------------------------------
A `Disposition` is a recommendation, and `requires_human_approval` is hard-coded true on the
type. Nothing in this system applies one. That is the interesting engineering here, not the
autonomy: an "agentic" layer whose entire design effort went into bounding what it may
touch. Only about 2% of financial-services AI deployments are fully autonomous, and the
reason is this one.
"""

from __future__ import annotations

from ai.contract import (
    DISPOSITIONS,
    CaseFile,
    Disposition,
    disposition_unavailable,
)
from ai.provider import ProviderError, complete
from ai.schema import ValidationError, validate_disposition

SYSTEM_RULES = f"""You are assisting a fraud analyst reviewing a flagged cluster of payment
entities. A deterministic statistical engine has already flagged this cluster and computed
every figure below. You are drafting a recommended disposition for a human to approve or
reject.

Absolute rules:
1. You do NOT decide anything. Your output is a recommendation that a human reviews. Write
   it as advice, not as a verdict.
2. You MUST NOT state any number that does not already appear in the case file. Do not add,
   total, average, convert or estimate any figure. Describe quantities in words if you need
   one you were not given.
3. Weigh the findings on BOTH sides. The case file deliberately includes findings that
   argue against concern, and a recommendation that ignores them is not useful to a
   reviewer.
4. Respond with a single JSON object and nothing else. No markdown, no code fence.

The JSON object must have exactly these fields:
  "recommendation": one of {' | '.join(DISPOSITIONS)}
  "confidence":     one of high | medium | low
  "rationale":      one paragraph explaining the recommendation to the analyst, engaging
                    with the findings on both sides
  "key_factors":    a list of 2 to 4 short strings, each naming one factor that drove the
                    recommendation

Choose "dismiss" when the evidence looks like ordinary shared infrastructure. Choose
"escalate" when the evidence is genuinely ambiguous and a human should look harder — that
is a legitimate and often correct answer, not a failure to decide. Choose "confirm" only
when the findings supporting concern clearly outweigh those against.

Your confidence is measured against outcomes, so overstating it is penalised.
"""


def build_prompt(case: CaseFile, correction: str | None = None) -> str:
    prompt = f"{SYSTEM_RULES}\n\nCASE FILE\n---------\n{case.as_prompt_facts()}\n"
    if correction:
        prompt += (
            "\nYour previous response was REJECTED for this reason:\n"
            f"  {correction}\n"
            "Return a corrected JSON object obeying every rule above.\n"
        )
    return prompt


def draft_disposition(case: CaseFile, use_cache: bool = True) -> Disposition:
    """One validated recommendation, or an honest DISPOSITION_UNAVAILABLE.

    One correction round, exactly as the narrative layer does. There is deliberately no
    third path in which a partly-valid response is patched up and used: a reviewer cannot
    tell which sentences of a repaired recommendation were the model's.
    """
    correction: str | None = None

    for attempt in range(2):
        try:
            response = complete(build_prompt(case, correction), use_cache=use_cache)
        except ProviderError as exc:
            return disposition_unavailable(case.case_id, f"provider unavailable: {exc}")

        try:
            return validate_disposition(response.text, case)
        except ValidationError as exc:
            correction = str(exc)
            if attempt == 1:
                return disposition_unavailable(
                    case.case_id, f"validation failed twice; last error: {correction}"
                )

    return disposition_unavailable(case.case_id, "exhausted attempts")


def draft_all(cases: list[CaseFile], use_cache: bool = True) -> list[Disposition]:
    return [draft_disposition(case, use_cache=use_cache) for case in cases]
