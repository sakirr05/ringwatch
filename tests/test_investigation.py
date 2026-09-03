"""Tests for the bounded investigation orchestrator.

Three things are being pinned here, in descending order of importance:

1. **The orchestrator cannot state a number it was not given.** The provenance guard now
   covers `key_factors` as well as the rationale, and a recommendation is exactly the kind
   of confident prose an analyst rubber-stamps, so an invented figure there is worse than
   one in a descriptive narrative.
2. **The case file argues both ways.** `contradicting` is not decoration. If the engine
   only ever assembled incriminating detail, the drafter would be reasoning from a
   prosecutor's brief and its recommendations would be worth less than nothing.
3. **Nothing is applied.** `requires_human_approval` is true on the type, the audit trail
   is append-only, and the decision route records rather than acts.
"""

from __future__ import annotations

import json

import pytest

from ai.contract import CaseFile, ClusterEvidence, Disposition, disposition_unavailable
from ai.schema import MAX_FACTORS, MAX_RATIONALE_CHARS, ValidationError, validate_disposition
from core.investigation import BURST_WINDOW_DAYS, build_case_files


def make_evidence(**overrides) -> ClusterEvidence:
    defaults = dict(
        cluster_id=1,
        entity_count=4,
        transaction_count=37,
        flagged_transaction_count=12,
        component_size=4,
        core_number=2,
        max_degree=3,
        shared_attributes=("card1", "addr1_pemail"),
        distinct_cards=4,
        distinct_addresses=1,
        distinct_email_domains=1,
        span_days=21,
        total_amount_inr=184320.50,
        max_risk_score=0.8123,
        mean_risk_score=0.4471,
    )
    defaults.update(overrides)
    return ClusterEvidence(**defaults)  # type: ignore[arg-type]


def make_case(**overrides) -> CaseFile:
    evidence = overrides.pop("cluster", make_evidence())
    defaults = dict(
        case_id="CASE-001",
        cluster=evidence,
        rank=1,
        total_flagged_clusters=12,
        risk_percentile=91.67,
        entities_in_other_clusters=2,
        transactions_per_entity=9.25,
        flagged_share=0.32,
        population_mean_risk=0.3311,
        corroborating=("all 4 entities share 2 identifying attributes",),
        contradicting=("activity spans 21 days",),
    )
    defaults.update(overrides)
    return CaseFile(**defaults)  # type: ignore[arg-type]


def response(**overrides) -> str:
    payload = {
        "recommendation": "escalate",
        "confidence": "medium",
        "rationale": "The 4 entities share attributes but activity spans 21 days.",
        "key_factors": ["shared attributes", "long activity span"],
    }
    payload.update(overrides)
    return json.dumps(payload)


# --------------------------------------------------------------------------
# the case file: comparative context
# --------------------------------------------------------------------------


def test_rank_orders_by_max_risk_descending():
    items = [
        make_evidence(cluster_id=1, max_risk_score=0.42),
        make_evidence(cluster_id=2, max_risk_score=0.91),
        make_evidence(cluster_id=3, max_risk_score=0.65),
    ]
    files = build_case_files(items)
    assert [f.rank for f in files] == [3, 1, 2]
    assert [f.case_id for f in files] == ["CASE-001", "CASE-002", "CASE-003"]
    assert all(f.total_flagged_clusters == 3 for f in files)


def test_percentile_is_the_share_scoring_below():
    items = [make_evidence(cluster_id=i, max_risk_score=0.1 * i) for i in range(1, 5)]
    files = build_case_files(items)
    assert files[0].risk_percentile == pytest.approx(0.0)  # lowest
    assert files[-1].risk_percentile == pytest.approx(75.0)  # 3 of 4 below


def test_derived_ratios_match_the_evidence():
    case = build_case_files([make_evidence()])[0]
    assert case.transactions_per_entity == pytest.approx(37 / 4)
    assert case.flagged_share == pytest.approx(12 / 37)


def test_empty_input_produces_no_case_files():
    assert build_case_files([]) == []


def test_zero_transaction_cluster_does_not_divide_by_zero():
    case = build_case_files([make_evidence(transaction_count=0, flagged_transaction_count=0)])[0]
    assert case.flagged_share == 0.0
    assert case.transactions_per_entity == 0.0


def test_case_files_are_deterministic():
    items = [make_evidence(cluster_id=i, max_risk_score=0.5 + i / 100) for i in range(6)]
    assert build_case_files(items) == build_case_files(items)


# --------------------------------------------------------------------------
# findings on both sides
# --------------------------------------------------------------------------


def test_a_long_span_argues_against_a_burst():
    case = build_case_files([make_evidence(span_days=BURST_WINDOW_DAYS + 30)])[0]
    assert any("longer than a typical" in item for item in case.contradicting)
    assert not any("consistent with a burst" in item for item in case.corroborating)


def test_a_short_span_supports_a_burst():
    case = build_case_files([make_evidence(span_days=BURST_WINDOW_DAYS - 1)])[0]
    assert any("consistent with a burst" in item for item in case.corroborating)


def test_a_mostly_unflagged_cluster_argues_against_concern():
    case = build_case_files(
        [make_evidence(transaction_count=100, flagged_transaction_count=5)]
    )[0]
    assert any("scored normally" in item for item in case.contradicting)


def test_a_mostly_flagged_cluster_supports_concern():
    case = build_case_files(
        [make_evidence(transaction_count=100, flagged_transaction_count=80)]
    )[0]
    assert any("independently flagged" in item for item in case.corroborating)


def test_an_indirect_only_link_is_recorded_as_a_weakness():
    case = build_case_files(
        [make_evidence(shared_attributes=("indirect_attribute_chain",))]
    )[0]
    assert any("indirect chain" in item for item in case.contradicting)


def test_a_shared_card_across_entities_gets_its_innocent_explanation():
    """One card, several entities is also one household. The case file must say so."""
    case = build_case_files([make_evidence(distinct_cards=1, entity_count=3)])[0]
    assert any("household" in item for item in case.contradicting)


def test_the_benign_shaped_cluster_produces_contradicting_findings():
    """The whole point: an ordinary-looking cluster must not read as incriminating."""
    benign = make_evidence(
        span_days=140,
        transaction_count=200,
        flagged_transaction_count=6,
        distinct_cards=1,
        entity_count=3,
        shared_attributes=("indirect_attribute_chain",),
        core_number=1,
    )
    case = build_case_files([benign])[0]
    assert len(case.contradicting) >= 3
    assert not case.corroborating


def test_findings_appear_in_the_prompt_under_labelled_headings():
    case = build_case_files([make_evidence(span_days=90)])[0]
    facts = case.as_prompt_facts()
    assert "findings_supporting_concern:" in facts
    assert "findings_arguing_against_concern:" in facts


# --------------------------------------------------------------------------
# the provenance guard, over the new output shape
# --------------------------------------------------------------------------


def test_a_valid_disposition_is_accepted():
    result = validate_disposition(response(), make_case())
    assert isinstance(result, Disposition)
    assert result.recommendation == "escalate"
    assert result.key_factors == ("shared attributes", "long activity span")
    assert result.status == "OK"


def test_an_invented_number_in_the_rationale_is_rejected():
    raw = response(rationale="The cluster moved 999999 across 88 cards.")
    with pytest.raises(ValidationError, match="absent from the case file"):
        validate_disposition(raw, make_case())


def test_an_invented_number_in_a_key_factor_is_rejected():
    """The new surface. A figure smuggled into a bullet is as misleading as one in prose."""
    raw = response(key_factors=["shared attributes", "linked to 4712 other accounts"])
    with pytest.raises(ValidationError, match="4712"):
        validate_disposition(raw, make_case())


def test_numbers_from_the_case_file_context_are_quotable():
    """Rank, percentile and overlap are computed by core/, so the drafter may cite them."""
    case = make_case()
    raw = response(
        rationale="Ranked 1 of 12 at the 91.67 percentile, with 2 shared entities.",
        key_factors=["rank 1 of 12"],
    )
    assert validate_disposition(raw, case).confidence == "medium"


def test_numbers_quoted_inside_a_deterministic_finding_are_quotable():
    """The findings are shown to the drafter, so their figures must be allowed."""
    case = make_case(corroborating=("all 4 entities share 2 identifying attributes",))
    raw = response(rationale="All 4 entities share 2 identifying attributes.")
    assert validate_disposition(raw, case).recommendation == "escalate"


@pytest.mark.parametrize("bad", ["approve", "block", "CONFIRM", "", "confirm "])
def test_a_recommendation_outside_the_vocabulary_is_rejected(bad: str):
    with pytest.raises(ValidationError, match="not one of"):
        validate_disposition(response(recommendation=bad), make_case())


def test_a_confidence_outside_the_vocabulary_is_rejected():
    with pytest.raises(ValidationError, match="not one of"):
        validate_disposition(response(confidence="very high"), make_case())


def test_a_missing_field_is_rejected():
    payload = json.loads(response())
    del payload["key_factors"]
    with pytest.raises(ValidationError, match="missing required field"):
        validate_disposition(json.dumps(payload), make_case())


def test_an_extra_field_is_rejected():
    """Rejecting rather than ignoring: an unexpected field means a misunderstood contract."""
    payload = json.loads(response())
    payload["action"] = "block_card"
    with pytest.raises(ValidationError, match="unexpected field"):
        validate_disposition(json.dumps(payload), make_case())


def test_non_json_is_rejected():
    with pytest.raises(ValidationError, match="not valid JSON"):
        validate_disposition("Here is my recommendation: escalate.", make_case())


def test_a_json_array_is_rejected():
    with pytest.raises(ValidationError, match="expected a JSON object"):
        validate_disposition('["escalate"]', make_case())


def test_a_code_fence_is_stripped():
    fenced = f"```json\n{response()}\n```"
    assert validate_disposition(fenced, make_case()).recommendation == "escalate"


@pytest.mark.parametrize("factors", [[], "shared attributes", [""], ["  "], [7]])
def test_malformed_key_factors_are_rejected(factors):
    with pytest.raises(ValidationError):
        validate_disposition(response(key_factors=factors), make_case())


def test_too_many_key_factors_are_rejected():
    raw = response(key_factors=[f"factor {i}" for i in range(MAX_FACTORS + 1)])
    with pytest.raises(ValidationError, match="limit"):
        validate_disposition(raw, make_case())


def test_an_overlong_rationale_is_rejected():
    raw = response(rationale="a" * (MAX_RATIONALE_CHARS + 1))
    with pytest.raises(ValidationError, match="limit"):
        validate_disposition(raw, make_case())


@pytest.mark.parametrize("bad", ["", "   ", 12, None])
def test_a_malformed_rationale_is_rejected(bad):
    with pytest.raises(ValidationError):
        validate_disposition(response(rationale=bad), make_case())


# --------------------------------------------------------------------------
# how much the guard actually catches -- measured, not assumed
#
# `CaseFile.allowed_numbers()` is a wider set than `ClusterEvidence`'s, and a guard that
# widened until it stopped rejecting anything would be theatre. These two tests measure
# what it catches and what it provably does not, so neither claim rests on assertion.
# --------------------------------------------------------------------------


def test_the_guard_rejects_essentially_all_fabricated_figures():
    """Sampled plausible-looking figures: counts, risk scores, rupee amounts."""
    import random

    from ai.contract import _normalise_number

    rng = random.Random(0)
    case, allowed = make_case(), make_case().allowed_numbers()

    def sample() -> str:
        kind = rng.random()
        if kind < 0.35:
            return str(rng.randint(11, 999))
        if kind < 0.70:
            return f"{rng.uniform(0, 1):.4f}"
        return f"{rng.uniform(1000, 500000):.2f}"

    trials = 5000
    accepted = sum(1 for _ in range(trials) if _normalise_number(sample()) in allowed)
    assert accepted / trials < 0.01, f"guard accepts {accepted}/{trials} invented figures"
    assert case.allowed_numbers() == allowed  # and it is deterministic


def test_the_guard_cannot_catch_a_figure_borrowed_from_another_cluster():
    """A known, deliberate gap — pinned so the README's claim stays exactly this narrow.

    The guard checks provenance against the case file, not identity across case files.
    Two small clusters often share a figure honestly (both span 26 days), so a number
    misattributed from a sibling cluster passes. It prevents *fabrication*, which is the
    failure that makes prose authoritatively wrong; it does not prevent *misattribution*.
    """
    a = make_case(cluster=make_evidence(cluster_id=1, span_days=26, transaction_count=37))
    b = make_case(cluster=make_evidence(cluster_id=2, span_days=26, transaction_count=91))

    # A figure above the small-integer allowance IS caught when the clusters differ.
    borrowed = response(rationale="91 transactions over 26 days.")
    validate_disposition(borrowed, b)
    with pytest.raises(ValidationError, match="91"):
        validate_disposition(borrowed, a)

    # Where the two clusters genuinely coincide, nothing can catch it.
    coincide = response(rationale="Activity spans 26 days.")
    validate_disposition(coincide, a)
    validate_disposition(coincide, b)

    # And a small integer is never caught, by design: 0-10 are allowed unconditionally
    # because they appear in ordinary prose. That widens this gap and is the reason the
    # measured cross-case leakage is 100% for small integers and ~29% above them.
    small = response(rationale="The cluster contains 9 entities.")
    validate_disposition(small, a)  # 9 is not cluster 1's entity count, and passes anyway


# --------------------------------------------------------------------------
# the gate is on the type, not just the UI
# --------------------------------------------------------------------------


def test_every_disposition_requires_human_approval():
    assert validate_disposition(response(), make_case()).requires_human_approval
    assert disposition_unavailable("CASE-001", "provider down").requires_human_approval


def test_the_unavailable_fallback_is_unmistakably_empty():
    fallback = disposition_unavailable("CASE-004", "provider down")
    assert fallback.status == "DISPOSITION_UNAVAILABLE"
    assert fallback.rationale == "DISPOSITION_UNAVAILABLE"
    assert fallback.key_factors == ()
    assert fallback.rejection_reason == "provider down"


def test_the_fallback_defaults_to_escalate_not_dismiss():
    """When the drafter fails, the safe default is 'a human should look', not 'ignore'."""
    assert disposition_unavailable("CASE-004", "x").recommendation == "escalate"


def test_a_disposition_is_immutable():
    result = validate_disposition(response(), make_case())
    with pytest.raises(Exception):
        result.recommendation = "confirm"  # type: ignore[misc]


def test_a_case_file_is_immutable():
    case = make_case()
    with pytest.raises(Exception):
        case.risk_percentile = 100.0  # type: ignore[misc]


def test_the_disposition_carries_no_action_field():
    """There is no execution surface in the contract for the orchestrator to reach."""
    fields = set(Disposition.__dataclass_fields__)
    for forbidden in ("action", "apply", "execute", "block", "callback", "webhook"):
        assert forbidden not in fields
