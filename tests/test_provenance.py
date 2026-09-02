"""Tests for tracing LLM prose back to deterministic evidence.

Two properties matter above the rest, and both are about the annotator being a *view* over
the prose rather than an editor of it:

  * losslessness -- the segments must reconstruct the input exactly, so nothing the model
    wrote can be silently dropped or altered on its way to the screen;
  * no invention -- a value may only be marked if it genuinely appears in both the prose
    and the evidence.

If either fails, the dashboard would be showing something other than what the model
actually wrote, which is precisely the thing this feature exists to disprove.
"""

from __future__ import annotations

from core.provenance import Segment, annotate, annotate_to_dicts

EVIDENCE = {
    "cluster_id": 8,
    "entity_count": 3,
    "transaction_count": 12,
    "flagged_transaction_count": 6,
    "component_size": 4,
    "core_number": 2,
    "max_degree": 2,
    "span_days": 26,
    "distinct_cards": 1,
    "distinct_addresses": 1,
    "distinct_email_domains": 1,
    "total_amount_inr": 48114.00,
    "max_risk_score": 0.7464,
    "mean_risk_score": 0.3318,
    "shared_attributes": ["card1", "addr1", "P_emaildomain"],
}


def rebuild(segments: list[Segment]) -> str:
    return "".join(segment.text for segment in segments)


# --------------------------------------------------------------------------
# the two load-bearing properties
# --------------------------------------------------------------------------


def test_segments_reconstruct_the_input_exactly():
    """Lossless. Whatever the model wrote is what gets rendered."""
    text = (
        "Cluster 8 consists of 3 entities connected across 12 transactions over an "
        "activity span of 26 days, accumulating 48114.00 INR."
    )
    assert rebuild(annotate(text, EVIDENCE)) == text


def test_nothing_is_marked_that_is_not_in_the_evidence():
    """A figure absent from the evidence must be left untraced, never given a source."""
    text = "The cluster moved 999999 rupees across 4821 accounts."
    for segment in annotate(text, EVIDENCE):
        assert not segment.traced


def test_empty_text_yields_no_segments():
    assert annotate("", EVIDENCE) == []


def test_prose_without_evidence_values_is_a_single_segment():
    text = "This cluster appears consistent with ordinary shared household infrastructure."
    segments = annotate(text, EVIDENCE)
    assert len(segments) == 1
    assert not segments[0].traced


# --------------------------------------------------------------------------
# what gets traced
# --------------------------------------------------------------------------


def test_counts_are_traced_to_their_fields():
    segments = annotate("3 entities across 12 transactions", EVIDENCE)
    traced = {segment.text: segment.field for segment in segments if segment.traced}
    assert traced["3"] == "entity_count"
    assert traced["12"] == "transaction_count"


def test_attribute_names_point_at_a_specific_node():
    """The most precise highlight available: one named attribute node in the graph."""
    segments = annotate("all entities share card1 and addr1", EVIDENCE)
    traced = {segment.text: segment.target for segment in segments if segment.traced}
    assert traced["card1"] == "attr:card1"
    assert traced["addr1"] == "attr:addr1"


def test_amounts_are_traced_in_several_renderings():
    """Models write the same amount various ways; each should still trace."""
    for rendering in ("48114.00", "48,114.00"):
        segments = annotate(f"totalling {rendering} INR", EVIDENCE)
        assert any(s.field == "total_amount_inr" for s in segments), rendering


def test_risk_scores_are_traced():
    segments = annotate("a max risk score of 0.7464", EVIDENCE)
    assert any(segment.field == "max_risk_score" for segment in segments)


# --------------------------------------------------------------------------
# the ways naive matching goes wrong
# --------------------------------------------------------------------------


def test_longer_numbers_win_over_shorter_ones():
    """Scanning for 2 before 12 would mark the '1' of twelve as a core number."""
    segments = annotate("12 transactions", EVIDENCE)
    traced = [segment for segment in segments if segment.traced]
    assert len(traced) == 1
    assert traced[0].text == "12"
    assert traced[0].field == "transaction_count"


def test_numbers_embedded_in_larger_numbers_are_not_matched():
    """4 must not match inside 4821, or the highlight would point at nonsense."""
    segments = annotate("across 4821 accounts", EVIDENCE)
    assert not any(segment.traced for segment in segments)


def test_numbers_inside_words_are_not_matched():
    segments = annotate("reference card12345 was seen", EVIDENCE)
    assert not any(segment.traced for segment in segments)


def test_small_integers_are_not_traced():
    """1 and 2 appear constantly in ordinary prose; marking them would mean nothing."""
    segments = annotate("1 address and 2 hops away", EVIDENCE)
    assert not any(segment.traced for segment in segments)


def test_indirect_attribute_chain_is_not_treated_as_a_node():
    """It is a placeholder meaning 'no single shared attribute', not a graph node."""
    evidence = {**EVIDENCE, "shared_attributes": ["indirect_attribute_chain"]}
    segments = annotate("linked by an indirect_attribute_chain", evidence)
    assert not any(segment.traced for segment in segments)


# --------------------------------------------------------------------------
# safety and serialisation
# --------------------------------------------------------------------------


def test_markup_in_model_output_is_never_interpreted():
    """Prose is language-model output. It must survive as text, tags and all.

    The annotator returns plain segments and the template escapes them, so a model that
    emitted a script tag would render it visibly rather than execute it.
    """
    hostile = "<script>alert('x')</script> and 12 transactions"
    segments = annotate(hostile, EVIDENCE)
    assert rebuild(segments) == hostile
    assert any("<script>" in segment.text for segment in segments)


def test_annotate_to_dicts_is_json_shaped():
    payload = annotate_to_dicts("3 entities share card1", EVIDENCE)
    assert all(set(item) == {"text", "field", "target"} for item in payload)
    assert "".join(item["text"] for item in payload) == "3 entities share card1"


def test_annotation_is_deterministic():
    text = "3 entities, 12 transactions, card1 shared, 48114.00 INR"
    assert annotate(text, EVIDENCE) == annotate(text, EVIDENCE)


def test_missing_evidence_fields_are_tolerated():
    """Clusters vary; absent fields must not raise."""
    segments = annotate("3 entities", {"entity_count": 3})
    assert any(segment.traced for segment in segments)
