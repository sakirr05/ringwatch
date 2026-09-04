"""Tests for the hub-cap parameter and the coverage-sweep entry point.

Context, because the tests below look small for what they protect. The README publishes a
coverage sweep at hub caps 5 / 20 / 50 — the table answering the first objection anyone
raises about the graph layer. Until the Phase 11 audit those numbers had **no reproducible
path**: they were produced by editing `MAX_GROUP_SIZE` by hand. Every other figure in this
project regenerates from `run.py` or `scripts/export_results.py`; that one did not, which
made it the single published claim a reviewer had to take on trust.

The fix threads the cap through `build_features_for_split` as a parameter. The risk that
introduces is the one `test_the_default_is_unchanged` exists to catch: if the default ever
drifts from `MAX_GROUP_SIZE`, every published graph number silently changes.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.graph import (
    MAX_GROUP_SIZE,
    UID_COL,
    build_features_for_split,
    build_graph,
    entity_frame,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP = REPO_ROOT / "scripts" / "coverage_sweep.py"


def frame(n_entities: int, hub_members: int) -> pd.DataFrame:
    """`hub_members` entities sharing one (addr1, P_emaildomain); the rest isolated.

    The shared value has to be the whole composite: `LINK_KEYS` joins on
    `addr1_pemail = (addr1, P_emaildomain)`, never on addr1 alone, so entities sharing
    only an address are correctly not linked at all.
    """
    rows = []
    for i in range(n_entities):
        in_hub = i < hub_members
        rows.append(
            {
                UID_COL: f"u{i}",
                # card1/card3/card5 are in LINK_KEYS too, so entity_frame requires them.
                # Kept distinct per entity so the hub below is the only link.
                "card1": 1000 + i,
                "card3": 150 + i,
                "card5": 220 + i,
                "addr1": 500 if in_hub else 900 + i,
                "P_emaildomain": "hub.com" if in_hub else f"d{i}.com",
                "TransactionDT": i * 3600,
                "TransactionAmt": 100.0,
                "isFraud": 0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the default must not drift
# --------------------------------------------------------------------------


def test_the_default_is_unchanged():
    """If this drifts, every published graph figure silently changes."""
    default = inspect.signature(build_features_for_split).parameters["max_group_size"].default
    assert default == MAX_GROUP_SIZE == 5


def test_omitting_the_cap_matches_passing_the_shipped_one():
    """The parameter must be a pure widening: same call, same answer."""
    df = frame(12, hub_members=4)
    train, test = df.iloc[:8], df.iloc[8:]

    a_train, a_test, a_summary = build_features_for_split(train, test)
    b_train, b_test, b_summary = build_features_for_split(
        train, test, max_group_size=MAX_GROUP_SIZE
    )
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)
    assert a_summary == b_summary


# --------------------------------------------------------------------------
# the cap actually does something
# --------------------------------------------------------------------------


def test_a_higher_cap_admits_a_group_a_lower_cap_suppressed():
    """A 6-member hub is dropped at cap 5 and kept at cap 20 — the sweep's whole premise."""
    df = frame(10, hub_members=6)
    entities = entity_frame(df)

    suppressed = build_graph(entities, max_group_size=5)
    admitted = build_graph(entities, max_group_size=20)
    assert suppressed.n_edges < admitted.n_edges


def test_raising_the_cap_never_reduces_coverage():
    df = frame(30, hub_members=8)
    train, test = df.iloc[:20], df.iloc[20:]

    coverage = []
    for cap in (5, 20, 50):
        _, test_g, _ = build_features_for_split(train, test, max_group_size=cap)
        coverage.append(float((test_g["g_is_linked"].fillna(0) == 1).mean()))
    assert coverage == sorted(coverage)


def test_the_cap_reaches_both_graphs_not_just_one():
    """Train and test graphs are built separately; a cap applied to only one would leak
    an inconsistency into the features rather than raising."""
    source = inspect.getsource(build_features_for_split)
    assert source.count("max_group_size=max_group_size") == 2


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def test_the_sweep_script_exists_and_parses():
    assert SWEEP.exists(), "the README's coverage table needs a reproducible entry point"
    ast.parse(SWEEP.read_text())


def test_the_sweep_defaults_to_the_published_caps():
    tree = ast.parse(SWEEP.read_text())
    caps = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "DEFAULT_CAPS"
    )
    assert tuple(caps) == (5, 20, 50), "the defaults must be the caps the README publishes"


def test_the_sweep_reuses_the_shipped_scores_for_the_shipped_cap():
    """Retraining cap 5 would produce a second model for the same configuration, and any
    disagreement would put the table in conflict with the ablation above it."""
    source = SWEEP.read_text()
    assert "scores_graph_full.npy" in source
    assert "cap == MAX_GROUP_SIZE" in source


def test_the_sweep_computes_nothing_of_its_own():
    """It is a driver, not a new experiment: same training, evaluation and bootstrap."""
    tree = ast.parse(SWEEP.read_text())
    imported = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    for required in ("core.model", "core.evaluate", "core.graph"):
        assert required in imported
    source = SWEEP.read_text()
    for hand_rolled in ("def train", "def evaluate", "def bootstrap", "average_precision"):
        assert hand_rolled not in source
