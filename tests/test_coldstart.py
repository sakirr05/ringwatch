"""Tests for the cold-start work: probe semantics, figure loading, and the keepalive ping.

The load-bearing test here is `test_liveness_survives_a_missing_results_artifact`. The
previous `/health` returned 503 when `docs/results.json` could not be read, and
`render.yaml` points `healthCheckPath` at it — so a missing data file would have been
reported to the platform as "restart me," which cannot fix a missing file and buys a
restart loop instead. Liveness and readiness answer different questions and now have
different endpoints.

Nothing here claims the cold start is solved. It is caused by container scheduling on the
free tier, which no code in this repository can reach; the honest scope is measured in
`test_the_keepalive_workflow_states_what_it_cannot_fix`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "app" / "templates" / "dashboard.html"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "keepalive.yml"
RENDER_YAML = REPO_ROOT / "render.yaml"
RESULTS_PATH = REPO_ROOT / "docs" / "results.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# liveness vs readiness
# --------------------------------------------------------------------------


def test_liveness_survives_a_missing_results_artifact(client, monkeypatch):
    """The regression this phase existed to fix: a data problem must not read as 'restart'."""
    monkeypatch.setattr(main, "_RESULTS_META", None)
    monkeypatch.setattr(
        main, "load_results", lambda *a, **k: (_ for _ in ()).throw(
            main.ResultsUnavailable("gone")
        )
    )

    response = client.get("/health")
    assert response.status_code == 200, "liveness must not fail on a missing data file"
    body = response.json()
    assert body["status"] == "ok"
    assert body["results"] == "unavailable"
    assert "results_commit" not in body


def test_readiness_does_fail_on_a_missing_results_artifact(client, monkeypatch):
    """Readiness is the question whose answer legitimately depends on the filesystem."""
    monkeypatch.setattr(
        main, "load_results", lambda *a, **k: (_ for _ in ()).throw(
            main.ResultsUnavailable("gone")
        )
    )
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_render_health_check_points_at_the_liveness_probe():
    """If this ever moves to /ready, the restart-loop failure mode comes back."""
    config = yaml.safe_load(RENDER_YAML.read_text())
    service = config["services"][0]
    assert service["healthCheckPath"] == "/health"


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_liveness_reads_no_file_per_request(client, monkeypatch):
    """Primed once, then never again — the point is independence from the disk, not speed."""
    client.get("/health")  # prime

    calls = []
    monkeypatch.setattr(main, "load_results", lambda *a, **k: calls.append(1))
    for _ in range(5):
        assert client.get("/health").status_code == 200
    assert calls == [], "liveness touched the filesystem"


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_readiness_reports_what_the_dashboard_would_render(client):
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert body["results_commit"]
    assert body["clusters"] > 0


# --------------------------------------------------------------------------
# figures: reserved space, deferred bytes
# --------------------------------------------------------------------------


def _figure_imgs(html: str) -> list[str]:
    return re.findall(r"<img[^>]*class=\"(?:pr|rel)\"[^>]*>", html)


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_every_figure_reserves_its_space_before_loading(client):
    """No declared height means zero height until decode, then a snap. Both are ~123 KB."""
    imgs = _figure_imgs(client.get("/").text)
    assert imgs, "expected the PR-curve and reliability figures"
    for tag in imgs:
        assert 'width="' in tag and 'height="' in tag, f"no intrinsic size: {tag}"


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_below_the_fold_figures_are_deferred(client):
    """246 KB of plots should not compete with first paint."""
    for tag in _figure_imgs(client.get("/").text):
        assert 'loading="lazy"' in tag
        assert 'decoding="async"' in tag


def test_declared_dimensions_match_the_real_files():
    """A wrong aspect ratio reserves the wrong box and shifts the layout anyway."""
    import struct

    html = TEMPLATE.read_text()
    for cls, name in (("pr", "pr_curve.png"), ("rel", "reliability.png")):
        path = REPO_ROOT / "docs" / name
        if not path.exists():
            pytest.skip(f"{name} not generated")
        width, height = struct.unpack(">II", path.read_bytes()[16:24])
        tag = re.search(rf'<img class="{cls}"[^>]*>', html, re.S)
        assert tag, f"no <img class={cls}>"
        assert f'width="{width}"' in tag.group(), f"{name} is {width}px wide"
        assert f'height="{height}"' in tag.group(), f"{name} is {height}px tall"
        assert f"aspect-ratio: {width} / {height}" in html


def test_the_placeholder_clears_for_an_already_cached_image():
    """`complete` must be checked, or a cached figure shimmers forever under the picture."""
    html = TEMPLATE.read_text()
    assert "img.complete" in html and "naturalWidth" in html
    assert 'img.addEventListener("error"' in html  # a broken image must not shimmer either


def test_the_placeholder_respects_reduced_motion():
    assert "prefers-reduced-motion" in TEMPLATE.read_text()


# --------------------------------------------------------------------------
# the keepalive ping
# --------------------------------------------------------------------------


def test_the_keepalive_workflow_pings_the_cheap_endpoint():
    config = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML parses a bare `on:` key as the boolean True.
    triggers = config.get("on") or config[True]
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers, "should be runnable by hand before a demo"

    run = config["jobs"]["ping"]["steps"][0]
    assert "/health" in run["env"]["URL"], "ping the liveness probe, not the 250 KB page"
    assert run["continue-on-error"] is True, "a transient blip is not a broken build"


def test_the_ping_interval_is_shorter_than_the_spin_down_window():
    """Render idles out at ~15 minutes; a slower ping would not keep anything warm."""
    config = yaml.safe_load(WORKFLOW.read_text())
    triggers = config.get("on") or config[True]
    minutes = sorted(int(m) for m in triggers["schedule"][0]["cron"].split()[0].split(","))
    gaps = [b - a for a, b in zip(minutes, minutes[1:])] + [60 - minutes[-1] + minutes[0]]
    assert max(gaps) <= 15, f"largest gap between pings is {max(gaps)} min"


def test_the_keepalive_workflow_states_what_it_cannot_fix():
    """The caveats are the honest part; a keepalive that promises no cold starts is lying."""
    text = WORKFLOW.read_text().lower()
    assert "best effort" in text, "GitHub cron is not punctual and the file must say so"
    assert "60 days" in text, "scheduled workflows stop on an inactive repo"
    assert "750" in text, "free instance-hours are nearly exhausted by staying warm"


def test_the_readme_still_warns_about_the_cold_start():
    """The ping reduces cold starts; it does not remove the need to warn about them."""
    readme = (REPO_ROOT / "README.md").read_text()
    assert "30–60 seconds" in readme or "30-60s" in readme
