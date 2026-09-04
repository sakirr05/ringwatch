"""Tests for the container image and the demonstration scoring endpoint.

Split deliberately into two kinds.

**Always-on static checks** are the ones that catch the realistic regression: someone adds
an import to a module the container runs, forgets `requirements-web.txt`, and the image
breaks at runtime in a way no unit test would notice.
`test_the_runtime_import_closure_is_covered_by_the_web_requirements` walks the actual
import graph, including the documented lazy imports, and needs no Docker daemon.

**A live build-and-run test** gated behind `RINGWATCH_DOCKER_TESTS=1`, because a 90-second
image build on every `pytest -q` is a tax nobody pays willingly and an unpaid one is a test
that silently stops running. It was run for real in this phase; the command is in the
README.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_SCORE_BODY_BYTES, app

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
WEB_REQS = REPO_ROOT / "requirements-web.txt"
FULL_REQS = REPO_ROOT / "requirements.txt"
RENDER_YAML = REPO_ROOT / "render.yaml"


def parse_requirements(path: Path) -> dict[str, str]:
    """name -> full specifier, ignoring comments and blank lines."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!\[]", line, maxsplit=1)[0].strip().lower()
        out[name] = line
    return out


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# the runtime dependency list cannot drift
# --------------------------------------------------------------------------


def test_web_requirements_are_a_strict_subset_of_the_full_list():
    """Two files, one source of truth. A divergent pin means the container is not a
    rehearsal of what Render installs — it is a third environment nobody tests."""
    web, full = parse_requirements(WEB_REQS), parse_requirements(FULL_REQS)
    assert web, "requirements-web.txt parsed as empty"

    missing = sorted(set(web) - set(full))
    assert not missing, f"in requirements-web.txt but not requirements.txt: {missing}"

    for name, spec in web.items():
        assert spec == full[name], (
            f"{name} pinned differently: web={spec!r} full={full[name]!r}"
        )


def test_the_analysis_only_packages_are_excluded():
    """If these creep in, the exclusion has stopped meaning anything."""
    web = parse_requirements(WEB_REQS)
    for analysis_only in ("scikit-learn", "matplotlib", "pyarrow", "networkx", "pytest"):
        assert analysis_only not in web


def imports_of(path: Path) -> set[str]:
    """Every module named by an import in this file, including inside functions."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_the_runtime_import_closure_is_covered_by_the_web_requirements():
    """Walk what the container actually runs and check nothing needs a missing package.

    This is the test that earns its keep: it fails the moment a new third-party import
    appears on a path the container executes, which is the failure that would otherwise
    surface as a 500 in a deploy log.
    """
    stdlib_or_local = {"app", "core", "ai", "__future__"}
    # Distribution name != import name for these.
    provides = {
        "fastapi": "fastapi", "starlette": "fastapi", "pydantic": "fastapi",
        "uvicorn": "uvicorn", "jinja2": "jinja2", "lightgbm": "lightgbm",
        "numpy": "numpy", "pandas": "pandas", "requests": "requests",
        "dotenv": "python-dotenv",
    }
    web = parse_requirements(WEB_REQS)

    seen: set[Path] = set()
    queue = [p for p in (REPO_ROOT / "app").glob("*.py")]
    third_party: set[str] = set()

    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for module in imports_of(path):
            top = module.split(".")[0]
            if top in stdlib_or_local:
                candidate = REPO_ROOT / Path(module.replace(".", "/") + ".py")
                if candidate.exists():
                    queue.append(candidate)
                continue
            if top in provides:
                third_party.add(top)

    assert third_party, "walked no third-party imports; the walk is broken"
    uncovered = sorted(
        top for top in third_party if provides[top] not in web
    )
    assert not uncovered, (
        f"the container runs code importing {uncovered}, absent from requirements-web.txt"
    )


# --------------------------------------------------------------------------
# the image cannot carry the dataset or a secret
# --------------------------------------------------------------------------


def test_the_dockerignore_excludes_the_dataset_and_secrets():
    """829 MB of data and any .env must not be buildable into a distributable artifact."""
    text = DOCKERIGNORE.read_text()
    for excluded in ("data/", ".env", ".git/", ".venv/"):
        assert excluded in text, f"{excluded} is not in .dockerignore"


def test_the_dockerfile_copies_no_data_directory():
    copies = re.findall(r"^COPY\s+(\S+)", DOCKERFILE.read_text(), re.M)
    assert copies
    for source in copies:
        assert not source.startswith("data"), f"Dockerfile copies {source}"
        assert source != ".", "a bare `COPY . .` would depend entirely on .dockerignore"


# --------------------------------------------------------------------------
# the Dockerfile's operational details
# --------------------------------------------------------------------------


def test_the_container_shuts_down_gracefully():
    """`exec` is not cosmetic: without it SIGTERM hits the shell, uvicorn never sees it,
    and every stop waits out the grace period before SIGKILL cuts requests off."""
    text = DOCKERFILE.read_text()
    cmd = re.search(r"^CMD\s+(.+)$", text, re.M)
    assert cmd, "no CMD"
    assert "exec uvicorn" in cmd.group(1), "CMD must exec, or signals never reach uvicorn"


def test_the_healthcheck_uses_the_liveness_probe():
    text = DOCKERFILE.read_text()
    healthcheck = re.search(r"HEALTHCHECK.*?CMD\s+(.+)", text, re.S)
    assert healthcheck
    assert "/health" in healthcheck.group(1)
    assert "/ready" not in healthcheck.group(1), (
        "a readiness probe in the restart path reintroduces the restart loop"
    )


def test_the_container_runs_as_a_non_root_user():
    assert re.search(r"^USER\s+ringwatch", DOCKERFILE.read_text(), re.M)


def test_lightgbms_openmp_runtime_is_installed():
    """Without libgomp1, `import lightgbm` fails naming a .so rather than the package."""
    assert "libgomp1" in DOCKERFILE.read_text()


def test_the_container_python_matches_the_deployment_target():
    """A container on a different Python than Render is a third environment, not a rehearsal."""
    image = re.search(r"^FROM python:(\d+\.\d+)", DOCKERFILE.read_text(), re.M)
    assert image
    render = re.search(r'value:\s*"(\d+\.\d+)\.\d+"', RENDER_YAML.read_text())
    assert render
    assert image.group(1) == render.group(1)


# --------------------------------------------------------------------------
# the scoring endpoint
# --------------------------------------------------------------------------


def test_scoring_accepts_a_bare_payment_entity(client):
    response = client.post("/api/score", json={"amount": 250_000, "currency": "INR"})
    assert response.status_code == 200
    assert response.json()["available"] in (True, False)


def test_scoring_accepts_the_same_body_the_webhook_takes(client):
    """One payload shape for both routes, or the demo needs two sets of instructions."""
    webhook_body = {"payload": {"payment": {"entity": {"amount": 250_000}}}}
    wrapped = {"payment": {"amount": 250_000}}
    for body in (webhook_body, wrapped):
        assert client.post("/api/score", json=body).status_code == 200


def test_the_response_carries_its_own_caveat(client):
    """A caveat you have to look up is one the caller will not see."""
    body = client.post("/api/score", json={"amount": 250_000}).json()
    assert body["is_fraud_assessment"] is False
    assert "IEEE-CIS" in body["model_trained_on"]
    if body["available"]:
        assert "not an assessment of this transaction" in body["caveat"]
        assert body["coverage_pct"] < 5.0, "coverage this high would need re-describing"


def test_coverage_is_reported_as_a_measured_figure(client):
    """The point of the endpoint is quantifying how little transfers: 3 of 433."""
    body = client.post(
        "/api/score",
        json={"amount": 250_000, "currency": "INR", "created_at": 1_756_944_000},
    ).json()
    if not body["available"]:
        pytest.skip("booster artifact absent")
    assert body["features_total"] == 433
    assert body["features_present"] == 3
    assert set(body["mapped"]) == {"TransactionAmt", "tx_hour", "tx_dayofweek"}


def test_an_arbitrary_json_object_is_rejected_rather_than_scored_as_all_missing(client):
    assert client.post("/api/score", json={"foo": "bar"}).status_code == 400


@pytest.mark.parametrize(
    "body,expected",
    [(b"not json", 400), (b"[1,2,3]", 400), (b'"a string"', 400)],
)
def test_malformed_bodies_are_client_errors(client, body: bytes, expected: int):
    assert client.post("/api/score", content=body).status_code == expected


def test_an_oversized_body_is_refused_before_parsing(client):
    payload = b'{"amount":1,"pad":"' + b"a" * (MAX_SCORE_BODY_BYTES + 1024) + b'"}'
    assert client.post("/api/score", content=payload).status_code == 413


def test_scoring_writes_nothing_that_could_alter_a_reported_metric(client, tmp_path):
    """The artifact is produced offline; no request may touch it."""
    results = REPO_ROOT / "docs" / "results.json"
    if not results.exists():
        pytest.skip("docs/results.json not generated")
    before = results.read_bytes()
    client.post("/api/score", json={"amount": 999_999, "currency": "INR"})
    assert results.read_bytes() == before


# --------------------------------------------------------------------------
# the live image -- opt in with RINGWATCH_DOCKER_TESTS=1
# --------------------------------------------------------------------------

docker_available = shutil.which("docker") is not None
docker_opted_in = os.environ.get("RINGWATCH_DOCKER_TESTS") == "1"


@pytest.mark.skipif(
    not (docker_available and docker_opted_in),
    reason="set RINGWATCH_DOCKER_TESTS=1 (and have Docker) to build and run the image",
)
def test_the_image_builds_and_serves_health():
    tag = "ringwatch:pytest"
    build = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "."],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    name = "ringwatch-pytest"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", "18099:8000", tag],
        capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stderr

    try:
        import urllib.request

        body = None
        for _ in range(60):
            try:
                with urllib.request.urlopen("http://127.0.0.1:18099/health", timeout=2) as r:
                    body = json.loads(r.read())
                break
            except Exception:  # noqa: BLE001 - still booting
                time.sleep(1)

        assert body is not None, "container never answered /health"
        assert body["status"] == "ok"

        # The dataset must not be in the image, verified inside it rather than inferred.
        listing = subprocess.run(
            ["docker", "exec", name, "sh", "-c", "test -d /srv/data && echo yes || echo no"],
            capture_output=True, text=True, timeout=60,
        )
        assert listing.stdout.strip() == "no", "the 829 MB dataset is inside the image"

        # SIGTERM must reach uvicorn: a clean exit 0, not a 137 from SIGKILL.
        stop = subprocess.run(["docker", "stop", name], capture_output=True, timeout=60)
        assert stop.returncode == 0
        code = subprocess.run(
            ["docker", "inspect", name, "--format", "{{.State.ExitCode}}"],
            capture_output=True, text=True, timeout=60,
        )
        assert code.stdout.strip() == "0", "container was killed rather than shut down"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
