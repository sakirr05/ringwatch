"""Load the exported results artifact.

This module is the app's entire relationship with the analysis. It reads a JSON file and
returns a dict. It imports nothing from `core/` and computes nothing — which is what makes
the claim "the dashboard never produces a number" structurally true rather than a promise,
and is asserted by `tests/test_app.py::test_app_layer_computes_nothing`.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "docs" / "results.json"


class ResultsUnavailable(RuntimeError):
    """The export artifact is missing. The dashboard cannot render without it."""


def load_results(path: Path | None = None) -> dict:
    """Read docs/results.json.

    Deliberately not cached in a module global: on Render the file is baked into the image
    at deploy time, and re-reading it per request is far cheaper than reasoning about
    staleness. That was re-checked once the artifact grew to 112 KB (this docstring used to
    say 24 KB, which was four phases out of date): a full read and parse costs **0.34 ms**,
    against 5.2 ms to render the dashboard it feeds. Still not worth caching.

    `app/main.py` does keep the two-field `meta` block in memory, and for a different
    reason — a liveness probe must not depend on the filesystem. See `results_metadata`
    there.
    """
    path = path or RESULTS_PATH
    if not path.exists():
        raise ResultsUnavailable(
            f"{path} not found. Generate it locally with:\n"
            "    python run.py --stage ablation\n"
            "    python scripts/export_results.py"
        )
    return json.loads(path.read_text())
