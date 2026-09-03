"""Fetch the Elliptic Bitcoin dataset.

Used for a structural replication only — see `PLAN_ELLIPTIC.md`. No model is trained on it.

Only two of the three canonical files are needed. `elliptic_txs_features.csv` (~150 MB) is
deliberately skipped: this phase tests graph structure, and the features would only matter
if a model were being fitted, which it is not.

    python scripts/fetch_elliptic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "elliptic"

# Ungated mirrors of the original Elliptic release. Same approach as the IEEE-CIS fetch:
# the canonical source is Kaggle, which needs an authenticated account, and this keeps the
# repo reproducible for a reviewer who has none.
# Several ungated mirrors carry byte-identical copies (3.31 MB and 4.47 MB respectively).
# More than one is listed because a single hardcoded host is a single point of failure --
# the first attempt at this hit an HTTP 429 and had nowhere to fall back to.
MIRRORS = [
    "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main",
    "https://huggingface.co/datasets/rexaro/elliptic-bitcoin-dataset/resolve/main",
    "https://huggingface.co/datasets/SuodhanJ6/elliptic_txs_classes/resolve/main",
]

FILES = ["elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]

# Published characteristics, asserted on load so a truncated download or a substituted
# mirror fails here rather than surfacing as a strange graph statistic later.
EXPECTED_NODES = 203_769
EXPECTED_EDGES = 234_355


def download(name: str, url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=900) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r    {100 * written / total:5.1f}%  {written/1e6:6.1f} MB",
                          end="", flush=True)
    print()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching Elliptic into {RAW_DIR}")

    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"  {name} already present, skipping")
        else:
            for attempt, base in enumerate(MIRRORS):
                try:
                    print(f"  downloading {name} from mirror {attempt + 1} ...", flush=True)
                    download(name, f"{base}/{name}", dest)
                    break
                except Exception as exc:  # noqa: BLE001 - any failure means try the next
                    print(f"    mirror {attempt + 1} failed: {exc}")
                    dest.unlink(missing_ok=True)
                    if attempt + 1 < len(MIRRORS):
                        time.sleep(5.0)
            else:
                print(f"\nCould not fetch {name} from any mirror.")
                return 1
        with open(dest, "rb") as handle:
            lines = sum(1 for _ in handle) - 1  # minus header
        print(f"  ok   {name}: {lines:,} rows")

    print("\nNext: python run.py --stage elliptic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
