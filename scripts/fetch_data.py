"""Fetch the IEEE-CIS Fraud Detection training data.

The canonical source is Kaggle, which requires an authenticated account that has accepted
the competition rules. To keep this repo reproducible for anyone without Kaggle
credentials, this script pulls the identical raw CSVs from an ungated HuggingFace mirror
and then verifies them against the dataset's published characteristics, so a wrong or
truncated file fails here rather than silently degrading a metric later.

Usage:
    python scripts/fetch_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

MIRROR = "https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main"

# (filename, expected line count including header)
FILES = [
    ("train_transaction.csv", 590_541),
    ("train_identity.csv", 144_234),
]


def download(name: str, dest: Path) -> None:
    url = f"{MIRROR}/{name}"
    print(f"  downloading {name} ...", flush=True)
    with requests.get(url, stream=True, timeout=1800) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r    {pct:5.1f}%  {written/1e6:7.1f} MB", end="", flush=True)
    print()


def verify(path: Path, expected_lines: int) -> bool:
    with open(path, "rb") as handle:
        lines = sum(1 for _ in handle)
    if lines != expected_lines:
        print(f"  FAIL {path.name}: expected {expected_lines} lines, got {lines}")
        return False
    print(f"  ok   {path.name}: {lines} lines")
    return True


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching IEEE-CIS training data into {RAW_DIR}")

    ok = True
    for name, expected_lines in FILES:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"  {name} already present, skipping download")
        else:
            download(name, dest)
        ok &= verify(dest, expected_lines)

    if not ok:
        print("\nVerification failed. Delete data/raw/ and retry.")
        return 1

    print("\nAll files verified. Next: python run.py --stage data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
