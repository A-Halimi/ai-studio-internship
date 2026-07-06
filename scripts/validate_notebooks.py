"""Validate every notebook in the repo against the nbformat schema.

    python scripts/validate_notebooks.py

Requires:  pip install nbformat
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.validator import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paths = sorted(
        p
        for folder in ("solutions", "notebooks")
        for p in (ROOT / folder).glob("day*/*.ipynb")
    )
    if not paths:
        print("No notebooks found.")
        return 1

    bad = 0
    for p in paths:
        try:
            nb = nbformat.read(p, as_version=4)
            nbformat.validate(nb)
            print(f"OK    {p.relative_to(ROOT)}")
        except (ValidationError, Exception) as exc:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {p.relative_to(ROOT)}: {exc}")

    print(f"\n{len(paths) - bad}/{len(paths)} notebooks valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
