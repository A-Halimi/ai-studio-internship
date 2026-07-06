"""Generate the student notebooks from the solutions masters.

Solutions notebooks (solutions/dayNN/*_SOLUTIONS.ipynb) are the single source
of truth. This script:

  1. Replaces every  ### BEGIN SOLUTION ... ### END SOLUTION  region with a
     "YOUR CODE HERE" placeholder (keeping any  ### HINT: ...  lines).
  2. Drops whole cells whose first line is  ### SOLUTION CELL  .
  3. Strips all outputs and execution counts.
  4. Writes the result to notebooks/dayNN/<name>.ipynb (SOLUTIONS suffix removed).
  5. LEAK CHECK: fails if any solution marker survives in the output.
  6. Sanity check: every generated notebook must contain at least one
     exercise placeholder (notebooks with "demo" in the name are exempt).

Never edit notebooks/ by hand - rerun this script instead:
    python scripts/make_student_notebooks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS_DIR = ROOT / "solutions"
STUDENT_DIR = ROOT / "notebooks"

BEGIN = "### BEGIN SOLUTION"
END = "### END SOLUTION"
CELL_MARK = "### SOLUTION CELL"
HINT = "### HINT"
PLACEHOLDER = "# ==================== YOUR CODE HERE ===================="


def transform_code(src: str) -> tuple[str, int]:
    """Replace solution regions in one code cell. Returns (new_src, n_regions)."""
    lines = src.splitlines()
    out: list[str] = []
    i = 0
    n = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(BEGIN):
            n += 1
            indent = line[: len(line) - len(line.lstrip())]
            hints: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(END):
                if lines[i].strip().startswith(HINT):
                    hints.append(indent + lines[i].strip())
                i += 1
            if i >= len(lines):
                raise ValueError(f"{BEGIN} without matching {END}")
            i += 1  # skip the END line
            out.append(indent + PLACEHOLDER)
            out.extend(hints)
            out.append(indent + "...")
        else:
            out.append(line)
            i += 1
    return "\n".join(out), n


def process_notebook(src_path: Path, dst_path: Path) -> tuple[int, int]:
    nb = json.loads(src_path.read_text(encoding="utf-8"))
    kept_cells = []
    n_regions = 0
    n_dropped = 0
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(CELL_MARK):
            n_dropped += 1
            continue
        if cell.get("cell_type") == "code":
            new_src, n = transform_code(source)
            n_regions += n
            cell["source"] = new_src.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
        kept_cells.append(cell)
    nb["cells"] = kept_cells
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return n_regions, n_dropped


def main() -> int:
    masters = sorted(SOLUTIONS_DIR.glob("day*/*.ipynb"))
    if not masters:
        print("No solutions notebooks found - nothing to do.")
        return 1

    failures: list[str] = []
    for src in masters:
        rel_day = src.parent.name
        student_name = src.name.replace("_SOLUTIONS", "")
        dst = STUDENT_DIR / rel_day / student_name
        n_regions, n_dropped = process_notebook(src, dst)

        # Leak check
        text = dst.read_text(encoding="utf-8")
        if BEGIN in text or END in text or CELL_MARK in text:
            failures.append(f"LEAK: solution marker survived in {dst}")

        exempt = "demo" in student_name.lower()
        if n_regions == 0 and not exempt:
            failures.append(f"NO EXERCISES: {src.name} has no solution regions")

        print(f"OK  {rel_day}/{student_name}  ({n_regions} exercises, "
              f"{n_dropped} instructor-only cells removed)")

    if failures:
        print("\n".join(["", "BUILD FAILED:"] + failures))
        return 1
    print(f"\nGenerated {len(masters)} student notebooks in {STUDENT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
