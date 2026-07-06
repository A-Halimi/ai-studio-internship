#!/usr/bin/env bash
# Execute every SOLUTIONS notebook headlessly inside the course container.
#
#   docker run --rm --gpus all --ipc=host -e SMOKE_TEST=1 \
#     -v "E:\High_school_project_v3:/workspace" ai-studio:v3 \
#     bash /workspace/scripts/smoke_test.sh
#
# SMOKE_TEST=1 makes every notebook's CONFIG cell shrink its workload
# (tiny subsets, 1 epoch) so the whole suite finishes in minutes.
# Run without -e SMOKE_TEST=1 for a full-fidelity overnight pass.
set -uo pipefail

cd /workspace
export SMOKE_TEST="${SMOKE_TEST:-1}"
export MPLBACKEND=Agg
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

OUT_DIR=.smoke_out
mkdir -p "$OUT_DIR"

pass=0
fail=0
failed_list=()

for nb in solutions/day*/*.ipynb; do
  echo ""
  echo "============================================================"
  echo "  Executing: $nb   (SMOKE_TEST=$SMOKE_TEST)"
  echo "============================================================"
  if jupyter nbconvert --to notebook --execute "$nb" \
      --output-dir "$OUT_DIR" \
      --ExecutePreprocessor.timeout=1800; then
    echo "--- OK: $nb"
    pass=$((pass + 1))
  else
    echo "--- FAILED: $nb"
    fail=$((fail + 1))
    failed_list+=("$nb")
  fi
done

echo ""
echo "============================================================"
echo "  SMOKE TEST SUMMARY: $pass passed, $fail failed"
if [ "$fail" -gt 0 ]; then
  printf '  FAILED: %s\n' "${failed_list[@]}"
fi
echo "============================================================"
exit "$fail"
