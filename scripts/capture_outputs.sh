#!/usr/bin/env bash
# =====================================================================
#  The Lab · AI Studio — Intro to AI
#  Author: Dr. Abdelghafour HALIMI  |  https://ahalimi.com
#  Copyright (c) 2026 Dr. Abdelghafour HALIMI. All rights reserved.
# =====================================================================
# Execute the output-less solution notebooks IN PLACE, full fidelity
# (SMOKE_TEST=0), so the masters carry golden outputs for the web build.
#
# Run inside the course image with the repo mounted at /workspace:
#   docker run --rm --gpus all --ipc=host -v "${PWD}:/workspace" -w /workspace \
#     abdelghafour1/ai-studio:latest bash scripts/capture_outputs.sh
#
# day10 runs LAST: its 7/7 module roll-call reads model files that the
# earlier runs (and ai_studio/models/) provide.
# NOTE: do NOT set MPLBACKEND here — it would override ipykernel's inline
# backend and silently drop every matplotlib figure from the saved outputs.
set -e
export SMOKE_TEST=0 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1

NOTEBOOKS=(
  solutions/day02/day02_prediction_machine_SOLUTIONS.ipynb
  solutions/day04/day04_inside_the_brain_SOLUTIONS.ipynb
  solutions/day05/day05_digit_reader_SOLUTIONS.ipynb
  solutions/day06/day06_photo_detective_SOLUTIONS.ipynb
  solutions/day08/day08_language_lab_SOLUTIONS.ipynb
  solutions/day01/day01b_ai_wow_demos_SOLUTIONS.ipynb
  solutions/day10/day10_demo_day_SOLUTIONS.ipynb
)

for nb in "${NOTEBOOKS[@]}"; do
  echo "=== START $nb $(date +%H:%M:%S)"
  jupyter nbconvert --to notebook --execute --inplace "$nb" \
    --ExecutePreprocessor.timeout=1800
  echo "=== DONE  $nb $(date +%H:%M:%S)"
done
echo "ALL_NOTEBOOKS_EXECUTED"
