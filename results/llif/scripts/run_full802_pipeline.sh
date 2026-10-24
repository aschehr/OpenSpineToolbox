#!/bin/bash
# Full-dataset LLIF pipeline: Phase 1 (label-only report, all 802 cases,
# ~24.5h estimated at 3 workers) -> cohort_stats -> Phase 2 (CT-streaming
# HU/BMD, all 802 cases). Each phase script is checkpointed (resumes from
# its own output file), and this wrapper restarts a phase's script if it
# exits non-zero (memory-pressure kills observed earlier this project on
# long jobs) -- it stops restarting once the script itself reports
# "nothing to do".
set -uo pipefail
cd /Users/ashleyschehr/OpenSpineToolbox
LOG=results/llif/scripts/full802_pipeline.log

echo "=== PHASE 1: label-only report, full 802 ===" | tee -a "$LOG"
while true; do
  python3 results/llif/scripts/run_llif_report_full802.py 2>&1 | tee -a "$LOG"
  if tail -50 "$LOG" | grep -q "nothing to do\|DONE ->.*llif_report_full802"; then
    break
  fi
  echo "[pipeline] phase 1 script exited without finishing -- restarting in 10s" | tee -a "$LOG"
  sleep 10
done

echo "=== Building cohort_stats from full 802 ===" | tee -a "$LOG"
python3 results/llif/scripts/build_cohort_stats_full802.py 2>&1 | tee -a "$LOG"

echo "=== PHASE 2: CT-streaming HU/BMD, full 802 ===" | tee -a "$LOG"
while true; do
  python3 results/llif/scripts/run_vertebral_hu_full802_streaming.py 2>&1 | tee -a "$LOG"
  if tail -50 "$LOG" | grep -q "nothing to do\|DONE ->.*vertebral_hu_full802"; then
    break
  fi
  if tail -5 "$LOG" | grep -q "^ABORT:"; then
    echo "[pipeline] phase 2 aborted on low disk -- NOT auto-restarting, needs a look" | tee -a "$LOG"
    exit 1
  fi
  echo "[pipeline] phase 2 script exited without finishing -- restarting in 10s" | tee -a "$LOG"
  sleep 10
done

echo "=== PIPELINE COMPLETE ===" | tee -a "$LOG"
