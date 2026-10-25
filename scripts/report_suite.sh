#!/usr/bin/env bash
# Every table of the comparison in docs/benchmarks.md, regenerated from a suite's saved runs
# (`scripts/run_suite.sh`, then `compare mesh` on each configuration's first round) into
# <out>/report_<table>.txt. Each table compares with its first configuration.
#
#   scripts/report_suite.sh <out>   # e.g. data/bench/final
set -euo pipefail
out=$1
cd "$(dirname "$0")/.."
py=.venv-roboplan/bin/python
table() {  # name part config...
  local name=$1 part=$2
  shift 2
  $py scripts/report.py "${@/#/$out/$part/}" --per-set > "$out/report_$name.txt"
}
table paths_exact exact batchplan-A roboplan-A moveit-A
table paths_exact_matched exact batchplan-A-matched roboplan-A-matched moveit-A-matched
table paths_capsules capsules batchplan-A vamp-A ompl-A
table paths_capsules_matched capsules batchplan-A-matched vamp-A-matched ompl-A-matched
table trajectories exact batchplan-B roboplan-B-Hermite roboplan-B-Cubic roboplan-B-Adaptive \
  roboplan-B-LinearBlend moveit-B-totg moveit-B-ruckig
table one_core_paths one-core batchplan-A roboplan-A moveit-A
table one_core_capsules one-core-capsules batchplan-A vamp-A ompl-A
table one_core_capsules_matched one-core-capsules batchplan-A-matched vamp-A-matched
table one_core_trajectories one-core batchplan-B roboplan-B-Adaptive moveit-B-totg
table one_scene onescene batchplan-A roboplan-A moveit-A
ls "$out"/report_*.txt
