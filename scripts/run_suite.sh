#!/usr/bin/env bash
# The whole comparison on one split (`scripts/run_comparison.sh` for each part), run on an idle host:
# the frozen baseline on the held-out problems, the published run on the final ones. Each part writes
# <out>/<part>/<config>/round<k>.
#
#   scripts/run_suite.sh <out> [split=heldout]
set -euo pipefail
out=$1 split=${2:-heldout}
cd "$(dirname "$0")/.."
run=scripts/run_comparison.sh
seeds=1,2,3,4,5

# 1. Independent problems on the exact scenes: batchplan, RoboPlan and MoveIt, geometric paths (A) and
#    complete trajectories (B), each library's defaults and matched edge checks; 16 cores, 5 seeds,
#    3 rounds in rotating order.
$run "$out/exact" "$split" sixteen 3 $seeds \
  batchplan-A roboplan-A moveit-A batchplan-A-matched roboplan-A-matched moveit-A-matched \
  batchplan-B roboplan-B-Hermite roboplan-B-Cubic roboplan-B-Adaptive roboplan-B-LinearBlend moveit-B-totg moveit-B-ruckig

# 2. The capsule scenes VAMP and OMPL take: batchplan, VAMP and OMPL, geometric paths.
SCENE=capsules $run "$out/capsules" "$split" sixteen 3 $seeds \
  batchplan-A vamp-A ompl-A batchplan-A-matched vamp-A-matched ompl-A-matched

# 3. Repeated queries in 20 scenes (scene setup paid once per scene per worker).
SET=onescene $run "$out/onescene" "$split" sixteen 3 $seeds batchplan-A roboplan-A moveit-A

# 4. One query at a time on one core: single-query latency without contention, one seed.
$run "$out/one-core" "$split" one 1 1 batchplan-A roboplan-A moveit-A batchplan-B roboplan-B-Adaptive moveit-B-totg
SCENE=capsules $run "$out/one-core-capsules" "$split" one 1 1 batchplan-A vamp-A ompl-A batchplan-A-matched vamp-A-matched
