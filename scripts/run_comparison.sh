#!/usr/bin/env bash
# Runs the comparison's configurations round by round, rotating the planners' order each round, and
# judges every run. Each configuration's rounds go to <out>/<config>/round<k>; `scripts/report.py`
# reads them.
#
#   scripts/run_comparison.sh <out> <split> <budget: one|sixteen> <rounds> <seeds> <config>...
#
# Configurations: batchplan-A, batchplan-B, roboplan-A, roboplan-B-<Mode>, moveit-A, moveit-B-totg,
# moveit-B-ruckig, vamp-A, ompl-A, each optionally suffixed -matched (e.g. roboplan-A-matched).
# SCENE=capsules plans every configuration on the capsule scenes (VAMP's and OMPL's need it), and
# SET=onescene the repeated queries in 20 scenes instead of the independent problems.
# Needs .venv-roboplan (roboplan 0.7.0, pyyaml), the moveit/moveit2:jazzy-release image, and for VAMP
# and OMPL the `compare_cpp` binary (scripts/compare_cpp.cc) at data/bench/cpp/compare_cpp.
set -euo pipefail
out=$1 split=$2 budget=$3 rounds=$4 seeds=$5
shift 5
configs=("$@")
cd "$(dirname "$0")/.."
case $budget in
  one) cores=2 workers=1 ;;
  sixteen) cores=0-15 workers=16 ;;
  *) echo "budget: one or sixteen" >&2; exit 2 ;;
esac
py=.venv-roboplan/bin/python
image=moveit/moveit2:jazzy-release
cargo build --release --example compare >/dev/null

run() {  # config round
  local config=$1 round=$2 family=defaults
  [[ $config == *-matched ]] && family=matched && config=${config%-matched}
  local planner=${config%%-*} rest=${config#*-}
  local workload=${rest%%-*} variant=""
  [[ $rest == *-* ]] && variant=${rest#*-}
  local dir="$out/$1/round$round"
  rm -rf "$dir"
  local args=(run data/bench/inputs "$dir" --planner "$planner" --workload "$workload" --family "$family"
              --split "$split" --set "${SET:-problems}" --scene "${SCENE:-exact}" --seeds "$seeds" --reps 1
              --cores "$cores" --workers "$workers")
  echo "$(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null) $1 round $round"
  case $planner in
    moveit)
      [[ $variant == ruckig ]] && args+=(--smoothing ruckig)
      docker run --rm --cpuset-cpus "$cores" --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/batchplan" \
        -w /batchplan "$image" bash -c "source /opt/ros/jazzy/setup.bash && python3 scripts/compare.py ${args[*]}" \
        2>&1 | grep -E "^moveit |Error|Traceback" || true
      ;;
    roboplan)
      [[ -n $variant ]] && args+=(--mode "$variant")
      taskset -c "$cores" $py scripts/compare.py "${args[@]}"
      ;;
    vamp|ompl)
      # compare_cpp links the image's OMPL, so the orchestrator runs in the image too.
      docker run --rm --cpuset-cpus "$cores" --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/batchplan" \
        -w /batchplan "$image" bash -c "python3 scripts/compare.py ${args[*]}"
      ;;
    batchplan)
      $py scripts/compare.py "${args[@]}"
      ;;
  esac
  $py scripts/validate.py run data/bench/inputs "$dir"
}

n=${#configs[@]}
for round in $(seq 1 "$rounds"); do
  for k in $(seq 0 $((n - 1))); do
    run "${configs[$(( (k + round - 1) % n ))]}" "$round"
  done
done
