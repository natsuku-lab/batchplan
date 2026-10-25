#!/usr/bin/env bash
# Builds `compare_cpp` (scripts/compare_cpp.cc) inside the MoveIt Jazzy image, which has Eigen and OMPL
# (1.7.0, whose RRTConnect and simplification match 2.0's), against VAMP 0.6.4 and the nigh and pdqsort
# commits VAMP 0.6.4 pins. The binary runs in the same image (`scripts/run_comparison.sh`).
#
#   scripts/build_compare_cpp.sh   # -> data/bench/cpp/compare_cpp
set -euo pipefail
cd "$(dirname "$0")/.."
src=data/bench/cpp/src
mkdir -p "$src"
fetch() {  # repository commit directory
  if [[ ! -d $src/$3 ]]; then
    git clone -q "https://github.com/$1" "$src/$3"
  fi
  git -C "$src/$3" checkout -q "$2"
}
fetch KavrakiLab/vamp v0.6.4 vamp
fetch kavrakilab/nigh 97130999440647c204e0265d05a997dbd8da4e70 nigh
fetch orlp/pdqsort b1ef26a55cdb60d236a5cb199c4234c704f46726 pdqsort
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/batchplan" -w /batchplan moveit/moveit2:jazzy-release bash -c '
  ompl=$(dirname "$(find /opt/ros/jazzy/include -maxdepth 2 -type d -name ompl | head -1)")
  g++ -std=c++17 -O3 -march=native -ffp-contract=fast -fno-math-errno -fno-signed-zeros -fno-trapping-math \
    -fno-rounding-math -I data/bench/cpp/src/vamp/src/impl -I /usr/include/eigen3 \
    -I data/bench/cpp/src/nigh/src -I data/bench/cpp/src/pdqsort -I "$ompl" \
    scripts/compare_cpp.cc -L /opt/ros/jazzy/lib/x86_64-linux-gnu -Wl,-rpath,/opt/ros/jazzy/lib/x86_64-linux-gnu -lompl -lpthread \
    -o data/bench/cpp/compare_cpp
  echo "built against OMPL in $ompl"'
