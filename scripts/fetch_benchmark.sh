#!/usr/bin/env bash
# Downloads the MotionBenchMaker and MπNets Panda problem sets, which robometrics packages as plain
# YAML (MIT, https://github.com/fishbotics/robometrics), and VAMP's Panda collision spheres and
# disabled collision pairs (Apache-2.0, https://github.com/KavrakiLab/vamp), which
# `benchmark --spheres vamp` plans on. Verifies their checksums.
#
#   scripts/fetch_benchmark.sh [dir=data/robometrics]
set -euo pipefail
ROBOMETRICS=https://raw.githubusercontent.com/fishbotics/robometrics/81e3d1d605de84100d8ab880b43096aba221a48b/robometrics/content/dataset
VAMP=https://raw.githubusercontent.com/KavrakiLab/vamp/8fd768f9974a185a8aa2b4f370e595a8c3ca13cf/resources/panda # v0.6.4
DIR="${1:-data/robometrics}"
mkdir -p "$DIR"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

while read -r sum url file; do
  path="$DIR/$file"
  if [[ -f "$path" && "$(sha256 "$path")" == "$sum" ]]; then
    echo "$path is up to date"
    continue
  fi
  curl -fsSL -o "$path.part" "$url"
  got="$(sha256 "$path.part")"
  if [[ "$got" != "$sum" ]]; then
    rm -f "$path.part"
    echo "checksum mismatch for $file: expected $sum, got $got" >&2
    exit 1
  fi
  mv "$path.part" "$path"
  echo "fetched $path"
done <<SUMS
5165ad4fb55f93c63dbbdda6f25a14f3ceb24ee3305e27ad317f38f1e0d6ed0d $ROBOMETRICS/mb_set.yaml mb_set.yaml
9189186d83e51600a2c768aa7657933aa052150501852771994757052b797bbf $ROBOMETRICS/mpinets_set.yaml mpinets_set.yaml
d7d891b892e805d399d506a5c3f3cfc698ac75e48ec55449bde025dc7403291c $VAMP/panda_spherized.urdf vamp_panda_spherized.urdf
1150719ea9d81139418198a50faea17e155323547d056c4edcb7ecc82fd8d317 $VAMP/panda.srdf vamp_panda.srdf
SUMS
