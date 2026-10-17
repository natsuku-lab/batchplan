#!/usr/bin/env bash
# Downloads the Franka Panda's visual meshes (franka_ros 0.7.0, Apache-2.0,
# https://github.com/frankaemika/franka_ros), which `assets/franka/franka_panda.urdf` names but the
# repository leaves out, and verifies every file's checksum. The Rerun demos draw them; without
# them they draw the collision meshes.
#
#   scripts/fetch_panda_meshes.sh [dir=data/franka_visual]
set -euo pipefail
REPO=https://raw.githubusercontent.com/frankaemika/franka_ros/0.7.0
DIR="${1:-data/franka_visual}"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

fetch() {
  local sum="$1" url="$2" path="$3"
  if [[ -f "$path" && "$(sha256 "$path")" == "$sum" ]]; then
    return
  fi
  mkdir -p "$(dirname "$path")"
  curl -fsSL -o "$path.part" "$url"
  local got
  got="$(sha256 "$path.part")"
  if [[ "$got" != "$sum" ]]; then
    rm -f "$path.part"
    echo "checksum mismatch for $url: expected $sum, got $got" >&2
    exit 1
  fi
  mv "$path.part" "$path"
}

fetch a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9 "$REPO/LICENSE" "$DIR/LICENSE"
while read -r sum file; do
  fetch "$sum" "$REPO/franka_description/meshes/visual/$file" "$DIR/$file"
done <<SUMS
caefea762f2d18ca9412cf6c2e64e5007ad03571181d790a8e0c828b1b2035cf finger.dae
fe5d445d509e44a9bd107ff78f0b03c49752c98e5e0a8ebafddf2b6cf5a8b380 hand.dae
050fe608136d11913287eaea6906538ddc9112cdbf7337a3181d9ed03f55f619 link0.dae
b061af1bcfb57f32719a1ffda44290619549a41b33a37a49bfebe781b20039a4 link1.dae
01ec435e7aee5812eb401489a83bf1717834eb5f21f8f6233e553b2da9be00bb link2.dae
ce28221fe968ee86787d594222b24d00aa0f47bb64f53eaacbaf6172e81c29b4 link3.dae
d3a4029d1396471ad78ea5fd06e440167cb46c3ed7ab57befb2bcdbf1360efbf link4.dae
07bab2ced87c546bfbc15ce8249038db0e0e582de1607b4243ebd3669dac1f21 link5.dae
4475cb620e2e7293eb0f317b5f968f6520f703a4e5d4f7b95f75249e4302fd21 link6.dae
463227bda9eb9901637107068f59e861fcb81ac89880706d41c26d18c4b46abe link7.dae
SUMS
echo "Franka Panda visual meshes in $DIR (Apache-2.0, see $DIR/LICENSE)"
