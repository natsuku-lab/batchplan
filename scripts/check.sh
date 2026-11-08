#!/usr/bin/env bash
# Every check CI's lint, demos and python jobs run, on the library, the demos crate and the Python
# bindings: rustfmt with the style guide's options (stable, then the ones only nightly has), clippy
# with each feature set, the docs, and ruff on the Python, all with warnings denied. Run it before
# every push.
#
#   scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# As CI sets them; the checks build in a directory of their own, so the flags leave the usual
# build alone.
export RUSTFLAGS="-D warnings" RUSTDOCFLAGS="-D warnings"
export CARGO_TARGET_DIR="${CHECK_TARGET_DIR:-$PWD/target/check}"
nightly=(--config imports_granularity=Module,hex_literal_case=Lower,format_macro_matchers=true)
for crate in . demos python; do
  (cd "$crate" && cargo fmt --check && cargo +nightly fmt --check -- "${nightly[@]}")
done
cargo clippy --all-targets
cargo clippy --all-targets --features lerobot,usd
cargo clippy --all-targets --no-default-features
cargo doc --no-deps --features lerobot,usd
(cd demos && CARGO_TARGET_DIR="$CARGO_TARGET_DIR/demos" cargo clippy --release --all-targets)
(cd python && CARGO_TARGET_DIR="$CARGO_TARGET_DIR/python" cargo clippy --all-targets)
(cd python && uv run --only-group dev ruff check . && uv run --only-group dev ruff format --check .)
# rustfmt leaves the bodies of the bindings' `value_class!` alone, so their lines are checked here.
if long=$(grep -n '.\{101,\}' python/src/*.rs); then
  echo "python/src: lines over 100 columns:"; echo "$long"; exit 1
fi
echo "every check passes"
