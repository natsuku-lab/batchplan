# Contributing

Thanks for helping. Issues and pull requests are welcome at
[natsuku-lab/batchplan](https://github.com/natsuku-lab/batchplan).

## Building and testing

```bash
cargo build --release --all-targets --features lerobot,usd
BATCHPLAN_REQUIRE_GPU=1 cargo test --release --features lerobot,usd   # fails rather than skips the GPU tests without an adapter
cargo test --release --no-default-features                            # the CPU-only build
cargo clippy --release --all-targets --features lerobot,usd           # zero warnings; CI denies them
cargo doc --no-deps --features lerobot,usd                            # zero warnings
(cd demos && cargo clippy --release --all-targets)                    # the Rerun demos, a crate of their own
scripts/check.sh                                                      # every lint CI runs, on both crates: run it before pushing
```

CI runs the same on Linux, with the GPU kernels on Mesa's llvmpipe, and builds on macOS, Windows
and aarch64 Linux. [Verification](docs/verification.md) says what each test pins down.

## Style

The code follows [Google's Rust style guide](https://google.github.io/styleguide/rustguide.html).
Format with stable rustfmt, then check the guide's options that only nightly rustfmt has:

```bash
cargo fmt
rustup toolchain install nightly --profile minimal -c rustfmt
cargo +nightly fmt --check -- --config imports_granularity=Module,hex_literal_case=Lower,format_macro_matchers=true
```

Clippy checks what it can of the guide (`[lints]` in `Cargo.toml`, and `src/lib.rs` for the
library). Where the guide leaves a choice to the codebase:

- **Imports.** No glob imports, except `use super::*` in test modules and the rayon and
  fearless_simd preludes. Call free functions through their module (`world::box_distance`,
  `datagen::handover`, `batchplan::plan`), and import types by name, unless the name would clash
  or comes from wgpu or rerun.
- **Modules.** A module with children only declares and re-exports them.
- **API.** Document every public item; mark pure functions `#[must_use]`; name a constructor that
  returns a `Result` `try_new`; return named fields rather than tuples; spell names out (the
  field's own terms, such as `dof` and `q`, excepted). Results and enums that may grow are
  `#[non_exhaustive]`; options and problems keep struct literals, so struct update syntax works.
- **Implementation.** `expect` says why it cannot fail, `for` loops replace sequential
  `for_each`, iterator chains are named before a `for` loop, and assertions say why they hold.

## Changes

- **A new test must fail on the unfixed code.** Plant the bug and watch the test fail.
- **The CPU and the GPU implement the same math.** A change to one (`src/cpu.rs`, the kernels in
  `src/*.wgsl` and `gpu.rs`) lands in the other in the same change; the parity tests catch drift.
- **Performance claims need an A/B comparison on one machine.** Build the previous commit in a
  `git worktree` with its own `CARGO_TARGET_DIR`, and alternate runs with yours after checking the
  machine is idle. One run proves nothing.
- **Numbers in the docs come from runs you can name.** [Benchmarks](docs/benchmarks.md) says how
  each was measured; rerun what a change moves and update every page that quotes it.

[CLAUDE.md](CLAUDE.md) holds the architecture and the working rules in full, for people and coding
agents alike.

## License

Contributions are dual-licensed under MIT or Apache-2.0, as the project is, unless you say
otherwise.
