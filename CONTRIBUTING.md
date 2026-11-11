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
(cd python && BATCHPLAN_REQUIRE_GPU=1 uv run pytest)                  # the Python bindings, built and tested through the package
scripts/check.sh                                                      # every lint CI runs, on every crate: run it before pushing
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

## Review

Before pushing, run `scripts/check.sh`, then have the reviewer agent read the change. In
[Claude Code](https://claude.com/claude-code), ask for the `reviewer` agent on your branch
([`.claude/agents/reviewer.md`](.claude/agents/reviewer.md)). It checks the change against the
rules in [CLAUDE.md](CLAUDE.md): correctness, the CPU/GPU twin and the other invariants,
deep-module design, the style guide, tests that fail on the unfixed code, measured performance and
docs that match the code. It reports each defect with the input that shows it. Fix what it finds,
or say in the pull request why a finding does not apply.

[CLAUDE.md](CLAUDE.md) holds the design rules, invariants and working rules, for people and coding
agents alike; [architecture.md](docs/architecture.md) says how the code fits together.

## Releases

Merging is not releasing. Changes collect on `main` and ship together, so a version stands for a
batch of changes rather than for each one.

- **The changelog.** A change a user would notice adds a line to `## [Unreleased]` in
  [CHANGELOG.md](CHANGELOG.md), under Keep a Changelog's headings (Added, Changed, Removed,
  Fixed). Mark with **Breaking:** whatever
  breaks code written against the last release.
- **Before a release.** A project that needs a merged change before it ships pins its commit,
  `batchplan = { git = "https://github.com/natsuku-lab/batchplan", rev = "<commit>" }`, and goes
  back to a version when the release is out. crates.io takes no crate with a git dependency, so
  this is for applications and unpublished crates. To try the changes from crates.io, cut a
  pre-release: `0.10.0-beta.1`, then `0.10.0-beta.2`. Cargo picks a pre-release only for a
  requirement that names one, so users of `0.9` never get it. One beta may break the last, and a
  requirement of one takes the next, so a project trying a beta pins it exactly
  (`batchplan = "=0.10.0-beta.1"`).
- **Patch or minor.** `cargo semver-checks --release-type minor`
  (`cargo install cargo-semver-checks --locked`) fails on a change to the interface that breaks
  the latest release on crates.io. It calls such a change major, which before 1.0 means the next
  minor version. A call that now behaves differently can break callers too, which only a reader
  sees. A batch that breaks nothing is a patch, even one that adds to the interface. Breaking
  changes wait on `main` for the next minor version.
- **When.** A patch whenever a fix should reach crates.io users. A minor release at most every
  two weeks, once a beta of it has run in the projects that use batchplan,
  [batchplan-ros](https://github.com/natsuku-lab/batchplan-ros) among them. A patch for a release
  that `main` has moved past with breaking changes goes on a branch from that release's last tag
  (`0.9.x` from `v0.9.0`). Changes reach the branch by pull request, so CI runs on them, and their
  changelog lines go to `main` too.
- **Cutting one.**
  1. Set `version` in `Cargo.toml` and `python/Cargo.toml` (the Python package's). The next build
     of each crate updates `Cargo.lock`, `demos/Cargo.lock` and `python/Cargo.lock`, and
     `scripts/check.sh` builds all three.
  2. For a release, not a beta, move the lines under `## [Unreleased]` to a new
     `## [0.10.0] - date`, and point the compare links at the end of the file at it.
  3. Commit it as `batchplan 0.10.0`, to `main` or by pull request to a release's branch, and
     wait for CI.
  4. `git tag -a v0.10.0 -m "batchplan 0.10.0" && git push origin v0.10.0`. The release workflow
     checks that the tag names both crates' versions and that `cargo semver-checks` passes. It
     builds and tests Python wheels for Linux (x86-64 and ARM64), macOS (Intel and Apple Silicon)
     and Windows (x86-64), and builds and tests the source distribution. Once these pass, it
     publishes the Rust crate to crates.io, then the Python package to PyPI as `batches`.
     Rust's `0.10.0-beta.1` becomes Python's `0.10.0b1`; the import remains `batchplan`.
     Both registries use trusted publishing through `.github/workflows/release.yml` in the
     `release` environment, restricted to `v*` tags. No upload token is stored in GitHub.
     Pull requests that change Python packaging run the builds without publishing; a manual
     run also only builds. If PyPI fails after crates.io succeeds, rerun the failed job, not
     the completed crates.io job. Published versions cannot be replaced: fix a bad release
     with a new version and tag.
  5. After a release, move batchplan-ros to it.
- **1.0** waits for an interface we mean to keep. Two-digit minor versions until then are usual
  for Rust crates: nalgebra, glam and rapier are past 0.30. One decision comes first: how options
  and problems grow. Today a new field breaks callers' struct literals, and after 1.0 each one would
  take a major version.

## License

Contributions are dual-licensed under MIT or Apache-2.0, as the project is, unless you say
otherwise.
