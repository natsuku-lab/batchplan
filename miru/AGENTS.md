# Agent instructions

miru is a Rust library that says what depth cameras show, as objects: a crate of batchplan's
repository, built on batchplan. Read `README.md` first: what it is, its one call, and where it
stands. The repository's `CLAUDE.md` holds the rules both crates keep (style, working rules, the
lints); this file holds miru's own.

## Commands

Run from the repository's root.

```bash
cargo test -p miru                                     # unit tests, tests/look.rs on rendered scenes, the scorer's own rules, the README's and lib.rs's code
scripts/check.sh                                       # the repository's lints, miru's among them: run before every push
cargo run --release -p miru --example score -- <set> [--recorded | --masks truth|one [--border <pixels>]] [--again] [--noise <factor>] [--shares] [--offsets] [--twin <set>] [--rooms] [--threads <n>] [--images <dir>] [--digest] [run ...]   # score recorded looks against the simulator's truth (--recorded: their executor's; --masks truth: miru's with the truth as its masks; --masks one: each truth object from a look of its own whose masks are its pixels alone, as a model prompted for one thing gives, refused with --again, --offsets, --shares, --twin, --rooms and --images; --border: every mask's border moved out by that many pixels, or in where negative, as a model's masks are never exact; --again: each look's frames a second time; --noise: every view's stated noise multiplied by the factor; --shares: per kind, how much of it is an object's, a known shape's or no one's; --offsets: per kind, the median distance from above of a matched object's middle from the truth's, and how many matched ones are solid; --twin: the same scenes on other depth, each object matched in both held against itself, per kind: how much its half sizes exceed the twin's, and how many are round and solid where the twin is; --rooms: per kind, how many of a thing's own readings lie in the room of its object's solid box open at the top, and how many of another thing's inside a solid object's solids, by more than three deviations of their camera's noise, and the ten objects that hold the most of each; --threads: how many threads a look may use, the machine's by default)
cargo run --release -p miru --example score -- <set> --again --digest [--masks truth] [--threads <n>]   # one hash over every scene the set's looks give, the second looks' too, and nothing else: a change meant to leave the results alone, as one for speed, prints the same digest before and after, on both sets, with and without masks, on 1, 2 and the default threads
```

A set is a folder of recorded looks as batchplan-orchestrator's benchmark writes them
(`examples/common/mod.rs` has the layout); the sets live outside this repository.

## Design rules

These are settled with the user. Keep to them unless the user decides otherwise.

- **Built on batchplan, never the other way.** miru takes batchplan's types where the two
  meet: a `View` holds its `DepthImage`, the shapes a look knows are its `Obstacle`s,
  `Shape::solids` gives obstacles, errors are its `Error`, and a look runs on a `Device`'s
  threads (`Device::install`). batchplan does not depend on miru, in code, tests or examples:
  what a look needs of it that it does not offer becomes a public primitive of batchplan's
  first. The crate is not published.
- **The same on any machine's threads.** A scene does not depend on how many threads the device
  has, to the bit: each piece of work is one thread's alone and results are taken in the
  pieces' order (`src/threads.rs`). The scorer's `--digest` on 1, 2 and the machine's threads is
  the check.
- **Any embodiment.** Nothing names or assumes a robot, a hand, a camera or a table. The robot
  is a mask per pixel, known shapes are obstacles, noise is each camera's own polynomial in
  depth, stereo or time of flight alike: all inputs. A look reads nothing of its device's robot.
- **One call, state in the result.** `look(&device, &Look, &previous, &options) -> Scene`. No
  handle of miru's, no run-time configuration, no plugins. What the next look needs is in the
  `Scene`.
- **Geometry first.** Objects are told apart from depth, and a look without masks is whole
  without a model. A mask model sits behind one seam, `View::masks`, and what crosses it is
  data: its instance masks as a number per pixel, which divide and join what geometry made
  (`src/masks.rs`). miru runs no model, links none and takes no callback, and a model is never
  asked for coordinates; a caller whose model needs prompts looks twice.
- **Every tolerance comes from the camera's noise.** No tuned constants: each constant in
  `src/segment.rs` and `src/view.rs` says what it is derived from (deviations of a named
  difference, a pixel's width) or cites its source (LCCP's 10 degrees). A new one needs the
  same, or it does not go in.
- **Real depth and exact depth both.** A change holds on both paths, the simulator's exact depth
  and depth as a real camera reads it, or it does not land. What differs between them follows
  from `View::noise` and from which pixels have no reading, never from a switch.
- **Ground truth stays outside.** Only `examples/` and `tests/` read a simulator's truth.
- **Small interface.** Prefer deepening `look` to adding a public item. Internals are
  `pub(crate)`; tests go through `look`, with unit tests only for internal seams whose answers
  are known in closed form.

## Style

The repository's (`CLAUDE.md`, Style), with its lints (`Cargo.toml`'s workspace lints and
`src/lib.rs`'s attributes).

## Working rules

The repository's (`CLAUDE.md`, Working rules), and:

- **Numbers come from named runs.** The README's table names its sets and command; rerun it
  when a change moves a number.
- **A gap is written down with its run**, in the README's "Where it stands", not worked around
  with a rule.
- **An A/B for speed runs over both kinds of set**, exact depth and a camera's, and prints the
  same digest before and after.
