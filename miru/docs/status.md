# Where miru stands

What is measured, with the run that gives each number.

## The recorded sets

Looks recorded by batchplan-orchestrator's LIBERO benchmark, kept outside this repository
(`examples/common/mod.rs` has their layout). Each look has two cameras: a table camera stated as
an Intel RealSense D435i (2.8 mm at 1 m) and a hand camera stated as a D405 (14.2 mm at 1 m).
Each view carries the robot's mask as the executor computed it.

- `rows4`: the first looks of 20 LIBERO scenes, on the simulator's exact depth.
- `rows4-stereo`: the same tasks and seeds, with depth as the two stereo cameras read it.
- `hand-in-drawer`: two runs of LIBERO spatial-4 with the hand left in the open drawer; timed only.

## How a look is scored

`cargo run --release -p batchplan-miru --example score -- <set>` scores miru's labels against the
simulator's body under each pixel, by the rules of batchplan-orchestrator's perception scorecard.
`--recorded` scores the executor that recorded the set by the same rules. `--masks truth` gives
miru the simulator's truth as every view's masks: no mask model says more, so what those rows
leave is the look's own. The scorer gives `look` each view's depth as recorded, unfiltered, and
no corners of reach, so the `rows4-stereo` rows are not what `observe` gives on that depth.

Each true object that 50 or more pixels show is counted once over a look's two cameras together,
as the first of these that holds:

- masked: most of its pixels are under the robot's mask;
- merged: most of it is in a measured object that is not mostly it, or one that is mostly it
  holds most of another;
- in pieces: several measured objects are mostly it;
- matched: one measured object is mostly it and holds most of it;
- missed: none of these.

`examples/score.rs` has the rules in full. Each row counts the 140 true objects in view, none of
them masked.

## Objects

| Set | Measured by | Matched | Merged | In pieces | Missed |
|---|---|---|---|---|---|
| `rows4` | miru | 91 | 0 | 49 | 0 |
| | miru, the truth as masks | 135 | 0 | 5 | 0 |
| | the executor | 92 | 22 | 26 | 0 |
| `rows4-stereo` | miru | 70 | 14 | 55 | 1 |
| | miru, the truth as masks | 134 | 0 | 6 | 0 |
| | the executor | 58 | 34 | 48 | 0 |

`--again` gives each look's frames a second time: on `rows4`, 226 of 226 objects keep their ids,
and on `rows4-stereo` 226 of 226; none stays unseen.

## Masks off by a few pixels

`--masks truth|one --border <pixels>` moves every mask's border in (short) or out (over). With
`one`, each true object gets a look of its own whose masks are its pixels alone, as a model
prompted for one thing gives. Matched / merged / in pieces / missed:

| Set | Masks | 3 px short | 1 px short | exact | 1 px over | 3 px over |
|---|---|---|---|---|---|---|
| `rows4` | the truth | 126 / 0 / 14 / 0 | 133 / 0 / 7 / 0 | 135 / 0 / 5 / 0 | 135 / 0 / 5 / 0 | 135 / 0 / 5 / 0 |
| | one | 126 / 0 / 14 / 0 | 133 / 0 / 7 / 0 | 135 / 0 / 5 / 0 | 135 / 0 / 5 / 0 | 134 / 4 / 2 / 0 |
| `rows4-stereo` | the truth | 116 / 0 / 23 / 1 | 135 / 0 / 5 / 0 | 134 / 0 / 6 / 0 | 134 / 0 / 6 / 0 | 134 / 0 / 6 / 0 |
| | one | 107 / 14 / 18 / 1 | 124 / 9 / 6 / 1 | 129 / 7 / 4 / 0 | 129 / 7 / 4 / 0 | 127 / 12 / 1 / 0 |

## The truth table

`cargo test --release -p batchplan-miru --test truth -- --nocapture` builds scenes from known
things and reads each through three cameras: the simulator's exact depth, a D435 by its
datasheet, and a D435 with a D405 near. It prints a row per scene, camera and thing: how many
objects the thing came out as, and the kind, size, middle, support, solidity and share of the one
that holds most of it, each against the truth. On exact depth the rows that do not hold are the
knob's dial and bar (one object, on a box that is not solid), a plate on a box (one object with
the box) and two boxes touching (one object). Through either stereo camera no row holds whole:
most boxes come out larger than the noise reaches and not solid.

`tests/truth.rs` lists each gap by row, and `tests/memory.rs` (`GAPS`) each gap of the memory and
the grips by camera. Both tests fail when a listed gap closes or an unlisted one opens.

## Time

`score -- <set> --observe [--threads 1]` times `miru::observe` on each look, on a CPU device of
the run's own robot: the median over the set's looks, and the median of three runs, on an AMD
Ryzen AI Max+ 395 (32 threads) under `nice -n 19`, at a load of 1.5 to 4 from other work:

| | `rows4`, 1 thread | `rows4`, 32 threads | `hand-in-drawer`, 1 | `hand-in-drawer`, 32 |
|---|---|---|---|---|
| `observe` | 686 ms | 113 ms | 763 ms | 163 ms |
| the look inside it | 135 ms | 77 ms | 246 ms | 128 ms |
| `Memory::world` after it | 1.0 ms | 0.9 ms | 1.0 ms | 1.2 ms |

A scene does not depend on the thread count: `--again --digest` prints one digest on 1 and 32
threads, for each set.

## Gaps on the recorded sets

- **Things in parts that depth does not join.** On `rows4` without masks, the stove (18 of 18
  looks), the rack (8 of 8), the basket (2 of 2) and the cabinet (10 of 18) are in pieces. With
  the truth as masks, none of them is (`score -- rows4`, with and without `--masks truth`).
- **Two cameras' parts of one thing.** With the truth as masks, the cookie box is two objects in 4
  of its 10 looks, where a bowl on it hides from each camera what the other sees; the tomato
  sauce in 1 of 2, its two sides seen by the two cameras (`--masks truth`, both sets).
- **Stereo depth merges things that touch.** On `rows4-stereo`, 14 of the 140 merge: 6 bowls, 4
  cookie boxes, the cabinet twice, a ramekin and the rack (`score -- rows4-stereo`).
- **Stereo depth moves shapes off and leaves few of them solid.** A shape's faces lie at its
  outermost readings, which a stereo camera's noise puts farther out. By `--offsets`, a matched
  object's middle lies within 0.09 cm of the truth's on `rows4` but for the cabinet (0.36 cm), and
  on `rows4-stereo` from 0.19 cm (cream cheese) to 12.3 cm (the one stove matched). Every matched
  cream cheese, cookie box and wine bottle is solid on `rows4`; on `rows4-stereo` one cream cheese
  of 10 is, and none of the others.
- **Depth stated as more exact than the camera's does not help.** On `rows4`, 91 are matched at
  the stated noise, and 80, 83 and 81 at 0.5, 0.25 and 0.1 of it (`--noise <factor>`).

## Gaps on rendered scenes

Each is a test of `tests/look.rs`, ignored until it passes:

| Test | What it shows |
|---|---|
| `a_box_on_a_wider_box_keeps_its_rim_where_its_mask_stops_short` | A rim goes with a wide unmasked neighbour in its part. |
| `a_ring_leaves_the_table_inside_it_the_tables` | The table inside a ring reads as a plate's thin floor does. |
| `a_round_hollow_in_a_square_block_seen_from_50_degrees_is_no_box_open_at_the_top` | A round hollow seen in part, its turn within the noise, is open at the top. |
| `a_thick_walled_bowl_seen_on_noisy_depth_is_no_box_open_at_the_top` | A thick-walled bowl seen from 50 degrees on noisy depth opens at the top. |
| `a_rod_across_a_lidless_box_under_a_board_keeps_its_room` | A thing the views see too little of to tell from a panel is held as one. |
| `a_panel_at_a_boards_edge_with_an_arm_into_the_room_keeps_its_room` | A panel at an overhang's edge, with an arm into the room, has readings in the solids. |

Tests that pass and record a gap in a comment:

- `a_box_across_a_narrower_slab_stands_on_it_where_the_slab_reads_over_its_foot`: on noisy depth,
  on the draws where the box's lowest reading lies farther under the slab's top than its
  tolerance, the box stands on the table.
- `a_plate_on_a_known_wall_that_touches_a_box_at_an_edge_is_not_the_boxs`: a plate 1 cm thick that
  touches the box along a window is the box's.
- `what_comes_back_as_one_thing_stands_on_nothing_of_its_own`: a block behind a tray's near wall,
  with an arm beside it, is taken for a piece of the tray.

The truth table's tray is a box open at the top on exact depth and through the D435, there in four
objects, and no box open at the top through the D435 with the D405 near.

What a look cannot see is tested on rendered scenes alone, on exact depth: the recorded sets are
first looks.
