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
| `rows4` | miru | 92 | 0 | 48 | 0 |
| | miru, the truth as masks | 136 | 0 | 4 | 0 |
| | the executor | 92 | 22 | 26 | 0 |
| `rows4-stereo` | miru | 70 | 14 | 55 | 1 |
| | miru, the truth as masks | 134 | 0 | 6 | 0 |
| | the executor | 58 | 34 | 48 | 0 |

`--again` gives each look's frames a second time: on `rows4`, 224 of 224 objects keep their ids,
and on `rows4-stereo` 225 of 225; none stays unseen.

## Masks off by a few pixels

`--masks truth|one --border <pixels>` moves every mask's border in (short) or out (over). With
`one`, each true object gets a look of its own whose masks are its pixels alone, as a model
prompted for one thing gives. Matched / merged / in pieces / missed:

| Set | Masks | 3 px short | 1 px short | exact | 1 px over | 3 px over |
|---|---|---|---|---|---|---|
| `rows4` | the truth | 127 / 0 / 13 / 0 | 134 / 0 / 6 / 0 | 136 / 0 / 4 / 0 | 136 / 0 / 4 / 0 | 136 / 0 / 4 / 0 |
| | one | 127 / 0 / 13 / 0 | 134 / 0 / 6 / 0 | 136 / 0 / 4 / 0 | 136 / 0 / 4 / 0 | 135 / 4 / 1 / 0 |
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

## A held thing's surface

`tests/memory.rs` renders a plank 30 by 6 cm and 3 cm thick, a cube and the thin tray, reads
them through the three cameras, and holds `Kept::surface_held`'s surface of each against where
the true thing is (`cargo test --release -p batchplan-miru --test memory -- surface_seen
held_thing two_holds lifted_off falls_into_a_held rides_on_a_held`). The stance is padded by
three deviations of each camera's noise at the arm's farthest reading: about 2 mm on exact depth,
2 cm through the stereo cameras. One scene is read on exact depth at 2 cm as well, as far as a
real robot's spheres are off its links.

| | exact | exact, 2 cm padding | D435 | D435 and D405 |
|---|---|---|---|---|
| The plank seen standing, carried by a hold raised and tipped more than 0.5 rad | holds | | holds | holds |
| The plank seen in the hand, at the look's stance and carried from it | holds | | holds | holds |
| The same, the plank 5 mm farther from the hand than the hold says | holds | | holds | holds |
| A cube held by the other hand 1 cm over the plank: neither surface holds the other's readings | holds | | holds | holds |
| A box under the plank's far end, 1 cm under it: the plank's surface holds none of its readings | holds | gap | gap | gap |
| A cube riding on the plank: a small box set near it rests on the cube's top | holds | | holds | holds |
| The tray seen standing, carried level: what falls into it lands inside it | holds | | gap | gap |

A plank's row holds where its points lie within three deviations of the noise of the camera
that read each and a pixel's width of the plank's true faces, where the hold or the hand has
them, but for at most twice the 0.27% of a normal draw that lie past three deviations; where
their normals, at the median, lie within half the turn of the faces' own; and, in the hand,
where every rest of a 2 cm cube on it stands on its top. Slipped, the face toward the hand lies
within 1.5 mm of where it truly is, at the median. Through the stereo cameras the points past
three deviations lie at the camera's depth edges, a centimetre or two off the plank's faces by
the fingers and beside its sides: a rest's support may take one in, and every rest stands on the
plank's top all the same.

`tests/batchplan.rs` runs the two flows a program planned from one look takes, on exact depth, on
the CPU and the GPU: two Pandas pick a plank (or the tray) and a cube by batchplan's `pick` in
the memory's world, and the plank's surface where hand 0 is to hold it in the air gives rests of
the cube on its top (the tray's, falls into it), which a place onto it by hand 1
(`Place::onto`) plans.

Gaps:

- **What a held thing was lifted off.** Readings within the padding of the held thing are the
  hold's at the look, so the look fits a support under it from what lies past the padding: its
  top short of the true one by more than the noise. Those of the support's readings that lie
  farther from that solid than three deviations stay the held thing's. Under the plank, 18 of
  3,404 points are the box's through the D435, 20 of 3,484 through the D435 and D405, and 59 of
  4,000 on exact depth at a 2 cm padding (55, 60 and 62 without leaving any out); rests land on
  the plank's top all the same (`what_a_held_thing_was_lifted_off_is_not_its_surface`).
- **A thing the look gives in pieces.** Through either stereo camera the look gives the tray in
  four objects, and no one object's readings hold the hollow: nothing falls into the held tray's
  surface, as nothing falls into its floor's object's readings standing (`Kept::surface`), where
  every reading about it (`Kept::surface_about`) gives falls. A hold carries one object's
  readings (`what_falls_into_a_held_tray_lands_inside_it`).
- **A slip past the padding.** Readings of the held thing farther than the stance's padding from
  where the hold has it are not the hold's: the look makes what it makes of them, and they are not
  in the surface.

## What rides on a held thing

`tests/memory.rs` renders the plank in hand 0 with a 4 cm cube's shape riding on it, the thing
there, smaller, slipped or gone, reads them through the three cameras, and holds `Kept::riding`
and the memory after the look against the truth (`cargo test --release -p batchplan-miru --test
memory -- rider a_look_answers --nocapture`, which prints the windows of readings in the rider's
place and past it, counted by the box's own faces, which equal the rule's). Stereo cases are read
with three draws of the camera's noise each. A look only reports: in every case the memory after
it still has the rider, and `Act::Lost` of it then takes it off.

| | exact | D435 | D435 and D405 |
|---|---|---|---|
| The cube by the hand, in both views: there | holds | holds | holds |
| A cube 2 mm smaller in its place: there | holds | holds | holds |
| A cube 4 mm smaller in its place: there | holds | holds | gap (unseen) |
| A ball 2 cm across in its place: there | holds | gap (gone) | gap (gone) |
| The cube behind a box nearer each camera: unseen | holds | holds | holds |
| Slipped 3 mm along the plank: there | holds | holds | holds |
| A plate 4 mm thick: there and, fell, gone where a camera that reads it allows less, else unseen | holds | holds | holds |
| The cube by the hand fell: gone | holds | holds | holds |
| Of two cubes, the one by the hand fell: gone, the other not | holds | holds | holds |
| The cube fell to the table in view: gone, and a new object there the cube's size | holds | gap (size) | gap (size) |
| The cube at the far end, which the far view alone reads, fell: gone | holds | gap (unseen) | gap (unseen) |
| Slipped by its own width, under the arm: gone | holds | holds | holds |

The windows that decide each, per draw (the rule's, which the test's count by the box's own
faces equals):

| | exact | D435 | D435 and D405 |
|---|---|---|---|
| The cube by the hand: in its place | 43.9 | 4.5 to 4.6 | 1.3 to 1.5 |
| The cube by the hand fell: read through its place | 43.1 | 4.4 | 1.1 to 1.2 |
| Slipped by its own width: read through its place | 43.0 | 4.3 to 4.4 | 1.1 |
| The far end fell: read through its place | 14.3 | 0.73 to 0.77 | 0.72 to 0.76 |
| A cube 2 mm smaller: in its place, through | 36.3, 6.7 | 3.6 to 3.8, 0.3 to 0.4 | 1.06 to 1.11, 0.1 |
| A cube 4 mm smaller: in its place, through | 28.4, 14.2 | 2.6 to 2.7, 1.2 to 1.3 | 0.73 to 0.79, 0.15 to 0.19 |
| A ball 2 cm across: in its place, through | 2.1, 41.1 | 0.09 to 0.14, 4.3 | 0.04 to 0.05, 1.04 to 1.06 |

Gaps:

- **A thing much smaller than the shape carried for it** reads gone while it is there, where the
  camera reads less than a window of it at that range, the look's measure of a thing, and a window
  of rays through the place around it: the ball 2 cm across in the 4 cm cube's place through the
  D435, and through the D435 and D405 (`a_rider_smaller_than_its_shape_is_there`). There the cube
  4 mm smaller lies in under a window and few rays read past it: unseen. A shape a stereo look fits
  comes out centimetres larger than its thing ("a box" in the truth table), so a small thing whose
  shape a stereo look gave is the case to expect it in.
- **A rider's place read too far off.** At the plank's far end only the far view, 0.65 m off,
  reads the cube's place. Through a D435 there a window is 121 readings, and with the cube gone
  0.72 to 0.77 of one reads through its place: unseen
  (`a_rider_one_view_reads_far_off_is_gone_where_a_window_reads_past_it`).
- **A rider found again through a stereo camera** comes out 4.6 to 4.8 by 8.9 to 9.1 cm on the
  table, as the look sizes a box through a stereo camera ("a box" in the truth table)
  (`a_rider_gone_is_found_again_under_a_new_id`).

Untested: riders of two holds at once, a carrier two hands hold (`Held::also`; its riders are
held against the views where the hand its thing is fixed to has them), riders of the other convex
shapes (a cylinder, a sphere, a capsule), and a rider seen against the robot's links behind it.

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
  of its 10 looks, where a bowl on it hides from each camera what the other sees (`--masks truth`,
  both sets). On `rows4-stereo` the tomato sauce is two in 1 of 2, its two sides seen by the two
  cameras; on `rows4` it is one in both, the side the hand camera alone sees lying on the shape
  the table camera's readings give.
- **Stereo depth merges things that touch.** On `rows4-stereo`, 14 of the 140 merge: 6 bowls, 4
  cookie boxes, the cabinet twice, a ramekin and the rack (`score -- rows4-stereo`).
- **Stereo depth moves shapes off and leaves few of them solid.** A shape's faces lie at its
  outermost readings, which a stereo camera's noise puts farther out. By `--offsets`, a matched
  object's middle lies within 0.09 cm of the truth's on `rows4` but for the cabinet (0.36 cm), and
  on `rows4-stereo` from 0.19 cm (cream cheese) to 12.3 cm (the one stove matched). Every matched
  cream cheese, cookie box and wine bottle is solid on `rows4`; on `rows4-stereo` one cream cheese
  of 10 is, and none of the others.
- **Depth stated as more exact than the camera's does not help.** On `rows4`, 92 are matched at
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

The room of a box drawn in from under what overhangs it (the tests of a board over a lidless
box in `tests/look.rs`) is held on exact depth alone: as a D435 reads that scene from 0.8 m, the
box under the board does not come out a box open at the top, and no room is measured to draw in.

Tests that pass and record a gap in a comment:

- `a_box_across_a_narrower_slab_stands_on_it_where_the_slab_reads_over_its_foot`: on noisy depth,
  on the draws where the box's lowest reading lies farther under the slab's top than its
  tolerance, the box stands on the table.
- `a_plate_on_a_known_wall_that_touches_a_box_at_an_edge_is_not_the_boxs`: a plate 1 cm thick that
  touches the box along a window is the box's.
- `what_comes_back_as_one_thing_stands_on_nothing_of_its_own`: a block behind a tray's near wall,
  with an arm beside it, is taken for a piece of the tray.
- `a_card_against_the_face_one_camera_alone_sees_is_the_bars_unless_masks_say_otherwise`: a face
  one view alone sees is joined to the thing whose shape it lies on within the readings' noise
  and a pixel's footprint there, so a card 5 mm thick flat against the bar's hidden face, the views
  stating the recorded sets' noise, is taken for that face: one object, 45 mm across where the
  bar is 40. Masks keep the two apart.

A face one view alone sees joins only a thing whose shape stands for it: a block under a bowl's
rim whose face lies on the bowl's cylinder stays apart
(`a_block_under_a_bowls_rim_that_one_camera_alone_sees_is_not_the_bowls`), and a hollow thing's
own face is not joined either: the outer face of a tray's wall that one view alone sees stays an
object of its own (`a_trays_wall_that_one_camera_alone_sees_stays_apart`). Through either stereo
camera, the readings the filter keeps between a can's edge and the table lie off the can's shape
by more than that allowance, and the side of the can a near camera alone sees stays apart: the
truth table's `a can seen by its side near`, three objects through the D435 and through the D435
with the D405 near, one on exact depth.

The truth table's tray is a box open at the top on exact depth and through the D435, there in four
objects, and no box open at the top through the D435 with the D405 near.

What a look cannot see is tested on rendered scenes alone, on exact depth: the recorded sets are
first looks.
