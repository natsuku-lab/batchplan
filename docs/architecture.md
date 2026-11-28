# Architecture

How batchplan's code is organised, and why. [`CLAUDE.md`](../CLAUDE.md) holds the rules for changing it; each module's doc comment holds its details, and its constants say why they have the values they do.

A robot and its worlds go onto a device once. Every algorithm then takes the device, the worlds and a batch of problems, and returns each problem with its answer.

## The code

```
src/
├── lib.rs           # the public API: re-exports, and the guides' code compiled as tests
│
│   loading a robot
├── robot.rs         # Robot: the kinematic tree, limits, hands, held objects
├── description.rs   # the one description every format loads into
├── urdf.rs          # ┐
├── mjcf.rs          # ├ loaders: a file into the description, nothing more
├── usd.rs           # ┘
├── loops.rs         # closed loops become mimic joints of their one motor
├── spheres.rs       # collision spheres fitted to the meshes (cuRobo's voxel method), and held objects' covers
├── meshes.rs        # exact clearance to the meshes, to check finished plans
│
│   worlds
├── world.rs         # obstacles, allowed contacts, scenes
├── sdf.rs           # distance grids from meshes, points and depth images; occupancy maps
├── winding.rs       # which points of a grid a closed mesh winds around: the inside of a mesh grid
│
│   running a batch
├── device.rs        # Device: checks every batch, then hands it to a backend
├── cpu.rs           # the CPU backend, the scalar twin of the GPU's math
├── lanes.rs         # SIMD collision checks, eight configurations at a time
├── gpu.rs           # the GPU backend (wgpu), kernels written per robot
├── *.wgsl           # the GPU's math, and grid building
├── edges.rs         # how searches ask for edge checks, and the threads that answer
│
│   motion
├── reach.rs         # HandReach: an outer bound on the poses a hand can reach
├── ik.rs            # solve_ik: damped least squares from many seeds
├── rrt/             # RRT-Connect, and idle threads that help long searches
├── constraint.rs    # HandConstraint: a hand kept upright, in a box or in another's frame
├── shortcut.rs      # simplifying a found path
├── spline.rs        # uniform cubic B-splines, the shape of every path
├── trajopt.rs       # plan: search, simplify, fit a spline, optionally optimize it
├── tool_path.rs     # follow: the hand in straight lines through poses
├── join.rs          # join: consecutive motions without stopping between them
├── gripper.rs       # Gripper: a hand as a parallel-jaw gripper, its measured widths and finger joints
├── task.rs          # pick, place, turn and pull: a whole task in one call
├── surface.rs       # grasps of what cameras see: surfaces from depth and points, pairs of patches, the hand kept clear
├── rest/            # rests_on: a measured support as heights, an object set down on it, what it touches, whether that holds it; falls_into: the hollows it cannot leave
├── handover.rs      # one hand passing what it holds to another, on the task layer's steps
├── held.rs          # Held: a held object's spheres, its devices, problems grouped by hold
├── timing.rs        # Trajectory: a path timed within the limits; stops; moving starts
├── topp.rs          # time-optimal timing (TOPP-RA), smoothed for jerk
├── live.rs          # Follower and servo: running a motion while the world changes
│
│   training data
├── datagen.rs       # reaches with recoveries, pick-and-place, handovers, reset states
├── npy.rs           # export as NumPy arrays
├── lerobot.rs       # export as LeRobot v3.0 datasets
│
│   shared
├── types.rs         # Pose, JointTrajectory, StartMotion, Solved
├── error.rs         # Error: the kinds a host matches on
└── rng.rs           # the generator every algorithm seeds from
```

Modules are grouped by what they do, in the order a call meets them. `demos/` is a crate of its own (Rerun), so the library never depends on a viewer, and so is `python/`, the Python bindings (PyO3), so it never depends on Python.

## How it fits together

- **`Device` is the only way to run a batch.** It holds the robot and a CPU or GPU backend behind the crate-private `Backend` trait, and checks every batch (array shapes, world indices, worlds uploaded to it) before either backend sees it. Worlds go onto it once (`Device::upload`) and every algorithm takes them as `&Worlds`.
- **Both backends compute the same numbers.** The GPU kernels and `cpu.rs` implement the same math function by function, held together by parity tests. Searches run on CPU threads on either device, so results depend on neither the batch nor the device.
- **Algorithms are plain functions.** `solve_ik`, `plan`, `follow`, `join`, `pick`, `place`, `handover` and `datagen` take a device, its worlds and problems, and exchange the shared types. Each owns its seeding, validation and choice of answer. There are no planner plugins and no runtime configuration.
- **Batch-first.** Every call covers many problems, and each problem names its world by index, so one call spans many scenes. Results carry their problems (`solved()`).
- **Seeds come from the problem.** Each problem's generator mixes the options' seed with the problem's own numbers, so a problem draws alike in any batch and on either device.

## Principles

The library's engineering principles, and where the code shows them:

- **The simplest implementation that fully works.** One path shape (uniform cubic B-splines) serves planning, tool paths, joins, timing and checks. Loaders only translate; everything after them sees one description.
- **Layers on a working product.** Each capability was added on top of ones that already worked end to end: tool paths on IK and splines, tasks on tool paths and planning, joins on paths from both, datagen on tasks.
- **Separate concerns.** Collision lives in the backends, searching in `rrt/` and `edges.rs`, timing in `timing.rs` and `topp.rs`. A change to what collides touches the CPU, the SIMD lanes, the GPU and `MeshModel` together, and nothing else.
- **Proven designs, studied first.** RRT-Connect and its edge checks as VAMP runs them; spline stops eased as OpenRAVE's smoother does; spheres fitted as cuRobo fits them; TOPP-RA for timing; allowed contacts as MoveIt's matrix; blends at a radius as Pilz and UR blend. Each is adapted to batches, and documented where it differs.
- **Decisions for the long term, without compatibility layers.** Obsolete paths are removed rather than kept beside their replacement; minor versions before 1.0 may change the interface.

## Planning a path

`plan` runs each problem as one search on one CPU thread. Only optional trajectory optimization runs on the device, batched over every problem.

1. **Search** (`rrt::Connect`). RRT-Connect as VAMP runs it: Halton samples, dynamic domains, balanced trees, a k-d tree for nearest neighbours. Samples come first from a tube around the straight line, widening until it spans the joint ranges; the search keeps a clearance from obstacles for its first samples, then continues the same trees without it. A straight line that is free, or one bent around the collision, skips the search.
2. **Simplify** (`shortcut::Simplify`). The straight line, then passes of greedy shortcuts and smoothing.
3. **Fit** (`trajopt::Fitting`). A B-spline through the simplified path, its control points spread by arc length. Where it collides, the nearest waypoint is pinned, so the curve passes through it; each pin is then eased into the widest blend that keeps the spline valid. A spline is checked as edges are, except along a straight line that the search or the shortcuts already checked keeping at least its clearance. Splines keep the shortcuts' clearance, then a 2 mm margin, from the world, and 2 mm between the robot's links but in a last round, which the searches and shortcuts keep nothing between; the lanes check pairs of links against that margin in a copy of their own (`keeps_with`'s `APART`), out of the searches' loop (`edges::keeps_apart`), whose vectorized closures it changes by −1.5% to +2.2% at one codegen unit and −0.3% to +3.5% at sixteen.
4. **Repair.** A path searched but left without a valid spline has its spline optimized up to ten steps on its own thread, and takes the first step whose spline keeps the 2 mm margin, or the last step's spline where that is collision-free. A path found near the straight line that neither fits nor repairs is searched again over the joint ranges alone. If the last path searched still has no valid spline, the fit and the repair run again with twice the control points (`Fitting::grow`), as FITPACK adds knots until its fit holds. That stops once as many are free as the path has waypoints, the density of TOPP-RA's spline through every waypoint. So `PlanResult::paths` holds each seed's own control points, and optimized seeds keep `control_points`.
5. **Optimize** (optional, `PlanOptions::seeds`). The spline and seeds straight to the goal or bent through random points are optimized together on the device (L-BFGS) and validated densely; the shortest valid path wins.

Goals on joints whose range spans more than a turn are reached at whichever equivalent is nearest or free. Held joints never move. A start in motion pins the spline's first span to the current velocity and acceleration.

**Failure reasons** come from where each problem's planning stopped: a search that never started (`Connect::blocked`: the start, or every goal, collides), one that drew all its samples, a path whose splines all failed, or a planning still running at the deadline. Only the colliding ends get contacts, in one batch after the searches.

**Constraints** (`constraint.rs`) change four steps and add a test to every check. A `Region` holds a problem's constraints prepared once; each bound it breaks is a row with its rate per unit of the hand's velocity and of the velocity of the hand it is relative to, which both projection and optimization read. The search's extensions and connections walk in hops of one lane pass each, every hop projected back inside the bounds by damped least squares on the rows it breaks (CBiRRT), and the straight line's projected walk replaces its bends. Smoothing projects a moved waypoint before checking it. The spline is fitted through points along the path rather than near them (`through_knots`). Under bounds tighter than the room projection leaves (`Projector::tight`), shortcuts that stop short walk the same way toward farther waypoints, halving hops whose edges bulge out at their middles (`shortcut::Step::Walk`), and the spline's points are projected before it is fitted through them. Optimization adds the squared distance past each row to its cost, on the CPU (`Region::cost`) and in its GPU twin (`constraint_cost`), which the trajectory kernels compile in only when built with `CONSTRAINED`, the first time a constrained path is optimized; the constraints ride in the `targets` binding IK uses. Every check of a constrained problem tests the hands' frames in the lanes after their forward kinematics, before any collision test, so it costs a few dot products per eight states. A constrained search takes no help from idle threads, which forecast unconstrained extensions. Joins, tasks and the follower pass their constraints to the checks and plans they already make.

## Searches

A search (RRT-Connect, simplification, a spline's check, a tool path, a join) is a small state machine (`edges::Search`): it asks for an edge check, takes the answer and decides what to ask next. The SIMD lanes answer at once; a GPU device keeps a CPU copy of its worlds for them.

- **Edge checks** follow VAMP: eight states spread over the edge, each pass stepping back, stopping at the first collision. A link's spheres are placed only when its bounding sphere is near something, a sphere around each obstacle rules it out before the exact test, and distances are compared squared.
- **Scheduling.** Each thread takes the next search until none is left. Threads with nothing left help searches that have checked many samples (`rrt/ahead.rs`): they replay the trees from a log and check the edges the next samples will ask about. A search takes an answer only for a query equal to its own bit for bit, so it grows the same trees as alone.
- **Memory.** A search allocates nothing per query, and is built on the thread that runs it (`edges::Deferred`), so its memory belongs to that thread's allocator arena.

## Collision

The robot is a set of spheres; obstacles are boxes, spheres, cylinders, capsules or distance grids. A configuration's cost sums the squared penetration of every sphere into every obstacle, and of every self-collision pair worth checking, within a margin.

- **Gating.** Each link has a bounding sphere; an obstacle farther than the margin skips its spheres, a distance grid by its box and the least value on the box's faces. Clearances stay exact up to the margin. On a GPU the grids' rule is in kernels that only uploads holding a grid run, so that uploads without one keep kernels without it. Between two links in reach of each other, the CPU's scalar code skips a sphere's pairs with a link whose bounding sphere it is past the margin from, where the clearances are not read (the IK step, the optimizer): the kernels and `evaluate` visit every pair, and the cost and its gradient are the same.
- **Gradients** accumulate a wrench per link; one reverse pass gives each joint its share.
- **Links no joint moves** are not checked against the world.
- **Allowed contacts** become one link bit set per obstacle at upload, which every check skips.
- **Contacts** (`Device::contacts`, `cpu::contacts`) say what a configuration collides with: `collision`'s distances per link and obstacle and per link pair, without its gates, skipping what it skips. They run on the CPU copy of the worlds that searches check, so both devices list the same. Plans, IK diagnosis, tool paths and tasks call them only for what failed.
- **SIMD** (`lanes.rs`): forward kinematics and clearances for eight configurations in `f32x8` lanes, at the widest level the CPU has (SSE2 to AVX-512, NEON), chosen once per search and inlined all the way down. `cpu::collision` is the scalar twin the lanes are tested against.

## The GPU

- **The twin.** `kernels.wgsl` and `cpu.rs` implement the same functions: kinematics, the collision cost and gradient, the IK step, the optimizer's passes.
- **Layouts declared once.** Every struct the GPU reads is declared in Rust with `shader_struct!` and its WGSL generated from that, with shared constants written in from their Rust values.
- **Kernels written per robot.** `robot_wgsl` writes the kinematics, Jacobians and collision cost link by link, each in its own variable: run-time-indexed arrays spill to scratch memory on Mesa's RADV, and writing them out made GPU planning 4.8× faster on a Radeon and 2.7× on an M4 Pro. Small batches give each configuration 8 invocations that split its sphere loops: up to 4,096 configurations, and IK of robots with many spheres up to 25 configurations per sphere, since how far splitting pays grows with the robot's collision work. IK without the push away from obstacles checks no collisions, so it never splits.
- **Trajectory optimization** is L-BFGS with a parallel line search, five passes per round, split across submissions to stay under driver watchdogs. A path whose optimization is over (a steepest-descent search found nothing better) is marked in its state, and every pass leaves it out, as `cpu.rs` ends its rounds.
- **Loops written out.** naga counts the rounds of every shader loop. One hand's least squares is written term by term per robot (`least_squares_wgsl`), and `loops_written_out` writes out the kernels' loops of a known count when they are built; `kernels.wgsl` keeps them as loops, to read beside `cpu.rs`.
- **IK's lanes.** The lanes of a configuration split its collision gradient; its first lane then takes the step, and the others take the configuration from it (`lead`).
- **IK** is one round trip per chunk of 65,536 seeds: the screening steps, a pass in which every seed finds its place among its problem's (`ik_select`), the remaining steps of the seeds chosen, through their indices, and the clearances, read back together. Passes share a submission up to the work of 16 steps of a full chunk, so a small call of up to 64 steps is one submission, and its buffers (up to 64 KiB each) are kept for the next small call.

## Tool paths, joins and tasks

- **`follow`** splits a path into poses a step apart and solves each by damped least squares from the configuration before, halving a step whose joint motion strays from the path; eight halvings is a jump. A spline then passes through the configurations, checked densely; a collision cuts the motion back.
- **`join`** blends consecutive rest-to-rest paths: it drops the control points around each join within the blend radius at the hand, bridges the gap with a curve tangent to both paths whose curvature rises from zero, and checks only the spans that changed. A blend is kept only where it makes the stretch so far and the next motion finish sooner than stopping between them, so no stretch takes longer than its motions apart. Paths are first split at their spans' middles (the same curve) until they are spaced alike, because timing follows a path best where its pace is even.
- **`pick` and `place`** run every task's grasps or place poses as candidates through the same steps, each a batch across every task whose fingers and held object are alike: start, IK at the grasp or release, the straight move in (followed backward), the straight move away, the planned transit. A place that names how far its hand may open solves its release on the robot relocked where the fingers hold the object, and runs its move away on the robot relocked at each opening in turn, those the fingers stop going on to the next. A task that leaves its transit out passes the first and the last unchecked. Each step runs in the world its task names for it and allows only its own contacts besides (the fingers on the object, the held object on its support). A held object is a link attached to the robot, so lifts and carries run on a device for the holding robot. A place onto what another hand holds (`Place::onto`) runs on a device that holds that support, grouped with the tasks set on the same hold: the release and the move in on holders whose object has the support's link as a touch link, the start and the transit on holders without it, and every step with the joints that move the support held. Each object's spheres are fitted once, in its own frame, and placed by each grasp of it, where attaching it at each grasp would fit it again.
- **`turn`** runs a pick's steps with a turn checked after the reach, and the grasp waves and locked-joint groups of `pick` (`by_grasps`, over the `Gripping` tasks). Every configuration IK finds at a grasp becomes a try of its own, nearest the start first; every try's turn (`Screw::path`'s waypoints, at half the tool path's precision, followed at the other half) is followed holding the object, as a pick's lift holds it (a device per grasp, built where it is used, the object's spheres fitted once, in the task's world without it, touching its `support`), or for an obstacle that `stays`, one batch in the world the fingers touch it in; then the moves in from the tries that made it and the moves away from where each turn ends, in a world per grasp with the object turned. Each grasp keeps its first try through the move away, or, where none got there, the one that got farthest, and at the turn the one that turned farthest; the transit is planned for the kept tries alone. `pull` runs the same steps with a straight draw along the task's direction, the hand's rotation fixed, in place of the turn.
- **`turn`'s precedent.** MoveIt Task Constructor's `MoveRelative` takes a twist (`setDirection` with a `TwistStamped`) and its `GenerateGraspPose` samples grasps about a settable axis of the object (`setRotationAxis`, `setAngleDelta`); PickNik's `affordance_primitives` moves along a `ScrewAxis` of an origin, an axis and a pitch (`getTF(delta_theta)`); GPD filters its grasp candidates by approach direction (`filter_approach_direction`); MoveIt's perception tutorial fits a cylinder to a cloud by RANSAC and takes its axis. None of them orders grips by the turn they are to make, or checks many grips' turns, from every configuration of the arm, in one batched call: `turn` does both.
- **Holding per problem.** `plan`, `solve_ik`, `follow` and `join` check every problem, then group them by `holding` (`held::by_hold`) and run each group on a device holding its object, made with `with_robot`, in parallel, gathering the results back in the problems' order. Seeds come from each problem's own values, so a group gives each problem what a batch of it alone would. A device keeps the spheres fitted to the latest 64 held objects (`held::Fits`), shared with the devices made from it, so a call holding a round object does not fit it again.

## Grasps of what cameras see

`grasps_on` (`surface.rs`) closes on pairs of patches of measured points and checks the hand's spheres against the points, with no shape assumed; a wall seen from one side is closed on where the cameras saw past it. Its module doc has the method. Dex-Net's depth-image sampler grasps from one overhead image by pairs of depth edges whose gradients point away from each other, as across a rim, with the gripper coming down the camera's axis; GPD's candidates need points between the fingers and none in them, and it computes what no camera saw only for its classifier. `grasps_on` takes from Dex-Net that a depth edge bounds a wall, and from GPD the camera each point was seen from.

## Timing

A `Trajectory` is the path plus a time map `s = σ(t)`, itself a uniform cubic B-spline. `Trajectory::try_new` keeps the faster of uniform timing (exact along the whole curve) and time-optimal timing: TOPP-RA on the path parameter, with the jerk limit as a cap on speed, smoothed into the time map, solved again slower where the map overshoots a limit, and stretched by what overshoot remains. Map spans are sized by the path and, for finely split paths, by the time the jerk limit takes to swing acceleration. A start in motion is timed from its speed along the path, with the full limits at the start easing over two spans to those that leave the stretch's margin, and where no map fits, the nearest stretched with its start set again. Its candidate maps are turned down at a quarter of the checking density, and one kept is also measured where it crosses the path's knots. A controlled stop brakes to rest along the path.

## Live motion

- **`Follower`** is built from the pieces above: `evaluate` on the rest of its motion every tick (up to a pending takeover, then the replan, and any stop), `plan` with a `start_motion` and a time budget, `Trajectory::moving`, and `Trajectory::stop`. It replans from where the robot will be once the budget has passed (with a `Detour`, no sooner than its window opens); with `replan` off, it only stops and holds. One that holds an object (`FollowerOptions::holding`) checks on a device holding it, built from the tick's device (`held::holders`) and kept until a tick is given another device, and passes the hold to `plan`.
- **`servo`** solves the hand's velocity with the same damped least squares and analytic Jacobian as `follow` (`cpu::Reach`). It checks, in one `evaluate` batch, the braking paths from the state it would command at full speed and from braking now, with the margin at `slow_within` so the clearances it acts on are exact.

## Perception and training data

- **Distance grids** are a CPU/GPU twin too (`sdf.rs` / `grids.wgsl`): voxels classified from depth images or points, then an exact distance transform in integers, so both devices build identical grids. A robot mask keeps a camera that sees the arm from making it an obstacle to itself. A map's update when one image decides (`Weighing::Latest`) is a copy of its own: on the CPU `integrated::<MASK, true>`, on the GPU `map_update` from a second module of `grids.wgsl` whose `HIT` and `MISS` alone differ, so the default's code is as it was. The GPU's `map_update` visits every voxel; the CPU's walks each row of voxels over the span whose middles may project into the image (`sdf::span`, the image's five bounding planes moved out past what rounding moves a point), and past it only the voxels holding a reading or inside the mask (`sdf::flagged`), the only ones there an image changes. Each voxel it visits takes the update it took when every voxel was visited, so the map is the same to the byte, and a fuse's cost follows the view, not the map's size.
- **Grids on the device.** On a GPU the host computes a depth image's bounds and, with a mask, the grid points the robot fills, found sphere by sphere over the points near each, the same for both devices; everything after runs in compute kernels, and on the CPU device on its threads.
- **Shapes in a mask.** `with_shapes` measures each pixel's point against the shapes on the device's threads, a shape only in the pixels its box, grown by the margin, projects to once cut to the camera's view, and only where the point lies in that box. A fuse with the mask finds the space a shape fills a row of voxels at a time: every shape but a distance grid is convex, so a row of its box holds one run of voxels that reach into it, found by a few distance tests, and a shape costs a frame by the rows its box spans in the map. `sdf::on_shapes` turns each shape's rotation into its matrix once and takes the points a run at a time on the device's threads, a shape measured only against the runs that come within reach of its box.
- **The band** (`with_band`) runs on the device's threads. A pass along the columns and one along each row find, for each reading, the least depth of a flagged reading within the band less the padding or the margin, which is as much nearer as that reading's thing may lie; a reading no thing can lie nearer than by the margin is kept there. Only the readings that pass are measured against the flagged readings of the band and their things, the spheres first, and a shape only where the reading lies in its box.
- **Training data** (`datagen.rs`) runs the planners in batches: reaches with recoveries from perturbed states, pick-and-place as a `pick` then a `place`, handovers that move both arms at once, and reset states. `npy` and `lerobot` export them.

## The robot model

`Robot::load` reads the description, closes kinematic loops (each loop's passive joints become polynomial mimics of its one actuated joint, fitted to the loop's static equilibrium), expands ball, planar and floating joints into one-axis joints, numbers the actuated joints breadth first, and fits collision spheres unless a model is given. `Robot::assemble` does the same for several descriptions, their names prefixed.

## Collision spheres

`spheres.rs` fits a robot's spheres by the steps [Robots](robots.md#collision-spheres) lists; its module doc has the method.

- **Inside a mesh** is where its surface winds around the point (`winding.rs`): shells that overlap or share a face are their union, a shell given twice is itself, and a sheet with two sides has no inside. Without a limit on overhang a shell is solid whichever way it faces, and one facing inward inside one facing outward is a cavity. A shell that faces inward and is thinner than the grid still gets spheres on its outside, from its samples' own candidates, which is why a limit refuses it.
- **A limit on overhang** (`SphereOptions::max_overhang`) cuts each sphere's radius back to its center's depth plus the limit. A center's depth is its distance to the nearest face, a face buried in another shell among them, so near such a face it reads less than it is; where two shells that face opposite ways overlap, the overlap reads outside. Depths are measured in the geometry as the fit sees it, a round shape as its mesh of 24 sides, which lies inside it by up to 0.9% of a cylinder's radius and 1.7% of a ball's or a capsule's. A held shape covered without a fit is held to the shape itself, so `qualify`, measuring on the mesh, reads up to that much more for it.
- **What a hand holds** (`robot::covered`) is covered within its tolerance, whatever the budget, by the first of: a direct cover within the budget that takes no more spheres than the tiles (`spheres::cover_primitives`: a ball is itself, a box a grid of cells, a capsule a line of spheres, a disc one layer of rings and a rod one line); its surface's tiles (`spheres::cover_tiles`: a box's medial cover, a cylinder's rings along its medial structure or a thin one's layer, whichever takes fewer); a fit. Direct and tiled covers hold every point of the shape, inside as well, by construction: a sphere passes through the corners of its cell or its tiles. A shape's covers are laid out by `box_cover`, `cylinder_tiles` and `cylinder_layer`.
- **A held shape's fit** holds a grid of points on its surfaces (`spheres::surface_grid`), no farther apart than puts every point of them within `m` of one: half a grid cell's diagonal, about 80,000 points, and more where that keeps `m` within half the tolerance. It fits within the tolerance less `m`, each grid point held with `m` to spare, and grows every sphere by `m`: by the triangle inequality every point of the surface is inside a sphere, and none reaches past the tolerance. A round shape is fitted to a mesh that lies within half of `m` inside it. The candidates take in each sample's medial ball, the largest ball inside the shape that touches it, found by halving along the sample's inward normal (the shrinking ball of Ma, Bae and Choi): its center lies on the medial axis, the middle of a slab or the bisector of an edge, where a sphere reaches the most surface for its overhang, as in the medial-sphere approximations of Hubbard, of Bradshaw and O'Sullivan, and of Stolpner, Kry and Siddiqi. The greedy cover drops each pick whose every sample others reach, and a direct cover is taken instead where it takes no more spheres. Under a limit on overhang the fit's spheres are held to the limit rather than grown, which leaves some of the shape outside them.

## Performance

Performance claims come from alternating runs of two builds on one machine; [Benchmarks](benchmarks.md) gives each measurement with the run behind it.
