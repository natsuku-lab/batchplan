"""The planner comparison (see the spec in docs/benchmarks.md): runs one planner configuration over a
set of problems, seeds and repetitions, and writes what every run produced for `validate.py` and
`report.py`.

Each worker is a process pinned to one core. It loads its planner (timed as setup), plans one tuning
problem untimed, and waits at a barrier; then the workers take queries from a shared queue. A pass is
every problem once with one seed; its wall time runs from the barrier to the last answer, and
`setup_wall` from the workers' start. Each query records its phases separately: `setup` (building the
problem's scene), `search`, `post` (shortcutting, simplification), `timing` (time parameterization),
`validation` (any check the pipeline itself runs), and `total`, as a user would call it. The outputs
of the first repetition are kept, as float32 arrays, in one file per pass.

    python scripts/compare.py run <inputs> <out_dir> --planner roboplan --workload A --family defaults \\
        --split tuning --seeds 1,2,3,4,5 --reps 3 --cores 0-15

batchplan, VAMP and OMPL run in their own binaries (`examples/compare.rs`, `scripts/cpp_planners.cc`),
which write the same records.
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_model import shape_from_json  # noqa: E402

LIMITS = {
    "velocity": [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
    "acceleration": [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0],
    "jerk": [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0],
}
TIME_LIMIT = 5.0  # seconds per query, every planner
RESOLUTION = 1 / 32  # matched edge-check spacing, rad (VAMP's and batchplan's default)

# Settings per planner and family. "defaults" is each library's own, with RRT-Connect and the 5 s limit
# deciding; "matched" also checks edges as finely as batchplan and VAMP do.
SETTINGS = {
    "roboplan": {
        # RoboPlan's default max_nodes (1000) is raised so that, as for every planner, the time limit
        # decides when it gives up.
        "defaults": {"max_nodes": 100_000, "step": 0.05, "shortcut_step": 0.05},
        "matched": {"max_nodes": 100_000, "step": RESOLUTION, "shortcut_step": RESOLUTION},
    },
}
MODES = ["Hermite", "Cubic", "Adaptive", "LinearBlend"]
# MoveIt's OMPL checks motions every `longest_valid_segment_fraction` of the space's extent, which MoveIt
# measures in L1, as the sum of the joints' ranges: 0.01 by default; matched, 1/32 rad in L1 (so no more
# than 1/32 rad in L2).
SETTINGS["moveit"] = {
    "defaults": {"segment_fraction": 0.01, "smoothing": "totg"},
    "matched": {"segment_resolution": RESOLUTION, "smoothing": "totg"},
}


def problem_table(inputs):
    """Every problem and one-scene query by id, each with its scene (a query's is its `scene_ref`'s)."""
    inputs = Path(inputs)
    problems = {p["id"]: {**p, "scene_ref": p["id"]} for p in json.loads((inputs / "problems.json").read_text())}
    for q in json.loads((inputs / "queries_onescene.json").read_text()):
        ref = problems[q["scene_ref"]]
        problems[q["id"]] = {**q, "scene": ref["scene"], "scene_capsules": ref["scene_capsules"]}
    return problems


def seed_for(seed, number):
    """The RNG seed of a query: distinct per problem and per seed."""
    return seed * 1_000_003 + number


def roboplan_limits(path):
    """The trajectory limits as RoboPlan's joint-limits config, written once before the workers start."""
    Path(path).write_text("joint_limits:\n" + "".join(
        f"  panda_joint{i + 1}:\n    has_velocity_limits: true\n    max_velocity: [{LIMITS['velocity'][i]}]\n"
        f"    has_acceleration_limits: true\n    max_acceleration: [{LIMITS['acceleration'][i]}]\n"
        for i in range(7)))


class RoboPlan:
    def __init__(self, inputs, settings, workload):
        from roboplan import core, rrt, toppra

        self.core, self.rrt, self.toppra = core, rrt, toppra
        inputs = Path(inputs)
        self.scene = core.Scene("panda", core.loadUrdfSceneDescriptionFromXml((inputs / "panda.urdf").read_text(), []))
        self.scene.importSrdf((inputs / "panda.srdf").read_text())
        self.scene.importJointLimitsFromConfig(core.loadJointLimitsConfig(settings["limits_file"]))
        assert np.allclose(self.scene.getVelocityLimitVectors()[1], LIMITS["velocity"]), "velocity limits not loaded"
        assert np.allclose(self.scene.getAccelerationLimitVectors()[1], LIMITS["acceleration"]), "acceleration limits not loaded"
        self.settings, self.workload = settings, workload
        self.color = np.array([0.5, 0.5, 0.5, 1.0])
        self.loaded, self.names = None, []

    def plan(self, problem, seed):
        """Plans one query, loading its scene unless the last query's was the same (removing the
        previous scene is the `teardown` phase, outside `total`)."""
        from parity import add_roboplan_shapes

        core, rrt, toppra = self.core, self.rrt, self.toppra
        teardown = 0.0
        if problem["scene_ref"] != self.loaded:
            t = time.perf_counter()
            for name in self.names:
                self.scene.removeGeometry(name)
            teardown = time.perf_counter() - t
            self.names, self.loaded = [], None
        t0 = time.perf_counter()
        if self.loaded is None:
            shapes = [shape_from_json(s) for s in problem[self.settings["scene_key"]]]
            self.names = add_roboplan_shapes(self.scene, shapes, self.color, core.Box, core.Cylinder, core.Sphere)
            self.loaded = problem["scene_ref"]
        t1 = time.perf_counter()
        options = rrt.RRTOptions(group_name="", rrt_connect=True, max_nodes=self.settings["max_nodes"],
                                 max_planning_time=TIME_LIMIT, collision_check_step_size=self.settings["step"])
        planner = rrt.RRT(self.scene, options)
        planner.setRngSeed(seed)
        start, goal = core.JointConfiguration(), core.JointConfiguration()
        start.positions, goal.positions = np.array(problem["start"]), np.array(problem["goal"])
        error = "no path"
        try:
            path = planner.plan(start, goal)
        except RuntimeError as e:
            path, error = None, str(e)
        t2 = time.perf_counter()
        phases = {"teardown": teardown, "setup": t1 - t0, "search": t2 - t1}
        if path is None:
            phases["total"] = t2 - t0
            return {"status": "failed", "reason": error, "phases": phases}, None
        shortcut = core.PathShortcutter(self.scene, core.PathShortcuttingOptions(
            group_name="", seed=seed % (2**31), max_step_size=self.settings["shortcut_step"])).shortcut(path)
        t3 = time.perf_counter()
        phases["post"] = t3 - t2
        waypoints = np.array([list(q) for q in shortcut.positions], dtype=np.float32)
        if self.workload == "A":
            phases["total"] = t3 - t0
            return {"status": "solved", "phases": phases}, {"path": waypoints}
        # Workload B: one TOPP-RA mode, timed; for the other modes, Hermite afterwards (untimed) shows
        # whether the mode fell back to it.
        generator = toppra.PathParameterizerTOPPRA(self.scene, "")
        options = toppra.TOPPRAOptions()
        options.mode = getattr(toppra.SplineFittingMode, self.settings["mode"])
        try:
            traj = generator.generate(shortcut, options)
        except RuntimeError as e:
            t4 = time.perf_counter()
            phases.update({"timing": t4 - t3, "total": t4 - t0})
            return {"status": "failed", "reason": f"timing: {e}", "phases": phases}, {"path": waypoints}
        t4 = time.perf_counter()
        phases.update({"timing": t4 - t3, "total": t4 - t0})
        q = np.asarray(traj.positions, dtype=np.float32)
        fallback = False
        if self.settings["mode"] != "Hermite":
            hermite = toppra.TOPPRAOptions()
            hermite.mode = toppra.SplineFittingMode.Hermite
            h = np.asarray(generator.generate(shortcut, hermite).positions, dtype=np.float32)
            fallback = bool(h.shape == q.shape and np.array_equal(h, q))
        times = np.asarray(traj.times, dtype=np.float32)
        return {"status": "solved", "phases": phases, "fallback": fallback, "duration": float(times[-1])}, {
            "path": waypoints, "t": times, "q": q,
            "v": np.asarray(traj.velocities, dtype=np.float32), "a": np.asarray(traj.accelerations, dtype=np.float32)}


class MoveIt:
    """MoveIt 2 through moveit_py (in the Jazzy image), configured from the canonical URDF and SRDF:
    OMPL's RRTConnect with simplification. Workload A runs no response adapters; workload B runs the
    default pipeline's, TOTG (resampled every 10 ms, the comparison's output rate) and ValidateSolution,
    with Ruckig smoothing between them for `smoothing: ruckig`. They run inside one `plan` call, so B
    times them together. moveit_py exposes no seed for OMPL, so MoveIt runs unseeded."""

    def __init__(self, inputs, settings, workload):
        import rclpy
        from moveit.planning import MoveItPy

        from bench_model import Panda

        inputs = Path(inputs)
        self.settings, self.workload = settings, workload
        canonical = Panda(inputs / "panda.urdf", inputs / "panda.srdf")
        self.extent = float((canonical.upper - canonical.lower).sum())
        fraction = settings.get("segment_fraction") or settings["segment_resolution"] / self.extent
        if not rclpy.ok():
            rclpy.init()
        limits = {f"panda_joint{i + 1}": {
            "has_velocity_limits": True, "max_velocity": LIMITS["velocity"][i],
            "has_acceleration_limits": True, "max_acceleration": LIMITS["acceleration"][i],
            "has_jerk_limits": True, "max_jerk": LIMITS["jerk"][i],
        } for i in range(7)}
        pipeline = {
            "planning_plugins": ["ompl_interface/OMPLPlanner"],
            "request_adapters": ["default_planning_request_adapters/CheckStartStateBounds"],
            "planner_configs": {"RRTConnectkConfigDefault": {"type": "geometric::RRTConnect", "range": 0.0}},
            "panda_arm": {
                "planner_configs": ["RRTConnectkConfigDefault"],
                "longest_valid_segment_fraction": fraction,
                "simplify_solutions": True, "interpolate": False,
            },
        }
        if workload == "B":
            adapters = ["default_planning_response_adapters/AddTimeOptimalParameterization"]
            if settings["smoothing"] == "ruckig":
                adapters.append("default_planning_response_adapters/AddRuckigTrajectorySmoothing")
            pipeline["response_adapters"] = adapters + ["default_planning_response_adapters/ValidateSolution"]
            pipeline["totg"] = {"path_tolerance": 0.1, "resample_dt": 0.01, "min_angle_change": 0.001}
        # Workload A: no response adapters (the key left out: ROS parameters cannot be empty lists).
        config = {
            "robot_description": (inputs / "panda.urdf").read_text(),
            "robot_description_semantic": (inputs / "panda.srdf").read_text(),
            "robot_description_planning": {"joint_limits": limits},
            "planning_pipelines": {"pipeline_names": ["ompl"]},
            "ompl": pipeline,
            "plan_request_params": {
                "planning_attempts": 1, "planning_pipeline": "ompl", "planner_id": "RRTConnectkConfigDefault",
                "planning_time": TIME_LIMIT, "max_velocity_scaling_factor": 1.0, "max_acceleration_scaling_factor": 1.0,
            },
            "planning_scene_monitor_options": {
                "name": "planning_scene_monitor", "robot_description": "robot_description",
                "joint_state_topic": "/joint_states", "attached_collision_object_topic": "/attached_collision_object",
                "publish_planning_scene_topic": "/publish_planning_scene",
                "monitored_planning_scene_topic": "/monitored_planning_scene", "wait_for_initial_state_timeout": 0.0,
            },
        }
        self.robot = MoveItPy(node_name=f"compare_{os.getpid()}", config_dict=config)
        self.arm = self.robot.get_planning_component("panda_arm")
        self.loaded = None

    def state(self, q):
        from moveit.core.robot_state import RobotState

        s = RobotState(self.robot.get_robot_model())
        s.joint_positions = dict(zip(ARM_JOINTS, map(float, q)))
        s.update()
        return s

    def plan(self, problem, seed):
        t0 = time.perf_counter()
        monitor = self.robot.get_planning_scene_monitor()
        with monitor.read_write() as scene:
            if problem["scene_ref"] != self.loaded:
                scene.remove_all_collision_objects()
                for o in moveit_objects([shape_from_json(s) for s in problem[self.settings["scene_key"]]]):
                    scene.apply_collision_object(o)
                    scene.allowed_collision_matrix.set_entry("panda_link0", o.id, True)
                self.loaded = problem["scene_ref"]
            scene.current_state = self.state(problem["start"])
        self.arm.set_start_state(robot_state=self.state(problem["start"]))
        self.arm.set_goal_state(robot_state=self.state(problem["goal"]))
        t1 = time.perf_counter()
        result = self.arm.plan()  # with `plan_request_params`
        t2 = time.perf_counter()
        stage = "search+post" if self.workload == "A" else "search+post+timing+validation"
        phases = {"setup": t1 - t0, stage: t2 - t1, "total": t2 - t0}
        if not result:
            return {"status": "failed", "reason": "no plan", "phases": phases}, None
        msg = result.trajectory.get_robot_trajectory_msg().joint_trajectory
        order = [msg.joint_names.index(j) for j in ARM_JOINTS]
        pts = msg.points
        q = np.array([[p.positions[i] for i in order] for p in pts], dtype=np.float32)
        if self.workload == "A":
            return {"status": "solved", "phases": phases}, {"path": q}
        t = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts], dtype=np.float32)
        out = {
            "path": q, "t": t, "q": q,
            "v": np.array([[p.velocities[i] for i in order] for p in pts], dtype=np.float32),
            "a": np.array([[p.accelerations[i] for i in order] for p in pts], dtype=np.float32),
        }
        return {"status": "solved", "phases": phases, "duration": float(t[-1])}, out


ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def moveit_objects(shapes):
    """Collision objects for MoveIt's planning scene: a capsule as a cylinder and two spheres."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    from bench_model import Box, Capsule, quaternion
    from parity import _rotation_to

    def pose(centre, rot):
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, centre)
        p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = map(float, quaternion(rot))
        return p

    objects = []
    for k, s in enumerate(shapes):
        o = CollisionObject()
        o.header.frame_id = "panda_link0"
        o.id = f"obstacle{k}"
        o.operation = CollisionObject.ADD
        if isinstance(s, Box):
            o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(d) for d in 2 * s.half])]
            o.primitive_poses = [pose(s.centre, s.rot)]
        elif isinstance(s, Capsule):
            axis = s.b - s.a
            o.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[float(np.linalg.norm(axis)), s.radius]),
                            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[s.radius]),
                            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[s.radius])]
            o.primitive_poses = [pose((s.a + s.b) / 2, _rotation_to(axis)), pose(s.a, np.eye(3)), pose(s.b, np.eye(3))]
        else:
            o.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[2 * s.half_height, s.radius])]
            o.primitive_poses = [pose(s.centre, s.rot)]
        objects.append(o)
    return objects


PLANNERS = {"roboplan": RoboPlan, "moveit": MoveIt}
# Planners with binaries of their own, which write the same records.
EXTERNAL = {"batchplan", "vamp", "ompl"}
# VAMP and OMPL from C++ (`scripts/compare_cpp.cc`), on the capsule scenes. VAMP's RRT-Connect range is
# its Panda's (1 rad, as its Python package plans) and its sample cap its default; OMPL checks motions
# every `segment_fraction` of the space's L2 extent (13.42 rad): 0.01 by default, matched 1/32 rad.
# batchplan's defaults keep 1.5 cm while searching and 1 cm while shortcutting; matched, it keeps none, as
# no other planner does. It checks edges at 1/32 rad either way.
SETTINGS["batchplan"] = {"defaults": {}, "matched": {"search_clearance": 0.0, "shortcut_clearance": 0.0}}
# Matched, VAMP stops sampling where batchplan does, at 10,000 samples (its default is 100,000).
SETTINGS["vamp"] = {"defaults": {"range": 1.0, "max_iterations": 0}, "matched": {"range": 1.0, "max_iterations": 10_000}}
SETTINGS["ompl"] = {"defaults": {"segment_fraction": 0.01}, "matched": {"segment_resolution": RESOLUTION}}
CPP_BINARY = Path(os.environ.get("COMPARE_CPP", "data/bench/cpp/compare_cpp"))
REPO = Path(__file__).resolve().parent.parent


def worker(index, cores, job, queries, results, barrier):
    if hasattr(os, "sched_setaffinity") and cores:
        os.sched_setaffinity(0, {cores[index % len(cores)]})
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[var] = "1"
    problems = problem_table(job["inputs"])
    try:
        t = time.perf_counter()
        planner = PLANNERS[job["planner"]](job["inputs"], job["settings"], job["workload"])
        setup = time.perf_counter() - t
        planner.plan(problems[job["warmup"]], 0)
    except Exception:
        import traceback

        results.put(("broken", index, traceback.format_exc()))
        return
    results.put(("ready", index, setup))
    barrier.wait()
    while True:
        item = queries.get()
        if item is None:
            break
        pid, number, seed, rep = item
        t = time.perf_counter()
        try:
            record, output = planner.plan(problems[pid], seed_for(seed, number))
        except Exception as e:  # a query a library refuses counts as a failure, with its reason
            record, output = {"status": "error", "reason": f"{type(e).__name__}: {e}"}, None
        record.setdefault("phases", {}).setdefault("total", time.perf_counter() - t)
        record.update({"problem": pid, "seed": seed, "rep": rep, "worker": index})
        results.put(("done", record, output if rep == 1 else None))


def run(args):
    inputs, out = Path(args.inputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = "problems.json" if args.set == "problems" else "queries_onescene.json"
    problems = json.loads((inputs / source).read_text())
    chosen = [(p["id"], k) for k, p in enumerate(problems) if p["split"] == args.split]
    if args.limit:
        chosen = chosen[:args.limit]
    warmup = next(p["id"] for p in problems if p["split"] == "tuning")
    if args.set == "onescene":
        # Warm up in a scene no measured query uses.
        warmup = next(p["id"] for p in json.loads((inputs / "problems.json").read_text()) if p["split"] == "final")
    cores = parse_cores(args.cores)
    workers = args.workers or len(cores) or 1
    job = {
        "planner": args.planner, "workload": args.workload, "family": args.family, "split": args.split,
        "scene": args.scene, "set": args.set,
        "settings": {**SETTINGS.get(args.planner, {}).get(args.family, {}), **({"mode": args.mode} if args.mode else {}),
                     **({"smoothing": args.smoothing} if args.smoothing else {})},
        "inputs": str(inputs), "warmup": warmup,
        "time_limit": TIME_LIMIT, "limits": LIMITS, "workers": workers, "cores": cores,
        "seeds": args.seeds, "reps": args.reps, "problems": len(chosen),
    }
    job["settings"]["scene_key"] = "scene_capsules" if args.scene == "capsules" else "scene"
    if args.planner == "roboplan":
        job["settings"]["limits_file"] = str(out / "roboplan_limits.yaml")
        roboplan_limits(job["settings"]["limits_file"])
    (out / "manifest.json").write_text(json.dumps({**job, **environment(inputs)}, indent=2))
    if args.planner in EXTERNAL:
        run_external(args, job, chosen, out)
        return
    records = open(out / "records.jsonl", "w")
    passes = open(out / "passes.jsonl", "w")
    ctx = mp.get_context("spawn")
    for rep in range(1, args.reps + 1):
        for seed in args.seeds:
            queries, results, barrier = ctx.Queue(), ctx.Queue(), ctx.Barrier(workers + 1)
            started = time.perf_counter()
            procs = [ctx.Process(target=worker, args=(i, cores, job, queries, results, barrier)) for i in range(workers)]
            for p in procs:
                p.start()
            setups = []
            for _ in range(workers):
                kind, index, detail = results.get()
                if kind == "broken":
                    for p in procs:
                        p.terminate()
                    raise RuntimeError(f"worker {index} could not start:\n{detail}")
                setups.append(detail)
            for pid, number in chosen:
                queries.put((pid, number, seed, rep))
            for _ in procs:
                queries.put(None)
            barrier.wait()
            released = time.perf_counter()
            outputs = {}
            for _ in chosen:
                kind, record, output = results.get()
                record.update({"planner": args.planner, "workload": args.workload, "family": args.family,
                               "mode": args.mode or None})
                records.write(json.dumps(record) + "\n")
                if output is not None:
                    for key, value in output.items():
                        outputs[f"{record['problem']}|{key}"] = value
            wall = time.perf_counter() - released
            for p in procs:
                p.join()
            passes.write(json.dumps({"rep": rep, "seed": seed, "problems": len(chosen), "wall": wall,
                                     "setup_wall": time.perf_counter() - started, "worker_setup": setups}) + "\n")
            records.flush()
            passes.flush()
            if outputs:
                write_outputs(out / f"outputs_seed{seed}", outputs)
            print(f"{args.planner} {args.workload} {args.family} rep {rep} seed {seed}: {len(chosen)} in {wall:.2f} s", flush=True)
    records.close()
    passes.close()


def cpp_problems(inputs, chosen, path):
    """The chosen problems' capsule scenes as `compare_cpp` reads them, a problem per line."""
    from bench_model import Box, euler_xyz_from

    by_id = problem_table(inputs)
    lines = []
    for pid, _ in chosen:
        p = by_id[pid]
        shapes = [shape_from_json(s) for s in p["scene_capsules"]]
        boxes = [s for s in shapes if isinstance(s, Box)]
        capsules = [s for s in shapes if not isinstance(s, Box)]
        fields = [pid] + [repr(float(v)) for v in p["start"] + p["goal"]] + [str(len(boxes))]
        for b in boxes:
            fields += [repr(float(v)) for v in [*b.centre, *euler_xyz_from(b.rot), *b.half]]
        fields.append(str(len(capsules)))
        for c in capsules:
            fields += [repr(float(v)) for v in [*c.a, *c.b, c.radius]]
        lines.append(" ".join(fields))
    Path(path).write_text("\n".join(lines) + "\n")


def run_cpp(args, job, chosen, out):
    """VAMP or OMPL: one `compare_cpp` process per pass (OMPL's RNG is seeded once per process)."""
    from bench_model import Panda

    cpp_problems(job["inputs"], chosen, out / "problems.txt")
    canonical = Panda(Path(job["inputs"]) / "panda.urdf", Path(job["inputs"]) / "panda.srdf")
    extent = float(np.linalg.norm(canonical.upper - canonical.lower))
    s = job["settings"]
    fraction = s.get("segment_fraction") or s.get("segment_resolution", RESOLUTION) / extent
    cores = ",".join(map(str, job["cores"])) or "0"
    for rep in range(1, args.reps + 1):
        for seed in args.seeds:
            subprocess.run([str(CPP_BINARY), str(out / "problems.txt"), str(out), args.planner, str(seed), str(rep),
                            str(job["workers"]), cores, str(s.get("range", 1.0)), str(s.get("max_iterations", 0)),
                            str(TIME_LIMIT), repr(fraction)], check=True)
            print(f"{args.planner} {args.workload} {args.family} rep {rep} seed {seed}: done", flush=True)


def run_external(args, job, chosen, out):
    """A planner with its own binary: its job file, run on the given cores."""
    if args.planner in ("vamp", "ompl"):
        assert args.workload == "A", "VAMP and OMPL return geometric paths: workload A only"
        run_cpp(args, job, chosen, out)
        return
    job = {**job, "problems": chosen, "threads": job["workers"],
           "settings": {k: v for k, v in job["settings"].items() if isinstance(v, (int, float))}}
    (out / "job.json").write_text(json.dumps(job))
    binary = REPO / "target" / "release" / "examples" / "compare"
    command = [str(binary), "run", str(out / "job.json"), str(out)]
    if job["cores"] and sys.platform.startswith("linux"):
        command = ["taskset", "-c", ",".join(map(str, job["cores"]))] + command
    subprocess.run(command, check=True)


def write_outputs(stem, outputs):
    """Arrays as one float32 file (`stem.bin`) and an index (`stem.json`) of key: [offset, rows, cols]."""
    index, offset = {}, 0
    with open(f"{stem}.bin", "wb") as f:
        for key, value in outputs.items():
            a = np.ascontiguousarray(value, dtype="<f4")
            a = a.reshape(len(a), -1) if a.ndim > 1 else a.reshape(-1, 1)
            f.write(a.tobytes())
            index[key] = [offset, a.shape[0], a.shape[1]]
            offset += a.size
    Path(f"{stem}.json").write_text(json.dumps(index))


def read_outputs(stem):
    """The arrays `write_outputs` wrote, as {key: array}; vectors come back with one column."""
    data = np.fromfile(f"{stem}.bin", dtype="<f4")
    return {k: data[o:o + r * c].reshape(r, c) for k, (o, r, c) in json.loads(Path(f"{stem}.json").read_text()).items()}


def parse_cores(text):
    cores = []
    for part in (text or "").split(","):
        if "-" in part:
            a, b = part.split("-")
            cores += list(range(int(a), int(b) + 1))
        elif part:
            cores.append(int(part))
    return cores


def environment(inputs):
    """What the run ran on: the host, the inputs' checksums and the libraries' versions."""
    import importlib.metadata as md

    info = {
        "host": {"machine": platform.machine(), "processor": platform.processor(), "system": platform.platform(),
                 "logical_cpus": os.cpu_count(), "python": platform.python_version()},
        "inputs": json.loads((Path(inputs) / "manifest.json").read_text()),
        "packages": {},
    }
    # The exact sources that ran (the benchmark host's checkout arrives by rsync, not git).
    import hashlib

    digest = hashlib.sha256()
    files = sorted([*REPO.glob("src/**/*"), REPO / "Cargo.toml", REPO / "Cargo.lock", REPO / "examples/compare.rs",
                    *[REPO / "scripts" / n for n in ["bench_model.py", "compare.py", "validate.py", "compare_cpp.cc",
                                                     "parity.py", "run_comparison.sh"]]])
    for f in files:
        if f.is_file():
            digest.update(str(f.relative_to(REPO)).encode() + b"\0" + f.read_bytes())
    info["source_tree_sha256"] = digest.hexdigest()
    try:
        info["batchplan_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip()
        info["batchplan_dirty"] = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "src"],
                                                      capture_output=True, text=True).stdout.strip())
    except FileNotFoundError:
        pass
    for name in ["roboplan", "roboplan-core", "roboplan-rrt", "roboplan-toppra", "numpy", "vamp-planner", "ompl"]:
        try:
            info["packages"][name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    for path, key in [("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "governor"),
                      ("/sys/firmware/acpi/platform_profile", "platform_profile"), ("/proc/loadavg", "loadavg")]:
        if os.path.exists(path):
            info["host"][key] = Path(path).read_text().strip()
    try:
        info["host"]["cpu"] = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        pass
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("inputs")
    r.add_argument("out")
    r.add_argument("--planner", required=True, choices=sorted(PLANNERS) + sorted(EXTERNAL))
    r.add_argument("--workload", required=True, choices=["A", "B"])
    r.add_argument("--family", default="defaults", choices=["defaults", "matched"])
    r.add_argument("--split", default="tuning", choices=["tuning", "heldout", "final"])
    r.add_argument("--set", default="problems", choices=["problems", "onescene"],
                   help="independent problems, each in its own scene, or repeated queries in 20 scenes")
    r.add_argument("--scene", default="exact", choices=["exact", "capsules"],
                   help="the problems' exact cylinders, or the capsules around them (VAMP's and OMPL's)")
    r.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3, 4, 5])
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--cores", default="")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--mode", default="", choices=["", *MODES], help="workload B: the TOPP-RA mode")
    r.add_argument("--smoothing", default="", choices=["", "totg", "ruckig"], help="workload B: MoveIt's smoothing")
    r.add_argument("--limit", type=int, default=0, help="only the first N problems (a smoke test)")
    args = parser.parse_args()
    assert (args.workload == "B") == bool(args.mode) or args.planner != "roboplan", "RoboPlan's workload B needs --mode"
    assert args.scene == "capsules" or args.planner not in ("vamp", "ompl"), "VAMP and OMPL take capsules: --scene capsules"
    run(args)
