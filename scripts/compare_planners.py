"""Plans the robometrics Panda problems (MotionBenchMaker and MπNets, from scripts/fetch_benchmark.sh)
with other planners, from each start to the set's first IK goal, as `examples/benchmark.rs` does in
its "plan" mode, and writes every path for `cargo run --release --example benchmark -- --check`
to judge with the same checks as batchplan's.

- VAMP (vamp-planner): RRT-Connect, then its path simplification, on its own SIMD collision checker
  and Panda spheres. Cylinders become capsules around them as VAMP's own MotionBenchMaker loader does
  (boxes in the "box" set). Every problem's floor or table top is at the base's height, and VAMP checks
  its base sphere, which reaches 3 cm below it, against the scene; batchplan, RoboPlan, MoveIt 2 and
  the judge leave the base out. So the boxes under the base get a hole around it, where no other link
  can reach, and VAMP plans on the scene the judge checks everywhere else.
- OMPL (ompl 2.0): RRT-Connect, then `simplifySolution`, with VAMP's collision checker as the state
  validity checker, called from Python.
- RoboPlan (roboplan 0.7): RRT-Connect, then its path shortcutter, with its default settings, on
  Coal's collision checks of the Panda's own collision meshes against exact boxes and cylinders.
  Groups and allowed collisions come from MoveIt's Panda SRDF (moveit_resources, fetched at a
  pinned commit). The base link is not checked against the scene, as batchplan does not check it,
  and obstacles are not checked against each other.
- cuRobo (curobo 0.8, NVIDIA GPU): `MotionPlanner.plan_cspace`, one problem at a time, with its
  default 5 attempts and graph seeding from the second (`curobo`); and `BatchMotionPlanner`,
  `CUROBO_BATCH` problems at a time, each in its own world (`curobo-batch`). cuRobo's own benchmark
  allows 100 attempts, but on a T4 each problem that fails them takes about 45 s.
  Both on cuRobo's Panda spheres, which batchplan's are. Boxes stay oriented boxes and cylinders
  become meshes (`get_collision_check_world`), so the scenes are exact; as boxes, the cylinders that
  are grasped close the gap between the fingers at the goal. Timings include loading each problem's
  world.
- MoveIt 2 (Jazzy, the `moveit/moveit2:jazzy-release` image): `moveit_py` planning with the
  default OMPL pipeline (RRT-Connect, simplification, time parameterization, validation) and FCL collision
  checks of the Panda's meshes, configured by `moveit_resources_panda_moveit_config`. Obstacles
  join the planning scene as boxes and cylinders; the base may touch them.

Each problem runs on one core; problems run in parallel on every core, so the wall time per set
gives throughput and each problem's planning time gives latency.

    uv venv .venv-bench --python 3.12
    uv pip install --python .venv-bench/bin/python vamp-planner==0.6.4 ompl==2.0.1 pyyaml numpy
    .venv-bench/bin/python scripts/compare_planners.py data/robometrics data/compare.json vamp ompl
    # VAMP and OMPL from C++ (scripts/cpp_planners.cc) on the same scenes
    .venv-bench/bin/python scripts/compare_planners.py data/robometrics data/vamp_problems.txt vamp-export
    uv venv .venv-roboplan --python 3.12
    uv pip install --python .venv-roboplan/bin/python roboplan==0.7.0 pyyaml numpy
    .venv-roboplan/bin/python scripts/compare_planners.py data/robometrics data/compare_roboplan.json roboplan
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/batchplan" -w /batchplan \
      moveit/moveit2:jazzy-release bash -c \
      'source /opt/ros/jazzy/setup.bash && python3 scripts/compare_planners.py data/robometrics data/compare_moveit.json moveit'
    # on an NVIDIA machine, in a venv with torch and curobo 0.8 (`pip install -e .[cu12-torch]`)
    python scripts/compare_planners.py data/robometrics data/compare_curobo.json curobo curobo-batch
"""

import hashlib
import json
import math
import multiprocessing
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

OMPL_TIME = 5.0  # seconds to plan, then 1 to simplify
ROBOPLAN_TIME = 5.0  # seconds to plan
MOVEIT_TIME = 5.0  # seconds to plan
VAMP_RANGE = 1.0  # VAMP's Panda range (`vamp.constants.ROBOT_RRT_RANGES`); `RRTCSettings()` defaults to 2
CUROBO_ATTEMPTS = 5  # one problem at a time, `plan_cspace`'s default
CUROBO_BATCH = 32  # problems planned together, each in its own world
CUROBO_BATCH_ATTEMPTS = 5  # passes over a batch, until every problem in it succeeds

DATA = None  # the problems and VAMP's spheres, given on the command line (`use_data`)
PANDA_URDF = Path(__file__).resolve().parent.parent / "assets" / "franka" / "franka_panda.urdf"
PANDA_SRDF = (
    "https://raw.githubusercontent.com/moveit/moveit_resources/049f2db326565044f3588076921c02443bd44036/"
    "panda_moveit_config/config/panda.srdf",
    "1150719ea9d81139418198a50faea17e155323547d056c4edcb7ecc82fd8d317",
)


def euler_xyz(qw, qx, qy, qz):
    """VAMP's orientation: extrinsic x, y, z angles, R = Rz Ry Rx."""
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    r10 = 2 * (qx * qy + qw * qz)
    r00 = 1 - 2 * (qy * qy + qz * qz)
    return [math.atan2(r21, r22), -math.asin(max(-1.0, min(1.0, r20))), math.atan2(r10, r00)]


def rotation(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


_vamp_base = None


def vamp_base():
    """VAMP's spheres on the Panda's base, `panda_link0` (centre, radius), from the spherized URDF
    that `scripts/fetch_benchmark.sh` fetches with the problems."""
    global _vamp_base
    if _vamp_base is None:
        import xml.etree.ElementTree as ET

        link = ET.parse(DATA / "vamp_panda_spherized.urdf").find("link[@name='panda_link0']")
        _vamp_base = []
        for c in link.iter("collision"):
            centre = np.array([float(v) for v in c.find("origin").get("xyz").split()])
            _vamp_base.append((centre, float(c.find("geometry/sphere").get("radius"))))
    return _vamp_base


def around_base(centre, rot, half):
    """A box, rotated by `rot`, as boxes (centre, half extents) with the same rotation that leave a
    hole through it, along its z axis and a centimetre wider than each of VAMP's base spheres it
    touches."""
    boxes = [(np.array(centre), np.array(half))]
    for b, r in vamp_base():
        kept = []
        for c, h in boxes:
            local = rot.T @ (b - c)
            if np.linalg.norm(local - np.clip(local, -h, h)) >= r:
                kept.append((c, h))
                continue
            assert abs(rot[2, 2]) > 0.999, "only boxes lying flat are cut around the base"
            lo, hi = np.maximum(local[:2] - r - 0.01, -h[:2]), np.minimum(local[:2] + r + 0.01, h[:2])
            # Strips either side in x, then either side in y between them.
            for x0, x1, y0, y1 in [(-h[0], lo[0], -h[1], h[1]), (hi[0], h[0], -h[1], h[1]),
                                   (lo[0], hi[0], -h[1], lo[1]), (lo[0], hi[0], hi[1], h[1])]:
                if x1 > x0 and y1 > y0:
                    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
                    kept.append((c + rot @ mid, np.array([(x1 - x0) / 2, (y1 - y0) / 2, h[2]])))
        boxes = kept
    return boxes


def vamp_shapes(problem, set_name):
    """The problem's obstacles as VAMP takes them, with holes around its base (see `around_base`):
    cuboids (centre, Euler XYZ, half extents) and capsules (end, end, radius)."""
    cuboids, capsules = [], []
    for c in (problem["obstacles"].get("cuboid") or {}).values():
        x, y, z, qw, qx, qy, qz = c["pose"]
        for centre, half in around_base([x, y, z], rotation(qw, qx, qy, qz), [d / 2 for d in c["dims"]]):
            cuboids.append((list(centre), euler_xyz(qw, qx, qy, qz), list(half)))
    for c in (problem["obstacles"].get("cylinder") or {}).values():
        x, y, z, qw, qx, qy, qz = c["pose"]
        if set_name.startswith("box"):
            cuboids.append(([x, y, z], euler_xyz(qw, qx, qy, qz), [c["radius"], c["radius"], c["height"] / 2]))
        else:
            # The cylinder's axis is its local z.
            half = rotation(qw, qx, qy, qz)[:, 2] * c["height"] / 2
            centre = np.array([x, y, z])
            capsules.append((list(centre - half), list(centre + half), c["radius"]))
    return cuboids, capsules


def environment(problem, set_name):
    """The problem's obstacles for VAMP (`vamp_shapes`)."""
    import vamp

    env = vamp.Environment()
    cuboids, capsules = vamp_shapes(problem, set_name)
    for centre, angles, half in cuboids:
        env.add_cuboid(vamp.Cuboid(centre, angles, half))
    for a, b, radius in capsules:
        env.add_capsule(vamp.Cylinder(a, b, radius))
    return env


def export_vamp(sets, out):
    """Every problem's start, goal and VAMP scene, as numbers `scripts/cpp_planners.cc` reads."""
    lines = [str(len(sets))]
    for name, problems in sets.items():
        lines.append(f"{name} {len(problems)}")
        for p in problems:
            cuboids, capsules = vamp_shapes(p, name)
            real = lambda values: [repr(float(v)) for v in values]
            fields = real([*p["start"], *p["goal_ik"][0]]) + [str(len(cuboids))]
            fields += real(v for centre, angles, half in cuboids for v in [*centre, *angles, *half])
            fields += [str(len(capsules))] + real(v for a, b, radius in capsules for v in [*a, *b, radius])
            lines.append(" ".join(fields))
    out.write_text("\n".join(lines) + "\n")


def plan_vamp(job):
    import vamp

    set_name, problem = job
    env = environment(problem, set_name)
    start, goal = problem["start"], problem["goal_ik"][0]
    rng = vamp.panda.halton()
    settings = vamp.RRTCSettings()
    settings.range = VAMP_RANGE
    result = vamp.panda.rrtc(start, goal, env, settings, rng)
    if not result.solved:
        return {"solved": False, "seconds": result.nanoseconds * 1e-9}
    simplified = vamp.panda.simplify(result.path, env, vamp.SimplifySettings(), rng)
    seconds = (result.nanoseconds + simplified.nanoseconds) * 1e-9
    return {"solved": True, "seconds": seconds, "path": simplified.path.numpy().tolist()}


def plan_ompl(job):
    import vamp
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    ou.setLogLevel(ou.LogLevel.LOG_WARN)
    set_name, problem = job
    env = environment(problem, set_name)
    space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    for j, (lo, hi) in enumerate(zip(vamp.panda.lower_bounds(), vamp.panda.upper_bounds())):
        bounds.setLow(j, float(lo))
        bounds.setHigh(j, float(hi))
    space.setBounds(bounds)
    setup = og.SimpleSetup(space)
    setup.setStateValidityChecker(lambda s: vamp.panda.validate([s[j] for j in range(7)], env))
    setup.setPlanner(og.RRTConnect(setup.getSpaceInformation()))
    start, goal = space.allocState(), space.allocState()
    for j in range(7):
        start[j], goal[j] = problem["start"][j], problem["goal_ik"][0][j]
    setup.setStartAndGoalStates(start, goal)
    t = time.perf_counter()
    solved = bool(setup.solve(OMPL_TIME)) and setup.haveExactSolutionPath()
    if not solved:
        return {"solved": False, "seconds": time.perf_counter() - t}
    setup.simplifySolution(1.0)
    seconds = time.perf_counter() - t
    path = setup.getSolutionPath()
    states = [[path.getState(i)[j] for j in range(7)] for i in range(path.getStateCount())]
    return {"solved": True, "seconds": seconds, "path": states}


_roboplan_scene = None


def roboplan_scene():
    """The Panda, loaded once per worker, with its fingers open as batchplan's benchmark holds them."""
    global _roboplan_scene
    if _roboplan_scene is None:
        from roboplan.core import Scene, loadUrdfSceneDescriptionFromXml

        url, digest = PANDA_SRDF
        srdf = urllib.request.urlopen(url).read()
        assert hashlib.sha256(srdf).hexdigest() == digest, "unexpected SRDF"
        # RoboPlan wants frame names unique across links and joints; this URDF names a joint after its link.
        urdf = PANDA_URDF.read_text().replace('joint name="right_gripper"', 'joint name="right_gripper_joint"')
        scene = Scene("panda", loadUrdfSceneDescriptionFromXml(urdf, [str(PANDA_URDF.parent)]))
        scene.importSrdf(srdf.decode())
        q = scene.getCurrentJointPositions()
        for name in ["panda_finger_joint1", "panda_finger_joint2"]:
            q[scene.getJointPositionIndices([name])[0]] = 0.04
        scene.setJointPositions(q)
        _roboplan_scene = scene
    return _roboplan_scene


def plan_roboplan(job):
    from roboplan.core import Box, Cylinder, JointConfiguration, PathShortcutter, PathShortcuttingOptions
    from roboplan.rrt import RRT, RRTOptions

    set_name, problem = job
    scene = roboplan_scene()
    names = []
    for i, (kind, c) in enumerate(
        [("cuboid", c) for c in (problem["obstacles"].get("cuboid") or {}).values()]
        + [("cylinder", c) for c in (problem["obstacles"].get("cylinder") or {}).values()]
    ):
        x, y, z, qw, qx, qy, qz = c["pose"]
        pose = np.eye(4)
        pose[:3, :3] = rotation(qw, qx, qy, qz)
        pose[:3, 3] = [x, y, z]
        name = f"obstacle{i}"
        color = np.array([0.5, 0.5, 0.5, 1.0])
        if kind == "cuboid":
            scene.addBoxGeometry(name, "universe", Box(*c["dims"]), pose, color)
        else:
            scene.addCylinderGeometry(name, "universe", Cylinder(c["radius"], c["height"]), pose, color)
        names.append(name)
    # Obstacles do not move, so they need not be checked against each other or against the base.
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]] + [(n, "panda_link0") for n in names]
    scene.setCollisions(pairs, False)
    try:
        options = RRTOptions(group_name="panda_arm", rrt_connect=True, max_nodes=100_000, max_planning_time=ROBOPLAN_TIME)
        rrt = RRT(scene, options)
        rrt.setRngSeed(1)
        start, goal = JointConfiguration(), JointConfiguration()
        start.positions = np.array(problem["start"])
        goal.positions = np.array(problem["goal_ik"][0])
        t = time.perf_counter()
        try:
            path = rrt.plan(start, goal)
        except RuntimeError:
            path = None
        if path is None:
            return {"solved": False, "seconds": time.perf_counter() - t}
        path = PathShortcutter(scene, PathShortcuttingOptions(group_name="panda_arm")).shortcut(path)
        seconds = time.perf_counter() - t
        return {"solved": True, "seconds": seconds, "path": [list(map(float, q)) for q in path.positions]}
    finally:
        for name in names:
            scene.removeGeometry(name)


_moveit = None


def moveit():
    """MoveItPy for the Panda, started once per worker with its own node."""
    global _moveit
    if _moveit is None:
        import rclpy
        from moveit.planning import MoveItPy
        from moveit_configs_utils import MoveItConfigsBuilder

        rclpy.init()
        config = (
            MoveItConfigsBuilder("moveit_resources_panda")
            .robot_description(file_path="config/panda.urdf.xacro")
            .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
            .to_dict()
        )
        config["planning_scene_monitor_options"] = {
            "name": "planning_scene_monitor",
            "robot_description": "robot_description",
            "joint_state_topic": "/joint_states",
            "attached_collision_object_topic": "/attached_collision_object",
            "publish_planning_scene_topic": "/publish_planning_scene",
            "monitored_planning_scene_topic": "/monitored_planning_scene",
            "wait_for_initial_state_timeout": 0.0,
        }
        config["planning_pipelines"] = {"pipeline_names": ["ompl"]}
        config["plan_request_params"] = {
            "planning_attempts": 1,
            "planning_pipeline": "ompl",
            "planner_id": "RRTConnectkConfigDefault",
            "planning_time": MOVEIT_TIME,
            "max_velocity_scaling_factor": 1.0,
            "max_acceleration_scaling_factor": 1.0,
        }
        robot = MoveItPy(node_name=f"compare_planners_{os.getpid()}", config_dict=config)
        _moveit = (robot, robot.get_planning_component("panda_arm"))
    return _moveit


def plan_moveit(job):
    from geometry_msgs.msg import Pose
    from moveit.core.robot_state import RobotState
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    set_name, problem = job
    robot, arm = moveit()
    monitor = robot.get_planning_scene_monitor()
    arm_joints = [f"panda_joint{j}" for j in range(1, 8)]

    def state(q):
        s = RobotState(robot.get_robot_model())
        s.joint_positions = {**dict(zip(arm_joints, q)), "panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
        s.update()
        return s

    objects = []
    for kind, shapes in [("cuboid", problem["obstacles"].get("cuboid")), ("cylinder", problem["obstacles"].get("cylinder"))]:
        for name, c in (shapes or {}).items():
            o = CollisionObject()
            o.header.frame_id = "panda_link0"
            o.id = name
            box = kind == "cuboid"
            primitive = SolidPrimitive(type=SolidPrimitive.BOX if box else SolidPrimitive.CYLINDER)
            primitive.dimensions = list(map(float, c["dims"])) if box else [float(c["height"]), float(c["radius"])]
            x, y, z, qw, qx, qy, qz = c["pose"]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = map(float, (qw, qx, qy, qz))
            o.primitives, o.primitive_poses, o.operation = [primitive], [pose], CollisionObject.ADD
            objects.append(o)
    with monitor.read_write() as scene:
        scene.remove_all_collision_objects()
        for o in objects:
            scene.apply_collision_object(o)
            scene.allowed_collision_matrix.set_entry("panda_link0", o.id, True)
        scene.current_state = state(problem["start"])
    arm.set_start_state(robot_state=state(problem["start"]))
    arm.set_goal_state(robot_state=state(problem["goal_ik"][0]))
    t = time.perf_counter()
    result = arm.plan()
    seconds = time.perf_counter() - t
    if not result:
        return {"solved": False, "seconds": seconds}
    points = result.trajectory.get_robot_trajectory_msg().joint_trajectory
    order = [points.joint_names.index(j) for j in arm_joints]
    return {"solved": True, "seconds": seconds, "path": [[float(p.positions[i]) for i in order] for p in points.points]}


def use_data(data):
    global DATA
    DATA = data


def run_curobo(sets, batch):
    """cuRobo on the GPU, in this process: every set one problem at a time (`MotionPlanner`), or
    `CUROBO_BATCH` at a time, each problem in its own world (`BatchMotionPlanner`). Each problem's
    time includes loading its world; a batch's time is shared among its problems."""
    import copy

    import torch
    from curobo import runtime
    from curobo._src.geom.types import SceneCfg
    from curobo._src.motion import MotionPlanner, MotionPlannerCfg
    from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
    from curobo._src.state.state_joint import JointState

    # Joint-space goals shape cuRobo's buffers differently from its pose-goal warmup, and its CUDA
    # graphs may only be captured again with this on (CUDA 12+).
    runtime.cuda_graph_reset = True
    # CUROBO_PROBLEMS=n plans only each set's first n problems (a smoke test).
    limit = int(os.environ.get("CUROBO_PROBLEMS", "0")) or None
    sets = {name: problems[:limit] for name, problems in sets.items()}
    scenes = {
        (name, i): SceneCfg.create(copy.deepcopy(p["obstacles"])).get_collision_check_world()
        for name, problems in sets.items()
        for i, p in enumerate(problems)
    }
    cache = {
        "obb": max(len(scene.cuboid or []) for scene in scenes.values()),
        "mesh": max(len(scene.mesh or []) for scene in scenes.values()),
    }
    first = next(iter(scenes.values()))
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=[{}] * CUROBO_BATCH if batch else first,  # a list gives each problem its own world
        collision_cache={kind: n for kind, n in cache.items() if n > 0},
        max_batch_size=CUROBO_BATCH if batch else 1,
        multi_env=batch,
        use_cuda_graph=True,
    )
    planner = BatchMotionPlanner(config) if batch else MotionPlanner(config)
    planner.warmup(enable_graph=not batch)
    n = len(planner.joint_names)

    def state(qs):
        return JointState.from_position(planner.device_cfg.to_device(qs), joint_names=planner.joint_names)

    def arm(trajectory):
        """The planned joints' positions; cuRobo appends the locked fingers."""
        names = trajectory.joint_names or planner.joint_names
        return trajectory.position[..., [names.index(j) for j in planner.joint_names]]

    def path(position):
        return [list(map(float, q)) for q in position.reshape(-1, n).cpu().tolist()]

    # Joint-space planning warmed up too, untimed, as cuRobo's benchmark plans its first problem thrice.
    name, problems = next(iter(sets.items()))
    for _ in range(3):
        if batch:
            for env in range(CUROBO_BATCH):
                planner.scene_collision_checker.load_collision_model(scenes[(name, 0)], env_idx=env)
            goals, starts = [problems[0]["goal_ik"][0]] * CUROBO_BATCH, [problems[0]["start"]] * CUROBO_BATCH
            planner.plan_cspace(state(goals), state(starts), max_attempts=1)
        else:
            planner.scene_collision_checker.clear_cache()
            planner.update_world(scenes[(name, 0)])
            planner.plan_cspace(state([problems[0]["goal_ik"][0]]), state([problems[0]["start"]]), max_attempts=1)
    torch.cuda.synchronize()

    results = {}
    for name, problems in sets.items():
        attempts, wall = [], 0.0
        if not batch:
            for i, p in enumerate(problems):
                torch.cuda.synchronize()
                t = time.perf_counter()
                planner.scene_collision_checker.clear_cache()
                planner.update_world(scenes[(name, i)])
                planner.reset_seed()
                try:
                    r = planner.plan_cspace(
                        state([p["goal_ik"][0]]), state([p["start"]]), max_attempts=CUROBO_ATTEMPTS, enable_graph_attempt=1
                    )
                except Exception as e:  # a problem cuRobo refuses counts as unsolved
                    print(f"curobo {name} {i}: {e}", flush=True)
                    r = None
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t
                wall += seconds
                solved = r is not None and bool(r.success.reshape(-1)[0])
                q = path(arm(r.get_interpolated_plan())) if solved else None
                attempts.append({"solved": solved, "seconds": seconds, "path": q})
        else:
            for c in range(0, len(problems), CUROBO_BATCH):
                chunk = list(range(c, min(c + CUROBO_BATCH, len(problems))))
                padded = chunk + [chunk[-1]] * (CUROBO_BATCH - len(chunk))
                torch.cuda.synchronize()
                t = time.perf_counter()
                for env, i in enumerate(padded):
                    planner.scene_collision_checker.load_collision_model(scenes[(name, i)], env_idx=env)
                planner.reset_seed()
                try:
                    r = planner.plan_cspace(
                        state([problems[i]["goal_ik"][0] for i in padded]),
                        state([problems[i]["start"] for i in padded]),
                        max_attempts=CUROBO_BATCH_ATTEMPTS,
                    )
                except Exception as e:  # a batch cuRobo refuses counts as unsolved
                    print(f"curobo-batch {name} {c}: {e}", flush=True)
                    r = None
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t
                wall += seconds
                success = r.success.reshape(CUROBO_BATCH, -1)[:, 0] if r is not None else None
                trajectory = arm(r.interpolated_trajectory) if r is not None else None
                if trajectory is not None and trajectory.dim() == 4:
                    trajectory = trajectory[:, 0]
                last = r.interpolated_last_tstep if r is not None else None
                for env in range(len(chunk)):
                    solved = success is not None and bool(success[env])
                    q = None
                    if solved:
                        end = int(last.reshape(-1)[env]) + 1 if last is not None else trajectory.shape[1]
                        q = path(trajectory[env, :end])
                    attempts.append({"solved": solved, "seconds": seconds / len(chunk), "path": q})
        results[name] = {"wall_seconds": wall, "problems": attempts}
        solved = sum(a["solved"] for a in attempts)
        print(f"{'curobo-batch' if batch else 'curobo'} {name}: {solved}/{len(problems)} in {wall:.2f} s", flush=True)
    return results


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    use_data(data)
    chosen = sys.argv[3:] or ["vamp", "ompl"]
    sets = {}
    for name in ["mb_set.yaml", "mpinets_set.yaml"]:
        sets.update(yaml.safe_load((data / name).read_text()))
    results = {"machine": f"{platform.processor() or platform.machine()}, {os.cpu_count()} threads", "planners": {}}
    if chosen == ["vamp-export"]:
        export_vamp(sets, out)
        return
    # cuRobo plans on the GPU from this process.
    for planner in [name for name in chosen if name.startswith("curobo")]:
        results["planners"][planner] = run_curobo(sets, batch=planner == "curobo-batch")
    chosen = [name for name in chosen if not name.startswith("curobo")]
    with multiprocessing.Pool(initializer=use_data, initargs=(data,)) as pool:
        planners = {"vamp": plan_vamp, "ompl": plan_ompl, "roboplan": plan_roboplan, "moveit": plan_moveit}
        for planner, plan in [(name, planners[name]) for name in chosen]:
            results["planners"][planner] = {}
            for set_name, problems in sets.items():
                t = time.perf_counter()
                done = pool.map(plan, [(set_name, p) for p in problems], chunksize=1)
                wall = time.perf_counter() - t
                results["planners"][planner][set_name] = {"wall_seconds": wall, "problems": done}
                solved = sum(r["solved"] for r in done)
                print(f"{planner} {set_name}: {solved}/{len(problems)} in {wall:.2f} s", flush=True)
    out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
