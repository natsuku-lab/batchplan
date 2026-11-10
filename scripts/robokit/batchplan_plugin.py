"""batchplan as a planner plugin of robokit's motion-plan benchmark (`benchmark.py`'s plugin
interface): their datasets, warmups, timer placement, judge and summary, with batchplan planning.

Each problem is a pose goal: one timed `plan` call solves IK from the start, plans to the nearest
collision-free solution (optimizing `BATCHPLAN_SEEDS` seeds, 0 by default), and the path is timed
within the judge's limits (1% under them), sampled at most 25 ms apart (or as `BATCHPLAN_FRAMES`
samples per plan). Obstacles are the boxes robokit's planner and the judge use
(`obstacles_to_box_data`: cuboids, and cylinders as cuRobo's boxes). The robot is batchplan's Panda
with cuRobo's spheres (assets/franka), fingers locked at each split's judged opening.
"""

import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import batchplan as bp
from scripts.benchmark.motion_plan.benchmark import (
    MotionPlanMetrics,
    obstacles_to_box_data,
    trajectory_derivatives,
)

ASSETS = Path(os.environ.get("BATCHPLAN_ASSETS", "../batchplan/assets"))
DT = 0.025  # the judge's interpolation step
SEEDS = int(os.environ.get("BATCHPLAN_SEEDS", "0"))  # optimized seeds per problem
# Under the judge's velocity, acceleration (15) and jerk (500) limits: its 5-point stencils on
# 25 ms samples read a few trajectories timed at the limits themselves as past them.
MARGIN = float(os.environ.get("BATCHPLAN_MARGIN", "0.99"))
# The jerk limit trajectories are timed under: the judge's 500, or less for smoother motions.
JERK = float(os.environ.get("BATCHPLAN_JERK", "500"))
# The share of the judge's acceleration limit trajectories are timed under: less reads smoother to
# the judge's five-point jerk for less time than a lower jerk limit costs.
ACCELERATION = float(os.environ.get("BATCHPLAN_ACCELERATION", "1"))
# Samples per plan, as RoboKit returns a fixed count (31); 0 samples at most 25 ms apart.
FRAMES = int(os.environ.get("BATCHPLAN_FRAMES", "0"))

# Goal IK as 64 seeds of 30 steps of at most 1 rad, written out so the run does not depend on the
# library's defaults.
OPTIONS = bp.PlanOptions(seeds=SEEDS, goal_ik=bp.IkOptions(seeds=64, iterations=30, max_step=1.0))

_robots = {}


def _robot(finger: float, kind: str):
    key = (finger, kind)
    if key not in _robots:
        options = bp.RobotOptions(
            ee_links=["panda_hand"],
            lock_joints={"panda_finger_joint1": finger, "panda_finger_joint2": finger},
            collision_model=bp.CollisionModel.load(ASSETS / "franka/panda_collision.json"),
            max_acceleration=15.0,
            max_jerk=JERK,
            velocity_scale=MARGIN,
            acceleration_scale=MARGIN * ACCELERATION,
            jerk_scale=MARGIN,
        )
        robot = bp.Robot.load(ASSETS / "franka/franka_panda.urdf", options)
        device = bp.Device.gpu(robot) if kind == "gpu" else bp.Device.cpu(robot)
        device.warmup()
        _robots[key] = (robot, device)
    return _robots[key]


def _world(obstacles: dict) -> bp.World:
    half_extents, poses, _ = obstacles_to_box_data(obstacles)
    boxes = []
    for half, pose in zip(half_extents, poses, strict=True):
        rotation = Rotation.from_matrix(pose[:3, :3]).as_quat()  # x, y, z, w
        frame = bp.Pose(pose[:3, 3], rotation / np.linalg.norm(rotation))
        boxes.append(bp.Obstacle.cuboid((0.0, 0.0, 0.0), half).placed(frame))
    return bp.World(boxes)


def _plan(robot, device, worlds, problem):
    """One timed pose-goal plan: ((samples, their steps, a step's seconds) or None, seconds)."""
    start = np.asarray(problem["start"], dtype=np.float32)
    position = np.asarray(problem["goal_pose"]["position_xyz"], dtype=np.float32)
    w, x, y, z = problem["goal_pose"]["quaternion_wxyz"]
    goal = bp.Pose(position, (x, y, z, w))
    begin = time.perf_counter()
    plans = bp.plan(device, worlds, [bp.PlanProblem(0, start, goal)], OPTIONS)
    path = plans.best(0)
    if path is None:
        return None, time.perf_counter() - begin
    # Sampled at the largest step of at most 25 ms that divides its duration, so the samples end
    # where the motion does rather than up to a step later, or as `FRAMES` samples.
    trajectory = bp.Trajectory(robot, path)
    steps = FRAMES - 1 if FRAMES else max(int(np.ceil(trajectory.duration / DT)), 1)
    samples = trajectory.sample(steps / trajectory.duration)
    return (samples, steps, trajectory.duration / steps), time.perf_counter() - begin


def prepare(cfg, datasets):
    return datasets


def solve(planner, problems, spec, cfg):
    del planner  # robokit's, which this plugin leaves unused
    mpinets = problems[0]["case"].startswith("mpinets")
    robot, device = _robot(0.025 if mpinets else 0.04, spec["device"])
    metrics, trajs, extras = [], [], []
    warmed = False
    for problem in problems:
        worlds = device.upload([_world(problem["obstacles"])])
        if not warmed:
            for _ in range(cfg.warmup_count):
                _plan(robot, device, worlds, problem)
            warmed = True
        samples, seconds = _plan(robot, device, worlds, problem)
        extras.append({"device": device.name})
        if samples is None:
            metrics.append(MotionPlanMetrics(total_time=seconds))
            trajs.append(None)
            continue
        samples, steps, dt = samples
        q = np.asarray(samples.positions, dtype=np.float64)
        # Rounding can give that rate one sample more than its steps, just before the end.
        if len(q) == steps + 2:
            q = np.delete(q, -2, axis=0)
        _, _, jerk = trajectory_derivatives(q, dt)
        diffs = np.diff(q, axis=0)
        metrics.append(
            MotionPlanMetrics(
                success=True,
                total_time=seconds,
                cspace_path_length=float(np.linalg.norm(diffs, axis=1).sum()),
                max_jerk=float(np.abs(jerk).max()),
                motion_time=(len(q) - 1) * dt,
                num_frames=len(q),
                dt=dt,
            )
        )
        trajs.append(q.astype(np.float32))
    return metrics, trajs, extras


def score(extra, q, jerk):
    return True


def finish(cfg):
    pass
