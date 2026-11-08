"""IK, planning and timing through the package, on every device."""

import math
import threading
import time

import numpy as np
import pytest

import batchplan as bp
from conftest import ASSETS, devices, ur5e

DOWN = (1.0, 0.0, 0.0, 0.0)  # half a turn about x: the tool pointing down
TABLE = bp.Obstacle.cuboid((0.0, 0.0, -0.02), (1.0, 1.0, 0.02))


def same_rotation(a, b) -> bool:
    """Quaternions `a` and `b` turn the same way: equal up to sign."""
    return abs(float(np.dot(a, b))) > 1 - 1e-4


def test_ik_reaches_each_target(ur5e_robot):
    targets = [bp.Pose((0.4, 0.2, 0.3), DOWN), bp.Pose((0.3, -0.3, 0.4), DOWN)]
    for device in devices(ur5e_robot):
        worlds = device.upload([bp.World([TABLE])])
        result = bp.solve_ik(device, worlds, [bp.IkProblem(0, t) for t in targets])
        assert result.q.shape == (2, 32, ur5e_robot.dof)
        solved = result.solved()
        assert [s.index for s in solved] == [0, 1], device
        for s, target in zip(solved, targets, strict=True):
            reached = ur5e_robot.ee_pose(s.solution)
            np.testing.assert_allclose(reached.position, target.position, atol=1e-3)
            assert same_rotation(reached.rotation, target.rotation), device


def test_ik_reports_each_seed_as_forward_kinematics_measures_it(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    worlds = device.upload([bp.World()])
    target = bp.Pose((0.4, 0.2, 0.3), (0.0, 1.0, 0.0, 0.0))
    result = bp.solve_ik(device, worlds, [bp.IkProblem(0, target)], bp.IkOptions(iterations=3))
    for q, position, rotation in zip(
        result.q[0], result.position_error[0], result.rotation_error[0], strict=True
    ):
        reached = ur5e_robot.ee_pose(q)
        distance = np.linalg.norm(reached.position - target.position)
        angle = 2 * math.acos(min(1.0, abs(float(np.dot(reached.rotation, target.rotation)))))
        assert math.isclose(position, distance, abs_tol=1e-4), (position, distance)
        assert math.isclose(rotation, angle, abs_tol=2e-3), (rotation, angle)
    assert np.all(result.world_clearance > 1e20), "nothing to collide with is far away"
    assert np.all(result.self_clearance < 1.0), "the links are near one another"
    with pytest.raises(IndexError):
        result.best(1)
    with pytest.raises(IndexError):
        result.nearest(1, ur5e_robot.default_q)


def test_ik_holds_the_joints_it_is_told_to(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    worlds = device.upload([bp.World()])
    seed = ur5e_robot.default_q.copy()
    seed[0] = 0.3
    problem = bp.IkProblem(0, bp.Pose((0.4, 0.2, 0.3), DOWN), seed=seed, hold=[0])
    result = bp.solve_ik(device, worlds, [problem])
    assert np.all(result.q[0, :, 0] == np.float32(0.3)), "every seed keeps the held base joint"


def test_plans_run_from_start_to_goal_and_time_within_limits(ur5e_robot):
    goal = ur5e_robot.default_q.copy()
    goal[0] += 1.5
    for device in devices(ur5e_robot):
        worlds = device.upload([bp.World([TABLE])])
        problem = bp.PlanProblem(0, ur5e_robot.default_q, goal)
        result = bp.plan(device, worlds, [problem])
        assert result.failures == (None,), device
        assert result.valid.shape == (1, 1)
        path = result.best(0)
        assert path.shape[1] == ur5e_robot.dof
        trajectory = bp.Trajectory(ur5e_robot, path)
        trajectory.check(ur5e_robot)
        np.testing.assert_allclose(trajectory.start, ur5e_robot.default_q, atol=1e-6)
        np.testing.assert_allclose(trajectory.end, goal, atol=1e-6)
        samples = trajectory.sample(100.0)
        assert samples.positions.shape == (len(samples), ur5e_robot.dof)
        assert math.isclose(samples.duration, trajectory.duration, rel_tol=1e-6)
        mid = trajectory.at(trajectory.duration / 2)
        assert np.any(mid.velocity != 0), "the state halfway is moving"


def test_held_joints_stay_along_the_path(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    start = ur5e_robot.default_q
    goal = start.copy()
    goal[1:3] += (-1.0, 1.2)
    between = (ur5e_robot.ee_pose(start).position + ur5e_robot.ee_pose(goal).position) / 2
    worlds = device.upload([bp.World([TABLE, bp.Obstacle.cuboid(between, (0.05, 0.05, 0.05))])])
    free = bp.plan(device, worlds, [bp.PlanProblem(0, start, goal)]).best(0)
    assert np.ptp(free[:, 0]) > 0.1, "around the box, a free plan turns the base"
    held = bp.plan(device, worlds, [bp.PlanProblem(0, start, goal, hold=[0])]).best(0)
    assert np.all(held[:, 0] == start[0]), "a held base stays"


def test_a_start_in_collision_names_what_it_touches(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    block = bp.Obstacle.sphere(ur5e_robot.ee_pose(ur5e_robot.default_q).position, 0.1)
    worlds = device.upload([bp.World([TABLE, block])])
    goal = ur5e_robot.default_q.copy()
    goal[0] += 1.0
    result = bp.plan(device, worlds, [bp.PlanProblem(0, ur5e_robot.default_q, goal)])
    failure = result.failures[0]
    assert isinstance(failure, bp.PlanFailure.StartCollides)
    assert result.best(0) is None and result.solved() == []
    contact = failure.contacts[0]
    assert isinstance(contact, bp.Contact.World)
    assert contact.obstacle == 1 and contact.distance < 0
    assert contact.link.startswith("wrist"), contact
    assert str(contact.distance) == str(np.float32(contact.distance)), "f32 as written"
    backward = bp.plan(device, worlds, [bp.PlanProblem(0, goal, ur5e_robot.default_q)])
    assert isinstance(backward.failures[0], bp.PlanFailure.GoalCollides), backward.failures


def test_unreachable_targets_say_so(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    worlds = device.upload([bp.World()])
    problems = [bp.IkProblem(0, bp.Pose((3.0, 0.0, 0.3), DOWN))]
    result = bp.solve_ik(device, worlds, problems)
    failures = bp.ik_failures(device, worlds, result)
    assert isinstance(failures[0], bp.IkFailure.Unreachable)


def test_errors_raise_their_kind():
    robot_path = ASSETS / "ur5e/ur_description/urdf/ur5e.urdf"
    with pytest.raises(bp.LoadError, match=r"missing\.urdf"):
        bp.Robot.load(robot_path.with_name("missing.urdf"))
    robot = bp.Robot.load(robot_path)
    device = bp.Device.cpu(robot)
    worlds = device.upload([bp.World()])
    with pytest.raises(bp.InputError, match="world 3"):
        bp.solve_ik(device, worlds, [bp.IkProblem(3, bp.Pose((0.4, 0.0, 0.3), DOWN))])
    assert issubclass(bp.InputError, bp.Error)
    with pytest.raises(bp.InputError):
        robot.ee_pose(robot.default_q, hand=1)
    with pytest.raises(IndexError):
        worlds[1]
    with pytest.raises(IndexError):
        worlds[-2]
    assert isinstance(worlds[-1], bp.World)
    plans = bp.plan(device, worlds, [bp.PlanProblem(0, robot.default_q, robot.default_q)])
    with pytest.raises(IndexError):
        plans.best(1)


def test_result_arrays_are_built_once(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    worlds = device.upload([bp.World()])
    result = bp.solve_ik(device, worlds, [bp.IkProblem(0, bp.Pose((0.4, 0.2, 0.3), DOWN))])
    assert result.q is result.q, "a loop over `result.q[i]` copies the result once"
    assert result.problems is result.problems
    plans = bp.plan(device, worlds, [bp.PlanProblem(0, ur5e_robot.default_q, result.best(0))])
    assert plans.problems is plans.problems and plans.paths is plans.paths


def reaches(robot: bp.Robot, n: int, seed: int = 3):
    """`n` random reaches, each with a 10 cm ball where the hand is halfway through it."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(robot.lower, robot.upper, size=(2, n, robot.dof)).astype(np.float32)
    middles = [robot.ee_pose((a + b) / 2).position for a, b in zip(q[0], q[1], strict=True)]
    worlds = [bp.World([bp.Obstacle.sphere(m, 0.1)]) for m in middles]
    problems = [bp.PlanProblem(i, a, b) for i, (a, b) in enumerate(zip(q[0], q[1], strict=True))]
    return worlds, problems


def test_failures_say_which_step_stopped_them(tasks_robot):
    device = bp.Device.cpu(tasks_robot)
    scene, problems = reaches(tasks_robot, 64)
    worlds = device.upload(scene)
    no_samples = bp.PlanOptions(rrt=bp.RrtOptions(max_iterations=0))
    failures = bp.plan(device, worlds, problems, no_samples).failures
    ends = (bp.PlanFailure.StartCollides, bp.PlanFailure.GoalCollides)
    searched = [f for f in failures if f is not None and not isinstance(f, ends)]
    assert searched and all(isinstance(f, bp.PlanFailure.NoPath) for f in searched), searched
    no_time = bp.PlanOptions(time_budget=0.0)
    failures = bp.plan(device, worlds, problems, no_time).failures
    assert all(isinstance(f, bp.PlanFailure.OutOfTime) for f in failures), failures


def test_a_folded_arm_names_both_links_it_folds_onto(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    folded = ur5e_robot.default_q.copy()
    folded[2] = 2.9
    worlds = device.upload([bp.World()])
    result = bp.plan(device, worlds, [bp.PlanProblem(0, folded, ur5e_robot.default_q)])
    contacts = result.failures[0].contacts
    pairs = [c.links for c in contacts if isinstance(c, bp.Contact.SelfCollision)]
    assert ("upper_arm_link", "wrist_2_link") in pairs, contacts


def test_ik_says_what_a_target_collides_with(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    target = bp.Pose((0.4, 0.2, 0.3), DOWN)
    worlds = device.upload([bp.World([bp.Obstacle.sphere(target.position, 0.05)])])
    result = bp.solve_ik(device, worlds, [bp.IkProblem(0, target)])
    failure = bp.ik_failures(device, worlds, result)[0]
    assert isinstance(failure, bp.IkFailure.Collides)
    assert failure.contacts[0].obstacle == 0


def test_nearest_takes_the_successful_seed_closest_to_a_configuration(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    # A ball above the target, which every seed clears by its own distance.
    worlds = device.upload([bp.World([bp.Obstacle.sphere((0.4, 0.2, 0.6), 0.05)])])
    result = bp.solve_ik(device, worlds, [bp.IkProblem(0, bp.Pose((0.4, 0.2, 0.3), DOWN))])
    solved = result.q[0][result.success[0]]
    assert len(solved) > 1, "several seeds succeed, so nearest and best can differ"
    for toward in solved:
        np.testing.assert_array_equal(result.nearest(0, toward), toward)
    clearance = np.where(result.success[0], result.world_clearance[0], -np.inf)
    assert np.sum(clearance == clearance.max()) == 1, "one seed is the clearest"
    clearest = np.argmax(clearance)
    np.testing.assert_array_equal(result.best(0), result.q[0, clearest])


def test_options_given_as_maps_reach_the_robot(ur5e_robot):
    pan = "shoulder_pan_joint"
    locked = bp.Robot.load(
        ASSETS / "ur5e/ur_description/urdf/ur5e.urdf", bp.RobotOptions(lock_joints={pan: 0.5})
    )
    assert locked.dof == 5 and pan not in locked.joint_names
    limited = bp.Robot.load(
        ASSETS / "ur5e/ur_description/urdf/ur5e.urdf",
        bp.RobotOptions(joint_limits={pan: bp.JointLimits(position=(-1.0, 1.0), velocity=0.5)}),
    )
    assert (limited.lower[0], limited.upper[0], limited.max_velocity[0]) == (-1.0, 1.0, 0.5)
    assert isinstance(limited.joint_names, tuple)
    rates = bp.RobotOptions(joint_limits={pan: bp.JointLimits(acceleration=3.0, jerk=30.0)})
    rated = bp.Robot.load(ASSETS / "ur5e/ur_description/urdf/ur5e.urdf", rates)
    assert (rated.max_acceleration[0], rated.max_jerk[0]) == (3.0, 30.0)

    # A joint to relock is locked with spheres fitted while it was free.
    free = ur5e_robot.collision_model
    path = ASSETS / "ur5e/ur_description/urdf/ur5e.urdf"
    held = bp.Robot.load(path, bp.RobotOptions(lock_joints={pan: 0.5}, collision_model=free))
    q = held.default_q
    moved = held.relock({pan: 1.0})
    before, after = held.link_pose(q, "tool0"), moved.link_pose(q, "tool0")
    assert np.linalg.norm(before.position - after.position) > 0.1, "the base turned"
    text = (ASSETS / "ur5e/ur_description/urdf/ur5e.urdf").read_text()
    # From text, `package://` meshes are found through `package_dirs`.
    options = bp.RobotOptions(lock_joints={pan: 0.5}, package_dirs=[ASSETS / "ur5e"])
    from_text = bp.Robot.from_urdf(text, options)
    assert from_text.dof == 5


def test_hands_name_their_links_and_fingers(tasks_robot):
    links, fingers = tasks_robot.hand_links(0), tasks_robot.finger_links(0)
    assert "panda_hand" in links and "panda_hand" not in fingers
    assert set(fingers) == {"panda_leftfinger", "panda_rightfinger"}
    assert set(fingers) < set(links)


def test_collision_models_save_and_load(tasks_robot, tmp_path):
    model = tasks_robot.collision_model
    model.save(tmp_path / "model.json")
    loaded = bp.CollisionModel.load(tmp_path / "model.json")
    assert loaded == model and len(loaded.spheres) > 5


def test_trajectories_start_moving_stop_and_cap_the_hand(ur5e_robot):
    device = bp.Device.cpu(ur5e_robot)
    worlds = device.upload([bp.World([TABLE])])
    start, goal = ur5e_robot.default_q, ur5e_robot.default_q.copy()
    goal[0] += 1.5
    velocity = np.zeros(ur5e_robot.dof, np.float32)
    velocity[0] = 0.3
    motion = bp.StartMotion(velocity, np.zeros_like(velocity))
    moving = bp.PlanProblem(0, start, goal, start_motion=motion)
    path = bp.plan(device, worlds, [moving]).best(0)
    trajectory = bp.Trajectory.moving(ur5e_robot, path, motion)
    np.testing.assert_allclose(trajectory.at(0.0).velocity, velocity, atol=1e-4)
    trajectory.check_from(ur5e_robot, bp.JointState(start, velocity))
    with pytest.raises(bp.UnsafeError):
        trajectory.check(ur5e_robot)  # it doesn't start at rest

    still = bp.Trajectory(
        ur5e_robot, bp.plan(device, worlds, [bp.PlanProblem(0, start, goal)]).best(0)
    )
    t = still.duration / 3
    stopped = still.stop(ur5e_robot, t)
    np.testing.assert_allclose(stopped.start, still.at(t).position, atol=1e-5)
    stopped.check_from(ur5e_robot, still.at(t))
    assert still.sample(100.0).dt == 0.01

    slow = bp.Trajectory.with_hand_speed(ur5e_robot, still.control_points, bp.HandSpeed(0, 0.05))
    assert slow.duration > 2 * still.duration, "a 5 cm/s cap slows a long reach"

    slowed = bp.Robot.load(
        ASSETS / "ur5e/ur_description/urdf/ur5e.urdf", bp.RobotOptions(velocity_scale=0.1)
    )
    with pytest.raises(bp.UnsafeError):
        still.check(slowed)


def test_a_table_up_to_the_shoulder_is_always_touched(ur5e_robot):
    # The shoulder turns about the base's axis at one height, so a table up to it touches it in
    # every configuration.
    floor = bp.Obstacle.cuboid((0.0, 0.0, 0.05), (1.0, 1.0, 0.07))
    touches = bp.World([floor]).always_touching(ur5e_robot)
    assert touches and all(t.obstacle == 0 for t in touches), touches


def targets(robot, problems):
    return [bp.IkProblem(p.world, robot.ee_pose(p.goal)) for p in problems]


def out_of_reach(problems):
    return [bp.IkProblem(p.world, bp.Pose((3.0, 0.0, 0.3))) for p in problems]


def failures_of(robot, device, worlds, problems):
    """`ik_failures` for targets no seed reaches, so it solves every one again."""
    result = bp.solve_ik(device, worlds, out_of_reach(problems))
    return lambda: bp.ik_failures(device, worlds, result)


def tool_paths(robot, device, worlds, problems):
    """`follow`, 10 cm down from each reach's start."""
    paths = []
    for p in problems:
        hand = robot.ee_pose(p.start)
        down = bp.Pose(hand.position - np.array((0.0, 0.0, 0.1)), hand.rotation)
        paths.append(bp.ToolPath(p.world, p.start, [down]))
    return lambda: bp.follow(device, worlds, paths)


def picks(robot, device, _worlds, problems):
    """`pick`, of a box in front of the robot from each reach's start."""
    box = bp.Obstacle.cuboid((0.45, 0.2, 0.03), (0.02, 0.03, 0.03))
    worlds = device.upload([bp.World([box])])
    grasps = bp.grasps(box, bp.GraspOptions(tip=robot.tip(0)))
    tasks = [bp.Pick(0, robot.default_q, 0, grasps) for _ in problems]
    return lambda: bp.pick(device, worlds, tasks)


# Each call, with enough reaches that it takes a while on two threads: per call, the reaches and
# what makes the call to time from them.
LONG_CALLS = {
    "follow": (8000, tool_paths),
    "pick": (16, picks),
    "solve_ik": (300, lambda r, d, w, p: lambda: bp.solve_ik(d, w, targets(r, p))),
    "plan": (4000, lambda r, d, w, p: lambda: bp.plan(d, w, p)),
    "ik_failures": (1000, failures_of),
    "Robot.load": (1, lambda *_: lambda: [ur5e() for _ in range(3)]),
}


@pytest.mark.parametrize("call", LONG_CALLS)
def test_calls_let_other_threads_run(tasks_robot, call):
    # While a call works, the main thread runs: it resumes soon after the call begins and keeps
    # running through it. A call that held the interpreter would keep it waiting until the end.
    n, prepare = LONG_CALLS[call]
    device = bp.Device.cpu(tasks_robot, threads=2)
    scene, problems = reaches(tasks_robot, n)
    worlds = device.upload(scene)
    run = prepare(tasks_robot, device, worlds, problems)
    started, done = threading.Event(), threading.Event()
    began, took = [], []

    def work():
        began.append(time.perf_counter())
        started.set()
        run()
        took.append(time.perf_counter() - began[0])
        done.set()

    worker = threading.Thread(target=work)
    worker.start()
    started.wait()
    first = last = time.perf_counter()
    while not done.is_set():
        last = time.perf_counter()
    worker.join()
    assert took[0] > 0.1, f"{call} is long enough to tell: {took[0]:.3f} s"
    assert first - began[0] < took[0] / 4, "the main thread resumed while the call ran"
    assert last - first > took[0] / 2, "and kept running through it"
