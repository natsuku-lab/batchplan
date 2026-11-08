"""Picks, places, tool paths, held objects and hand constraints through the package."""

import math

import numpy as np
import pytest

import batchplan as bp
from conftest import devices

TABLE, BOX = 0, 1
HALF = (0.02, 0.03, 0.03)


def scene(*more: bp.Obstacle) -> bp.World:
    """A table with its top at z = 0 and a 4 x 6 x 6 cm box on it, then `more`."""
    table = bp.Obstacle.cuboid((0.4, 0.0, -0.02), (0.7, 0.8, 0.02))
    box = bp.Obstacle.cuboid((0.45, 0.2, HALF[2]), HALF)
    return bp.World([table, box, *more])


def box_grasps(robot: bp.Robot) -> list[bp.Pose]:
    options = bp.GraspOptions(tip=robot.tip(0), floor=0.005)
    return bp.grasps(scene().obstacles[BOX], options)


def test_grasps_and_place_poses_follow_their_options():
    box = scene().obstacles[BOX]
    floored = bp.grasps(box, bp.GraspOptions(floor=0.005))
    assert 0 < len(floored) < len(bp.grasps(box)), "a floor leaves out grasps from below"
    can = bp.Obstacle.cylinder((0.0, 0.0, 0.05), 0.03, 0.05)
    assert len(bp.place_poses(can, bp.Pose((0.4, 0.0, 0.05)), 6)) == 6


def test_a_picked_box_is_carried_and_set_down(tasks_robot):
    for device in devices(tasks_robot):
        worlds = device.upload([scene()])
        task = bp.Pick(0, tasks_robot.default_q, BOX, box_grasps(tasks_robot), support=[TABLE])
        outcome = bp.pick(device, worlds, [task]).outcomes[0]
        assert isinstance(outcome, tuple) and outcome, f"{outcome} on {device}"
        plan = outcome[0]
        for motion in (plan.transit, plan.approach, plan.depart):
            motion.check(tasks_robot)
        np.testing.assert_array_equal(plan.transit.start, tasks_robot.default_q)
        np.testing.assert_array_equal(plan.approach.start, plan.transit.end)
        grasp = task.grasps[plan.candidate]
        at = scene().obstacles[BOX].pose.mul_pose(grasp)
        grasped = tasks_robot.ee_pose(plan.approach.end)
        np.testing.assert_allclose(grasped.position, at.position, atol=2e-3)
        lifted = tasks_robot.ee_pose(plan.depart.end)
        np.testing.assert_allclose(lifted.position, at.position + task.lift, atol=2e-3)
        assert plan.held.hand == 0 and plan.held.grasp == grasp
        motions = plan.transit.duration + plan.approach.duration + plan.depart.duration
        assert math.isclose(plan.duration, motions, rel_tol=1e-5)
        assert str(plan.duration) == str(np.float32(plan.duration)), "f32 as written"

        # Set it down 20 cm along y, in the world without it.
        without = device.upload([bp.World(scene().obstacles[:BOX])])
        spot = bp.Pose((0.45, 0.0, HALF[2]))
        poses = bp.place_poses(plan.held.object, spot, 4)
        assert len(poses) == 2, "a box with unequal x and y sides sets down two ways"
        place = bp.Place(0, plan.end, plan.held, poses, support=[TABLE])
        placed = bp.place(device, without, [place]).outcomes[0]
        assert isinstance(placed, tuple) and placed, f"{placed} on {device}"
        np.testing.assert_allclose(placed[0].placed.pose.position, spot.position, atol=2e-3)


def test_a_task_no_grasp_gets_through_says_where_each_stopped(tasks_robot):
    device = bp.Device.cpu(tasks_robot)
    far = bp.Obstacle.cuboid((2.0, 0.0, HALF[2]), HALF)
    worlds = device.upload([bp.World([scene().obstacles[TABLE], far])])
    grasps = bp.grasps(far, bp.GraspOptions(tip=tasks_robot.tip(0), floor=0.005))
    failure = bp.pick(device, worlds, [bp.Pick(0, tasks_robot.default_q, 1, grasps)]).outcomes[0]
    assert isinstance(failure, bp.TaskFailure)
    assert failure.step == bp.TaskStep.Reach
    assert failure.at(bp.TaskStep.Reach) == len(grasps)
    assert all(isinstance(c.why, bp.ToolPathStop.Unreachable) for c in failure.candidates)
    assert "reach" in str(failure).lower(), str(failure)


def test_a_tool_path_goes_as_far_as_it_is_clear(tasks_robot):
    device = bp.Device.cpu(tasks_robot)
    q = tasks_robot.default_q
    hand = tasks_robot.ee_pose(q)
    down = bp.Pose(hand.position - np.array((0.0, 0.0, 0.1)), hand.rotation)
    clear = device.upload([bp.World()])
    result = bp.follow(device, clear, [bp.ToolPath(0, q, [down])])
    assert result.fraction[0] == 1.0 and result.stopped == (None,)
    reached = tasks_robot.ee_pose(result.paths[0][-1])
    np.testing.assert_allclose(reached.position, down.position, atol=2e-3)

    # A slab whose top is 5 cm below the hand, halfway down its path.
    block = bp.Obstacle.cuboid(hand.position - np.array((0.0, 0.0, 0.07)), (0.1, 0.1, 0.02))
    blocked = device.upload([bp.World([block])])
    result = bp.follow(device, blocked, [bp.ToolPath(0, q, [down])])
    assert 0 < result.fraction[0] < 1
    assert isinstance(result.stopped[0], bp.ToolPathStop.Collision)
    assert result.contacts[0][0].obstacle == 0
    for read in ("problems", "fraction", "stopped", "paths", "contacts"):
        assert getattr(result, read) is getattr(result, read), f"{read} is built once"


def test_holding_and_constraints_reach_the_planner(tasks_robot):
    device = bp.Device.cpu(tasks_robot)
    worlds = device.upload([bp.World()])
    q = tasks_robot.default_q
    hand = tasks_robot.ee_pose(q)
    nowhere = bp.Held(5, bp.Obstacle.sphere((0.0, 0.0, 0.0), 0.02), bp.Pose.IDENTITY)
    with pytest.raises(bp.InputError, match="hand 5"):
        bp.solve_ik(device, worlds, [bp.IkProblem(0, hand, holding=nowhere)])
    with pytest.raises(bp.InputError, match="hand 5"):
        bp.plan(device, worlds, [bp.PlanProblem(0, q, q, holding=nowhere)])
    with pytest.raises(bp.InputError, match="hand 5"):
        bp.follow(device, worlds, [bp.ToolPath(0, q, [hand], holding=nowhere)])
    away = bp.Pose(hand.position + np.array((0.5, 0.0, 0.0)), hand.rotation)
    pinned = bp.HandConstraint(0, away, position=(0.0, 0.0, 0.0))
    with pytest.raises(bp.InputError, match="start"):
        bp.plan(device, worlds, [bp.PlanProblem(0, q, q, constraints=[pinned])])


def test_an_attached_object_moves_with_the_hand(tasks_robot):
    q = tasks_robot.default_q
    ball = bp.Obstacle.sphere((0.0, 0.0, 0.05), 0.03)
    held = tasks_robot.attach(bp.AttachedObject("ball", "panda_hand", [ball]))
    assert len(held.spheres(q)) > len(tasks_robot.spheres(q))
    assert held.link_pose(q, "ball") is not None
    assert len(held.detach("ball").spheres(q)) == len(tasks_robot.spheres(q))
    device = bp.Device.cpu(tasks_robot)
    worlds = device.upload([bp.World()])
    holding = device.with_robot(held)
    assert holding.robot is held
    result = bp.solve_ik(holding, worlds, [bp.IkProblem(0, tasks_robot.ee_pose(q))])
    assert result.success.any()


def test_a_transit_that_breaks_its_constraints_stops_there(tasks_robot):
    # Pinned within a centimetre of where it starts, the hand can't reach where any move in
    # begins.
    device = bp.Device.cpu(tasks_robot)
    worlds = device.upload([scene()])
    start = tasks_robot.ee_pose(tasks_robot.default_q)
    pinned = bp.HandConstraint(0, start, position=(0.01, 0.01, 0.01))
    task = bp.Pick(
        0,
        tasks_robot.default_q,
        BOX,
        box_grasps(tasks_robot),
        support=[TABLE],
        constraints=[pinned],
    )
    failure = bp.pick(device, worlds, [task]).outcomes[0]
    assert isinstance(failure, bp.TaskFailure)
    assert failure.step == bp.TaskStep.Transit, failure
    assert isinstance(failure.candidates, tuple)
