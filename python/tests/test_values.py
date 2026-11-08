"""Value classes: what crosses into Rust is what was given, and nothing changes behind a value."""

import math

import numpy as np
import pytest

import batchplan as bp


def test_rotations_are_scalar_last():
    # A quarter turn about z, scalar last, takes the box's long x axis onto y.
    s = math.sqrt(0.5)
    turn = bp.Pose(rotation=(0.0, 0.0, s, s))
    box = bp.Obstacle.cuboid((0.0, 0.0, 0.0), (0.5, 0.1, 0.1)).placed(turn)
    assert box.distance((0.0, 0.45, 0.0)).value < 0, "the long axis lies along y once turned"
    assert box.distance((0.45, 0.0, 0.0)).value > 0, "and no longer along x"
    np.testing.assert_allclose(box.pose.rotation, (0.0, 0.0, s, s), atol=1e-6)


def test_placing_sets_the_frame_it_was_made_at():
    box = bp.Obstacle.cuboid((1.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    moved = box.placed(bp.Pose((0.0, 1.0, 0.0)))
    np.testing.assert_allclose(moved.pose.position, (0.0, 1.0, 0.0), atol=1e-6)


def test_poses_compose_and_invert():
    s = math.sqrt(0.5)
    a = bp.Pose((1.0, 0.0, 0.0), (0.0, 0.0, s, s))
    b = bp.Pose((1.0, 0.0, 0.0))
    np.testing.assert_allclose(a.mul_pose(b).position, (1.0, 1.0, 0.0), atol=1e-6)
    back = a.mul_pose(a.inverse())
    np.testing.assert_allclose(back.position, (0.0, 0.0, 0.0), atol=1e-6)
    assert bp.Pose() == bp.Pose.IDENTITY


def test_replace_changes_only_what_it_names():
    options = bp.PlanOptions()
    changed = options.replace(seeds=4, rrt=options.rrt.replace(max_iterations=5))
    assert (changed.seeds, changed.rrt.max_iterations) == (4, 5)
    assert (options.seeds, options.rrt.max_iterations) == (0, 10_000), "the original is unchanged"
    assert changed.control_points == options.control_points
    with pytest.raises(TypeError, match="no field 'seed'"):
        options.replace(seed=4)


def test_values_cannot_change_in_place():
    problem = bp.PlanProblem(0, [0.0, 0.0], [1.0, 1.0])
    with pytest.raises(AttributeError):
        problem.world = 1
    with pytest.raises(ValueError, match="read-only"):
        problem.start[0] = 5.0
    world = bp.World([bp.Obstacle.sphere((0.0, 0.0, 0.0), 0.1)])
    assert isinstance(world.obstacles, tuple), "sequences come out immutable"


def test_floats_read_back_as_written():
    # f32 defaults read as the decimals they were written as, not their f64 expansions.
    assert bp.IkOptions().damping == 0.02
    assert bp.PlanOptions().collision.margin == 0.012
    assert bp.PlanOptions(time_budget=0.25).time_budget == 0.25


def test_sequences_come_from_any_iterable_but_a_string():
    touch = bp.Touch(0, (link for link in ["a", "b"]))
    assert touch.links == ("a", "b")
    with pytest.raises(TypeError, match="string"):
        bp.Touch(0, "panda_hand")


def test_maps_and_pairs_cross_both_ways():
    options = bp.RobotOptions(lock_joints={"a": 0.5}, joint_limits={"b": bp.JointLimits()})
    assert dict(options.lock_joints) == {"a": 0.5}
    with pytest.raises(TypeError):
        options.lock_joints["c"] = 1.0  # a map comes out read-only
    again = options.replace(lock_joints=options.lock_joints)
    assert dict(again.lock_joints) == {"a": 0.5}, "a map read back goes in again"
    assert bp.JointLimits(position=[-1.0, 1.0]).position == (-1.0, 1.0)
    assert bp.JointLimits(position=np.array([-1.0, 1.0])).position == (-1.0, 1.0)
    with pytest.raises(bp.InputError, match="pair"):
        bp.JointLimits(position=[1.0])


def test_wrong_inputs_are_input_errors():
    with pytest.raises(bp.InputError, match="3 numbers"):
        bp.Obstacle.sphere((0.0, 0.0), 0.1)
    with pytest.raises(bp.InputError, match="duration"):
        bp.PlanOptions(time_budget=1e300)
    with pytest.raises(TypeError, match="not both"):
        bp.IkProblem(0, bp.Pose(), targets=[None])


def test_capsules_take_their_radius_and_half_length_in_order():
    capsule = bp.Obstacle.capsule((0.0, 0.0, 0.0), 0.1, 0.5)
    assert math.isclose(capsule.distance((0.2, 0.0, 0.0)).value, 0.1, abs_tol=1e-6)
    assert math.isclose(capsule.distance((0.0, 0.0, 0.55)).value, -0.05, abs_tol=1e-6)


def test_spheres_need_four_numbers_each():
    model = bp.CollisionModel(spheres={"a": [[0.0, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2]]})
    assert model.spheres["a"].shape == (2, 4)
    with pytest.raises(bp.InputError, match="shape"):
        bp.CollisionModel(spheres={"a": [[0.0, 0.0, 0.0]] * 4})
    with pytest.raises(bp.InputError, match="shape"):
        bp.CollisionModel(spheres={"a": np.zeros((4, 3))})
    assert bp.CollisionModel(spheres={"a": []}).spheres["a"].shape == (0, 4)


def test_a_world_without_an_obstacle_renumbers_the_touches_after_it():
    ball = bp.Obstacle.sphere((0.0, 0.0, 0.0), 0.1)
    world = bp.World([ball, ball, ball], touch=[bp.Touch(0, ["a"]), bp.Touch(2, ["b"])])
    picked = world.without(0)
    assert len(picked.obstacles) == 2 and picked.touch == (bp.Touch(1, ["b"]),)
    assert len(world.obstacles) == 3, "the world it was made from keeps its obstacles"
    with pytest.raises(bp.InputError):
        world.without(3)
