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


def test_a_limit_on_overhang_holds_fitted_and_given_spheres():
    urdf = """<robot name="box"><link name="base"/><link name="shape"><collision><geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry></collision></link><joint name="joint" type="revolute">
      <parent link="base"/><child link="shape"/><axis xyz="0 0 1"/>
      <limit lower="-1" upper="1" effort="1" velocity="1"/>
    </joint></robot>"""
    assert bp.SphereOptions().max_overhang is None
    spheres = bp.SphereOptions(budget=8, max_overhang=0.002, self_collision_samples=0)
    assert spheres.max_overhang == 0.002
    options = bp.RobotOptions(spheres=spheres)
    fitted = np.asarray(bp.Robot.from_urdf(urdf, options).collision_model.spheres["shape"])
    assert 0 < len(fitted) <= spheres.budget
    # No sphere reaches more than the limit past a face of the box.
    assert np.all(np.abs(fitted[:, :3]).max(axis=1) + fitted[:, 3] <= 0.052001)
    free = bp.Robot.from_urdf(urdf, options.replace(spheres=spheres.replace(max_overhang=None)))
    loose = np.asarray(free.collision_model.spheres["shape"])
    assert np.any(np.abs(loose[:, :3]).max(axis=1) + loose[:, 3] > 0.06), "the limit was needed"
    past = bp.CollisionModel(spheres={"shape": [[0.0, 0.0, 0.0, 0.06]]})
    with pytest.raises(bp.InputError, match="link 'shape': sphere 0,"):
        bp.Robot.from_urdf(urdf, options.replace(collision_model=past))
    with pytest.raises(bp.InputError, match="max_overhang must be finite"):
        bp.Robot.from_urdf(urdf, options.replace(spheres=spheres.replace(max_overhang=-0.001)))


def test_options_are_checked_as_their_calls_check_them():
    defaults = (
        bp.IkOptions(),
        bp.PlanOptions(),
        bp.RrtOptions(),
        bp.ShortcutOptions(),
        bp.ToolPathOptions(),
    )
    for options in defaults:
        options.check()
    with pytest.raises(bp.InputError, match="at least one seed"):
        bp.IkOptions(seeds=0).check()
    with pytest.raises(bp.InputError, match="IK needs at most 4294967295 seeds"):
        bp.IkOptions(seeds=10**12).check()
    with pytest.raises(bp.InputError, match="goal_ik needs positive tolerances"):
        bp.PlanOptions(goal_ik=bp.IkOptions(position_tolerance=float("nan"))).check()
    with pytest.raises(bp.InputError, match="range must be positive"):
        bp.RrtOptions(range=0.0).check()
    with pytest.raises(bp.InputError, match="smallest change must be positive"):
        bp.ShortcutOptions(min_change=0.0).check()
    with pytest.raises(bp.InputError, match="steps must be finite"):
        bp.ToolPathOptions(step=0.0).check()


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
    assert bp.Touch.held(2, 1) == bp.Touch(2, ["held by hand 1"]), "what hand 1 comes to hold"
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


def test_a_hold_names_the_other_hands_that_hold_it():
    bar = bp.Obstacle.cuboid((0.0, 0.0, 0.0), (0.015, 0.16, 0.015))
    left, right = bp.Pose((0.0, 0.12, 0.0)), bp.Pose((0.0, -0.12, 0.0))
    assert bp.Held(0, bar, left).also == (), "one hand's hold"
    held = bp.Held(0, bar, left, also=[bp.Grip(1, right)])
    (grip,) = held.also
    assert grip.hand == 1 and grip.grasp == right
    assert held.replace(also=[]).also == ()
    assert held.touch_links == (), "none but the hands that hold it"
    touching = held.replace(touch_links=["right_panda_hand"])
    assert touching.touch_links == ("right_panda_hand",)


def test_a_hold_names_what_rides_on_it():
    bar = bp.Obstacle.cuboid((0.0, 0.0, 0.0), (0.015, 0.16, 0.015))
    cube = bp.Obstacle.cuboid((0.0, -0.1, 0.035), (0.02, 0.02, 0.02))
    held = bp.Held(0, bar, bp.Pose((0.0, 0.12, 0.0)))
    assert held.carries == (), "nothing rides on it"
    (rider,) = held.replace(carries=[cube]).carries
    assert rider.pose == cube.pose, "a shape in the bar's frame"
    riding = bp.Held(0, bar, held.grasp, carries=[cube]).carries
    assert [rider.pose for rider in riding] == [cube.pose], "given to the constructor"


def test_a_place_names_what_its_object_is_set_on():
    cube = bp.Obstacle.cuboid((0.0, 0.0, 0.0), (0.02, 0.02, 0.02))
    held = bp.Held(1, cube, bp.Pose((0.0, 0.0, 0.0)))
    other = bp.Held(0, cube, bp.Pose((0.0, 0.1, 0.0)))
    release = bp.Pose((0.3, 0.0, 0.4))
    assert bp.Place(0, [0.0] * 14, held, [release]).onto is None, "set on nothing a hand holds"
    onto = bp.Place(0, [0.0] * 14, held, [release], onto=other).onto
    assert onto.hand == 0 and onto.grasp == other.grasp, "the other hand's hold"


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


def test_cylinders_take_their_radius_and_half_height_in_order():
    cylinder = bp.Obstacle.cylinder((0.0, 0.0, 0.0), 0.1, 0.5)
    assert math.isclose(cylinder.distance((0.2, 0.0, 0.0)).value, 0.1, abs_tol=1e-6)
    assert math.isclose(cylinder.distance((0.0, 0.0, 0.55)).value, 0.05, abs_tol=1e-6)


def test_distances_point_away_from_the_surface():
    distance = bp.Obstacle.sphere((0.0, 0.0, 0.0), 0.1).distance((0.3, 0.0, 0.0))
    assert math.isclose(distance.value, 0.2, abs_tol=1e-6)
    np.testing.assert_allclose(distance.gradient, (1.0, 0.0, 0.0), atol=1e-6)


def test_joint_states_keep_what_they_are_given():
    state = bp.JointState([0.1, 0.2], [0.3, 0.4], [0.5, 0.6])
    np.testing.assert_allclose(state.velocity, (0.3, 0.4))
    np.testing.assert_allclose(state.acceleration, (0.5, 0.6))
    np.testing.assert_array_equal(bp.JointState([0.1, 0.2]).acceleration, (0.0, 0.0))
