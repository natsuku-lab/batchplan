"""Qualifying a robot: its spheres, its scene, and what its configurations collide with."""

import math

import numpy as np
import pytest

import batchplan as bp
from conftest import ASSETS, devices

TABLE = bp.Obstacle.cuboid((0.4, 0.0, -0.02), (0.7, 0.8, 0.02))
SO101 = ASSETS / "so101/so101_new_calib.urdf"


def test_qualify_measures_every_link_with_spheres(panda_robot):
    model = panda_robot.collision_model
    qualification = panda_robot.qualify()
    # Every link of the Panda's model has geometry, so each is measured, once.
    assert sorted(link.link for link in qualification.links) == sorted(model.spheres)
    for link in qualification.links:
        assert link.spheres == len(model.spheres[link.link]), link
        measured = (link.overhang, link.uncovered, link.gap, link.thickness)
        assert all(math.isfinite(v) for v in measured), link
        assert 0 <= link.uncovered <= 1 and link.gap >= 0 and link.thickness > 0, link
    # As `a_hands_tip_is_how_far_its_spheres_reach_past_its_frame` asserts in Rust.
    assert qualification.tips == (panda_robot.tip(0),)


def test_qualify_reads_fitted_and_hand_tuned_spheres_as_rust_does():
    # The bounds of `qualification_measures_how_the_spheres_fit_each_link`, on the same robot.
    robot = bp.Robot.load(SO101)
    fitted = robot.qualify()
    assert fitted.links is fitted.links, "built once"
    names = [link.link for link in fitted.links]
    assert names == [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
        "gripper_link",
        "moving_jaw_so101_v1_link",
    ]
    # The fit's own account of its overhang, to the tenth of a millimeter it gives.
    model = robot.collision_model
    reported = float(model.source.split("up to ")[1].split(" ")[0])
    largest = max(link.overhang for link in fitted.links)
    assert abs(largest * 1e3 - reported) < 0.06, (largest, model.source)
    for link in fitted.links:
        assert link.spheres == len(model.spheres[link.link]), link
        assert link.uncovered < 0.005 and link.thickness > 0.01, link
    assert fitted.tips == (robot.tip(0),)
    # Halved spheres leave most of the surface out and reach less far.
    halved = np.array(model.spheres["upper_arm_link"])
    halved[:, 3] *= 0.5
    tuned = model.replace(spheres={**model.spheres, "upper_arm_link": halved})
    hand_tuned = bp.Robot.load(SO101, bp.RobotOptions(collision_model=tuned)).qualify()
    before, after = (
        next(link for link in q.links if link.link == "upper_arm_link")
        for q in (fitted, hand_tuned)
    )
    assert after.uncovered > 0.3 and after.overhang < before.overhang, after


SCENE = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="table" pos="0.5 0 0">
      <geom type="box" size="0.3 0.4 0.02" pos="0 0 0.38"/>
      <geom type="box" size="0.03 0.03 0.05" pos="0.1 0.1 0.45"/>
      <geom type="box" size="0.03 0.03 0.05" pos="-0.1 -0.1 0.45"/>
    </body>
    <body name="arm" pos="0 0 0.1">
      <joint name="j" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
  </worldbody>
</mujoco>"""


def test_a_scene_loads_as_a_world_without_its_robot(tmp_path):
    # As `scenes_load_as_worlds_without_their_robots` shows in Rust.
    path = tmp_path / "scene.xml"
    path.write_text(SCENE)
    world = bp.World.load(path)
    assert world.touch == ()
    floor, *boxes = world.obstacles
    assert len(boxes) == 3, "the table top and its two boxes; not the arm"
    assert abs(floor.distance((0.3, 0.2, 0.0)).value) < 1e-6, "the floor's surface is z = 0"
    centers = sorted(tuple(np.round(b.pose.position, 4)) for b in boxes)
    assert centers == [(0.4, -0.1, 0.45), (0.5, 0.0, 0.38), (0.6, 0.1, 0.45)]
    on_the_arm = (0.15, 0.0, 0.1)
    assert all(o.distance(on_the_arm).value > 0 for o in world.obstacles), "the arm is no obstacle"
    robot = bp.Robot.load(path)
    assert robot.joint_names == ("j",)
    assert list(robot.collision_model.spheres) == ["arm"], "the arm is the robot"
    with pytest.raises(bp.LoadError, match="--features usd"):
        bp.World.load(tmp_path / "scene.usda")


# A cube 2 units across, at its center: its corners, then two triangles per face.
CUBE = """\
v -1 -1 -1
v -1 -1 1
v -1 1 -1
v -1 1 1
v 1 -1 -1
v 1 -1 1
v 1 1 -1
v 1 1 1
f 1 3 4
f 1 4 2
f 5 6 8
f 5 8 7
f 1 2 6
f 1 6 5
f 3 7 8
f 3 8 4
f 1 5 7
f 1 7 3
f 2 4 8
f 2 8 6
"""


def test_a_scenes_meshes_become_grids_laid_out_as_asked(tmp_path):
    (tmp_path / "cube.obj").write_text(CUBE)
    path = tmp_path / "scene.xml"
    path.write_text("""<mujoco>
      <asset><mesh name="cube" file="cube.obj" scale="0.05 0.05 0.05"/></asset>
      <worldbody><geom type="mesh" mesh="cube" pos="0.5 0 0.5"/></worldbody>
    </mujoco>""")
    assert (bp.SdfOptions().voxel, bp.SdfOptions().padding) == (0.01, 0.15)
    (cube,) = bp.World.load(path, bp.SdfOptions(voxel=0.02)).obstacles
    assert cube.distance((0.5, 0.0, 0.5)).value < -0.03, "inside the cube"
    assert cube.distance((0.5, 0.0, 0.7)).value > 0.1, "above it"
    assert len(repr(cube)) < 200, "a grid is named by its frame, not by its values"
    with pytest.raises(bp.LoadError, match="voxel size must be positive"):
        bp.World.load(path, bp.SdfOptions(voxel=-0.02))


def test_contacts_name_what_a_configuration_collides_with(tasks_robot):
    q = tasks_robot.default_q
    ball = bp.Obstacle.sphere(tasks_robot.ee_pose(q).position, 0.1)
    listed = set()
    for device in devices(tasks_robot):
        worlds = device.upload([bp.World([TABLE]), bp.World([TABLE, ball])])
        free, into = device.contacts(worlds, [0, 1], [q, q], 0.0)
        assert free == (), "the default configuration clears the table and itself"
        assert into, "the ball is in the hand"
        assert all(isinstance(c, bp.Contact.World) and c.obstacle == 1 for c in into), into
        assert into[0].distance < 0 and into[0].link in tasks_robot.hand_links(0), into
        distances = [c.distance for c in into]
        assert distances == sorted(distances), "nearest first"
        (near,) = device.contacts(worlds, [0], [q], 1.0)
        assert all(0 <= c.distance < 1.0 for c in near), near
        assert any(isinstance(c, bp.Contact.World) for c in near), "the table is within a meter"
        listed.add(repr((free, into, near)))
    assert len(listed) == 1, "every device lists the same"


def test_contacts_name_both_links_a_folded_arm_folds_onto(ur5e_robot):
    folded = ur5e_robot.default_q.copy()
    folded[2] = 2.9
    device = bp.Device.cpu(ur5e_robot)
    worlds = device.upload([bp.World()])
    (contacts,) = device.contacts(worlds, [0], [folded], 0.0)
    pairs = [c.links for c in contacts if isinstance(c, bp.Contact.SelfCollision)]
    assert ("upper_arm_link", "wrist_2_link") in pairs, contacts
    assert all(c.distance < 0 for c in contacts), "a folded arm overlaps itself"


def test_evaluate_measures_clearances_and_the_cost(tasks_robot):
    q = tasks_robot.default_q
    ball = bp.Obstacle.sphere(tasks_robot.ee_pose(q).position, 0.1)
    # Exact up to 10 cm, as `default_pose_is_collision_free_on_table` reads them in Rust.
    weights = bp.CollisionWeights(world=1.0, margin=0.1, self_margin=0.1)
    for device in devices(tasks_robot):
        worlds = device.upload([bp.World([TABLE]), bp.World([TABLE, ball]), bp.World()])
        e = device.evaluate(worlds, [0, 1, 2], [q, q, q], weights)
        assert len(e) == 3 and e.gradient.shape == (3, tasks_robot.dof)
        assert e.world_clearance[0] > 0.05 and e.self_clearance[0] > 0, e.world_clearance
        assert e.collision_free(0) and not e.collision_free(1)
        assert e.world_clearance[1] < 0 and e.world_clearance[2] == np.float32(1e30), (
            "the ball overlaps the hand; an empty world has nothing to check"
        )
        assert e.cost[1] > e.cost[0] and np.any(e.gradient[1] != 0), "the ball pushes the hand"
        # Contacts measure what clearances do, as `contacts_name_what_each_clearance_measures`
        # holds them to in Rust.
        (into,) = device.contacts(worlds, [1], [q], 0.0)
        assert math.isclose(into[0].distance, e.world_clearance[1], abs_tol=1e-5), into
        unweighted = device.evaluate(worlds, [1], [q])
        assert unweighted.cost[0] == 0 and not np.any(unweighted.gradient), "the default weighs 0"
        with pytest.raises(IndexError):
            e.collision_free(3)


def test_configurations_go_in_one_row_each(tasks_robot):
    device = bp.Device.cpu(tasks_robot)
    worlds = device.upload([bp.World([TABLE])])
    q = np.stack([tasks_robot.default_q] * 2)
    assert len(device.contacts(worlds, [0, 0], q, 0.0)) == 2
    assert len(device.evaluate(worlds, [0, 0], q)) == 2
    # Turned the wrong way round, the batch holds as many numbers but is refused.
    with pytest.raises(bp.InputError, match=r"q needs the shape \(rows, 7\), got \[7, 2\]"):
        device.contacts(worlds, [0, 0], q.T, 0.0)
    with pytest.raises(bp.InputError, match=r"q needs the shape \(rows, 7\), got \[7, 2\]"):
        device.evaluate(worlds, [0, 0], q.T)
    # Rows of unequal lengths have no shape to check.
    ragged = [list(q[0]), list(q[0])[:6]]
    with pytest.raises(bp.InputError, match=r"q needs the shape \(rows, 7\)"):
        device.contacts(worlds, [0, 0], ragged, 0.0)
    with pytest.raises(bp.InputError, match="references world 1"):
        device.contacts(worlds, [0, 1], q, 0.0)
    with pytest.raises(bp.InputError, match="finite distance"):
        device.contacts(worlds, [0, 0], q, math.inf)
    with pytest.raises(bp.InputError, match="finite collision weights"):
        device.evaluate(worlds, [0, 0], q, bp.CollisionWeights(world=math.nan))
