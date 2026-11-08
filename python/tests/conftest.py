"""Robots from the repository's assets, and the devices the tests run on."""

import os
from pathlib import Path

import pytest

import batchplan as bp

ASSETS = Path(__file__).resolve().parents[2] / "assets"


def ur5e() -> bp.Robot:
    """The UR5e with its fitted spheres."""
    return bp.Robot.load(ASSETS / "ur5e/ur_description/urdf/ur5e.urdf")


def panda() -> bp.Robot:
    """The Panda with its committed collision model and its hand as the IK frame."""
    model = bp.CollisionModel.load(ASSETS / "franka/panda_collision.json")
    options = bp.RobotOptions(ee_links=["panda_hand"], collision_model=model)
    return bp.Robot.load(ASSETS / "franka/franka_panda.urdf", options)


def tasks_panda() -> bp.Robot:
    """The Panda as the examples load it: its tool frame as the hand, fingers held open."""
    model = bp.CollisionModel.load(ASSETS / "franka/panda_collision.json")
    options = bp.RobotOptions(
        ee_links=["ee_link"],
        lock_joints={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
        default_q=[0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8],
        collision_model=model,
    )
    return bp.Robot.load(ASSETS / "franka/franka_panda.urdf", options)


def devices(robot: bp.Robot) -> list[bp.Device]:
    """The CPU, and a GPU when one exists.

    Without one, GPU cases skip unless `BATCHPLAN_REQUIRE_GPU` is set, as in the Rust tests.
    """
    found = [bp.Device.cpu(robot)]
    try:
        found.append(bp.Device.gpu(robot))
    except bp.GpuError:
        if os.environ.get("BATCHPLAN_REQUIRE_GPU"):
            raise
    return found


@pytest.fixture(scope="session")
def ur5e_robot() -> bp.Robot:
    return ur5e()


@pytest.fixture(scope="session")
def panda_robot() -> bp.Robot:
    return panda()


@pytest.fixture(scope="session")
def tasks_robot() -> bp.Robot:
    return tasks_panda()
