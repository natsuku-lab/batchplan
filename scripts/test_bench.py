"""Regression fixtures for the comparison's judge (`validate.py`) and parity check (`parity.py`): each
fault it must catch, and a clean case it must pass. Needs the exported inputs
(`python scripts/bench_model.py export data/robometrics data/bench/inputs`).

    pytest scripts/test_bench.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import Judge, densify  # noqa: E402

INPUTS = Path(__file__).resolve().parent.parent / "data" / "bench" / "inputs"
pytestmark = pytest.mark.skipif(not (INPUTS / "problems.json").exists(), reason="no exported inputs")


@pytest.fixture(scope="module")
def judge():
    return Judge(INPUTS)


def free_straight(judge):
    """A problem whose straight line from start to goal is valid, and that line."""
    for pid, p in judge.problems.items():
        line = [p["start"], p["goal"]]
        if judge.path(pid, line)["valid"]:
            return pid, np.array(line)
    pytest.fail("no problem with a free straight line")


def corner_cut(path):
    """A path smoothed by cutting each interior corner at its neighbours' midpoints (one Chaikin pass)."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        out += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
    return np.array(out[:1] + out[2:-1] + [path[-1]])


def blocked_with_detour(judge, cut_collides=False):
    """A problem whose straight line hits the scene, and a valid path through one waypoint near the
    line's middle; with `cut_collides`, one whose corner-cut smoothing (`corner_cut`) collides.
    Candidates are rejected at a coarse step before the judge's full check."""
    rng = np.random.default_rng(3)
    r = judge.robot
    tried = 0
    for pid, p in judge.problems.items():
        start, goal = np.array(p["start"]), np.array(p["goal"])
        if judge.path(pid, [start, goal], step=0.05)["reason"] != "world collision":
            continue
        tried += 1
        if tried > 40:
            break
        for _ in range(60):
            via = np.clip((start + goal) / 2 + rng.normal(0, 0.6, 7), r.lower, r.upper)
            path = np.array([start, via, goal])
            if not judge.path(pid, path, step=0.05)["valid"] or not judge.path(pid, path)["valid"]:
                continue
            if not cut_collides or "collision" in judge.path(pid, corner_cut(path))["reason"]:
                return pid, path
    pytest.fail("no blocked problem with a suitable one-waypoint detour")


def test_a_free_straight_line_is_valid(judge):
    pid, line = free_straight(judge)
    v = judge.path(pid, line)
    assert v["valid"], v
    assert v["length"] == pytest.approx(np.linalg.norm(line[1] - line[0]))


def test_a_path_through_an_obstacle_is_caught(judge):
    pid, path = blocked_with_detour(judge)
    straight = judge.path(pid, [path[0], path[-1]])
    assert not straight["valid"] and "world collision" in straight["reason"], straight
    assert straight["min_world_clearance"] < 0


def test_smoothing_that_cuts_a_corner_into_the_scene_is_caught(judge):
    # The detour is valid; its smoothed version, which never passes its waypoint, is not.
    pid, path = blocked_with_detour(judge, cut_collides=True)
    assert judge.path(pid, path)["valid"]
    smoothed = judge.path(pid, corner_cut(path))
    assert not smoothed["valid"] and "collision" in smoothed["reason"], smoothed


def test_coarse_sampling_would_miss_what_the_judge_finds(judge):
    # The judge's step matters: the same blocked line checked only at its ends looks free.
    pid, path = blocked_with_detour(judge)
    ends = judge.path(pid, [path[0], path[-1]], step=10.0)
    fine = judge.path(pid, [path[0], path[-1]])
    assert ends["samples"] == 2 and fine["samples"] > 2
    assert not fine["valid"]


def test_wrong_joint_order_is_caught(judge):
    pid, line = free_straight(judge)
    swapped = line[:, [1, 0, 2, 3, 4, 5, 6]]
    if np.allclose(swapped, line):
        pytest.skip("joints 1 and 2 coincide")
    v = judge.path(pid, swapped)
    assert not v["valid"] and ("start" in v["reason"] or "goal" in v["reason"]), v


def test_a_wrong_start_is_caught(judge):
    # Starting partway along a free line: valid in every other way.
    pid, line = free_straight(judge)
    late = np.array([line[0] + 0.1 * (line[1] - line[0]), line[1]])
    v = judge.path(pid, late)
    assert not v["valid"] and v["reason"] == "start", v


def test_a_bounds_violation_is_caught(judge):
    pid, line = free_straight(judge)
    r = judge.robot
    out = np.array([line[0], r.upper + 0.05, line[1]])
    v = judge.path(pid, out)
    assert "bounds" in v["reason"], v


def test_a_wrong_endpoint_is_caught(judge):
    pid, line = free_straight(judge)
    short = np.array([line[0], line[0] + 0.9 * (line[1] - line[0])])
    v = judge.path(pid, short)
    assert not v["valid"] and v["reason"] == "goal", v


def rest_to_rest(line, dt, vmax):
    """A trapezoid-free cubic ease along a line, rest to rest, at peak speed `vmax`-scaled."""
    distance = np.abs(line[1] - line[0]).max()
    duration = max(1.5 * distance / vmax, 0.1)
    t = np.arange(0.0, duration + dt / 2, dt) / duration
    s = 3 * t**2 - 2 * t**3
    ds = (6 * t - 6 * t**2) / duration
    dds = (6 - 12 * t) / duration**2
    d = line[1] - line[0]
    return line[0] + s[:, None] * d, ds[:, None] * d, dds[:, None] * d


LIMITS = {"velocity": [2.175] * 4 + [2.61] * 3, "acceleration": [15, 7.5, 10, 12.5, 15, 20, 20]}


def test_a_trajectory_within_its_limits_is_valid(judge):
    pid, line = free_straight(judge)
    q, v, a = rest_to_rest(line, 0.001, 1.0)
    verdict = judge.trajectory(pid, np.arange(len(q)) * 0.001, q, v, a, LIMITS)
    assert verdict["valid"], verdict


def test_a_velocity_violation_is_caught(judge):
    pid, line = free_straight(judge)
    q, v, a = rest_to_rest(line, 0.001, 1.0)
    fast = dict(LIMITS, velocity=[0.5 * x for x in LIMITS["velocity"]])
    fast["velocity"] = [min(x, 0.5 * np.abs(v).max()) for x in fast["velocity"]]
    verdict = judge.trajectory(pid, np.arange(len(q)) * 0.001, q, v, a, fast)
    assert "velocity" in verdict["reason"], verdict


def test_velocities_that_contradict_the_positions_are_caught(judge):
    pid, line = free_straight(judge)
    q, v, a = rest_to_rest(line, 0.001, 1.0)
    verdict = judge.trajectory(pid, np.arange(len(q)) * 0.001, q, 0.5 * v, a, LIMITS)
    assert "inconsistent velocities" in verdict["reason"], verdict


def test_densify_bounds_the_step():
    pts = densify([[0.0] * 7, [1.0] + [0.0] * 6], 0.005)
    assert np.abs(np.diff(pts, axis=0)).max() <= 0.005 + 1e-12
    assert np.allclose(pts[0], 0) and np.allclose(pts[-1][0], 1)


def test_parity_catches_a_shifted_link(tmp_path):
    # A planner whose hand is 1 mm off must fail the parity comparison.
    from parity import compare

    parity = INPUTS.parent / "parity"
    if not (parity / "validator.json").exists():
        pytest.skip("no parity results")
    for name in ["fixtures.json", "validator.json"]:
        (tmp_path / name).write_text((parity / name).read_text())
    rows = json.loads((parity / "validator.json").read_text())["rows"]
    for r in rows:
        r["links"]["panda_hand"][0] += 1e-3
    (tmp_path / "roboplan.json").write_text(json.dumps({"rows": [{"links": r["links"]} for r in rows]}))
    with pytest.raises(SystemExit) as failed:
        compare(tmp_path)
    assert failed.value.code == 1
