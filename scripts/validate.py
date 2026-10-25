"""The comparison's judge: checks each pipeline's final output on the canonical Panda (`bench_model.py`),
sharing no code with any planner.

An output is a path (waypoints joined by straight lines) or a timed trajectory (positions, and the
velocities and accelerations the pipeline reports, every `dt` seconds). It is valid when:
- it starts at the problem's start (1e-4 rad) and ends at its joint-space goal (1e-3 rad);
- every joint stays within the URDF's limits (1e-4 rad);
- no sample collides with the scene or with itself, sampled at most `step` rad apart in every joint
  (0.005 by default), linearly between the output's own points. Sampled, not continuous.
A trajectory is also checked against velocity, acceleration and (optionally) jerk limits with a 1%
tolerance, on its motion: the velocities it reports (which must agree with finite differences of its
positions), their finite differences for acceleration, and theirs for jerk. Reported accelerations are
kept for information only: some pipelines report values that step at their own grid points.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_model import Panda, shape_from_json, world_clearance  # noqa: E402

START_TOLERANCE = 1e-4
GOAL_TOLERANCE = 1e-3
BOUNDS_TOLERANCE = 1e-4
LIMIT_TOLERANCE = 0.01


def densify(points, step):
    """Points no more than `step` apart in any joint, linearly between consecutive `points`."""
    points = np.asarray(points, dtype=float)
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        k = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        t = np.arange(1, k + 1)[:, None] / k
        out.append(a + t * (b - a))
    return np.concatenate(out)


class Judge:
    def __init__(self, inputs):
        inputs = Path(inputs)
        self.robot = Panda(inputs / "panda.urdf", inputs / "panda.srdf")
        self.problems = {p["id"]: p for p in json.loads((inputs / "problems.json").read_text())}
        # One-scene queries: their own start and goal, in their `scene_ref`'s scene.
        queries = inputs / "queries_onescene.json"
        if queries.exists():
            for q in json.loads(queries.read_text()):
                ref = self.problems[q["scene_ref"]]
                self.problems[q["id"]] = {**q, "scene": ref["scene"], "scene_capsules": ref["scene_capsules"]}
        self._shapes = {}

    def shapes(self, problem_id, capsules=False):
        problem_id = self.problems[problem_id].get("scene_ref", problem_id)
        key = (problem_id, capsules)
        if key not in self._shapes:
            scene = self.problems[problem_id]["scene_capsules" if capsules else "scene"]
            self._shapes[key] = [shape_from_json(s) for s in scene]
        return self._shapes[key]

    def path(self, problem_id, points, step=0.005, capsules=False):
        """Checks a path; returns a verdict with its reason and metrics."""
        p = self.problems[problem_id]
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 7 or len(points) < 2:
            return {"valid": False, "reason": "malformed"}
        verdict = {
            "start_error": float(np.abs(points[0] - p["start"]).max()),
            "goal_error": float(np.abs(points[-1] - p["goal"]).max()),
            "length": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        }
        dense = densify(points, step)
        r = self.robot
        bounds = float(max((r.lower - dense).max(), (dense - r.upper).max(), 0.0))
        centres = r.sphere_centres(dense)
        world = world_clearance(r, centres, self.shapes(problem_id, capsules))
        self_ = r.self_clearance(centres)
        verdict.update({
            "samples": len(dense),
            "bounds_excess": bounds,
            "min_world_clearance": float(world.min()),
            "min_self_clearance": float(self_.min()),
        })
        reasons = []
        if verdict["start_error"] > START_TOLERANCE:
            reasons.append("start")
        if verdict["goal_error"] > GOAL_TOLERANCE:
            reasons.append("goal")
        if bounds > BOUNDS_TOLERANCE:
            reasons.append("bounds")
        if world.min() <= 0:
            reasons.append("world collision")
        if self_.min() <= 0:
            reasons.append("self collision")
        verdict["valid"] = not reasons
        verdict["reason"] = ", ".join(reasons)
        return verdict

    def trajectory(self, problem_id, times, positions, velocities, accelerations, limits, step=0.005, capsules=False):
        """Checks a timed trajectory: its path as `path` does, plus its timing against `limits`
        ({"velocity": [7], "acceleration": [7], optional "jerk": [7]}), starting and ending at rest."""
        times = np.asarray(times, dtype=float).reshape(-1)
        q = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        a = np.asarray(accelerations, dtype=float)
        verdict = self.path(problem_id, q, step, capsules)
        if verdict.get("reason") == "malformed":
            return verdict
        reasons = [verdict["reason"]] if verdict["reason"] else []
        if len(times) != len(q) or np.any(np.diff(times) <= 0):
            verdict.update({"valid": False, "reason": "malformed times"})
            return verdict
        verdict["duration"] = float(times[-1] - times[0])
        vmax, amax = np.asarray(limits["velocity"]), np.asarray(limits["acceleration"])
        verdict["velocity_ratio"] = float((np.abs(v) / vmax).max())
        motion = np.diff(v, axis=0) / np.diff(times)[:, None]
        verdict["acceleration_ratio"] = float((np.abs(motion) / amax).max()) if len(motion) else 0.0
        verdict["reported_acceleration_ratio"] = float((np.abs(a) / amax).max())
        if verdict["velocity_ratio"] > 1 + LIMIT_TOLERANCE:
            reasons.append("velocity")
        if verdict["acceleration_ratio"] > 1 + LIMIT_TOLERANCE:
            reasons.append("acceleration")
        if "jerk" in limits and len(motion) > 1:
            mid = (times[1:] + times[:-1]) / 2
            jerk = np.diff(motion, axis=0) / np.diff(mid)[:, None]
            verdict["jerk_ratio"] = float((np.abs(jerk) / np.asarray(limits["jerk"])).max())
            if verdict["jerk_ratio"] > 1 + LIMIT_TOLERANCE:
                reasons.append("jerk")
        # The reported velocities must be the positions' derivatives (central differences).
        if len(q) > 2:
            fd = (q[2:] - q[:-2]) / (times[2:] - times[:-2])[:, None]
            mid = v[1:-1]
            verdict["velocity_consistency"] = float((np.abs(fd - mid) / vmax).max())
            if verdict["velocity_consistency"] > 0.05:
                reasons.append("inconsistent velocities")
        rest = max(np.abs(v[0]).max(), np.abs(v[-1]).max())
        if rest / vmax.min() > LIMIT_TOLERANCE:
            reasons.append("not at rest")
        verdict["valid"] = not reasons
        verdict["reason"] = ", ".join(reasons)
        return verdict


def read_outputs(stem):
    """The arrays an adapter wrote (`compare.py`'s `write_outputs`): {key: array}."""
    data = np.fromfile(f"{stem}.bin", dtype="<f4")
    index = json.loads(Path(f"{stem}.json").read_text())
    return {k: data[o:o + r * c].reshape(r, c) for k, (o, r, c) in index.items()}


def _judge_chunk(args):
    """Judges a chunk of a run's records (in a worker process): their verdict lines."""
    inputs, run_dir, step, chunk = args
    judge = Judge(inputs)
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    capsules = manifest.get("scene") == "capsules"
    outputs, lines = {}, []
    for rec in chunk:
        seed = rec["seed"]
        if seed not in outputs:
            outputs[seed] = read_outputs(run_dir / f"outputs_seed{seed}")
        o, key = outputs[seed], rec["problem"]
        if manifest["workload"] == "B" and f"{key}|q" in o:
            v = judge.trajectory(key, o[f"{key}|t"], o[f"{key}|q"], o[f"{key}|v"], o[f"{key}|a"],
                                 manifest["limits"], step, capsules)
        else:
            v = judge.path(key, o[f"{key}|path"], step, capsules)
        lines.append(json.dumps({"problem": key, "seed": seed, **v}))
    return lines


def judge_run(inputs, run_dir, step=0.005, processes=None):
    """Adds `verdicts.jsonl` to a run directory: every first-repetition answer a planner gave, judged
    on its final output (the timed trajectory in workload B, the path in A), on all cores."""
    import multiprocessing as mp

    run_dir = Path(run_dir)
    with open(run_dir / "records.jsonl") as f:
        records = [r for r in map(json.loads, f)
                   if r["rep"] == 1 and r.get("api", "single") == "single" and r["status"] == "solved"]
    processes = processes or mp.cpu_count()
    size = max(1, -(-len(records) // (4 * processes)))
    chunks = [(str(inputs), str(run_dir), step, records[i:i + size]) for i in range(0, len(records), size)]
    with mp.get_context("spawn").Pool(processes) as pool, open(run_dir / "verdicts.jsonl", "w") as out:
        for lines in pool.imap(_judge_chunk, chunks):
            for line in lines:
                out.write(line + "\n")


if __name__ == "__main__":
    # validate.py run <inputs> <run_dir> [step]: judges a run's outputs into its verdicts.jsonl.
    command, *rest = sys.argv[1:]
    assert command == "run", "usage: validate.py run <inputs> <run_dir> [step]"
    judge_run(rest[0], rest[1], float(rest[2]) if len(rest) > 2 else 0.005)
