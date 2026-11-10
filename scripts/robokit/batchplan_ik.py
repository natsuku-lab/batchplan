# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "torch==2.9.1",
#   "robot_descriptions",
#   "yourdfpy",
#   "batchplan",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = ".", editable = true }
# torch = { index = "pytorch-cu126" }
# batchplan = { path = "../batchplan/python" }
# ///
"""batchplan in robokit's IK benchmark: their targets, timer and scorer (IKBenchmarkHelper.evaluate),
with batchplan as the solver. Run from robokit's root, in an environment with robokit's benchmark
dependencies and batchplan installed:

    python batchplan_ik.py --device gpu --batch-sizes 1,10,100,1000,2000 --out results.csv

Collision is off, as for every solver in their IK benchmark: the robot loads without spheres.
Problems are built from the targets outside the timer, as their warp arrays are; the timed solve is
`solve_ik` and picking each problem's seed.
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())  # robokit's root, for `scripts.benchmark`
from scripts.benchmark.ik.helper import IKBenchmarkHelper, load_panda_urdf_fixed  # noqa: E402

import batchplan as bp  # noqa: E402


class BatchplanIK(IKBenchmarkHelper):
    EXPECTED_TENSOR_TYPE = "warp"  # the scorer's arrays; batchplan itself takes NumPy
    EXPECTED_DEVICE = os.environ.get("RK_SCORER_DEVICE", "cuda")

    def __init__(self, batch_size: int, device: str, options: bp.IkOptions):
        super().__init__(batch_size=batch_size)
        urdf = load_panda_urdf_fixed()
        path = Path(tempfile.mkdtemp()) / "panda.urdf"
        urdf.write_xml_file(path)
        from robot_descriptions import panda_description

        robot_options = bp.RobotOptions(
            ee_links=[self.target_link_name],
            collision_model=bp.CollisionModel(),  # no spheres: collision off
            package_dirs=[Path(panda_description.PACKAGE_PATH).parent],
        )
        self.robot = bp.Robot.load(path, robot_options)
        self.device = bp.Device.gpu(self.robot) if device == "gpu" else bp.Device.cpu(self.robot)
        self.worlds = self.device.upload([bp.World()])
        self.options = options
        names = list(self.robot.joint_names)
        self.order = [names.index(n) for n in self.reference_joint_names]

    def _convert_target_poses_to_native(self, T_world_target):
        # robokit's targets are x, y, z, qw, qx, qy, qz; batchplan's rotations are scalar last.
        position, wxyz = T_world_target[:, :3], T_world_target[:, 3:]
        xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=1)
        return [bp.IkProblem(0, bp.Pose(p, r)) for p, r in zip(position, xyzw, strict=True)]

    def _synchronize(self, arrays):
        pass  # solve_ik returns once the batch is done

    def warmup(self):
        self.device.warmup()

    def solve_ik(self, problems):
        result = bp.solve_ik(self.device, self.worlds, problems, self.options)
        # Each problem's successful seed if any, else its closest: the combined error, as IK
        # weighs it, with successes first.
        error = result.position_error + self.options.rotation_weight * result.rotation_error
        error = np.where(result.success, error - 1e9, error)
        best = error.argmin(axis=1)
        return result.q[np.arange(len(problems)), best]

    def reindex_joints_to_reference(self, q):
        return q[:, self.order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--batch-sizes", default="1,10,100,1000,2000")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-step", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    options = bp.IkOptions(seeds=args.seeds, iterations=args.iterations, max_step=args.max_step)
    rows = []
    for batch in [int(b) for b in args.batch_sizes.split(",")]:
        helper = BatchplanIK(batch, args.device, options)
        out = helper.evaluate(num_runs=args.runs)
        row = {
            "solver": f"batchplan {args.device} ({options.seeds} x {options.iterations}, step {options.max_step})",
            "device": helper.device.name,
            "batch": batch,
            "time_ms": float(np.mean(out.time_ms)),
            "time_ms_std": float(np.std(out.time_ms)),
            "succ": float(np.mean(out.success)),
            "pos_err_mm": float(np.mean(out.position_mm)),
            "ori_err_rad": float(np.mean(out.orientation_rad)),
        }
        rows.append(row)
        print(row, flush=True)
    if args.out:
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
