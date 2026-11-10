# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "robometrics @ git+https://github.com/fishbotics/robometrics.git@81e3d1d",
#   "trimesh",
#   "tabulate",
#   "tyro",
#   "tqdm",
#   "robot_descriptions",
#   "yourdfpy",
#   "scipy",
#   "nvidia-curobo==0.7.8+pt2.9.1cu126",
#   "torch==2.9.1",
#   "batchplan",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [[tool.uv.index]]
# name = "torch-packages-builder"
# url = "https://jianglongye.github.io/torch_packages_builder"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = ".", editable = true }
# torch = { index = "pytorch-cu126" }
# nvidia-curobo = { index = "torch-packages-builder" }
# batchplan = { path = "../batchplan/python" }
# ///
"""robokit's motion-plan benchmark (`benchmark.py`, unchanged) with two more configs, batchplan on
the CPU and on the GPU, through `batchplan_plugin`. Run from robokit's root:

    uv run batchplan_mp.py --config batchplan-cpu --sample-per-split 100 --output-dir out/mp200
"""

import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import batchplan_plugin  # noqa: E402
from scripts.benchmark.motion_plan import benchmark  # noqa: E402

for kind in ("cpu", "gpu"):
    benchmark.CONFIGS[f"batchplan-{kind}"] = {
        "backend": "robokit",
        "plugin": batchplan_plugin,
        "device": kind,
    }

if __name__ == "__main__":
    benchmark.main()
