"""Checks a batchplan LeRobot export with the real `lerobot` package.

    pip install "lerobot[dataset]"
    python scripts/validate_lerobot.py data/lerobot_demo
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
extension = json.loads((root / "meta/batchplan.json").read_text())

ds = LeRobotDataset("local/batchplan", root=root)
meta = ds.meta
assert ds.num_episodes == info["total_episodes"] and ds.num_frames == info["total_frames"]
print(f"loaded {ds.num_episodes} episodes / {ds.num_frames} frames at {ds.fps} fps, robot_type={meta.robot_type}")

item = ds[0]
dof = info["features"]["observation.state"]["shape"][0]
assert info["features"]["observation.state"]["names"][-1].endswith("gripper")
assert item["task"] in set(meta.tasks.index) and len(meta.tasks) == info["total_tasks"], item["task"]
assert item["observation.state"].shape == (dof,) and item["action"].shape == (dof,)
episode_files = sorted((root / "meta/episodes").glob("*/*.parquet"))
print(f"{len(episode_files)} episode metadata file(s), {len(meta.tasks)} task(s)")
print("sample:", {k: tuple(v.shape) if isinstance(v, torch.Tensor) else v for k, v in sorted(item.items())})

# Every episode is contiguous, correctly bounded, and its action is the next frame's state.
frames = ds.hf_dataset.with_format("numpy")
state = np.stack(frames["observation.state"])
action = np.stack(frames["action"])
episode = np.asarray(frames["episode_index"])
is_recovery = np.asarray(frames["is_recovery"])
parent = np.asarray(frames["parent_episode_index"])
task_index = np.asarray(frames["task_index"])
for e, row in enumerate(meta.episodes):
    lo, hi = row["dataset_from_index"], row["dataset_to_index"]
    assert hi - lo == row["length"] and (episode[lo:hi] == e).all()
    assert np.array_equal(action[lo : hi - 1], state[lo + 1 : hi]) and np.array_equal(action[hi - 1], state[hi - 1])
    origin = extension["episodes"][e]
    assert bool(is_recovery[lo]) == (origin["origin"] == "recovery")
    assert parent[lo] == origin.get("parent", -1)
    assert row["tasks"] == [meta.tasks.index[task_index[lo]]] and (task_index[lo:hi] == task_index[lo]).all()
print(f"episode boundaries, action = next state, recovery labels: ok ({int(is_recovery.sum())} recovery frames)")

# Normalization stats are what LeRobot loaded from meta/stats.json and match the data.
for key, values in [("observation.state", state), ("action", action)]:
    values = values.astype(np.float64)  # float32 sums drift by 1e-5 over 200k frames
    assert np.allclose(meta.stats[key]["mean"], values.mean(0), atol=1e-5)
    assert np.allclose(meta.stats[key]["std"], values.std(0), atol=1e-5)
print("stats.json matches the frames")

# Action chunks across episode ends: LeRobot pads past the boundary using episode metadata.
horizon = [i / ds.fps for i in range(10)]
chunked = LeRobotDataset("local/batchplan", root=root, delta_timestamps={"action": horizon})
last = meta.episodes[0]["dataset_to_index"] - 1
sample = chunked[last]
assert sample["action"].shape == (10, dof) and sample["action_is_pad"][1:].all() and not sample["action_is_pad"][0]
print("10-step action chunk at an episode end is padded correctly")

batch = next(iter(torch.utils.data.DataLoader(chunked, batch_size=32, shuffle=True)))
print("dataloader batch:", {k: tuple(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)})
print("OK")
