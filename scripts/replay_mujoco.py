"""Renders camera videos for a batchplan LeRobot export by replaying its episodes in MuJoCo.

batchplan plans the motions and renders nothing. This replays each frame kinematically:
- the robot's parts (`meta/batchplan.json`), loaded from their URDF or MJCF and placed at their bases;
- each obstacle where `observation.environment_state` has it, so the carried object moves with the hand;
- the fingers following each gripper's opening as `meta/replay.json` maps it, closing until they
  touch the carried object.

It writes a new LeRobot dataset: the same frames, plus a video from each camera `replay.json` places.

    uv pip install --python .venv/bin/python mujoco fast-simplification
    .venv/bin/python scripts/replay_mujoco.py <dataset> <out_dir>
"""

import json
import re
import shutil
import sys
import time
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np
from lerobot.datasets import LeRobotDataset

HEIGHT, WIDTH = 480, 640
# Rendering time grows with triangles: CAD exports carry far more than a camera image shows.
MAX_FACES = 2000
# Obstacle kinds as `observation.environment_state` numbers them. A distance grid shows as its box.
BOX = mujoco.mjtGeom.mjGEOM_BOX
KINDS = [BOX, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE, BOX]
GOAL, OBSTACLE = 7, 12  # floats for the goal pose, then for each obstacle
# Features LeRobot writes itself.
AUTOMATIC = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def wxyz(q):
    """MuJoCo's quaternion order from batchplan's xyzw."""
    return [q[3], q[0], q[1], q[2]]


def load_part(path: Path) -> mujoco.MjSpec:
    """One description, with its visual meshes (or its collision ones if it has none) simplified."""
    if path.suffix != ".urdf":
        spec = mujoco.MjSpec.from_file(str(path))
    else:
        # MuJoCo reads URDF compiler options from a <mujoco> element: keep visual geoms, keep fixed
        # links (the hands' frames carry cameras), resolve mesh paths from the file, and accept
        # massless links (nothing here is simulated).
        text = re.sub(r'(<(?:visual|collision)) name="[^"]*"', r"\1", path.read_text())
        options = (
            '<mujoco><compiler discardvisual="false" fusestatic="false" strippath="false"'
            ' boundmass="1e-6" boundinertia="1e-9"/></mujoco>'
        )
        spec = mujoco.MjSpec.from_string(re.sub(r"(<robot[^>]*>)", r"\1" + options, text, count=1))
        spec.modelfiledir = str(path.parent)
        # MuJoCo puts a URDF's collision geoms in group 0 and its visual ones in group 1.
        if any(g.group == 1 for g in spec.geoms):
            for g in [g for g in spec.geoms if g.group == 0]:
                spec.delete(g)
    for mesh in [m for m in spec.meshes if m.file]:
        simplify(mesh, Path(spec.modelfiledir) / spec.meshdir / mesh.file)
    return spec


def simplify(mesh, path: Path):
    """Replaces a binary STL mesh of more than `MAX_FACES` triangles with a simplified copy."""
    raw = path.read_bytes() if path.suffix.lower() == ".stl" else b""
    count = int(np.frombuffer(raw, np.uint32, 1, 80)[0]) if len(raw) >= 84 else 0
    if count <= MAX_FACES or len(raw) != 84 + 50 * count:
        return
    record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
    corners = np.frombuffer(raw, record, count, 84)["corners"].reshape(-1, 3)
    vertices, faces = np.unique(corners, axis=0, return_inverse=True)
    vertices, faces = fast_simplification.simplify(vertices, faces.reshape(-1, 3), 1 - MAX_FACES / count)
    mesh.file = ""
    mesh.uservert = vertices.astype(np.float32).ravel()
    mesh.userface = faces.astype(np.int32).ravel()


def xyaxes(pos, target, up):
    """A camera at `pos` looking at `target`, `up` upward in its image."""
    forward = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return [*right, *np.cross(right, forward)]


def robot_spec(robot: dict, cameras: dict) -> mujoco.MjSpec:
    """The robot's parts at their bases, a light, and the cameras."""
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = WIDTH, HEIGHT
    spec.add_texture(
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.9, 0.92, 0.95],
        rgb2=[0.55, 0.6, 0.66],
        width=256,
        height=256,
    )
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    for part in robot["parts"]:
        frame = spec.worldbody.add_frame(pos=part["base"][:3], quat=wxyz(part["base"][3:]))
        spec.attach(load_part(Path(part["path"])), prefix=part["prefix"], frame=frame)
    for name, c in cameras.items():
        body = spec.body(c["frame"]) if "frame" in c else spec.worldbody
        body.add_camera(name=name, pos=c["pos"], xyaxes=xyaxes(c["pos"], c["target"], c["up"]), fovy=c["fovy"])
    return spec


def obstacle(env: np.ndarray, k: int) -> np.ndarray:
    """Obstacle `k` of a frame's environment state: present, kind, center, size, quaternion xyzw."""
    return env[GOAL + OBSTACLE * k : GOAL + OBSTACLE * (k + 1)]


def episode_model(spec: mujoco.MjSpec, env: np.ndarray, carried: int | None) -> mujoco.MjModel:
    """`spec` with a body `obstacle{k}` for each obstacle present at the episode's first frame."""
    spec = spec.copy()
    present = [k for k in range((len(env) - GOAL) // OBSTACLE) if obstacle(env, k)[0]]
    largest = max(present, key=lambda k: np.prod(obstacle(env, k)[5:8]))
    for k in present:
        o = obstacle(env, k)
        kind = int(o[1])
        size = o[5:8] if KINDS[kind] == BOX else [o[5], o[7], 0]
        # The carried object red, the largest (a table) light wood, the rest slate; grids see-through.
        rgba = [0.85, 0.25, 0.2, 1] if k == carried else [0.76, 0.66, 0.52, 1] if k == largest else [0.4, 0.47, 0.6, 1]
        rgba[3] = 0.4 if kind == 4 else 1
        body = spec.worldbody.add_body(name=f"obstacle{k}", mocap=True)
        body.add_geom(type=KINDS[kind], size=size, rgba=rgba, contype=0, conaffinity=0, group=2)
    return spec.compile()


def finger_positions(fingers: list, opening: float, stop: float) -> list:
    """Each finger joint's position at `opening` (1 open, 0 closed), closed fingers `stop` from the middle."""
    out = []
    for f in fingers:
        half = abs(f["open"])
        closed = min(stop, half)
        out.append((f["joint"], np.sign(f["open"]) * (closed + opening * (half - closed))))
    return out


def pose(model, data, names, state, env, carried, grippers):
    """Sets the joints, fingers and obstacles of one frame."""
    for i, name in enumerate(names):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            data.qpos[model.joint(name).qposadr[0]] = state[i]
    # Grasps close across the carried object's narrower side.
    stop = min(obstacle(env, carried)[5:7]) if carried is not None else 0.0
    for gripper, fingers in grippers.items():
        for joint, q in finger_positions(fingers, state[names.index(gripper)], stop):
            data.qpos[model.joint(joint).qposadr[0]] = q
    for body in range(model.nbody):
        if model.body_mocapid[body] >= 0:
            o = obstacle(env, int(model.body(body).name.removeprefix("obstacle")))
            data.mocap_pos[model.body_mocapid[body]] = o[2:5]
            data.mocap_quat[model.body_mocapid[body]] = wxyz(o[8:12])
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    info = json.loads((source / "meta/info.json").read_text())
    extension = json.loads((source / "meta/batchplan.json").read_text())
    replay = json.loads((source / "meta/replay.json").read_text())
    src = LeRobotDataset("local/source", root=source)
    frames = src.hf_dataset.with_format("numpy")
    extra = {k: v for k, v in info["features"].items() if k not in AUTOMATIC}
    columns = {k: np.asarray(frames[k]) for k in [*extra, "task_index"]}
    names = info["features"]["observation.state"]["names"]

    cameras = {name: {"up": [0, 0, 1], "fovy": 58, **c} for name, c in replay["cameras"].items()}
    spec = robot_spec(extension["robot"], cameras)
    video = {"dtype": "video", "shape": [HEIGHT, WIDTH, 3], "names": ["height", "width", "channels"]}
    features = extra | {f"observation.images.{name}": video for name in cameras}
    shutil.rmtree(out, ignore_errors=True)
    dataset = LeRobotDataset.create(
        f"local/{out.name}", info["fps"], features, root=out, robot_type=info["robot_type"], streaming_encoding=True
    )
    started = time.time()
    for e, row in enumerate(src.meta.episodes):
        lo, hi = row["dataset_from_index"], row["dataset_to_index"]
        env = columns["observation.environment_state"]
        carried = extension["episodes"][e].get("carried_obstacle")
        model = episode_model(spec, env[lo], carried)
        data = mujoco.MjData(model)
        with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
            for f in range(lo, hi):
                pose(model, data, names, columns["observation.state"][f], env[f], carried, replay["grippers"])
                frame = {k: np.asarray(columns[k][f], dtype=v["dtype"]).reshape(v["shape"]) for k, v in extra.items()}
                frame["task"] = src.meta.tasks.index[columns["task_index"][f]]
                for name in cameras:
                    renderer.update_scene(data, camera=name)
                    frame[f"observation.images.{name}"] = renderer.render()
                dataset.add_frame(frame)
        dataset.save_episode()
        print(f"episode {e + 1} of {src.num_episodes}: {hi - lo} frames, {time.time() - started:.0f} s", flush=True)
    dataset.finalize()
    for name in ["batchplan.json", "replay.json"]:
        shutil.copy(source / "meta" / name, out / "meta" / name)
    print(f"wrote {out}: {src.num_episodes} episodes, {src.num_frames} frames, cameras {', '.join(cameras)}")


if __name__ == "__main__":
    main()
