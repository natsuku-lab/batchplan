"""The canonical Panda model and scenes of the planner comparison, independent of every planner.

One spherized URDF (`vamp_panda_spherized.urdf`) and one SRDF (`vamp_panda.srdf`, the same 34 disabled
pairs as MoveIt's Panda SRDF) describe the robot for every planner and for the validator. This module
reads them with nothing but numpy: forward kinematics of the URDF's joints, each link's spheres, the
self-collision pairs the SRDF leaves enabled, and exact distances from spheres to the scene's boxes,
cylinders and capsules. Scenes are the robometrics problems' obstacles, with boxes the base sphere
touches cut around it in every planner's scene (`around_base`), and, for planners that only take
capsules, every cylinder as the capsule around it (`capsules=True`).
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

URDF_SHA256 = "d7d891b892e805d399d506a5c3f3cfc698ac75e48ec55449bde025dc7403291c"
ARM = [f"panda_joint{i}" for i in range(1, 8)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def rpy_matrix(r, p, y):
    """URDF's fixed-axis roll, pitch, yaw: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = [float(v) for v in element.get("xyz", "0 0 0").split()]
        t[:3, :3] = rpy_matrix(*[float(v) for v in element.get("rpy", "0 0 0").split()])
    return t


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class Panda:
    """The canonical spherized Panda: kinematics, spheres and self-collision pairs."""

    def __init__(self, urdf, srdf):
        root = ET.parse(urdf).getroot()
        self.joints = []
        for j in root.findall("joint"):
            axis = j.find("axis")
            self.joints.append(Joint(
                j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                _origin(j.find("origin")),
                np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.zeros(3),
            ))
        children = {j.child for j in self.joints}
        self.links = [l.get("name") for l in root.findall("link")]
        self.base = next(l for l in self.links if l not in children)
        limits = {j.get("name"): j.find("limit") for j in root.findall("joint")}
        self.lower = np.array([float(limits[n].get("lower")) for n in ARM])
        self.upper = np.array([float(limits[n].get("upper")) for n in ARM])
        # Spheres in their link's frame: (link, centre, radius), in the URDF's order.
        self.spheres = []
        for link in root.findall("link"):
            for c in link.findall("collision"):
                sphere = c.find("geometry/sphere")
                assert sphere is not None, f"{link.get('name')}: only spheres are canonical"
                self.spheres.append((link.get("name"), _origin(c.find("origin"))[:3, 3], float(sphere.get("radius"))))
        self.sphere_link = np.array([self.links.index(l) for l, _, _ in self.spheres])
        self.sphere_centre = np.array([c for _, c, _ in self.spheres])
        self.sphere_radius = np.array([r for _, _, r in self.spheres])
        disabled = {frozenset((d.get("link1"), d.get("link2"))) for d in ET.parse(srdf).getroot().findall("disable_collisions")}
        with_spheres = sorted({l for l, _, _ in self.spheres}, key=self.links.index)
        self.self_pairs = [
            (a, b) for i, a in enumerate(with_spheres) for b in with_spheres[i + 1:] if frozenset((a, b)) not in disabled
        ]
        # Sphere index pairs checked for self-collision: spheres of two different, enabled links.
        enabled = {frozenset(p) for p in self.self_pairs}
        pairs = [(i, k) for i in range(len(self.spheres)) for k in range(i + 1, len(self.spheres))
                 if frozenset((self.spheres[i][0], self.spheres[k][0])) in enabled]
        self.pair_i = np.array([i for i, _ in pairs])
        self.pair_k = np.array([k for _, k in pairs])

    def link_transforms(self, q):
        """Every link's pose in the base frame, `[n, links, 4, 4]`, for configurations `q` `[n, 7]`."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        n = len(q)
        out = np.zeros((n, len(self.links), 4, 4))
        out[:, self.links.index(self.base)] = np.eye(4)
        done = {self.base}
        pending = list(self.joints)
        while pending:
            for j in list(pending):
                if j.parent not in done:
                    continue
                t = out[:, self.links.index(j.parent)] @ j.origin
                if j.kind in ("revolute", "continuous"):
                    angle = q[:, ARM.index(j.name)]
                    rot = np.zeros((n, 4, 4))
                    rot[:, 3, 3] = 1
                    rot[:, :3, :3] = _axis_angle(j.axis, angle)
                    t = t @ rot
                else:
                    assert j.kind == "fixed", j.kind
                out[:, self.links.index(j.child)] = t
                done.add(j.child)
                pending.remove(j)
        return out

    def sphere_centres(self, q):
        """Every sphere's centre in the base frame, `[n, spheres, 3]`."""
        t = self.link_transforms(q)[:, self.sphere_link]
        return np.einsum("nsij,sj->nsi", t[..., :3, :3], self.sphere_centre) + t[..., :3, 3]

    def self_clearance(self, centres):
        """Smallest surface distance between spheres of enabled link pairs, per configuration."""
        d = np.linalg.norm(centres[:, self.pair_i] - centres[:, self.pair_k], axis=-1)
        return (d - self.sphere_radius[self.pair_i] - self.sphere_radius[self.pair_k]).min(axis=1)


def _axis_angle(axis, angle):
    """Rotations about a unit `axis` by each angle, `[n, 3, 3]` (Rodrigues)."""
    x, y, z = axis / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


@dataclass
class Box:
    centre: np.ndarray
    rot: np.ndarray
    half: np.ndarray


@dataclass
class Cylinder:
    """Axis along the local z."""
    centre: np.ndarray
    rot: np.ndarray
    radius: float
    half_height: float


@dataclass
class Capsule:
    a: np.ndarray
    b: np.ndarray
    radius: float


def distance(shape, points):
    """Signed distance from `points` `[..., 3]` to `shape` (negative inside)."""
    if isinstance(shape, Capsule):
        ab = shape.b - shape.a
        t = np.clip(((points - shape.a) @ ab) / (ab @ ab), 0.0, 1.0)
        return np.linalg.norm(points - (shape.a + t[..., None] * ab), axis=-1) - shape.radius
    local = (points - shape.centre) @ shape.rot
    if isinstance(shape, Box):
        q = np.abs(local) - shape.half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return outside + np.minimum(q.max(axis=-1), 0.0)
    radial = np.linalg.norm(local[..., :2], axis=-1) - shape.radius
    axial = np.abs(local[..., 2]) - shape.half_height
    outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
    return outside + np.minimum(np.maximum(radial, axial), 0.0)


def load_problems(data):
    """The robometrics sets, as {set name: [problem, ...]} in their files' order."""
    sets = {}
    for name in ["mb_set.yaml", "mpinets_set.yaml"]:
        sets.update(yaml.safe_load((Path(data) / name).read_text()))
    return sets


def around_base(panda, centre, rot, half):
    """A box as boxes (centre, half extents) with its rotation, leaving a hole along its z axis a
    centimetre wider than the base sphere wherever that sphere touches it."""
    base = [(panda.sphere_centre[i], panda.sphere_radius[i]) for i in range(len(panda.spheres))
            if panda.spheres[i][0] == panda.base]
    boxes = [(np.array(centre, dtype=float), np.array(half, dtype=float))]
    for b, r in base:
        kept = []
        for c, h in boxes:
            local = rot.T @ (b - c)
            if np.linalg.norm(local - np.clip(local, -h, h)) >= r:
                kept.append((c, h))
                continue
            assert abs(rot[2, 2]) > 0.999, "only boxes lying flat are cut around the base"
            lo, hi = np.maximum(local[:2] - r - 0.01, -h[:2]), np.minimum(local[:2] + r + 0.01, h[:2])
            for x0, x1, y0, y1 in [(-h[0], lo[0], -h[1], h[1]), (hi[0], h[0], -h[1], h[1]),
                                   (lo[0], hi[0], -h[1], lo[1]), (lo[0], hi[0], hi[1], h[1])]:
                if x1 > x0 and y1 > y0:
                    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
                    kept.append((c + rot @ mid, np.array([(x1 - x0) / 2, (y1 - y0) / 2, h[2]])))
        boxes = kept
    return boxes


def scene(panda, problem, capsules=False):
    """The problem's obstacles in the base frame: boxes cut around the base sphere, and cylinders,
    exact or (`capsules`) as the capsule around each."""
    shapes = []
    for c in (problem["obstacles"].get("cuboid") or {}).values():
        x, y, z, qw, qx, qy, qz = c["pose"]
        rot = rotation(qw, qx, qy, qz)
        for centre, half in around_base(panda, [x, y, z], rot, [d / 2 for d in c["dims"]]):
            shapes.append(Box(centre, rot, half))
    for c in (problem["obstacles"].get("cylinder") or {}).values():
        x, y, z, qw, qx, qy, qz = c["pose"]
        rot = rotation(qw, qx, qy, qz)
        centre = np.array([x, y, z], dtype=float)
        if capsules:
            half = rot[:, 2] * c["height"] / 2
            shapes.append(Capsule(centre - half, centre + half, float(c["radius"])))
        else:
            shapes.append(Cylinder(centre, rot, float(c["radius"]), c["height"] / 2))
    return shapes


def world_clearance(panda, centres, shapes):
    """Smallest surface distance between the robot's spheres, the base's left out, and the scene,
    per configuration (`centres` `[n, spheres, 3]`)."""
    moving = panda.sphere_link != panda.links.index(panda.base)
    pts, radii = centres[:, moving], panda.sphere_radius[moving]
    if not shapes:
        return np.full(len(centres), np.inf)
    return np.min([(distance(s, pts) - radii).min(axis=1) for s in shapes], axis=0)


def quaternion(rot):
    """A rotation matrix as a unit quaternion (w, x, y, z)."""
    t = np.trace(rot)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s]
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        q = [(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s]
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        q = [(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        q = [(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def euler_xyz_from(rot):
    """VAMP's orientation of a rotation matrix: extrinsic x, y, z angles, R = Rz Ry Rx."""
    return [float(np.arctan2(rot[2, 1], rot[2, 2])), float(-np.arcsin(np.clip(rot[2, 0], -1, 1))),
            float(np.arctan2(rot[1, 0], rot[0, 0]))]


def shape_json(shape):
    if isinstance(shape, Box):
        return {"kind": "box", "centre": shape.centre.tolist(), "wxyz": quaternion(shape.rot).tolist(), "half": shape.half.tolist()}
    if isinstance(shape, Cylinder):
        return {"kind": "cylinder", "centre": shape.centre.tolist(), "wxyz": quaternion(shape.rot).tolist(),
                "radius": shape.radius, "half_height": shape.half_height}
    return {"kind": "capsule", "a": shape.a.tolist(), "b": shape.b.tolist(), "radius": shape.radius}


def shape_from_json(s):
    if s["kind"] == "capsule":
        return Capsule(np.array(s["a"]), np.array(s["b"]), s["radius"])
    rot = rotation(*s["wxyz"])
    if s["kind"] == "box":
        return Box(np.array(s["centre"]), rot, np.array(s["half"]))
    return Cylinder(np.array(s["centre"]), rot, s["radius"], s["half_height"])


def split(index, count):
    """Each set's first 20% tunes, its last 10% is the final evaluation, the rest is held out."""
    if index < round(0.2 * count):
        return "tuning"
    if index >= count - round(0.1 * count):
        return "final"
    return "heldout"


def export(data, out):
    """Writes the canonical inputs every adapter reads: the URDF without its visuals (kinematics and
    spheres unchanged), the SRDF, and every problem with both scenes."""
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    urdf, srdf = data / "vamp_panda_spherized.urdf", data / "vamp_panda.srdf"
    assert sha256(urdf) == URDF_SHA256, f"{urdf} is not the canonical spherized URDF"
    panda = Panda(urdf, srdf)
    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):
        for v in link.findall("visual"):
            link.remove(v)
    ET.ElementTree(root).write(out / "panda.urdf", encoding="utf-8", xml_declaration=True)
    (out / "panda.srdf").write_bytes(srdf.read_bytes())
    problems = []
    for name, ps in load_problems(data).items():
        for i, p in enumerate(ps):
            problems.append({
                "id": f"{name}/{i}", "set": name, "index": i, "split": split(i, len(ps)),
                "start": [float(v) for v in p["start"]], "goal": [float(v) for v in p["goal_ik"][0]],
                "scene": [shape_json(s) for s in scene(panda, p)],
                "scene_capsules": [shape_json(s) for s in scene(panda, p, capsules=True)],
            })
    (out / "problems.json").write_text(json.dumps(problems))
    # Repeated queries in one scene: per split, the first problem of each set's scene, with 25 start and
    # goal pairs drawn from configurations at least 1 cm clear of it and of the robot itself.
    rng = np.random.default_rng(11)
    one = []
    for split_name in ["tuning", "heldout", "final"]:
        for set_name in load_problems(data):
            ref = next(p for p in problems if p["set"] == set_name and p["split"] == split_name)
            shapes = [shape_from_json(x) for x in ref["scene"]]
            ends = []
            while len(ends) < 50:
                q = panda.lower + rng.random((256, 7)) * (panda.upper - panda.lower)
                c = panda.sphere_centres(q)
                clear = (world_clearance(panda, c, shapes) > 0.01) & (panda.self_clearance(c) > 0.01)
                ends += [x.tolist() for x in q[clear]]
            for k in range(25):
                one.append({"id": f"one/{split_name}/{set_name}/{k}", "set": f"one/{set_name}", "index": k,
                            "split": split_name, "scene_ref": ref["id"], "start": ends[2 * k], "goal": ends[2 * k + 1]})
    (out / "queries_onescene.json").write_text(json.dumps(one))
    manifest = {
        "source_urdf": {"file": urdf.name, "sha256": sha256(urdf)},
        "source_srdf": {"file": srdf.name, "sha256": sha256(srdf)},
        "panda.urdf": sha256(out / "panda.urdf"), "panda.srdf": sha256(out / "panda.srdf"),
        "problems.json": sha256(out / "problems.json"),
        "queries_onescene.json": sha256(out / "queries_onescene.json"),
        "robometrics": {n: sha256(data / n) for n in ["mb_set.yaml", "mpinets_set.yaml"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import sys

    assert sys.argv[1] == "export", "usage: bench_model.py export <robometrics dir> <out dir>"
    print(json.dumps(export(sys.argv[2], sys.argv[3]), indent=2))
