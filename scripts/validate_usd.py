"""Cross-checks batchplan's OpenUSD loader against Pixar's reference implementation (usd-core).

    uv pip install --python .venv/bin/python usd-core
    .venv/bin/python scripts/validate_usd.py assets/usd/arm.usda [more.usda ...]

For each file this script builds the kinematic tree with pxr: rigid bodies, UsdPhysics joints
(flipping ones authored child to parent, skipping loop closures), NewtonMimicAPI and
PhysxMimicJointAPI, stage units and up axis. It then compares every rigid body's pose with
batchplan's (examples/usd_fk.rs) at random joint values. Composition, value resolution and the
kinematics are computed independently of batchplan's Rust code.
"""

import json
import math
import subprocess
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def joint_frame(joint, side, meters):
    pos = joint.GetPrim().GetAttribute(f"physics:localPos{side}").Get() or Gf.Vec3f(0)
    rot = joint.GetPrim().GetAttribute(f"physics:localRot{side}").Get() or Gf.Quatf(1)
    m = Gf.Matrix4d()
    m.SetTransform(Gf.Rotation(Gf.Quatd(rot)), Gf.Vec3d(pos) * meters)
    return m


def rigid(m, meters):
    """The rotation and scaled translation of a (possibly scaled) stage transform."""
    t = m.ExtractTranslation() * meters
    r = m.ExtractRotationMatrix().GetOrthonormalized()
    out = Gf.Matrix4d()
    out.SetRotate(r)
    out.SetTranslateOnly(t)
    return out


def applied(prim):
    """API schema names as authored; pxr's HasAPI only knows registered schemas (not Newton's)."""
    listop = prim.GetMetadata("apiSchemas")
    if not listop:
        return []
    return [str(s) for s in listop.prependedItems + listop.appendedItems + listop.explicitItems]


def mimic(prim, kind):
    if "NewtonMimicAPI" in applied(prim):
        targets = prim.GetRelationship("newton:mimicJoint").GetTargets()
        if targets:
            c0 = prim.GetAttribute("newton:mimicCoef0").Get() or 0.0
            c1 = prim.GetAttribute("newton:mimicCoef1").Get()
            c1 = 1.0 if c1 is None else c1
            return str(targets[0]), (c0 if kind == "prismatic" else math.radians(c0)), c1
    for api in applied(prim):
        if api.startswith("PhysxMimicJointAPI:"):
            axis = api.split(":")[1]
            targets = prim.GetRelationship(f"physxMimicJoint:{axis}:referenceJoint").GetTargets()
            gearing = prim.GetAttribute(f"physxMimicJoint:{axis}:gearing").Get()
            offset = prim.GetAttribute(f"physxMimicJoint:{axis}:offset").Get() or 0.0
            return str(targets[0]), -offset, -(1.0 if gearing is None else gearing)
    return None


def build(stage):
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    up = Gf.Matrix4d()
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        up.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
    cache = UsdGeom.XformCache()
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    bodies = {str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = []
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        if not j.GetJointEnabledAttr().Get() or j.GetExcludeFromArticulationAttr().Get():
            continue
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        joints.append((p, b0[0] if b0 else None, b1[0] if b1 else None))
    # Each body: (parent body or None, function of joint values giving its pose in the parent).
    placed = {}
    used = set()
    while True:
        progress = False
        for k, (p, b0, b1) in enumerate(joints):
            if k in used:
                continue
            side = lambda b: b not in bodies or b in placed
            if side(b0) and side(b1):
                used.add(k)
                continue
            if side(b0):
                parent, child, flip = b0, b1, False
            elif side(b1):
                parent, child, flip = b1, b0, True
            else:
                continue
            j = UsdPhysics.Joint(p)
            l_parent, l_child = (joint_frame(j, 1, meters), joint_frame(j, 0, meters)) if flip else (joint_frame(j, 0, meters), joint_frame(j, 1, meters))
            if parent not in bodies:
                base = up if parent is None else rigid(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent)) * up, meters)
                l_parent = l_parent * base
                parent = None
            if p.IsA(UsdPhysics.RevoluteJoint):
                kind, axis = "revolute", p.GetAttribute("physics:axis").Get()
            elif p.IsA(UsdPhysics.PrismaticJoint):
                kind, axis = "prismatic", p.GetAttribute("physics:axis").Get()
            else:
                kind, axis = "fixed", "X"
            unit = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[axis] * (-1 if flip else 1)
            placed[child] = (parent, str(p.GetPath()), kind, unit, l_parent, l_child, mimic(p, kind))
            used.add(k)
            progress = True
        if progress:
            continue
        rest = sorted((b for b in bodies if b not in placed), key=lambda b: (b.count("/"), b))
        if not rest:
            break
        root = rest[0]
        placed[root] = (None, None, "fixed", None, rigid(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root)) * up, meters), Gf.Matrix4d(), None)
    return placed, meters


def poses(placed, meters, values):
    """World pose of every body (row-vector Gf matrices) for joint values by path."""
    out = {}

    def value(path, kind, mim):
        if mim:
            leader, c0, c1 = mim
            return c0 + c1 * values[leader]
        return values.get(path, 0.0)

    def world(body):
        if body in out:
            return out[body]
        parent, path, kind, unit, l_parent, l_child, mim = placed[body]
        motion = Gf.Matrix4d()
        if kind == "revolute":
            motion.SetRotate(Gf.Rotation(unit, math.degrees(value(path, kind, mim))))
        elif kind == "prismatic":
            motion.SetTranslate(unit * value(path, kind, mim))
        m = l_child.GetInverse() * motion * l_parent * (world(parent) if parent else Gf.Matrix4d())
        out[body] = m
        return m

    for b in placed:
        world(b)
    return out


def check(path, variants):
    stage = Usd.Stage.Open(path)
    for prim_path, (variant_set, selection) in variants.items():
        stage.GetPrimAtPath(prim_path).GetVariantSet(variant_set).SetVariantSelection(selection)
    placed, meters = build(stage)
    names = {b: b.rsplit("/", 1)[1] for b in placed}
    request = {"file": path, "links": list(names.values()), "samples": 25, "seed": 3,
               "variants": {s: v for (s, v) in variants.values()}}
    run = subprocess.run(["cargo", "run", "--quiet", "--release", "--features", "usd", "--example", "usd_fk"],
                         input=json.dumps(request), capture_output=True, text=True, check=True)
    ours = json.loads(run.stdout)
    paths_by_name = {}
    for p, (_, jpath, _, _, _, _, _) in placed.items():
        if jpath:
            paths_by_name[jpath.rsplit("/", 1)[1]] = jpath
    worst = 0.0
    by_body = {}
    for q, row in zip(ours["q"], ours["poses"]):
        values = {paths_by_name[name]: v for name, v in zip(ours["joints"], q)}
        reference = poses(placed, meters, values)
        for body, pose in zip(names, row):
            m = reference[body]
            # Rotation matrices compared entry by entry; quaternion angles are too noisy near zero.
            ours_rot = Gf.Matrix3d(Gf.Rotation(Gf.Quatd(pose[6], pose[3], pose[4], pose[5])).GetQuat())
            ref_rot = m.ExtractRotationMatrix()
            rot_err = max(abs(ours_rot[i][k] - ref_rot[i][k]) for i in range(3) for k in range(3))
            err = max((Gf.Vec3d(*pose[:3]) - m.ExtractTranslation()).GetLength(), rot_err)
            by_body[names[body]] = max(by_body.get(names[body], 0.0), err)
            worst = max(worst, err)
    if worst > 1e-5:
        print("  per body:", {k: f"{v:.1e}" for k, v in by_body.items()})
    print(f"{path}: {len(placed)} bodies, {len(ours['joints'])} joints, worst difference {worst:.2e}")
    return worst


if __name__ == "__main__":
    files = sys.argv[1:] or [
        "assets/usd/arm.usda",
        "assets/franka/usd/panda.usda",
        "assets/newton/universal_robots_ur5e/usd_structured/ur5e.usda",
        "assets/newton/robotiq_2f85_v4/usd_structured/Dual_wrist_camera.usda",
    ]
    worst = max(check(f, {}) for f in files)
    worst = max(worst, check("assets/usd/arm.usda", {"/arm": ("reach", "short")}))
    sys.exit(0 if worst < 1e-5 else 1)
