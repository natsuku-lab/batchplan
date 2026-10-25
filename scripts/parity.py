"""Parity of the canonical Panda across the planners and the validator, before any timing.

Fixtures are configurations in 20 problems' scenes: random ones, near contacts (bisected to half a
millimetre either side of a surface), and clear collisions. Each implementation reports forward
kinematics and its collision verdicts on them, and `compare` holds them to the validator's
(`bench_model.py`): link poses within 1e-5 m and 1e-5 rad, the same verdicts wherever the validator's
clearance is more than 1e-4 m from zero, and, where an implementation reports clearances, those too.

    python scripts/parity.py fixtures data/bench/inputs data/bench/parity/fixtures.json
    python scripts/parity.py validator data/bench/inputs data/bench/parity/fixtures.json data/bench/parity/validator.json
    cargo run --release --example compare -- parity data/bench/inputs data/bench/parity/fixtures.json data/bench/parity/batchplan.json
    .venv-roboplan/bin/python scripts/parity.py roboplan data/bench/inputs data/bench/parity/fixtures.json data/bench/parity/roboplan.json
    .venv-bench/bin/python scripts/parity.py vamp data/bench/inputs data/bench/parity/fixtures.json data/bench/parity/vamp.json
    python scripts/parity.py moveit ...   (in the MoveIt image)
    python scripts/parity.py compare data/bench/parity
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_model import ARM, Capsule, Panda, rotation, shape_from_json, world_clearance  # noqa: E402

BOUNDARY = 1e-4  # verdicts are compared where the validator's clearance is farther than this from zero


def panda(inputs):
    return Panda(Path(inputs) / "panda.urdf", Path(inputs) / "panda.srdf")


def problems(inputs):
    return {p["id"]: p for p in json.loads((Path(inputs) / "problems.json").read_text())}


def clearance(robot, q, shapes):
    c = robot.sphere_centres(np.atleast_2d(q))
    return world_clearance(robot, c, shapes), robot.self_clearance(c)


def fixtures(inputs, out):
    robot, by_id = panda(inputs), problems(inputs)
    # One tuning problem from each set, the one with the most shapes.
    sets = {}
    for p in by_id.values():
        if p["split"] == "tuning" and len(p["scene"]) > len(sets.get(p["set"], {"scene": []})["scene"]):
            sets[p["set"]] = p
    scenes = sorted(p["id"] for p in sets.values())
    shapes = [[shape_from_json(s) for s in by_id[i]["scene"]] for i in scenes]
    rng = np.random.default_rng(7)
    random = lambda: robot.lower + rng.random(7) * (robot.upper - robot.lower)
    both = lambda q, w: min(v[0] for v in clearance(robot, q, shapes[w]))
    configs = [(random().tolist(), k % len(scenes), "random") for k in range(2000)]
    # Near contacts: a free and a colliding configuration in one scene, bisected to clearances of
    # +0.5 mm and -0.5 mm.
    pairs = 0
    while pairs < 100:
        w = int(rng.integers(len(scenes)))
        a, b = random(), random()
        ca, cb = both(a, w), both(b, w)
        if (ca > 0) == (cb > 0):
            continue
        if ca < 0:
            a, b = b, a
        for target in (5e-4, -5e-4):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if both(a + mid * (b - a), w) > target:
                    lo = mid
                else:
                    hi = mid
            configs.append(((a + lo * (b - a)).tolist(), w, "near"))
        pairs += 1
    hits = 0
    while hits < 50:
        w, q = int(rng.integers(len(scenes))), random()
        if both(q, w) < -0.01:
            configs.append((q.tolist(), w, "collision"))
            hits += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "scenes": scenes,
        "configurations": [[q, w] for q, w, _ in configs],
        "kinds": [k for _, _, k in configs],
    }))
    print(f"{len(configs)} configurations in {len(scenes)} scenes")


def load_fixtures(path):
    f = json.loads(Path(path).read_text())
    return f["scenes"], [(np.array(q), w) for q, w in f["configurations"]], f["kinds"]


def validator(inputs, fixture_path, out):
    robot, by_id = panda(inputs), problems(inputs)
    scenes, configs, _ = load_fixtures(fixture_path)
    exact = [[shape_from_json(s) for s in by_id[i]["scene"]] for i in scenes]
    capsules = [[shape_from_json(s) for s in by_id[i]["scene_capsules"]] for i in scenes]
    q = np.array([c for c, _ in configs])
    t = robot.link_transforms(q)
    centres = robot.sphere_centres(q)
    rows = []
    for k, (_, w) in enumerate(configs):
        rows.append({
            "links": {l: _pose(t[k, i]) for i, l in enumerate(robot.links)},
            "spheres": centres[k].tolist(),
            "world": float(world_clearance(robot, centres[k:k + 1], exact[w])[0]),
            "world_capsules": float(world_clearance(robot, centres[k:k + 1], capsules[w])[0]),
            "self": float(robot.self_clearance(centres[k:k + 1])[0]),
        })
    # Does the base sphere, which the validator leaves out, touch anything in these scenes?
    base = robot.sphere_link == robot.links.index(robot.base)
    base_hits = [i for i, (e, c) in enumerate(zip(exact, capsules))
                 if any((_dist(s, robot.sphere_centre[base]) - robot.sphere_radius[base] < 0).any() for s in e + c)]
    Path(out).write_text(json.dumps({"rows": rows, "base_touches_scene": base_hits}))


def _dist(shape, points):
    from bench_model import distance
    return distance(shape, points)


def _pose(t):
    from bench_model import quaternion
    return t[:3, 3].tolist() + quaternion(t[:3, :3]).tolist()


def roboplan(inputs, fixture_path, out):
    from roboplan.core import Box, Cylinder, Scene, Sphere, loadUrdfSceneDescriptionFromXml

    robot, by_id = panda(inputs), problems(inputs)
    scenes, configs, _ = load_fixtures(fixture_path)
    urdf = (Path(inputs) / "panda.urdf").read_text()
    scene = Scene("panda", loadUrdfSceneDescriptionFromXml(urdf, []))
    scene.importSrdf((Path(inputs) / "panda.srdf").read_text())
    names = scene.getJointNames()
    assert list(names) == ARM, names
    grey = np.array([0.5, 0.5, 0.5, 1.0])
    rows = []
    by_scene = {}
    for k, (_, w) in enumerate(configs):
        by_scene.setdefault(w, []).append(k)
    verdicts = {}
    # Self-collision alone, with no obstacles.
    for k, (q, _) in enumerate(configs):
        verdicts[k] = {"self_free": not scene.hasCollisions(q)}
    for w, ks in by_scene.items():
        added = add_roboplan_shapes(scene, [shape_from_json(s) for s in by_id[scenes[w]]["scene"]], grey, Box, Cylinder, Sphere)
        for k in ks:
            verdicts[k]["free"] = not scene.hasCollisions(configs[k][0])
        for name in added:
            scene.removeGeometry(name)
    for k, (q, _) in enumerate(configs):
        links = {l: _pose(np.asarray(scene.forwardKinematics(q, l))) for l in robot.links}
        rows.append({"links": links, **verdicts[k]})
    Path(out).write_text(json.dumps({"rows": rows}))


def add_roboplan_shapes(scene, shapes, color, Box, Cylinder, Sphere):
    """Adds shapes to a RoboPlan scene, a capsule as a cylinder and two spheres (exactly the capsule),
    and leaves obstacle pairs and the base unchecked against them. Returns the names added."""
    names = []

    def add(kind, shape, centre, rot):
        name = f"obstacle{len(names)}"
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rot, centre
        getattr(scene, f"add{kind}Geometry")(name, "universe", shape, np.asfortranarray(pose), color)
        names.append(name)

    for s in shapes:
        if isinstance(s, Capsule):
            axis = s.b - s.a
            rot = _rotation_to(axis)
            add("Cylinder", Cylinder(s.radius, float(np.linalg.norm(axis))), (s.a + s.b) / 2, rot)
            add("Sphere", Sphere(s.radius), s.a, np.eye(3))
            add("Sphere", Sphere(s.radius), s.b, np.eye(3))
        elif hasattr(s, "half"):
            add("Box", Box(*(2 * s.half)), s.centre, s.rot)
        else:
            add("Cylinder", Cylinder(s.radius, 2 * s.half_height), s.centre, s.rot)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]] + [(n, "panda_link0") for n in names]
    scene.setCollisions(pairs, False)
    return names


def _rotation_to(axis):
    """A rotation taking z to `axis`."""
    z = axis / np.linalg.norm(axis)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def vamp(inputs, fixture_path, out):
    import vamp as v

    by_id = problems(inputs)
    scenes, configs, _ = load_fixtures(fixture_path)
    envs = [vamp_environment(v, [shape_from_json(s) for s in by_id[i]["scene_capsules"]]) for i in scenes]
    empty = v.Environment()
    rows = []
    for q, w in configs:
        spheres = v.panda.fk(q.tolist())
        rows.append({
            "spheres": [[s.x, s.y, s.z, s.r] for s in spheres],
            "free_capsules": bool(v.panda.validate(q.tolist(), envs[w])),
            "self_free": bool(v.panda.validate(q.tolist(), empty)),
        })
    Path(out).write_text(json.dumps({"rows": rows}))


def vamp_environment(v, shapes):
    """A VAMP environment of boxes and capsules (the capsule scene)."""
    from bench_model import Box

    env = v.Environment()
    for s in shapes:
        if isinstance(s, Box):
            env.add_cuboid(v.Cuboid(s.centre.tolist(), euler_xyz(s.rot), s.half.tolist()))
        else:
            assert isinstance(s, Capsule), "VAMP's scene is boxes and capsules"
            env.add_capsule(v.Cylinder(s.a.tolist(), s.b.tolist(), s.radius))
    return env


def euler_xyz(rot):
    """VAMP's orientation: extrinsic x, y, z angles, R = Rz Ry Rx."""
    return [float(np.arctan2(rot[2, 1], rot[2, 2])), float(-np.arcsin(np.clip(rot[2, 0], -1, 1))),
            float(np.arctan2(rot[1, 0], rot[0, 0]))]


def moveit(inputs, fixture_path, out):
    from compare import MoveIt, moveit_objects

    by_id = problems(inputs)
    scenes, configs, _ = load_fixtures(fixture_path)
    m = MoveIt(inputs, {"segment_fraction": 0.01, "smoothing": "totg"}, "A")
    robot = panda(inputs)
    monitor = m.robot.get_planning_scene_monitor()
    rows = [None] * len(configs)
    with monitor.read_write() as scene:
        scene.remove_all_collision_objects()
        for k, (q, _) in enumerate(configs):
            s = m.state(q)
            links = {}
            for l in robot.links:
                t = np.asarray(s.get_global_link_transform(l))
                links[l] = _pose(t)
            rows[k] = {"links": links, "self_free": not scene.is_state_colliding(s, "panda_arm", False)}
        for w, pid in enumerate(scenes):
            objects = moveit_objects([shape_from_json(x) for x in by_id[pid]["scene"]])
            for o in objects:
                scene.apply_collision_object(o)
                scene.allowed_collision_matrix.set_entry("panda_link0", o.id, True)
            for k, (q, cw) in enumerate(configs):
                if cw == w:
                    rows[k]["free"] = not scene.is_state_colliding(m.state(q), "panda_arm", False)
            scene.remove_all_collision_objects()
    Path(out).write_text(json.dumps({"rows": rows}))


def compare(directory):
    d = Path(directory)
    kinds = json.loads((d / "fixtures.json").read_text())["kinds"]
    ref = json.loads((d / "validator.json").read_text())
    rows = ref["rows"]
    print(f"{len(rows)} configurations; base sphere touches scenes {ref['base_touches_scene'] or 'none'}")
    world_free = np.array([r["world"] > 0 for r in rows])
    caps_free = np.array([r["world_capsules"] > 0 for r in rows])
    self_free = np.array([r["self"] > 0 for r in rows])
    decided = np.array([abs(r["world"]) > BOUNDARY and abs(r["self"]) > BOUNDARY for r in rows])
    decided_caps = np.array([abs(r["world_capsules"]) > BOUNDARY and abs(r["self"]) > BOUNDARY for r in rows])
    print(f"validator: {int((~(world_free & self_free)).sum())} in collision, {int((~decided).sum())} within {BOUNDARY} m of a surface")
    failures = 0
    for name in ["batchplan", "roboplan", "vamp", "moveit"]:
        path = d / f"{name}.json"
        if not path.exists():
            print(f"{name}: no results")
            continue
        other = json.loads(path.read_text())["rows"]
        report = []
        if "links" in other[0]:
            pos = max(np.linalg.norm(np.array(o["links"][l][:3]) - np.array(r["links"][l][:3]))
                      for o, r in zip(other, rows) for l in r["links"] if l in o["links"])
            ang = max(_angle(o["links"][l][3:], r["links"][l][3:]) for o, r in zip(other, rows) for l in r["links"] if l in o["links"])
            report.append(f"link poses within {pos:.1e} m, {ang:.1e} rad")
            failures += pos > 1e-5 or ang > 1e-5
        if "spheres" in other[0]:
            err = max(_sphere_error(o["spheres"], r["spheres"]) for o, r in zip(other, rows))
            report.append(f"sphere centres within {err:.1e} m")
            failures += err > 1e-5
        for key, truth, mask in [("free", world_free & self_free, decided), ("self_free", self_free, decided),
                                 ("free_capsules", caps_free & self_free, decided_caps)]:
            if key in other[0]:
                got = np.array([o[key] for o in other])
                bad = np.flatnonzero((got != truth) & mask)
                report.append(f"{key}: {len(bad)} disagreements" + (f" (e.g. {[(int(i), kinds[i]) for i in bad[:5]]})" if len(bad) else ""))
                failures += len(bad) > 0
        if "world" in other[0]:
            for key in ["world", "self"]:
                got = np.array([o[key] for o in other])
                want = np.array([r[key] for r in rows])
                # batchplan's clearances are exact below its 10 cm margins, lower bounds above.
                near = want < 0.1
                err = np.abs(got - want)[near].max()
                bad = np.flatnonzero(((got > 0) != (want > 0)) & (np.abs(want) > BOUNDARY))
                report.append(f"{key} clearance within {err:.1e} m below 10 cm, {len(bad)} verdict disagreements")
                failures += len(bad) > 0 or err > 1e-4
        print(f"{name}: " + "; ".join(report))
    sys.exit(1 if failures else 0)


def _angle(a, b):
    """The angle between two rotations given as quaternions, well conditioned near zero."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    b = b if np.dot(a, b) >= 0 else -b
    return 2 * np.arctan2(np.linalg.norm(a - b), np.linalg.norm(a + b))


def _sphere_error(theirs, ours):
    """Largest distance from each of our sphere centres to the nearest of theirs (VAMP orders its
    spheres its own way)."""
    t = np.array(theirs)[:, :3]
    return max(np.linalg.norm(t - np.array(c), axis=1).min() for c in ours)


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    {"fixtures": fixtures, "validator": validator, "roboplan": roboplan, "vamp": vamp, "moveit": moveit,
     "compare": compare}[command](*rest)
