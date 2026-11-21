#!/usr/bin/env python3
"""Lets go of each rest `rests_on` found, in MuJoCo, and sees whether it stays; and of each fall
`falls_into` found, and sees whether it falls in and stays.

    cargo run --release --example surface_rests -- --dump data/rests
    uv run --with mujoco --with numpy python scripts/validate_rests.py data/rests

Each scene's support becomes a height field of the highest point in each cell `touching` wide, as
`rests_on` reads it, and the object a body of its own shape, its mass at its middle, with the
friction the rests were found with. A rest is let go where it was found and stepped for a second:
it stays if it has moved less than 5 mm and turned less than 5 degrees. A rest that moves is a
defect to explain, in `rests_on` or in this check. A fall (a scene that gives its hollow's `rim`)
is let go where it was found and stepped for as long as its middle takes to fall to the lowest of
the support, and then a rest's second to settle: it fell in and stays if its middle then lies under
the rim, over a cell of the support, and it moves slower than 5 mm and turns slower than 5 degrees
a second, no more in a second than a rest may in all.
"""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

# A rest's time to settle: ManiSkill2 settles a placement for half a second of steps, and another
# half where the object still moves.
SECONDS = 1.0
# The balls the object's surface is made of for MuJoCo, and how far apart they lie (meters).
BALL = 0.002
STEP = 0.004
MOST_MOVED = 0.005
MOST_TURNED = math.radians(5.0)


def surface_of(kind, size):
    """Centres of balls of radius `BALL` whose outsides lie on the shape's surface, `STEP` apart.

    MuJoCo keeps at most 50 contacts between one shape and a height field, the first it finds: a
    box on cells 2 mm wide keeps a clump at one corner and sinks. Each ball touches a few cells.
    """
    def along(half):
        inside = max(half - BALL, 0.0)
        count = max(int(math.ceil(2 * inside / STEP)), 1)
        return np.linspace(-inside, inside, count + 1)

    at = []
    if kind == "box":
        xs, ys, zs = (along(h) for h in size)
        for x in xs:
            for y in ys:
                for z in zs:
                    if x in (xs[0], xs[-1]) or y in (ys[0], ys[-1]) or z in (zs[0], zs[-1]):
                        at.append((x, y, z))
    elif kind == "sphere":
        return [(0.0, 0.0, 0.0)] if size[0] <= BALL else fibonacci(size[0] - BALL)
    else:
        radius, half = size[0] - BALL, size[1] if kind == "capsule" else size[1] - BALL
        around = max(int(math.ceil(2 * math.pi * radius / STEP)), 8)
        for z in along(half + BALL):
            z = float(np.clip(z, -half, half))
            for k in range(around):
                a = 2 * math.pi * k / around
                at.append((radius * math.cos(a), radius * math.sin(a), z))
        for end in (-half, half):
            if kind == "cylinder":
                for r in np.arange(0.0, radius, STEP):
                    ring = max(int(math.ceil(2 * math.pi * r / STEP)), 1)
                    for k in range(ring):
                        a = 2 * math.pi * k / ring
                        at.append((r * math.cos(a), r * math.sin(a), end))
            else:
                at += [(x, y, z + end) for x, y, z in fibonacci(radius) if z * end >= 0]
    return at


def fibonacci(radius):
    """Points spread over a sphere of `radius`, about `STEP` apart."""
    count = max(int(4 * math.pi * radius**2 / STEP**2), 12)
    golden = math.pi * (3 - math.sqrt(5))
    out = []
    for k in range(count):
        z = 1 - 2 * (k + 0.5) / count
        r = math.sqrt(1 - z * z)
        out.append((radius * r * math.cos(golden * k), radius * r * math.sin(golden * k), radius * z))
    return out


def mass_of(kind, size):
    """The shape's mass at a density of 500 kg/m3, and its moments about its own axes."""
    if kind == "box":
        x, y, z = (2 * h for h in size)
        m = 500 * x * y * z
        return m, (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12)
    if kind == "sphere":
        m = 500 * 4 / 3 * math.pi * size[0] ** 3
        return m, (0.4 * m * size[0] ** 2,) * 3
    r, h = size[0], 2 * size[1]
    m = 500 * math.pi * r * r * h
    across = m * (3 * r * r + h * h) / 12
    return m, (across, across, 0.5 * m * r * r)


def model_of(points, shape, friction, cell):
    """The scene: the support's points as a height field, the object free over it."""
    first = np.round(points[:, :2].min(axis=0) / cell).astype(int) - 2
    last = np.round(points[:, :2].max(axis=0) / cell).astype(int) + 2
    columns, rows = (last - first + 1).tolist()
    floor = float(points[:, 2].min()) - 0.02
    heights = np.full((rows, columns), floor, dtype=np.float64)
    at = np.round(points[:, :2] / cell).astype(int) - first
    np.maximum.at(heights, (at[:, 1], at[:, 0]), points[:, 2])
    top = float(heights.max())
    middle = (first + last) * cell / 2.0
    half = (last - first) * cell / 2.0
    kind, size = next(iter(shape.items()))
    balls = "".join(
        f'<geom type="sphere" size="{BALL}" pos="{x:.5f} {y:.5f} {z:.5f}" '
        f'friction="{friction} 0.005 0.0001"/>'
        for x, y, z in surface_of(kind, size)
    )
    mass, inertia = mass_of(kind, size)
    xml = f"""
    <mujoco>
      <option timestep="0.001" cone="elliptic" impratio="10"/>
      <asset>
        <hfield name="support" nrow="{rows}" ncol="{columns}"
                size="{half[0]} {half[1]} {top - floor} 0.01"/>
      </asset>
      <worldbody>
        <geom type="hfield" hfield="support" pos="{middle[0]} {middle[1]} {floor}"
              friction="{friction} 0.005 0.0001"/>
        <body name="object">
          <freejoint/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>
          {balls}
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.hfield_data[:] = ((heights - floor) / (top - floor)).ravel()
    return model


def stays(model, data, rest):
    """How far a rest moves and turns in `SECONDS` once let go."""
    mujoco.mj_resetData(model, data)
    x, y, z, qx, qy, qz, qw = rest[:7]
    data.qpos[:] = [x, y, z, qw, qx, qy, qz]
    mujoco.mj_forward(model, data)
    for _ in range(int(SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
    moved = float(np.linalg.norm(data.qpos[:3] - rest[:3]))
    was = np.array([qw, qx, qy, qz])
    turned = 2.0 * math.acos(min(1.0, abs(float(was @ data.qpos[3:7]))))
    return moved, turned


def falls_in(model, data, fall, rim, cells, cell, floor):
    """Whether a fall let go where it was found lies in the hollow and still once its middle has had
    the time to fall to `floor`, the lowest of the support, and a rest's `SECONDS` to settle, and
    where its middle is then."""
    mujoco.mj_resetData(model, data)
    x, y, z, qx, qy, qz, qw = fall[:7]
    data.qpos[:] = [x, y, z, qw, qx, qy, qz]
    mujoco.mj_forward(model, data)
    falling = math.sqrt(2.0 * max(z - floor, 0.0) / -model.opt.gravity[2])
    for _ in range(int((falling + SECONDS) / model.opt.timestep)):
        mujoco.mj_step(model, data)
    at = data.qpos[:3].copy()
    over = (round(at[0] / cell), round(at[1] / cell)) in cells
    moving, turning = (float(np.linalg.norm(data.qvel[k : k + 3])) for k in (0, 3))
    still = moving < MOST_MOVED / SECONDS and turning < MOST_TURNED / SECONDS
    return at[2] < rim and over and still, at


def main():
    folder = Path(sys.argv[1])
    most = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    failed = 0
    for about in sorted(folder.glob("*.json")):
        scene = json.loads(about.read_text())
        name = about.stem
        points = np.fromfile(folder / f"{name}.points.f32", dtype="<f4").reshape(-1, 3)
        if "rim" in scene:
            failed += check_falls(folder, name, scene, points, most)
            continue
        rests = np.fromfile(folder / f"{name}.rests.f32", dtype="<f4").reshape(-1, 8)
        if len(rests) == 0:
            print(f"{name}: no rests")
            continue
        model = model_of(points.astype(np.float64), scene["object"], scene["friction"], scene["touching"])
        data = mujoco.MjData(model)
        # Spread over the rests, the first few among them.
        chosen = np.unique(np.linspace(0, len(rests) - 1, min(most, len(rests))).astype(int))
        went = []
        for k in chosen:
            moved, turned = stays(model, data, rests[k].astype(np.float64))
            if moved > MOST_MOVED or turned > MOST_TURNED:
                went.append((k, moved, turned))
        failed += len(went)
        print(f"{name}: {len(chosen) - len(went)} of {len(chosen)} rests tried stay ({len(rests)} found)")
        for k, moved, turned in went[:8]:
            up = rests[k, 3:7]
            print(f"    rest {k} at {rests[k, :3].round(4)} turned {up.round(3)} tipping "
                  f"{math.degrees(rests[k, 7]):.1f} deg: moved {moved * 1e3:.1f} mm, "
                  f"turned {math.degrees(turned):.1f} deg")
    sys.exit(1 if failed else 0)


def check_falls(folder, name, scene, points, most):
    """How many of a scene's falls, spread over them, do not fall in and stay."""
    falls = np.fromfile(folder / f"{name}.falls.f32", dtype="<f4").reshape(-1, 8)
    if len(falls) == 0:
        print(f"{name}: no falls")
        return 0
    cell = scene["touching"]
    cells = {(round(x / cell), round(y / cell)) for x, y in points[:, :2].astype(np.float64)}
    floor = float(points[:, 2].min())
    model = model_of(points.astype(np.float64), scene["object"], scene["friction"], cell)
    data = mujoco.MjData(model)
    chosen = np.unique(np.linspace(0, len(falls) - 1, min(most, len(falls))).astype(int))
    out = []
    for k in chosen:
        fall = falls[k].astype(np.float64)
        inside, at = falls_in(model, data, fall, scene["rim"], cells, cell, floor)
        if not inside:
            out.append((k, at))
    print(f"{name}: {len(chosen) - len(out)} of {len(chosen)} falls tried fall in and stay "
          f"({len(falls)} found)")
    for k, at in out[:8]:
        print(f"    fall {k} let go at {falls[k, :3].round(4)} turned {falls[k, 3:7].round(3)}: "
              f"its middle ends at {at.round(4)}")
    return len(out)


if __name__ == "__main__":
    main()
