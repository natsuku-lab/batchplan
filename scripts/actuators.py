"""Plays trajectories from `examples/actuators.rs` on actuator models in MuJoCo, and measures how
closely the arm follows them, or the torque they ask of it.

  python scripts/actuators.py servo <run> [--kp 16] [--vin 5]  # the SO-101 on BAM's STS3215 model
  python scripts/actuators.py motor <run> [--gains soft]  # the reBot on Damiao datasheet models
  python scripts/actuators.py torque <run>                # inverse dynamics against `effort`

`servo` uses the Feetech STS3215 (7.4 V) model that BAM (Better Actuator Models, Rhoban) identified
from recorded trajectories: the firmware's rate-limited P controller with gain `kp` (LeRobot sets
16; the firmware's default is 32), PWM and supply limits, and load-dependent friction (model m6).
`vin` is the supply: the stock SO-101 runs these servos from 5 V; BAM identified them at 7.4 V.

`motor` models the reBot B601-DM's Damiao DM-J4340-2EC (joints 1-3) and DM-J4310-2EC V1.1 (joints
4-6) from their datasheets at 24 V, since BAM has not identified them: MIT mode, torque =
kp (q* - q) + kd (q̇* - q̇) + gravity at q, capped by the torque-speed curve the torque constant,
phase resistance, no-load speed and peak torque give, with the rotor's inertia through the gearbox
(Seeed's table does not say which side of the gearbox it is given at; it is taken as the rotor's).
It has no friction. Its gains are chosen here, not reBot's: kp per motor, and each joint's kd for
a damping ratio of 0.7 at the ready pose, from the joint's inertia there, the rotor's included.

Each motion starts from its first position held for a second, so gravity has settled the arm, and
ends with its last held for two. Every step sends that step's target. Per variant it prints:

- error: the largest gap between the arm and the trajectory while it runs, any joint (deg);
- at end: the gap to the last position when the trajectory ends (deg);
- settle: from the end until the arm stays within 0.25 deg of where it comes to rest (s);
- overshoot: how far any joint passes where it comes to rest, along its travel (deg);
- saturated: the share of steps in which some actuator is at its limit.

Python 3.12, with better-actuator-models[mujoco]==1.0.3, mujoco==3.17.0 and numpy==2.5.4.
"""

import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HOLD_BEFORE = 1.0
HOLD_AFTER = 2.0
SETTLED = np.deg2rad(0.25)

# Gear ratio, peak torque (N·m), no-load speed (rad/s), torque constant at the output (N·m/A),
# phase resistance (Ω) and rotor inertia (kg·m²), from Seeed's Damiao 43-series table, 24 V.
DAMIAO = {
    "4340": dict(ratio=40, peak=27.0, no_load=52.5 * 2 * np.pi / 60, kt=4.074, r=0.88, rotor=2.0e-5),
    "4310": dict(ratio=10, peak=7.0, no_load=200 * 2 * np.pi / 60, kt=0.945, r=0.85, rotor=1.8e-5),
}
REBOT_MOTORS = {f"joint{i}": "4340" if i <= 3 else "4310" for i in range(1, 7)}
# MIT position gains (kp N·m/rad) per motor; each joint's kd follows from it (`DAMPING`).
GAINS = {"soft": {"4340": 60.0, "4310": 20.0}, "stiff": {"4340": 150.0, "4310": 40.0}}
DAMPING = 0.7
# The reBot's ready pose, as examples/handover/rebot.rs has it: where kd is set.
READY = [0.0, -1.5, -1.5, 1.45, 0.0, 0.0]


def load_run(run):
    """The manifest and, per variant, each motion's samples as `[len, 3, dof]`."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    samples = np.fromfile(run / "samples.bin", dtype="<f4")
    dof = len(manifest["joints"])
    for variant in manifest["variants"]:
        for motion in variant["motions"]:
            start = motion["offset"]
            motion["samples"] = samples[start : start + motion["len"] * 3 * dof].reshape(motion["len"], 3, dof)
    return manifest


def arm_spec(urdf):
    """The description as a MuJoCo spec, its links' masses and inertias only: no geometry, so
    nothing touches anything, and the base fixed to the world. Links the description leaves
    massless (the reBot's fingers) get a microgram, which MuJoCo needs to move them."""
    text = re.sub(r"<(visual|collision)\b.*?</\1>", "", Path(urdf).read_text(), flags=re.S)
    spec = mujoco.MjSpec.from_string(text)
    spec.compiler.boundmass = 1e-9
    spec.compiler.boundinertia = 1e-9
    return spec


def damiao_cap(motor, speed):
    """The largest torque a Damiao motor gives at an output speed (N·m): its peak, until back-EMF
    leaves less current than that, then falling to zero at the no-load speed."""
    m = DAMIAO[motor]
    return np.clip(m["kt"] * m["kt"] / m["r"] * (m["no_load"] - np.abs(speed)), 0.0, m["peak"])


class Servo:
    """The SO-101 on BAM's STS3215 model: one BAM controller drives every joint."""

    def __init__(self, urdf, joints, hz, setting):
        from bam.model import load_model
        from bam.mujoco import MujocoController

        kp, vin = setting
        spec = arm_spec(urdf)
        for joint in joints:
            spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        self.model = spec.compile()
        self.model.opt.timestep = 1.0 / hz
        self.data = mujoco.MjData(self.model)
        servo = load_model(motor_name="feetech_sts3215_7_4V", model="m6")
        servo.actuator.kp = kp
        servo.actuator.vin = vin
        self.controller = MujocoController(servo, joints, self.model, self.data)
        self.joints = joints
        self.qpos = self.controller.qpos_indexes

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = q
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()

    def step(self, target, _velocity):
        for joint, q in zip(self.joints, target):
            self.controller.set_q_target(joint, q)
        self.controller.update()
        actuator = self.controller.model.actuator
        saturated = bool(np.any(np.abs(actuator.duty_cycle) >= actuator.max_pwm))
        mujoco.mj_step(self.model, self.data)
        return saturated


class Motor:
    """The reBot on Damiao datasheet models in MIT mode, its fingers fixed to the hand."""

    def __init__(self, urdf, joints, hz, gains):
        spec = arm_spec(urdf)
        for joint in joints:
            spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        # batchplan locks the fingers, and the description leaves them massless, so they cannot
        # be held by actuators of their own: their joints, and the right one's mimic, go.
        for equality in list(spec.equalities):
            spec.delete(equality)
        for finger in ["finger_left", "finger_right"]:
            spec.delete(spec.joint(finger))
        self.model = spec.compile()
        self.model.opt.timestep = 1.0 / hz
        ids = [self.model.joint(j).id for j in joints]
        self.qpos, self.dofs = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.motors = [REBOT_MOTORS[j] for j in joints]
        for dof, motor in zip(self.dofs, self.motors):
            self.model.dof_armature[dof] = DAMIAO[motor]["rotor"] * DAMIAO[motor]["ratio"] ** 2
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.kp = np.array([GAINS[gains][m] for m in self.motors])
        # kd = 2 ζ √(kp M_ii), with M_ii the joint's inertia at the ready pose.
        self.scratch.qpos[self.qpos] = READY
        mujoco.mj_forward(self.model, self.scratch)
        mass = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.scratch, mass)
        self.kd = 2.0 * DAMPING * np.sqrt(self.kp * mass[self.dofs, self.dofs])

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = q
        mujoco.mj_forward(self.model, self.data)

    def step(self, target, velocity):
        q, dq = self.data.qpos[self.qpos], self.data.qvel[self.dofs]
        # Gravity where the arm is, at rest.
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.scratch)
        torque = self.kp * (target - q) + self.kd * (velocity - dq) + self.scratch.qfrc_bias[self.dofs]
        limit = np.array([damiao_cap(m, s) for m, s in zip(self.motors, dq)])
        self.data.ctrl[:] = np.clip(torque, -limit, limit)
        mujoco.mj_step(self.model, self.data)
        return bool(np.any(np.abs(torque) >= limit))


_arm = None


def setup(kind, urdf, joints, hz, setting):
    global _arm
    _arm = Servo(urdf, joints, hz, setting) if kind == "servo" else Motor(urdf, joints, hz, setting)


def play(samples):
    """Plays one motion on the worker's arm and returns its metrics."""
    arm = _arm
    reference, velocity = samples[:, 0, :].astype(np.float64), samples[:, 1, :].astype(np.float64)
    first, last = reference[0], reference[-1]
    rest = np.zeros_like(first)
    dt = arm.model.opt.timestep
    arm.reset(first)
    for _ in range(round(HOLD_BEFORE / dt)):
        arm.step(first, rest)
    positions, saturated = [], 0
    for target, target_velocity in zip(reference, velocity):
        positions.append(arm.data.qpos[arm.qpos].copy())
        saturated += arm.step(target, target_velocity)
    after = []
    for _ in range(round(HOLD_AFTER / dt)):
        after.append(arm.data.qpos[arm.qpos].copy())
        arm.step(last, rest)
    positions, after = np.array(positions), np.array(after)
    settled_at = after[-1]
    off = np.abs(after - settled_at).max(axis=1) > SETTLED
    return {
        "error": float(np.abs(positions - reference).max()),
        "at_end": float(np.abs(after[0] - last).max()),
        "settle": float((np.nonzero(off)[0][-1] + 1) * dt) if off.any() else 0.0,
        "overshoot": max(0.0, float(((after - settled_at) * np.sign(last - first)).max())),
        "saturated": saturated / len(reference),
    }


def track(kind, manifest, setting, workers):
    init = (kind, manifest["urdf"], manifest["joints"], manifest["hz"], setting)
    deg = np.rad2deg
    print(f"{kind} {setting}, {len(manifest['variants'][0]['motions'])} motions; medians, worst in brackets")
    if kind == "motor":
        print("kd per joint:", np.round(Motor(*init[1:]).kd, 2).tolist())
    print(f"{'variant':<18}{'duration s':>12}{'error deg':>16}{'at end deg':>16}{'settle s':>14}{'overshoot deg':>16}{'saturated':>12}")
    with Pool(workers, initializer=setup, initargs=init) as pool:
        for variant in manifest["variants"]:
            metrics = pool.map(play, [m["samples"] for m in variant["motions"]])
            r = {k: np.array([m[k] for m in metrics]) for k in metrics[0]}

            def cell(key, scale=lambda x: x):
                values = scale(r[key])
                return f"{np.median(values):.2f} [{values.max():.2f}]"

            duration = np.mean([m["duration"] for m in variant["motions"]])
            print(
                f"{variant['name']:<18}{duration:>12.2f}{cell('error', deg):>16}{cell('at_end', deg):>16}"
                f"{cell('settle'):>14}{cell('overshoot', deg):>16}{np.mean(r['saturated']):>12.1%}"
            )


def torque(manifest):
    joints = manifest["joints"]
    model = arm_spec(manifest["urdf"]).compile()
    data = mujoco.MjData(model)
    ids = [model.joint(j).id for j in joints]
    qpos, dofs = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    text = Path(manifest["urdf"]).read_text()
    efforts = dict(re.findall(r'<joint\s+name="([^"]+)"[^>]*>.*?<limit[^>]*effort="([^"]+)"', text, flags=re.S))
    limit = np.array([float(efforts[j]) for j in joints])
    print(f"{len(manifest['variants'][0]['motions'])} motions; effort limits {limit.tolist()}")
    print(f"{'variant':<18}{'duration s':>12}{'worst share':>14}{'joint':>14}{'median share':>15}{'over':>7}")
    for variant in manifest["variants"]:
        shares = []
        for motion in variant["motions"]:
            peak = np.zeros(len(joints))
            for q, dq, ddq in motion["samples"].astype(np.float64):
                data.qpos[qpos], data.qvel[dofs], data.qacc[dofs] = q, dq, ddq
                mujoco.mj_inverse(model, data)
                peak = np.maximum(peak, np.abs(data.qfrc_inverse[dofs]))
            shares.append(peak / limit)
        shares = np.array(shares)
        worst = np.unravel_index(shares.argmax(), shares.shape)
        duration = np.mean([m["duration"] for m in variant["motions"]])
        print(
            f"{variant['name']:<18}{duration:>12.2f}{shares.max():>14.0%}{joints[worst[1]]:>14}"
            f"{np.median(shares.max(axis=1)):>15.0%}{int((shares > 1).any(axis=1).sum()):>7}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("kind", choices=["servo", "motor", "torque"])
    parser.add_argument("run")
    parser.add_argument("--kp", type=float, default=16.0, help="servo: the STS3215 firmware's P gain")
    parser.add_argument("--vin", type=float, default=5.0, help="servo: the supply voltage (V)")
    parser.add_argument("--gains", choices=sorted(GAINS), default="soft", help="motor: the MIT gains")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = load_run(args.run)
    if args.kind == "torque":
        torque(manifest)
    else:
        setting = (args.kp, args.vin) if args.kind == "servo" else args.gains
        track(args.kind, manifest, setting, args.workers)


if __name__ == "__main__":
    main()
