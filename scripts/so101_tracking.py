"""Plays trajectories from `examples/so101_tracking.rs` on the SO-101's servos in MuJoCo, through
the Feetech STS3215 (7.4 V) model that BAM (Better Actuator Models, Rhoban) identified from
recorded trajectories, and measures how closely the arm follows them.

  python scripts/so101_tracking.py <run> [--kp 16] [--vin 5] [--workers 8]

The model is BAM's m6: the firmware's rate-limited P controller with gain `kp` (LeRobot sets 16;
the firmware's default is 32), PWM and supply limits, and load-dependent friction. `vin` is the
supply: the stock SO-101 runs these servos from 5 V; BAM identified them at 7.4 V.

Each motion starts from its first position held for a second, so gravity has settled the arm, and
ends with its last held for two. Every step sends that step's target. Per variant it prints:

- error: the largest gap between the arm and the trajectory while it runs, any joint (deg);
- at end: the gap to the last position when the trajectory ends (deg);
- settle: from the end until the arm stays within 0.25 deg of where it comes to rest (s);
- overshoot: how far any joint passes where it comes to rest, along its travel (deg);
- saturated: the share of steps in which some servo's PWM is at its limit.

Python 3.12, with better-actuator-models[mujoco]==1.0.3, mujoco==3.17.0 and numpy==2.5.4.
"""

import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from bam.model import load_model
from bam.mujoco import MujocoController

HOLD_BEFORE = 1.0
HOLD_AFTER = 2.0
SETTLED = np.deg2rad(0.25)


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


class Arm:
    """The SO-101 with its links' masses and inertias only (no geometry, so nothing touches
    anything), one torque actuator per joint, and one BAM controller driving them all."""

    def __init__(self, urdf, joints, hz, kp, vin):
        text = re.sub(r"<(visual|collision)\b.*?</\1>", "", Path(urdf).read_text(), flags=re.S)
        spec = mujoco.MjSpec.from_string(text)
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

    def step(self, target):
        for joint, q in zip(self.joints, target):
            self.controller.set_q_target(joint, q)
        self.controller.update()
        actuator = self.controller.model.actuator
        saturated = bool(np.any(np.abs(actuator.duty_cycle) >= actuator.max_pwm))
        mujoco.mj_step(self.model, self.data)
        return saturated


_arm = None


def setup(*args):
    global _arm
    _arm = Arm(*args)


def play(samples):
    """Plays one motion on the worker's arm and returns its metrics."""
    arm = _arm
    reference = samples[:, 0, :].astype(np.float64)
    first, last = reference[0], reference[-1]
    dt = arm.model.opt.timestep
    arm.reset(first)
    for _ in range(round(HOLD_BEFORE / dt)):
        arm.step(first)
    positions, saturated = [], 0
    for target in reference:
        positions.append(arm.data.qpos[arm.qpos].copy())
        saturated += arm.step(target)
    after = []
    for _ in range(round(HOLD_AFTER / dt)):
        after.append(arm.data.qpos[arm.qpos].copy())
        arm.step(last)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("run")
    parser.add_argument("--kp", type=float, default=16.0, help="the STS3215 firmware's P gain")
    parser.add_argument("--vin", type=float, default=5.0, help="the supply voltage (V)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = load_run(args.run)
    init = (manifest["urdf"], manifest["joints"], manifest["hz"], args.kp, args.vin)
    deg = np.rad2deg
    print(f"gain {args.kp:g}, {args.vin:g} V, {len(manifest['variants'][0]['motions'])} motions; medians, worst in brackets")
    print(f"{'variant':<18}{'duration s':>12}{'error deg':>16}{'at end deg':>16}{'settle s':>14}{'overshoot deg':>16}{'saturated':>12}")
    with Pool(args.workers, initializer=setup, initargs=init) as pool:
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


if __name__ == "__main__":
    main()
