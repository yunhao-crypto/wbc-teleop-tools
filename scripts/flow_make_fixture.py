#!/usr/bin/env python3
"""Create deterministic raw Pico/state/button input; no SDK or inference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from flow_common import digest


def axis_quaternion(v):
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.empty_like(angle)
    np.divide(np.sin(angle / 2), angle, out=scale, where=angle != 0)
    scale[angle == 0] = .5
    return np.concatenate((np.cos(angle / 2), v * scale), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames", type=int, default=480)
    a = ap.parse_args()
    if a.frames < 480:
        raise ValueError("This scenario needs 480 frames for all event branches")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from deployment.vita_s100_aorta_30dof.onboard_runtime import load_contracts
    from deployment.vita_s100_aorta_30dof.robot_adapter import values_in_order, VITA_CONTROLLER_JOINT_NAMES
    contracts = load_contracts(a.contracts)
    motion = contracts.motion
    defaults = values_in_order(motion.joint_names, motion.default_joint_pos, VITA_CONTROLLER_JOINT_NAMES)
    n = a.frames
    t = np.arange(n, dtype=np.float64) * .02
    rng = np.random.default_rng(20260929)
    amplitudes = rng.uniform(.02, .15, (24, 3))
    phases = rng.uniform(-np.pi, np.pi, (24, 3))
    rotations = amplitudes[None] * np.sin(t[:, None, None] * .8 + phases[None])
    q = axis_quaternion(rotations)
    bodies = np.zeros((n, 24, 7), np.float32)
    bodies[:, :, 3:] = q[:, :, [1, 2, 3, 0]].astype(np.float32)
    # Equivalent quaternion signs exercise the original pipeline convention.
    bodies[::17, :, 3:] *= -1
    bodies[:, :, 0] = (.03 * np.sin(t * .5))[:, None]
    bodies[:, :, 1] = (1.0 + .01 * np.cos(t * .4))[:, None]
    bodies[:, :, 2] = (.02 * np.cos(t * .4))[:, None]
    phase = np.arange(30, dtype=np.float64) * .31
    positions = defaults[None].astype(np.float64) + .005 * np.sin(t[:, None] * .7 + phase[None])
    velocities = .0035 * np.cos(t[:, None] * .7 + phase[None])
    root_aa = np.stack((.01 * np.sin(t * .3), .015 * np.cos(t * .25), .08 * np.sin(t * .2)), axis=1)
    root_q = axis_quaternion(root_aa).astype(np.float32)
    angular = np.stack((.003 * np.cos(t * .3), -.00375 * np.sin(t * .25), .016 * np.cos(t * .2)), axis=1).astype(np.float32)
    buttons = np.full(n, "", dtype="<U16")
    events = {1:"start",3:"a",20:"b",63:"b",64:"b",80:"b",96:"y",98:"b",128:"y",130:"b",
        150:"y",152:"b",166:"y",168:"b",215:"b",235:"b",250:"y",252:"b",280:"x",
        282:"start",284:"a",286:"b",320:"y",322:"b",350:"y",352:"b",380:"y",382:"b",
        410:"y",412:"b",430:"y",432:"b",460:"back",461:"back",462:"start",466:"b,y"}
    for i, name in events.items():
        buttons[i] = name
    sequence = np.arange(n, dtype=np.int64)
    sequence[64] = sequence[63]  # identical controller report, no duplicate edge
    sequence[461] = sequence[460]
    axes = np.zeros((n, 4), np.float32)
    axes[35:48, 1] = -.25
    axes[102:106, 2] = .2  # ignored command while in Motion
    present = np.ones(n, np.uint8)
    present[190:232] = 0  # >.6s reference age causes Motion -> Velocity
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez(output, body_poses=bodies, sample_times=1700000000.0+t,
        wall_times=1000.0+t, monotonic_ns=np.rint((500.0+t)*1e9).astype(np.int64),
        joint_pos=positions.astype(np.float32), joint_vel=velocities.astype(np.float32),
        root_quat_wxyz=root_q, root_ang_vel_b=angular, state_ids=((65500+np.arange(n))%65536).astype(np.uint16),
        controller_sequence=sequence, buttons=buttons, axes=axes, pico_present=present,
        pico_frame_indices=np.arange(n, dtype=np.int64), robot_ready=np.ones(n, np.uint8))
    report = {"kind":"synthetic raw Pico + small finite joint perturbations about actual Vita defaults; not real robot dynamics",
        "frames":n,"policy_hz":50.0,"input_npz":str(output),"sha256":digest(output),
        "events":events,"pico_absent_frames":[190,231],"duplicate_controller_sequences":[64,461],
        "state_id_wrap":True,"own_policy_state_required":True,
        "restriction":"No SDK, inference, hardware mode request or actuator output executed by fixture generator"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
