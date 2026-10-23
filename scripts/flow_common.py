"""Deterministic no-actuation full-flow replay helpers (no SDK imports)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def transition_record(transition):
    if transition is None:
        return None
    return {"previous": transition.previous.value, "current": transition.current.value,
            "event": transition.event.value, "accepted": bool(transition.accepted),
            "changed": bool(transition.changed), "reason": transition.reason}


def frame_arguments(data, index):
    from deployment.common.controller_input import Button, ControllerSample
    from deployment.vita_s100_aorta_30dof.robot_adapter import VitaRobotState, VITA_CONTROLLER_JOINT_NAMES
    state = VitaRobotState(joint_names=VITA_CONTROLLER_JOINT_NAMES,
        joint_pos=data["joint_pos"][index], joint_vel=data["joint_vel"][index],
        root_quat_wxyz=data["root_quat_wxyz"][index], root_ang_vel_b=data["root_ang_vel_b"][index],
        state_id=int(data["state_ids"][index]), received_monotonic_ns=int(data["monotonic_ns"][index]))
    button_names = str(data["buttons"][index]).split(",") if str(data["buttons"][index]) else []
    controller = ControllerSample(sequence=int(data["controller_sequence"][index]),
        monotonic_time=float(data["monotonic_ns"][index]) / 1e9,
        buttons=frozenset(Button(name) for name in button_names),
        axes=dict(zip(("left_x", "left_y", "right_x", "right_y"), data["axes"][index].tolist())))
    pico = None
    if bool(data["pico_present"][index]):
        pico = (data["body_poses"][index], float(data["sample_times"][index]), int(data["pico_frame_indices"][index]))
    return dict(state=state, controller=controller, pico=pico, wall_time=float(data["wall_times"][index]),
                robot_ready=bool(data["robot_ready"][index]))


def check_fixture(data):
    count = len(data["wall_times"])
    if count < 200:
        raise ValueError("Full-flow coverage requires at least 200 frames")
    for key in data.files:
        value = data[key]
        if value.shape[0] != count:
            raise ValueError(f"Frame count mismatch: {key}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite fixture values: {key}")
    if data["body_poses"].shape != (count, 24, 7):
        raise ValueError("Pico input must be N,24,7")
    if not np.all(np.diff(data["sample_times"])[data["pico_present"][1:].astype(bool)] > 0):
        raise ValueError("Pico times must increase")
    return count


def stats(values):
    v = np.asarray(values, np.float64)
    if not len(v):
        return None
    s = np.sort(v)
    q = lambda fraction: float(s[max(0, int(np.ceil(fraction * len(s))) - 1)])
    return dict(samples=len(v), mean=float(v.mean()), p50=q(.5), p95=q(.95), p99=q(.99), maximum=float(v.max()))
