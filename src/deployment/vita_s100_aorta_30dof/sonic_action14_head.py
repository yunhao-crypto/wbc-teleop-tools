"""Pico neck targets for an action14 reference. Import opens no device.

The filter and calibration advance only after the reference is published.
A discarded candidate must not move the neck command.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .sonic_action14 import NECK_NAMES


@dataclass(frozen=True)
class _Joint:
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0


@dataclass(frozen=True)
class _Command:
    joints: tuple


class Action14Head:
    def __init__(self, control):
        self.control = control
        self.command = _Command(joints=tuple(_Joint() for _ in NECK_NAMES))

    def preview_targets(self, snapshot, now):
        command, diagnostics, candidate = self.control.preview(
            self.command, list(NECK_NAMES), now=now, snapshot=snapshot, tracking=True)
        targets = {name: float(value) for name, value in zip(NECK_NAMES, diagnostics['target_q'])}
        return targets, (command, candidate)

    def commit(self, pending):
        command, candidate = pending
        self.control.commit(candidate)
        self.command = command


def make_head(root, config):
    """Same chest frame the GMR waist task uses, including operator calibration."""
    from .actor_pico_gmr_profile import apply_calibration, operator_calibration
    from .actor_pico_head import PicoHeadControl, head_settings
    settings = head_settings(config)
    if settings is None:
        raise ValueError('follow-head requires enabled head_control')
    root = Path(root)
    gmr_path = root / config['artifacts']['gmr_source_4']['path']
    gmr_config = json.loads(gmr_path.read_text())
    calibration, _, _ = operator_calibration(root, config['retargeting'], gmr_path)
    if calibration is not None:
        apply_calibration(gmr_config, calibration)
    offset = gmr_config['ik_match_table1']['waist_pitch_link'][4]
    control = PicoHeadControl(settings, offset, period=0.02,
                               hold_age=float(config['pico_hold_age_s']),
                               maximum_age=float(config['pico_max_age_s']))
    return Action14Head(control)
