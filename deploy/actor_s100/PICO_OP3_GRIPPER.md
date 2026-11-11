# PICO side-grip control of OP3 grippers (opt-in, not hardware-validated)

`--gripper-config` lets each PICO controller's side grip drive the AgiLink OP3
gripper on the same side while the body runs under EXTERNAL. It is off unless
the flag is given. Status: offline tests only. No OP3, robot or PICO run has
been made with it.

## Prerequisites (all required, otherwise startup refuses before any control handle)

1. A rebuilt PICO binding that exports `get_controller_grip_snapshot`
   (`third_party/pico_binding`, same build as before). The archived prebuilt
   extension in this delivery predates it.
2. The pinned SDK 0.3.0 (humanoid-sdk 535cbbc: main abe21ab plus PR #22) with
   `locomotion_aorta.end_effector.EndEffectorClient` and its generated EE
   bindings. The body path uses the same SDK (body protocol 3, control
   protocol 4); Body and EE commands share one controller's session and
   sequence. An older SDK stops with `requires a humanoid-sdk with EndEffectorClient`.
3. Robot firmware that publishes OP3 Info/State and accepts EE commands
   (vita-robot #12162, merged as 896385d5b0; its docs state no hardware/HIL
   acceptance yet), with `end_effector` set to `op3` and a site `op3.json`.

## Use

```bash
python3.10 scripts/actor_pico.py live --manual-pico ... \
    --gripper-config /path/to/pico_op3_gripper.json
```

Only live manual-PICO mode is allowed (also `--auto-start`, `--wait-pico`,
`--auto-pico`, `--local-start`); `--stand-only` and `--motion-config` are refused.
Start from `config/pico_op3_gripper.example.json` and review every value on
site. The example values are placeholders, not measured OP3 settings.

## Behaviour

| Input | Gripper command |
| --- | --- |
| Grip at or below `open_threshold` | open; also arms the side |
| Grip above `close_threshold` for `close_hold_s`, after arming | close to `close_m` with `current_a` limit |
| Grip between the thresholds | keep the current command (hysteresis) |
| No PICO yet, stale (`input_max_age_s`), frozen or regressed packet, value outside [0, 1] | open, and disarm until the grip is released once |

- Left and right are independent; one config may list only one side.
- Before EXTERNAL entry, each side waits for Info, checks it is a one-axis OP3
  with metre `TOTAL_OPENING`, the chosen `control_mode`, and POS, VEL_LIMIT and
  CURRENT command fields with proven limits. The SDK then checks both targets
  against the actual Info limits, and the first State and output preparation
  must arrive. Any failure stops startup with no EXTERNAL request.
- After entry, a side publishes only while State reports ready, output allowed
  and no inhibit. It waits up to `output_ready_timeout_s` for the first
  permission. After that, lost permission, a nonzero error code, or stale or
  out-of-range opening feedback (`feedback_max_age_s`) is a fault.
- `on_fault`: `open_and_disable_side` sends a best-effort open, stops that side
  and keeps the body running. `stop_session` does the same and then ends the
  session (checked CANCEL, like any other live fault).
- On a normal exit the gripper thread stops publishing and sends nothing, so
  each OP3 holds its last command (a grasp stays closed). Only a gripper fault
  sends a best-effort open. An inhibit or fault cannot guarantee that the
  gripper physically opens.
- vita-robot master (#12162) advertises OP3 with `POSITION_TORQUE` only, so
  `control_mode` must be `POSITION_TORQUE`; `POSITION` is refused at startup.
  The firmware sends `current_a` as the mode5 current (mA) and uses
  `vel_limit_m_s` as a host-side opening rate (must be above 0 and at most the
  robot's `max_opening_rate_m_s`). The CAN speed byte is fixed by firmware.
- The firmware drops a target older than its `command_timeout_us` and then holds
  the last position; it never opens by itself. After EXTERNAL exit the gripper
  therefore stays where it was, holding any object. Release the grips before
  stopping if it should open.

## Grasp detection and logging

While closed, a side reports `grasped` when, after `grasp_settle_s`, the
measured opening stays above `close_m + grasp_min_opening_m` and either moved
less than `grasp_stall_m` over that window or the measured current reached
`grasp_current_ratio * current_a`. It reports `closed_empty` when it stalls at
the closed target. This is a report only and never changes the command. OP3 State on current
firmware measures opening only (no current, velocity, temperature or error
code), so the current rule never fires, `current_a` in logs is null, and
error-code faults cannot be detected; stale feedback and lost permission still are.

The live log gets `gripper_ready` (firmware, serial, actual limits),
`gripper_command` (side, open/close, reason, measured opening and current),
`gripper_grasp` (state changes), `gripper_fault` and `gripper_closed` (publish
counts and the command each side holds after exit). The console prints `PICO_GRIPPER:` lines for the same events.

## Controller vibration

Not available. The PXREA SDK only offers generic device-control calls, and the
PICO streaming app has no vibration handler, so a vibration needs a PICO app
change first. `GripperRunner` accepts an optional `haptics` object with
`pulse(side, event)` (events `grasped`, `fault`); live passes none today. A
failing sink is logged once and dropped without affecting control.
