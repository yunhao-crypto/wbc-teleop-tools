"""Frozen Vita02A policy and S100 slot-order contracts."""

VITA_STATE_SLOT_COUNT = 44
VITA_COMMAND_SLOT_COUNT = 42
VITA_SLOT_ROLE_ACTIVE = 1
VITA_SLOT_ROLE_RESERVED = 2

VITA_COMMAND_SLOT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_thumb_pitch_joint",
    "left_thumb_roll_joint",
    "left_index_finger_pitch_joint",
    "left_middle_finger_pitch_joint",
    "left_ring_finger_pitch_joint",
    "left_little_finger_pitch_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_thumb_pitch_joint",
    "right_thumb_roll_joint",
    "right_index_finger_pitch_joint",
    "right_middle_finger_pitch_joint",
    "right_ring_finger_pitch_joint",
    "right_little_finger_pitch_joint",
    "neck_pitch_joint",
    "neck_yaw_joint",
)

VITA_EXPECTED_STATE_SLOT_ROLES: tuple[int, ...] = (
    *(VITA_SLOT_ROLE_ACTIVE for _ in range(VITA_COMMAND_SLOT_COUNT)),
    VITA_SLOT_ROLE_RESERVED,
    VITA_SLOT_ROLE_RESERVED,
)

VITA_OBSERVATION_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "neck_yaw_joint",
    "right_shoulder_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_roll_joint",
    "neck_pitch_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
)

VITA_OBSERVATION_TO_STATE_SLOTS: tuple[int, ...] = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    41,
    18,
    3,
    9,
    15,
    40,
    19,
    4,
    10,
    16,
    20,
    5,
    11,
    17,
    21,
    22,
    31,
    23,
    32,
    24,
    33,
)

VITA_DOF_JOINT_NAMES: tuple[str, ...] = VITA_OBSERVATION_JOINT_NAMES
VITA_DOF_TO_STATE_SLOTS: tuple[int, ...] = VITA_OBSERVATION_TO_STATE_SLOTS
VITA_TARGET_DOF_JOINT_NAMES: tuple[str, ...] = VITA_DOF_JOINT_NAMES
VITA_TARGET_TO_COMMAND_SLOTS: tuple[int, ...] = VITA_DOF_TO_STATE_SLOTS

VITA_ACTION_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
)

VITA_ACTION_TO_COMMAND_SLOTS: tuple[int, ...] = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    18,
    3,
    9,
    15,
    19,
    4,
    10,
    16,
    20,
    5,
    11,
    17,
    21,
    22,
    31,
    23,
    32,
    24,
    33,
    41,
    40,
)

VITA_POLICY_STATE_JOINT_NAMES: tuple[str, ...] = VITA_ACTION_JOINT_NAMES
VITA_POLICY_STATE_TO_STATE_SLOTS: tuple[int, ...] = (
    VITA_ACTION_TO_COMMAND_SLOTS
)


def _validate_contract() -> None:
    if len(VITA_COMMAND_SLOT_NAMES) != VITA_COMMAND_SLOT_COUNT:
        raise RuntimeError("Vita command slot roster must contain 42 joints")
    if len(set(VITA_COMMAND_SLOT_NAMES)) != VITA_COMMAND_SLOT_COUNT:
        raise RuntimeError(
            "Vita command slot roster contains duplicate joints"
        )
    if len(VITA_EXPECTED_STATE_SLOT_ROLES) != VITA_STATE_SLOT_COUNT:
        raise RuntimeError(
            "Vita state slot role contract must contain 44 slots"
        )
    if len(VITA_OBSERVATION_JOINT_NAMES) != 30:
        raise RuntimeError("Vita observation contract must contain 30 joints")
    if len(VITA_ACTION_JOINT_NAMES) != 30:
        raise RuntimeError("Vita action contract must contain 30 joints")
    if len(VITA_TARGET_DOF_JOINT_NAMES) != 30:
        raise RuntimeError("Vita target DOF contract must contain 30 joints")
    if set(VITA_OBSERVATION_JOINT_NAMES) != set(VITA_ACTION_JOINT_NAMES):
        raise RuntimeError("Vita observation and action joint sets differ")
    if VITA_TARGET_DOF_JOINT_NAMES != VITA_OBSERVATION_JOINT_NAMES:
        raise RuntimeError("Vita target DOF and observation orders differ")
    if VITA_POLICY_STATE_JOINT_NAMES != VITA_ACTION_JOINT_NAMES:
        raise RuntimeError("Vita policy-state and action orders differ")
    if len(set(VITA_OBSERVATION_TO_STATE_SLOTS)) != 30:
        raise RuntimeError("Vita observation state slots are not unique")
    if len(set(VITA_ACTION_TO_COMMAND_SLOTS)) != 30:
        raise RuntimeError("Vita action command slots are not unique")
    if len(set(VITA_TARGET_TO_COMMAND_SLOTS)) != 30:
        raise RuntimeError("Vita target command slots are not unique")
    if len(set(VITA_POLICY_STATE_TO_STATE_SLOTS)) != 30:
        raise RuntimeError("Vita policy-state slots are not unique")
    if not all(
        0 <= slot < VITA_STATE_SLOT_COUNT
        for slot in VITA_OBSERVATION_TO_STATE_SLOTS
    ):
        raise RuntimeError("Vita observation state slot is out of range")
    if not all(
        0 <= slot < VITA_COMMAND_SLOT_COUNT
        for slot in VITA_ACTION_TO_COMMAND_SLOTS
    ):
        raise RuntimeError("Vita action command slot is out of range")
    if not all(
        0 <= slot < VITA_COMMAND_SLOT_COUNT
        for slot in VITA_TARGET_TO_COMMAND_SLOTS
    ):
        raise RuntimeError("Vita target command slot is out of range")
    if not all(
        0 <= slot < VITA_STATE_SLOT_COUNT
        for slot in VITA_POLICY_STATE_TO_STATE_SLOTS
    ):
        raise RuntimeError("Vita policy-state slot is out of range")
    for joint_name, slot in zip(
        VITA_ACTION_JOINT_NAMES, VITA_ACTION_TO_COMMAND_SLOTS, strict=True
    ):
        if VITA_COMMAND_SLOT_NAMES[slot] != joint_name:
            raise RuntimeError(
                f"Vita action mapping disagrees with slot roster at {slot}"
            )
    for joint_name, slot in zip(
        VITA_OBSERVATION_JOINT_NAMES,
        VITA_OBSERVATION_TO_STATE_SLOTS,
        strict=True,
    ):
        if VITA_COMMAND_SLOT_NAMES[slot] != joint_name:
            raise RuntimeError(
                "Vita observation mapping disagrees with slot roster at "
                f"{slot}"
            )
    for joint_name, slot in zip(
        VITA_TARGET_DOF_JOINT_NAMES,
        VITA_TARGET_TO_COMMAND_SLOTS,
        strict=True,
    ):
        if VITA_COMMAND_SLOT_NAMES[slot] != joint_name:
            raise RuntimeError(
                f"Vita target mapping disagrees with slot roster at {slot}"
            )


_validate_contract()


__all__ = [
    "VITA_ACTION_JOINT_NAMES",
    "VITA_ACTION_TO_COMMAND_SLOTS",
    "VITA_COMMAND_SLOT_NAMES",
    "VITA_COMMAND_SLOT_COUNT",
    "VITA_DOF_JOINT_NAMES",
    "VITA_DOF_TO_STATE_SLOTS",
    "VITA_EXPECTED_STATE_SLOT_ROLES",
    "VITA_OBSERVATION_JOINT_NAMES",
    "VITA_OBSERVATION_TO_STATE_SLOTS",
    "VITA_POLICY_STATE_JOINT_NAMES",
    "VITA_POLICY_STATE_TO_STATE_SLOTS",
    "VITA_SLOT_ROLE_ACTIVE",
    "VITA_SLOT_ROLE_RESERVED",
    "VITA_STATE_SLOT_COUNT",
    "VITA_TARGET_DOF_JOINT_NAMES",
    "VITA_TARGET_TO_COMMAND_SLOTS",
]
