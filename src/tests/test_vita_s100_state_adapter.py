import unittest

import numpy as np

from deployment.vita_s100_aorta_30dof.policy_contract import (
    VITA_ACTION_TO_COMMAND_SLOTS,
    VITA_DOF_TO_STATE_SLOTS,
    VITA_EXPECTED_STATE_SLOT_ROLES,
    VITA_POLICY_STATE_TO_STATE_SLOTS,
)
from deployment.vita_s100_aorta_30dof.state_adapter import (
    IMU_LOCATION_HIP,
    IMU_LOCATION_TRUNK,
    IMU_STATUS_OK,
    decode_humanoid_lowstate_view,
    policy_state_to_state_slots,
)


class FakeImuState:
    def __init__(self, quaternion, gyroscope, accelerometer, rpy):
        self.values = {
            "Quaternion": quaternion,
            "Gyroscope": gyroscope,
            "Accelerometer": accelerometer,
            "Rpy": rpy,
        }

    def __getattr__(self, name):
        values = self.values[name]
        return lambda index: values[index]


class FakeImuSample:
    def __init__(self, location, status, state):
        self.values = {
            "Location": location,
            "Status": status,
            "State": state,
            "TsSampledUs": 100,
            "TsAcquiredHwUs": 101,
            "TsIngestedUs": 102,
            "TsAssembledUs": 103,
            "DeviceTimestampMs": 99,
            "HostTimestampUs": 104,
        }

    def __getattr__(self, name):
        value = self.values[name]
        return lambda: value


class FakeMotor:
    def __init__(self, slot):
        self.values = {
            "SlotIndex": slot,
            "SlotRole": VITA_EXPECTED_STATE_SLOT_ROLES[slot],
            "Q": slot + 0.25,
            "Dq": -slot - 0.5,
            "TauEst": slot * 0.1,
            "Temperature": 30.0 + slot,
            "Lost": 0,
        }

    def __getattr__(self, name):
        value = self.values[name]
        return lambda: value


class FakeLowState:
    def __init__(self, slots=None):
        slot_values = range(44) if slots is None else slots
        self.motors = [FakeMotor(slot) for slot in slot_values]
        self.imus = [
            FakeImuSample(
                IMU_LOCATION_HIP,
                1,
                FakeImuState(
                    [1.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3],
                    [0.0, 0.0, 9.81],
                    [1.0, 2.0, 3.0],
                ),
            ),
            FakeImuSample(
                IMU_LOCATION_TRUNK,
                6,
                FakeImuState(
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ),
            ),
        ]
        self.values = {
            "StateId": 123,
            "TsStateReal": 1_000_000,
            "TsStatePub": 1_000_100,
            "TsConsumedUs": 1_000_050,
            "AttitudeValid": True,
        }

    def __getattr__(self, name):
        if name == "MotorStatesLength":
            return lambda: len(self.motors)
        if name == "MotorStates":
            return lambda index: self.motors[index]
        if name == "ImuStatesLength":
            return lambda: len(self.imus)
        if name == "ImuStates":
            return lambda index: self.imus[index]
        value = self.values[name]
        return lambda: value


class VitaStateAdapterTest(unittest.TestCase):
    def test_expands_policy_state_using_action_order(self):
        values = np.arange(30, dtype=np.float32)

        expanded = policy_state_to_state_slots(values, fill_value=-1.0)

        for policy_index, state_slot in enumerate(
            VITA_POLICY_STATE_TO_STATE_SLOTS
        ):
            self.assertEqual(expanded[state_slot], values[policy_index])
        self.assertEqual(expanded[25], -1.0)
        self.assertEqual(expanded[42], -1.0)

    def test_decodes_slots_and_keeps_explicit_joint_orders(self):
        state = decode_humanoid_lowstate_view(
            FakeLowState(), received_monotonic_ns=42
        )
        self.assertEqual(state.state_id, 123)
        self.assertEqual(state.received_monotonic_ns, 42)
        observation_q, observation_dq = state.observation_joint_state()
        action_q, action_dq = state.action_order_joint_state()
        dof_q, dof_dq = state.dof_order_joint_state()
        np.testing.assert_allclose(
            observation_q,
            np.asarray(VITA_POLICY_STATE_TO_STATE_SLOTS) + 0.25,
        )
        np.testing.assert_allclose(
            observation_dq,
            -np.asarray(VITA_POLICY_STATE_TO_STATE_SLOTS) - 0.5,
        )
        np.testing.assert_allclose(
            action_q,
            np.asarray(VITA_ACTION_TO_COMMAND_SLOTS) + 0.25,
        )
        np.testing.assert_allclose(
            action_dq,
            -np.asarray(VITA_ACTION_TO_COMMAND_SLOTS) - 0.5,
        )
        np.testing.assert_allclose(
            dof_q,
            np.asarray(VITA_DOF_TO_STATE_SLOTS) + 0.25,
        )
        np.testing.assert_allclose(
            dof_dq,
            -np.asarray(VITA_DOF_TO_STATE_SLOTS) - 0.5,
        )
        np.testing.assert_array_equal(observation_q, action_q)
        self.assertFalse(np.array_equal(observation_q, dof_q))

    def test_requires_explicit_imu_location_and_status(self):
        state = decode_humanoid_lowstate_view(FakeLowState())
        hip = state.require_imu(
            IMU_LOCATION_HIP, allowed_statuses={IMU_STATUS_OK}
        )
        np.testing.assert_allclose(
            hip.normalized_quaternion_wxyz(), [1.0, 0.0, 0.0, 0.0]
        )
        with self.assertRaisesRegex(RuntimeError, "rejected status 6"):
            state.require_imu(IMU_LOCATION_TRUNK, allowed_statuses={1})
        with self.assertRaisesRegex(ValueError, "near-zero quaternion"):
            state.require_imu(IMU_LOCATION_TRUNK).normalized_quaternion_wxyz()

    def test_rejects_duplicate_motor_slots(self):
        slots = list(range(44))
        slots[-1] = 0
        with self.assertRaisesRegex(
            ValueError, "duplicate motor slot index 0"
        ):
            decode_humanoid_lowstate_view(FakeLowState(slots))

    def test_rejects_missing_hip_imu(self):
        view = FakeLowState()
        view.imus = [view.imus[1]]
        with self.assertRaisesRegex(ValueError, "required HIP IMU"):
            decode_humanoid_lowstate_view(view)

    def test_rejects_unexpected_reserved_slot_role(self):
        view = FakeLowState()
        view.motors[42].values["SlotRole"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected slot roles"):
            decode_humanoid_lowstate_view(view)


if __name__ == "__main__":
    unittest.main()
