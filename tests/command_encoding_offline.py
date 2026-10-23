"""Compare complete command bytes with vendor code, without a Node or transport."""

import importlib
import json
from pathlib import Path
import sys
import time
import threading
import types
import unittest

import flatbuffers
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deployment.vita_s100_aorta_30dof import onboard_command_encoding as packed
from locomotion_aorta.transport import _fill_low_cmd
from locomotion_aorta import LowCmd, MotorCommand

original = importlib.import_module("lowlevel.MotorCmd")
low = importlib.import_module("lowlevel.LowCmd")


def encode(module, motors, capacity=256):
    builder = flatbuffers.Builder(capacity)
    command = LowCmd(
        cmd_id=65535,
        motor_cmds=tuple(motors),
        external_version=1,
        external_session_id=123456789,
        external_sequence=65535,
    )
    bindings = types.SimpleNamespace(
        low_cmd_module=low, motor_cmd_module=module
    )
    # A valid empty table represents a fixed test header; layout and offsets are real.
    builder.StartObject(0)
    header = builder.EndObject()
    root = _fill_low_cmd(bindings, command, builder, header)
    builder.Finish(root)
    return bytes(builder.Output())


def motors(n, seed):
    rng = np.random.default_rng(seed)
    return [
        MotorCommand(
            mode=i % 256,
            q=float(rng.normal()),
            dq=float(rng.normal()),
            tau=float(rng.normal()),
            kp=float(rng.uniform(0, 500)),
            kd=float(rng.uniform(0, 30)),
        )
        for i in range(n)
    ]


class CommandPackingTest(unittest.TestCase):
    def test_original_publisher_header_and_sequence_identical(self):
        # Bypass Publisher.__init__: fake clock and byte sink, no FFI handle.
        from aorta.publisher import Publisher
        from deployment.vita_s100_aorta_30dof.onboard_publish_profile import (
            ProfiledCommandPublisher,
        )

        def inert():
            obj = Publisher.__new__(Publisher)
            obj._closed = False
            obj._sequence = 0
            obj._seq_lock = threading.Lock()
            obj._node_name = "offline-test"
            obj._include_node = True

            def clock(pointer):
                pointer._obj.value = 1791522000000000000
                return 0

            obj._lib = types.SimpleNamespace(aorta_now_ns=clock)
            out = []
            obj.publish_bytes = lambda b: out.append(bytes(b))
            return obj, out

        old, old_bytes = inert()
        new, new_bytes = inert()
        wrapped = ProfiledCommandPublisher(new)
        commands = LowCmd(
            cmd_id=1,
            motor_cmds=tuple(motors(42, 1)),
            external_version=1,
            external_session_id=123,
            external_sequence=1,
        )
        for _ in range(3):
            for publisher, module in ((old, original), (wrapped, packed)):
                bindings = types.SimpleNamespace(
                    low_cmd_module=low, motor_cmd_module=module
                )
                publisher.publish_typed(
                    lambda b, h: _fill_low_cmd(bindings, commands, b, h)
                )
        self.assertEqual(old_bytes, new_bytes)
        self.assertEqual(old._sequence, new._sequence)

    def test_complete_command_is_byte_identical(self):
        for n in (0, 1, 30, 42, 44):
            for seed in range(20):
                commands = motors(n, seed)
                for capacity in (0, 1, 256, 2048):
                    self.assertEqual(
                        encode(original, commands, capacity),
                        encode(packed, commands, capacity),
                    )

    def test_float32_edges(self):
        f = np.finfo(np.float32)
        for value in (
            -0.0,
            0.0,
            float(f.max),
            -float(f.max),
            float(f.tiny),
            1e-40,
        ):
            commands = [
                MotorCommand(
                    mode=255, q=value, dq=value, tau=value, kp=value, kd=value
                )
            ]
            self.assertEqual(
                encode(original, commands), encode(packed, commands)
            )

    def test_invalid_modes_rejected(self):
        for mode in (-1, 256):
            for module in (original, packed):
                b = flatbuffers.Builder(256)
                with self.assertRaises(Exception):
                    module.CreateMotorCmd(b, mode, 0.0, 0.0, 0.0, 0.0, 0.0)


def benchmark():
    commands = motors(42, 20261009)
    result = dict(
        action_sent=False,
        scope="FlatBuffers encoding only; no publisher or SDK controller",
    )
    for name, module in (("vendor", original), ("packed", packed)):
        costs = []
        for _ in range(2000):
            start = time.perf_counter_ns()
            encode(module, commands)
            costs.append((time.perf_counter_ns() - start) / 1e6)
        result[name] = dict(
            mean=float(np.mean(costs)),
            p99=float(np.percentile(costs, 99)),
            max=max(costs),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        CommandPackingTest
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    benchmark()
