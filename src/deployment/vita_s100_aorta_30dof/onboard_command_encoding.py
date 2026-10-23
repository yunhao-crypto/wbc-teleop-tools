"""Bulk scalar packing for the pinned 24-byte MotorCmd FlatBuffers struct.

The SDK still validates commands, assigns session/sequence, builds the LowCmd
table and header, and publishes with the original QoS. Only the struct's six
scalar writes are combined into one standard-library C pack call.
"""

import struct


_MOTOR = struct.Struct("<B3x5f")


def CreateMotorCmd(builder, mode, q, dq, tau, kp, kd):
    builder.Prep(4, 24)
    builder.head -= 24
    _MOTOR.pack_into(builder.Bytes, builder.head, mode, q, dq, tau, kp, kd)
    return builder.Offset()
