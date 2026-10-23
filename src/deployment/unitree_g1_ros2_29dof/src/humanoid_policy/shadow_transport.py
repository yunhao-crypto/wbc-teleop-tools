"""Compatibility import for the shared ROS-free MuJoCo shadow transport."""

from deployment.holomotion_shadow.transport import (
    ShadowActionStateBridge,
    ShadowStateSample,
)

__all__ = ["ShadowActionStateBridge", "ShadowStateSample"]
