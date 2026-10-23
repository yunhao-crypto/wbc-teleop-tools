"""Pinned SDK 0.2.1 PullController session; no policy-rate Python RX callback.

The SDK owns ACK/status, leases, validation, sequencing, native encoding and
cancellation. ACK remains its low-rate compatibility callback. Body feedback
and ControlStatus are polled by the caller. Importing creates no handles.
"""
from .aorta_adapter import VitaExternalSession
from .sdk021_guard import VENDOR_SOURCE_SHA256, verify_vendor_sources


class OnboardLiveSession(VitaExternalSession):
    def __init__(self, group: str) -> None:
        super().__init__(group=group)

    @property
    def last_publish_profile(self):
        # These overlapping intervals are SDK measurements, not delivery/ACK
        # timestamps. Do not intercept the old publish_typed encoder interface.
        return dict(self.controller._last_publish_timing() or {})
