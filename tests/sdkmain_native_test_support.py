"""Inert publisher/header fixture; native codec only, no Core send or Node."""
import ctypes
import threading
from types import SimpleNamespace as NS


def native_test_transport(transport_class, frames):
    import body_cmd_schema_meta
    from lowlevel.BodyCmd import BodyCmd
    from locomotion_aorta._native_body_codec import NativeBodyPublisher
    from locomotion_aorta.transport import BODY_CMD_TOPIC

    class HeaderClock:
        def aorta_now_ns(self, pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64))[0] = 123456789
            return 0

    class Publisher:
        def __init__(self):
            self._seq_lock = threading.Lock()
            self._sequence = 0
            self._lib = HeaderClock()
            self._node_name = 'inert-header-fixture'
            self._include_node = True
            self._closed = False

        def publish_typed(self, *args, **kwargs):
            raise AssertionError('native BodyCmd must not use publish_typed')

        def publish_bytes(self, payload):
            frames.append(BodyCmd.GetRootAs(bytes(payload)))

    publisher = Publisher()
    transport = transport_class.__new__(transport_class)
    transport._closed = False
    transport._capture_publish_timing = True
    transport._publish_trace = threading.local()
    transport._body_publisher = NativeBodyPublisher(publisher, body_cmd_schema_meta)
    transport.publishers = {BODY_CMD_TOPIC: publisher}
    transport.subscribe = lambda *args: NS(close=lambda: None)
    transport.poll_control_status = lambda *args: None
    transport.close = lambda: None
    return transport
