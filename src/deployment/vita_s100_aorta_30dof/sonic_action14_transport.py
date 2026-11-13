"""Native action14 data-plane adapter. No connections on import.

Control-plane admission is deliberately separate: constructing this publisher is
not permission to send motion. The caller must own and continuously validate
an action14 request before calling publish().
"""


class NativeReferencePublisher:
    def __init__(self, *, transport=None, node_name='wtt_sonic_action14', group='default'):
        if transport is None:
            from locomotion_aorta.transport import _AortaTransport
            transport = _AortaTransport(node_name=node_name, group=group)
        self.transport = transport
        try:
            self.transport._prepare_body_output()
        except BaseException:
            self.transport.close()
            raise

    def robot_info(self, timeout=2.0):
        response = self.transport.get_robot_info(timeout)
        if response.info is None or response.runtime_id <= 0:
            raise RuntimeError('RobotInfo missing or runtime identity unavailable')
        return response

    def publish(self, command, *, authorized):
        # The installed generated FlatBuffers publisher is used, not the
        # ExternalController (which unconditionally selects action EXTERNAL).
        if not authorized():
            raise RuntimeError('action14 admission/ownership expired')
        if (command.action != 14 or type(command.control_id) is not int or
                not 0 < command.control_id < 2**64 or command.input_state_id != 0 or
                command.input_state_ts_real_us != 0):
            raise ValueError('not an independent action14 reference')
        self.transport._publish_body(command, lambda serialize: serialize() if authorized() else
                                     (_ for _ in ()).throw(RuntimeError('action14 ownership changed')))

    def close(self):
        self.transport.close()
