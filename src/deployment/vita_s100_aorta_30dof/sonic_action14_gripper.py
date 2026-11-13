"""Opt-in OP3 gripper on an existing action14 transport.

Importing this module does not import EndEffectorClient. describe() only
reads JSON. attach() is the first call that can construct a client, and it
refuses a missing connection instead of requesting EXTERNAL.
"""


def describe(path):
    from .actor_pico_gripper import load_config
    config = load_config(path)
    return dict(gripper_schema=config['schema'], gripper_device=config['device_model'],
                gripper_sides=sorted(config['sides']), gripper_opened=False)


def attach(path, connection, shared, *, sdk=None):
    if connection is None:
        raise ValueError('gripper requires the action14 body transport; EXTERNAL is not requested')
    from .actor_pico_gripper import GripperRunner, load_config
    return GripperRunner(load_config(path), shared, connection=connection, sdk=sdk)
