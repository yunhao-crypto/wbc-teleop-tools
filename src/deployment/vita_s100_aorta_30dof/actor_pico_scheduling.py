"""Process-local control scheduling; no robot or service configuration changes."""
import os
from pathlib import Path


def control_scheduler():
    """Match the original onboard HoloMotion operator's SCHED_RR priority 1.

    Called before transport/control handles. Permission failures are startup
    errors rather than silently falling back to ordinary scheduling.
    """
    os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(1))
    policy = os.sched_getscheduler(0)
    priority = os.sched_getparam(0).sched_priority
    if policy != os.SCHED_RR or priority != 1:
        raise RuntimeError('Control scheduler did not enter SCHED_RR priority 1')
    return dict(policy='SCHED_RR', priority=priority)


def input_worker_scheduler():
    """GMR and its vendor service must not inherit the control priority."""
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))


def control_affinity(cpus):
    """Pin every existing control-process thread; later threads inherit the mask.

    Call only after the Pico worker has been spawned: a child starts with its
    parent's mask, and the worker accepts only worker/GMR CPUs inside it.
    Native inference and transport threads that already exist are moved too.
    Unavailable CPUs are a startup error. An empty list leaves the mask as is.
    """
    available = os.sched_getaffinity(0)
    if cpus:
        if not set(cpus) <= available:
            raise ValueError(f'Requested control CPUs {list(cpus)} outside available CPUs {sorted(available)}')
        for task in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(task.name), cpus)
            except ProcessLookupError:
                pass
    return sorted(os.sched_getaffinity(0))
