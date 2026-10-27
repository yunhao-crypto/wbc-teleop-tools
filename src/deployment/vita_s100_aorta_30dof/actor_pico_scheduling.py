"""Process-local control scheduling; no robot or service configuration changes."""
import os


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
