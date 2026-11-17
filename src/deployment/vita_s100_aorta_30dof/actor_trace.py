"""WTT stage names and cold trace contracts; the SDK owns the sole writer."""
from contextlib import nullcontext

STAGES = ('actor_observe', 'actor_infer', 'actor_decode', 'planner_infer', 'planner_map',
          'planner_preprocess', 'planner_postprocess', 'map_preprocess', 'map_postprocess',
          'planner_submit', 'planner_ready', 'planner_accept', 'planner_reject', 'planner_consume',
          'planner_fusion', 'body_submit', 'cycle', 'pico_input', 'gmr', 'reference_filter',
          'grip_input', 'grip_decision', 'grip_submit', 'fault_open', 'firmware_counters',
          'actor_observation', 'actor_action')


def span(runtime, name, **identities):
    recorder = getattr(runtime, 'trace', None)
    return nullcontext() if recorder is None else recorder.span(name, **identities)
