"""Deterministic latest-only schedules for raw Pico deployment replay.

The real Pico reader runs asynchronously while the policy consumes only the
latest sample at 50 Hz.  This module reproduces that timing contract without
interpolating raw Pico frames.  It also prepends the deployment history needed
by :class:`GpuOnlineSmplReference` and appends an explicit final-pose hold so a
finite recording can expose every source target frame to the delayed policy.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawPicoClip:
    """One XRoboToolkit CSV with exact raw timestamps and body poses."""

    path: Path
    timestamp_realtime: np.ndarray
    timestamp_monotonic: np.ndarray
    timestamp_ns: np.ndarray
    pico_dt: np.ndarray
    pico_fps: np.ndarray
    body_poses: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.body_poses.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(
            self.timestamp_monotonic[-1] - self.timestamp_monotonic[0]
        )


@dataclass(frozen=True)
class PicoReplaySchedule:
    """50 Hz latest-only source indices and online-ring feed events."""

    policy_hz: float
    source_indices: np.ndarray
    latest_source_indices: np.ndarray
    policy_sample_times: np.ndarray
    feed_event_indices: np.ndarray
    event_source_indices: np.ndarray
    event_sample_times: np.ndarray
    prefill_event_count: int
    history_frames: int
    future_lag_frames: int

    @property
    def policy_frame_count(self) -> int:
        return int(self.source_indices.shape[0])

    def target_source_index(self, policy_frame: int) -> int:
        return int(self.source_indices[int(policy_frame)])

    def latest_source_index(self, policy_frame: int) -> int:
        return int(self.latest_source_indices[int(policy_frame)])

    def feed_event_index(self, policy_frame: int) -> int | None:
        """Return a new event, or None when the reader has no new frame."""

        event = int(self.feed_event_indices[int(policy_frame)])
        return None if event < 0 else event


def load_raw_pico_csv(path: str | Path) -> RawPicoClip:
    """Load the production 5 + 24x7 Pico CSV contract fail-closed."""

    source = Path(path).expanduser().resolve()
    values = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 173:
        raise ValueError(
            f"Raw Pico CSV must have 173 columns, got {values.shape}: {source}"
        )
    if values.shape[0] < 12:
        raise ValueError(
            f"Raw Pico CSV needs at least 12 frames, got {values.shape[0]}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Raw Pico CSV contains NaN or Inf: {source}")

    monotonic = values[:, 1]
    if not np.all(np.diff(monotonic) > 0.0):
        raise ValueError(
            f"Pico monotonic timestamps are not increasing: {source}"
        )
    body = values[:, 5:].reshape(-1, 24, 7).astype(np.float32)
    quat_norm = np.linalg.norm(body[:, :, 3:7], axis=-1)
    if float(quat_norm.min()) < 0.5:
        raise ValueError(
            f"Pico input contains a degenerate quaternion: {source}"
        )

    return RawPicoClip(
        path=source,
        timestamp_realtime=values[:, 0].copy(),
        timestamp_monotonic=monotonic.copy(),
        timestamp_ns=np.rint(values[:, 2]).astype(np.int64),
        pico_dt=values[:, 3].astype(np.float32),
        pico_fps=values[:, 4].astype(np.float32),
        body_poses=np.ascontiguousarray(body),
    )


def build_latest_only_schedule(
    timestamp_monotonic: np.ndarray,
    *,
    policy_hz: float = 50.0,
    history_frames: int = 50,
    future_lag_frames: int = 11,
) -> PicoReplaySchedule:
    """Build the deployment reader/policy timing schedule.

    Raw arrivals are never interpolated.  At each policy tick, the newest raw
    frame at or before that tick is selected.  If no new frame arrived, the
    online ring remains unchanged exactly as in the production reader.
    """

    timestamps = np.asarray(timestamp_monotonic, dtype=np.float64).reshape(-1)
    if timestamps.size < future_lag_frames + 1:
        raise ValueError("Pico replay is too short for the future window")
    if not np.isfinite(timestamps).all() or not np.all(
        np.diff(timestamps) > 0
    ):
        raise ValueError(
            "Pico timestamps must be finite and strictly increasing"
        )
    policy_hz = float(policy_hz)
    if policy_hz <= 0.0:
        raise ValueError("policy_hz must be positive")
    if history_frames <= 0 or future_lag_frames < 0:
        raise ValueError("history/future lag must be non-negative")

    dt = 1.0 / policy_hz
    relative = timestamps - timestamps[0]
    tick_count = int(np.ceil(relative[-1] * policy_hz - 1.0e-9)) + 1
    tick_times = np.arange(tick_count, dtype=np.float64) * dt
    polled_indices = np.searchsorted(relative, tick_times, side="right") - 1
    polled_indices = np.clip(polled_indices, 0, timestamps.size - 1)
    if polled_indices[0] != 0:
        raise AssertionError(
            "Latest-only replay must start from source frame zero"
        )
    if polled_indices[-1] != timestamps.size - 1:
        raise AssertionError(
            "Final policy poll must expose the last raw frame"
        )
    new_mask = np.concatenate(
        (np.asarray([True]), polled_indices[1:] != polled_indices[:-1])
    )
    unique_poll_positions = np.flatnonzero(new_mask)
    unique_source_indices = polled_indices[unique_poll_positions]
    if unique_source_indices.size < future_lag_frames + 1:
        raise ValueError("Pico replay has too few unique samples for prefill")

    first_time = float(timestamps[0])
    history_indices = np.zeros(history_frames, dtype=np.int64)
    history_times = first_time - dt * np.arange(
        history_frames, 0, -1, dtype=np.float64
    )
    stream_times = timestamps[unique_source_indices]
    tail_indices = np.full(
        future_lag_frames, timestamps.size - 1, dtype=np.int64
    )
    tail_times = float(stream_times[-1]) + dt * np.arange(
        1, future_lag_frames + 1, dtype=np.float64
    )
    event_indices = np.concatenate(
        (history_indices, unique_source_indices.astype(np.int64), tail_indices)
    )
    event_times = np.concatenate((history_times, stream_times, tail_times))
    if not np.all(np.diff(event_times) > 0.0):
        raise AssertionError(
            "Replay feed event times must be strictly increasing"
        )

    prefill = history_frames + future_lag_frames + 1
    if prefill > event_indices.size:
        raise ValueError("Pico stream is too short to prefill the online ring")
    # Motion mode becomes ready when the first 12 unique policy samples fill
    # the delayed current+future window.  Subsequent policy ticks either append
    # one newer Pico sample or leave the ring unchanged.
    motion_start_poll = int(unique_poll_positions[future_lag_frames])
    ring = list(event_indices[:prefill])
    next_unique_event = prefill
    target_by_policy: list[int] = []
    latest_by_policy: list[int] = []
    sample_time_by_policy: list[float] = []
    feed_by_policy: list[int] = []
    for poll in range(motion_start_poll, polled_indices.size):
        feed_event = -1
        if (
            poll > motion_start_poll
            and polled_indices[poll] != polled_indices[poll - 1]
        ):
            feed_event = next_unique_event
            ring.append(int(event_indices[feed_event]))
            ring = ring[-prefill:]
            next_unique_event += 1
        target_by_policy.append(int(ring[-(future_lag_frames + 1)]))
        latest_by_policy.append(int(ring[-1]))
        sample_time_by_policy.append(float(timestamps[int(ring[-1])]))
        feed_by_policy.append(feed_event)

    for _ in range(future_lag_frames):
        feed_event = next_unique_event
        ring.append(int(event_indices[feed_event]))
        ring = ring[-prefill:]
        next_unique_event += 1
        target_by_policy.append(int(ring[-(future_lag_frames + 1)]))
        latest_by_policy.append(int(ring[-1]))
        sample_time_by_policy.append(float(event_times[feed_event]))
        feed_by_policy.append(feed_event)

    source_indices = np.asarray(target_by_policy, dtype=np.int64)
    schedule = PicoReplaySchedule(
        policy_hz=policy_hz,
        source_indices=source_indices,
        latest_source_indices=np.asarray(latest_by_policy, dtype=np.int64),
        policy_sample_times=np.asarray(
            sample_time_by_policy, dtype=np.float64
        ),
        feed_event_indices=np.asarray(feed_by_policy, dtype=np.int64),
        event_source_indices=event_indices,
        event_sample_times=event_times,
        prefill_event_count=prefill,
        history_frames=int(history_frames),
        future_lag_frames=int(future_lag_frames),
    )
    if schedule.source_indices[0] != 0:
        raise AssertionError("Online target must begin from source frame zero")
    if schedule.source_indices[-1] != timestamps.size - 1:
        raise AssertionError("Tail hold must expose the final source frame")
    if next_unique_event != event_indices.size:
        raise AssertionError("Replay did not consume every unique/tail event")
    return schedule


def select_shortest_per_source(raw_root: str | Path) -> tuple[Path, ...]:
    """Select one shortest complete CSV from every immediate source folder."""

    root = Path(raw_root).expanduser().resolve()
    by_parent: dict[str, list[tuple[float, Path]]] = {}
    for path in sorted(root.glob("*/*.csv")):
        timestamps = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=(1,),
            dtype=np.float64,
        )
        timestamps = np.asarray(timestamps).reshape(-1)
        if timestamps.size < 2 or not np.all(np.diff(timestamps) > 0.0):
            raise ValueError(f"Invalid Pico timestamps: {path}")
        duration = float(timestamps[-1] - timestamps[0])
        by_parent.setdefault(path.parent.name, []).append((duration, path))
    if not by_parent:
        raise FileNotFoundError(f"No Pico CSV files found below {root}")
    return tuple(
        min(by_parent[parent], key=lambda item: (item[0], item[1].name))[1]
        for parent in sorted(by_parent)
    )


__all__ = [
    "PicoReplaySchedule",
    "RawPicoClip",
    "build_latest_only_schedule",
    "load_raw_pico_csv",
    "select_shortest_per_source",
]
