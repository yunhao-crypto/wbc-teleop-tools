"""Schedule fresh inputs before inference without replaying missed ticks."""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CycleTick:
    scheduled_ns: int
    started_ns: int
    wait_ms: float
    skipped_ticks: int


class PolicyCycleClock:
    """Periodic start deadlines; wait before acquiring state.

    If work overruns, coalesce expired ticks into one fresh-state cycle now.
    The following deadline is the next future tick, never a catch-up replay.
    Command completion intervals can vary with computation time.
    """

    def __init__(
        self,
        period_s: float,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("policy period must be finite and positive")
        self.period_ns = math.ceil(period_s * 1e9)
        self.clock = monotonic_ns
        self.sleep = sleep
        self.next_ns: int | None = None

    def reset(self, *, after_ns: int) -> None:
        self.next_ns = after_ns + self.period_ns

    def wait(self) -> CycleTick:
        before = self.clock()
        deadline = before if self.next_ns is None else self.next_ns
        now = before
        while now < deadline:
            self.sleep((deadline - now) / 1e9)
            now = self.clock()
        # A delayed wake is one cycle, never a burst of stale ticks.
        skipped = max(0, (now - deadline) // self.period_ns)
        self.next_ns = deadline + (skipped + 1) * self.period_ns
        return CycleTick(
            scheduled_ns=deadline,
            started_ns=now,
            wait_ms=(now - before) / 1e6,
            skipped_ticks=skipped,
        )
