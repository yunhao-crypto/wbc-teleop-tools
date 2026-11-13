"""Causal 50 Hz source reference grid. No policy inference or model history.

Source stamps are local diagnostics, never BodyState IDs. Each output is
interpolated only between two *received* Pico samples; no extrapolation,
synthetic freshness, or repeat publication is allowed.
"""
import math


class SourceGrid:
    def __init__(self, period_ns=20_000_000, max_gap_ns=600_000_000):
        if (type(period_ns) is not int or period_ns <= 0 or
                type(max_gap_ns) is not int or not period_ns <= max_gap_ns <= 2_000_000_000):
            raise ValueError('invalid reference grid timing')
        self.period_ns = period_ns
        self.max_gap_ns = max_gap_ns
        self.previous = None
        self.next_stamp = None
        self.grid_sequence = 0

    def push(self, stamp_ns, received_monotonic, q, root_wxyz):
        q = tuple(float(x) for x in q)
        root = tuple(float(x) for x in root_wxyz)
        if (type(stamp_ns) is not int or stamp_ns <= 0 or len(q) != 28 or
                len(root) != 4 or not all(math.isfinite(x) for x in q + root) or
                not math.isfinite(received_monotonic)):
            raise ValueError('invalid Pico reference')
        norm = math.sqrt(sum(x*x for x in root))
        if abs(norm-1) > .001:
            raise ValueError('nonunit Pico root')
        current = (stamp_ns, received_monotonic, q, root)
        previous = self.previous
        if previous is None:
            self.previous = current
            self.next_stamp = stamp_ns + self.period_ns
            return []
        if stamp_ns <= previous[0] or received_monotonic < previous[1]:
            raise ValueError('out of order Pico reference')
        if stamp_ns - previous[0] > self.max_gap_ns:
            # A discontinuity must be re-armed explicitly by a new session.
            raise ValueError('Pico sample gap exceeded')
        outputs = []
        while self.next_stamp <= stamp_ns:
            fraction = (self.next_stamp - previous[0]) / (stamp_ns - previous[0])
            position = tuple(a+(b-a)*fraction for a,b in zip(previous[2],q))
            # Shortest-path normalized lerp. This is not the original WTT slerp
            # and is not a claim of numerical equivalence with that window.
            before = previous[3]
            after = root if sum(a*b for a,b in zip(before,root)) >= 0 else tuple(-x for x in root)
            quat = tuple(a+(b-a)*fraction for a,b in zip(before,after))
            magnitude = math.sqrt(sum(x*x for x in quat))
            if magnitude < 1e-9:
                raise ValueError('degenerate root interpolation')
            # Receipt stays on the segment between the two real receipts. An
            # earlier grid point must not inherit the later packet's freshness.
            receipt = previous[1] + (received_monotonic-previous[1])*fraction
            self.grid_sequence += 1
            outputs.append(dict(grid_sequence=self.grid_sequence, source_ns=self.next_stamp,
                                received_monotonic=receipt,
                                q=position, root_wxyz=tuple(x/magnitude for x in quat)))
            self.next_stamp += self.period_ns
        self.previous = current
        return outputs
