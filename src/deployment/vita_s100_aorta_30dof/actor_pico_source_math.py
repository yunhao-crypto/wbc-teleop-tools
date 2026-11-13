"""Minimal GMR quaternion contract, independent of policy inference or SDK."""
import numpy as np


def quaternion(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('invalid GMR quaternion')
    norm = np.linalg.norm(q)
    if not .5 <= norm <= 1.5:
        raise ValueError('invalid GMR quaternion norm')
    return q / norm
