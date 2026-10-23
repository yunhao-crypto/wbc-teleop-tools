"""Line-buffered control-aligned Vita session recording."""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class JsonlSessionRecorder:
    """Flush every control record so Ctrl-C or faults retain evidence."""

    def __init__(self, root: str | Path, *, prefix: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = (
            Path(root).expanduser().resolve()
            / f"{prefix}-{stamp}-{os.getpid()}"
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "control.jsonl"
        descriptor = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)

    def write(self, payload: Mapping[str, Any]) -> None:
        self.stream.write(
            json.dumps(_jsonable(payload), sort_keys=True) + "\n"
        )

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

    def __enter__(self) -> "JsonlSessionRecorder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["JsonlSessionRecorder"]
