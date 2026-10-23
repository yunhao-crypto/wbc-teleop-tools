"""Host-local process lock for exclusive robot command ownership."""

from __future__ import annotations
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path


class ControlProcessLock:
    """Hold a non-blocking host-local lock for one robot command process."""

    def __init__(self, path: Path, *, purpose: str) -> None:
        self.path = path.expanduser().resolve()
        self.purpose = purpose
        self._descriptor: int | None = None

    def __enter__(self) -> "ControlProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.read(descriptor, 4096).decode(
                "utf-8", errors="replace"
            )
            os.close(descriptor)
            raise RuntimeError(
                "another Vita control process holds the lock: "
                f"{holder.strip() or self.path}"
            ) from None
        metadata = {
            "pid": os.getpid(),
            "purpose": self.purpose,
            "started_at": datetime.now().astimezone().isoformat(),
        }
        os.ftruncate(descriptor, 0)
        payload = (json.dumps(metadata, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
        self._descriptor = descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        if self._descriptor is None:
            return
        try:
            os.ftruncate(self._descriptor, 0)
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._descriptor = None


__all__ = ["ControlProcessLock"]
