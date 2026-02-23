"""Read-write lock for model access during non-blocking sleep.

Multiple readers (chat inference) can proceed concurrently.
A writer (weight update during fuse/reload) gets exclusive access.

Most sleep phases don't need the lock:
  - Curation, data prep, replay buffer: no lock (no model access)
  - LoRA training: no lock (runs as subprocess on MLX)
  - Pre/post evaluation: read lock (brief)
  - MEMIT scale operations: write lock (brief)
  - Fuse + reload: write lock (1-2 seconds, the critical section)
"""

import threading
import time
from contextlib import contextmanager


class ModelLock:
    """Read-write lock for model access.

    Uses a condition variable to coordinate readers and writers.
    Writers have priority to prevent starvation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writer = False
        self._writer_waiting = False
        # Diagnostics
        self._last_write_duration = 0.0
        self._total_write_time = 0.0
        self._write_count = 0

    @contextmanager
    def read_lock(self, timeout=30.0):
        """Acquire read access. Multiple readers can hold this concurrently.

        Args:
            timeout: Max seconds to wait for read access. Raises TimeoutError if exceeded.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._writer or self._writer_waiting:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for model read access")
                self._cond.wait(timeout=remaining)
            self._readers += 1

        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self, timeout=60.0):
        """Acquire exclusive write access. Blocks all readers and other writers.

        Args:
            timeout: Max seconds to wait for write access. Raises TimeoutError if exceeded.
        """
        deadline = time.monotonic() + timeout
        t_start = time.monotonic()

        with self._cond:
            self._writer_waiting = True
            try:
                while self._writer or self._readers > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for model write access")
                    self._cond.wait(timeout=remaining)
                self._writer = True
            finally:
                self._writer_waiting = False

        try:
            yield
        finally:
            duration = time.monotonic() - t_start
            with self._cond:
                self._writer = False
                self._last_write_duration = duration
                self._total_write_time += duration
                self._write_count += 1
                self._cond.notify_all()

    @property
    def is_writing(self) -> bool:
        """Check if a writer currently holds the lock."""
        return self._writer

    @property
    def reader_count(self) -> int:
        """Current number of active readers."""
        return self._readers

    def stats(self) -> dict:
        """Return lock usage statistics."""
        return {
            "readers": self._readers,
            "writer_active": self._writer,
            "writer_waiting": self._writer_waiting,
            "last_write_duration": round(self._last_write_duration, 3),
            "total_write_time": round(self._total_write_time, 3),
            "write_count": self._write_count,
        }
