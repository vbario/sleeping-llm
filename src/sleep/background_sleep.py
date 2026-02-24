"""Background sleep manager — runs sleep in a background thread.

Allows chat to continue during sleep. Uses a read-write lock to coordinate:
  - Chat inference holds read locks (can proceed concurrently)
  - Sleep training runs out-of-process (no lock needed)
  - Weight fuse/reload holds write lock (brief exclusive period)

Sleep phases and their lock requirements:
  Phase                     Lock        Duration
  ─────────────────────────────────────────────────
  Curation/data prep        none        seconds
  LoRA training (subprocess) none       10-100s
  Pre/post evaluation       read        seconds
  MEMIT scale operations    write       <1s
  Fuse + reload             write       1-2s
"""

import threading
import time
from enum import Enum
from typing import Callable, Optional


class SleepState(Enum):
    """Current state of the background sleep system."""
    IDLE = "idle"
    SLEEPING = "sleeping"         # Full sleep in progress
    NAPPING = "napping"           # Nap in progress
    MAINTAINING = "maintaining"      # Maintenance phase (refresh/prune)
    ERROR = "error"


class BackgroundSleepManager:
    """Manages non-blocking sleep execution in a background thread.

    Usage:
        manager = BackgroundSleepManager(model_lock)
        manager.start_sleep(sleep_fn, callback=on_done)
        # Chat continues normally while sleep runs in background
        # During fuse/reload, chat briefly blocks on read_lock
    """

    def __init__(self, model_lock):
        self.model_lock = model_lock
        self._state = SleepState.IDLE
        self._thread: Optional[threading.Thread] = None
        self._progress: list = []
        self._result: Optional[dict] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()  # Protects _state, _progress, _result, _error

    @property
    def state(self) -> SleepState:
        with self._lock:
            return self._state

    @property
    def is_sleeping(self) -> bool:
        with self._lock:
            return self._state in (SleepState.SLEEPING, SleepState.NAPPING, SleepState.CONSOLIDATING)

    @property
    def progress(self) -> list:
        with self._lock:
            return list(self._progress)

    @property
    def result(self) -> Optional[dict]:
        with self._lock:
            return self._result

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def start_sleep(self, sleep_fn: Callable, callback: Optional[Callable] = None,
                    sleep_type: str = "sleep"):
        """Start a sleep cycle in a background thread.

        Args:
            sleep_fn: Callable that executes the sleep pipeline.
                      Should be a generator yielding progress dicts.
            callback: Optional callback invoked when sleep completes.
                      Receives the result dict.
            sleep_type: "sleep" or "nap"

        Returns:
            True if sleep started, False if already sleeping.
        """
        with self._lock:
            if self._state in (SleepState.SLEEPING, SleepState.NAPPING, SleepState.CONSOLIDATING):
                return False
            self._state = SleepState.SLEEPING if sleep_type == "sleep" else SleepState.NAPPING
            self._progress = []
            self._result = None
            self._error = None

        self._thread = threading.Thread(
            target=self._run_sleep,
            args=(sleep_fn, callback),
            daemon=True,
            name=f"background-{sleep_type}",
        )
        self._thread.start()
        return True

    def _run_sleep(self, sleep_fn: Callable, callback: Optional[Callable]):
        """Execute sleep in background thread, collecting progress."""
        try:
            for progress_dict in sleep_fn():
                with self._lock:
                    self._progress.append(progress_dict)

                    # Track maintenance phase
                    label = progress_dict.get("label", "")
                    status = progress_dict.get("status", "")
                    if "Maintenance" in label and status == "running":
                        self._state = SleepState.MAINTAINING
                    elif self._state == SleepState.MAINTAINING and status == "done":
                        self._state = SleepState.SLEEPING

            with self._lock:
                # Extract result from final progress entry
                if self._progress:
                    last = self._progress[-1]
                    self._result = {
                        "status": "completed",
                        "facts_refreshed": last.get("facts_refreshed", 0),
                        "facts_pruned": last.get("facts_pruned", 0),
                        "steps": len(self._progress),
                    }
                else:
                    self._result = {"status": "completed", "steps": 0}
                self._state = SleepState.IDLE

        except Exception as e:
            with self._lock:
                self._state = SleepState.ERROR
                self._error = str(e)
                self._result = {"status": "error", "error": str(e)}

        # Invoke callback
        if callback:
            try:
                callback(self._result)
            except Exception:
                pass  # Don't let callback errors propagate

    def wait(self, timeout: float = None):
        """Block until sleep completes. Returns result dict."""
        if self._thread:
            self._thread.join(timeout=timeout)
        return self._result

    def to_dict(self) -> dict:
        """Return current state as a serializable dict."""
        with self._lock:
            return {
                "state": self._state.value,
                "progress_steps": len(self._progress),
                "last_progress": self._progress[-1] if self._progress else None,
                "result": self._result,
                "error": self._error,
            }
