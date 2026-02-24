"""Tests for Item 5: Non-blocking / incremental sleep.

Verifies:
  - ModelLock read/write semantics
  - Multiple concurrent readers allowed
  - Writer gets exclusive access
  - Writer priority (no starvation)
  - BackgroundSleepManager state machine
  - Timeout behavior
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.concurrency.model_lock import ModelLock
from src.sleep.background_sleep import BackgroundSleepManager, SleepState


# ─── ModelLock basic tests ───

def test_read_lock_basic():
    """A single read lock can be acquired and released."""
    lock = ModelLock()
    with lock.read_lock():
        assert lock.reader_count == 1
    assert lock.reader_count == 0


def test_multiple_readers():
    """Multiple readers can hold locks concurrently."""
    lock = ModelLock()
    results = []
    barrier = threading.Barrier(3)

    def reader(idx):
        with lock.read_lock():
            barrier.wait(timeout=5)
            results.append(lock.reader_count)
            time.sleep(0.05)

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # All 3 readers should have seen at least 2 concurrent readers
    assert any(r >= 2 for r in results), f"Expected concurrent readers, got: {results}"


def test_write_lock_basic():
    """A single write lock can be acquired and released."""
    lock = ModelLock()
    with lock.write_lock():
        assert lock.is_writing
    assert not lock.is_writing


def test_write_excludes_read():
    """A writer blocks new readers."""
    lock = ModelLock()
    events = []

    def writer():
        with lock.write_lock():
            events.append("write_start")
            time.sleep(0.2)
            events.append("write_end")

    def reader():
        time.sleep(0.05)  # Ensure writer starts first
        events.append("read_waiting")
        with lock.read_lock():
            events.append("read_acquired")

    wt = threading.Thread(target=writer)
    rt = threading.Thread(target=reader)
    wt.start()
    rt.start()
    wt.join(timeout=5)
    rt.join(timeout=5)

    # Reader should acquire AFTER writer finishes
    write_end_idx = events.index("write_end")
    read_acquired_idx = events.index("read_acquired")
    assert read_acquired_idx > write_end_idx, f"Reader should wait for writer. Events: {events}"


def test_read_excludes_write():
    """Active readers block a writer until they finish."""
    lock = ModelLock()
    events = []

    def reader():
        with lock.read_lock():
            events.append("read_start")
            time.sleep(0.2)
            events.append("read_end")

    def writer():
        time.sleep(0.05)  # Ensure reader starts first
        events.append("write_waiting")
        with lock.write_lock():
            events.append("write_acquired")

    rt = threading.Thread(target=reader)
    wt = threading.Thread(target=writer)
    rt.start()
    wt.start()
    rt.join(timeout=5)
    wt.join(timeout=5)

    # Writer should acquire AFTER reader finishes
    read_end_idx = events.index("read_end")
    write_acquired_idx = events.index("write_acquired")
    assert write_acquired_idx > read_end_idx, f"Writer should wait for reader. Events: {events}"


def test_write_lock_stats():
    """Stats track write operations."""
    lock = ModelLock()
    with lock.write_lock():
        time.sleep(0.01)
    stats = lock.stats()
    assert stats["write_count"] == 1
    assert stats["total_write_time"] > 0
    assert not stats["writer_active"]


def test_read_lock_timeout():
    """Read lock raises TimeoutError if writer holds too long."""
    lock = ModelLock()
    timed_out = threading.Event()

    def hold_write():
        with lock.write_lock():
            time.sleep(2)

    def try_read():
        try:
            with lock.read_lock(timeout=0.1):
                pass
        except TimeoutError:
            timed_out.set()

    wt = threading.Thread(target=hold_write)
    wt.start()
    time.sleep(0.05)  # Let writer acquire
    rt = threading.Thread(target=try_read)
    rt.start()
    rt.join(timeout=3)
    wt.join(timeout=5)

    assert timed_out.is_set(), "Reader should have timed out"


def test_write_lock_timeout():
    """Write lock raises TimeoutError if readers hold too long."""
    lock = ModelLock()
    timed_out = threading.Event()

    def hold_read():
        with lock.read_lock():
            time.sleep(2)

    def try_write():
        try:
            with lock.write_lock(timeout=0.1):
                pass
        except TimeoutError:
            timed_out.set()

    rt = threading.Thread(target=hold_read)
    rt.start()
    time.sleep(0.05)
    wt = threading.Thread(target=try_write)
    wt.start()
    wt.join(timeout=3)
    rt.join(timeout=5)

    assert timed_out.is_set(), "Writer should have timed out"


# ─── BackgroundSleepManager tests ───

def test_background_sleep_idle():
    """Manager starts in idle state."""
    lock = ModelLock()
    mgr = BackgroundSleepManager(lock)
    assert mgr.state == SleepState.IDLE
    assert not mgr.is_sleeping


def test_background_sleep_basic():
    """Manager runs a sleep generator and collects progress."""
    lock = ModelLock()
    mgr = BackgroundSleepManager(lock)

    def fake_sleep():
        yield {"step": 1, "total": 2, "label": "Curating", "status": "running"}
        time.sleep(0.05)
        yield {"step": 1, "total": 2, "label": "Curating", "status": "done"}
        yield {"step": 2, "total": 2, "label": "Training", "status": "done",
               "facts_refreshed": 3}

    started = mgr.start_sleep(fake_sleep)
    assert started
    assert mgr.is_sleeping

    result = mgr.wait(timeout=5)
    assert result is not None
    assert result["status"] == "completed"
    assert mgr.state == SleepState.IDLE
    assert len(mgr.progress) == 3


def test_background_sleep_prevents_double_start():
    """Can't start a second sleep while one is running."""
    lock = ModelLock()
    mgr = BackgroundSleepManager(lock)

    def slow_sleep():
        yield {"step": 1, "total": 1, "label": "Sleeping", "status": "running"}
        time.sleep(1)
        yield {"step": 1, "total": 1, "label": "Sleeping", "status": "done"}

    first = mgr.start_sleep(slow_sleep)
    assert first
    second = mgr.start_sleep(slow_sleep)
    assert not second, "Should not allow double start"

    mgr.wait(timeout=5)


def test_background_sleep_callback():
    """Callback is invoked when sleep completes."""
    lock = ModelLock()
    mgr = BackgroundSleepManager(lock)
    callback_result = {}

    def fake_sleep():
        yield {"step": 1, "total": 1, "label": "Done", "status": "done", "facts_refreshed": 5}

    def on_done(result):
        callback_result.update(result)

    mgr.start_sleep(fake_sleep, callback=on_done)
    mgr.wait(timeout=5)

    assert "status" in callback_result
    assert callback_result["status"] == "completed"


def test_background_sleep_error_handling():
    """Manager captures errors and sets ERROR state."""
    lock = ModelLock()
    mgr = BackgroundSleepManager(lock)

    def failing_sleep():
        yield {"step": 1, "total": 1, "label": "Starting", "status": "running"}
        raise RuntimeError("simulated failure")

    mgr.start_sleep(failing_sleep)
    mgr.wait(timeout=5)

    assert mgr.state == SleepState.ERROR
    assert mgr.error is not None
    assert "simulated failure" in mgr.error


def test_background_sleep_to_dict():
    """to_dict() returns serializable state."""
    lock = ModelLock()
    mgr = BackgroundSleepManager(lock)
    d = mgr.to_dict()
    assert d["state"] == "idle"
    assert d["progress_steps"] == 0


def test_concurrent_read_during_sleep():
    """Readers can proceed while sleep is running (no write lock phase)."""
    lock = ModelLock()
    mgr = BackgroundSleepManager(lock)
    read_ok = threading.Event()

    def fake_sleep():
        yield {"step": 1, "total": 2, "label": "Training", "status": "running"}
        time.sleep(0.3)
        yield {"step": 2, "total": 2, "label": "Done", "status": "done"}

    mgr.start_sleep(fake_sleep)
    time.sleep(0.05)  # Let sleep start

    # Reader should be able to acquire read lock during sleep
    def reader():
        try:
            with lock.read_lock(timeout=1):
                read_ok.set()
        except TimeoutError:
            pass

    rt = threading.Thread(target=reader)
    rt.start()
    rt.join(timeout=3)
    mgr.wait(timeout=5)

    assert read_ok.is_set(), "Reader should succeed during sleep (no write lock held)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
