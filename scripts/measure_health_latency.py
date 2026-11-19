#!/usr/bin/env python3
"""Measure what servo-health polling adds to a streaming control loop (mock bus only).

    SO101_MOCK=1 python scripts/measure_health_latency.py [--seconds 20] [--fps 30]
        [--poll-hz 2] [--read-ms 0]

Runs the mock teleop loop through RobotManager twice: once with the health
poll held, once polling at ``--poll-hz``. It reports how long the loop thread
waited for the motor bus lock each iteration, the loop period, and how long
each health read held the lock. ``--read-ms`` makes every mock health read
hold the bus that long, to model a real bus (an STS3215 health read is five
sync reads of six servos; the time per read on real hardware is unmeasured).

It starts a teleop loop, so it refuses unless SO101_MOCK=1. Data goes to a
temporary directory that is deleted afterwards. docs/bring-up.md "Servo health
monitor" records the numbers.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

OPERATOR = "latency-probe"


class TimedLock:
    """``threading.Lock`` that records each acquisition's wait and hold, per thread name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._acquired_at: dict[int, float] = {}
        self.waits: dict[str, list[float]] = {}
        self.holds: dict[str, list[float]] = {}
        self.starts: dict[str, list[float]] = {}

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        started = time.perf_counter()
        ok = self._lock.acquire(blocking, timeout)
        if ok:
            now = time.perf_counter()
            name = threading.current_thread().name
            self.waits.setdefault(name, []).append(now - started)
            self.starts.setdefault(name, []).append(now)
            self._acquired_at[threading.get_ident()] = now
        return ok

    def release(self) -> None:
        started = self._acquired_at.pop(threading.get_ident(), None)
        if started is not None:
            name = threading.current_thread().name
            self.holds.setdefault(name, []).append(time.perf_counter() - started)
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


def _ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    p99 = ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))]
    return (
        f"p50 {statistics.median(ordered) * 1000:.3f} ms, p99 {p99 * 1000:.3f} ms, "
        f"max {ordered[-1] * 1000:.3f} ms (n={len(ordered)})"
    )


def _slow(read: Callable[[], Any], delay_s: float) -> Callable[[], Any]:
    def read_health() -> Any:
        time.sleep(delay_s)  # holds the bus lock like a real register read
        return read()

    return read_health


def run(*, polling: bool, seconds: float, fps: int, read_ms: float) -> dict[str, Any]:
    from so101.config import get_settings
    from so101.services.robot_manager import get_manager, reset_manager

    get_settings.cache_clear()
    reset_manager()
    manager = get_manager()
    lock = TimedLock()
    manager._io_lock = lock  # type: ignore[assignment]
    if not polling:
        manager._health_tick = lambda: None  # type: ignore[method-assign]
    for device in (manager.follower(), manager.leader()):
        bus: Any = device.bus
        if read_ms > 0:
            bus.read_health = _slow(bus.read_health, read_ms / 1000.0)
    manager.operator_heartbeat(OPERATOR)
    manager.start_teleop(fps, operator_id=OPERATOR)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        manager.operator_heartbeat(OPERATOR)
        time.sleep(0.2)
    snapshot = manager.health_snapshot()
    manager.soft_stop(reason="latency probe end")
    reset_manager()
    loop = "so101-teleop"
    starts = lock.starts.get(loop, [])
    periods = [later - earlier for earlier, later in zip(starts, starts[1:], strict=False)]
    return {
        "loop_waits": lock.waits.get(loop, []),
        "loop_periods": periods,
        "health_holds": lock.holds.get("so101-health", []),
        "polls": {arm: view["polls"] for arm, view in snapshot["arms"].items()},
        "skipped": {arm: view["skipped_busy"] for arm, view in snapshot["arms"].items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--poll-hz", type=float, default=2.0)
    parser.add_argument("--read-ms", type=float, default=0.0)
    args = parser.parse_args()
    if os.environ.get("SO101_MOCK") != "1":
        print("refused: set SO101_MOCK=1 (this starts a teleop loop)", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="so101-latency-") as root:
        os.environ["SO101_DATA_DIR"] = os.path.join(root, "data")
        os.environ["SO101_CAMERAS"] = ""
        os.environ["SO101_HEALTH_POLL_HZ"] = str(args.poll_hz)
        results = {
            name: run(polling=polling, seconds=args.seconds, fps=args.fps, read_ms=args.read_ms)
            for name, polling in (("health held", False), ("health polling", True))
        }
    print(
        f"mock teleop at {args.fps} fps for {args.seconds:g} s; health at {args.poll_hz:g} Hz, "
        f"each mock read holding the bus {args.read_ms:g} ms extra"
    )
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  loop wait for the bus lock: {_ms(result['loop_waits'])}")
        print(f"  loop period:                {_ms(result['loop_periods'])}")
        print(f"  health read holding the bus: {_ms(result['health_holds'])}")
        print(f"  polls {result['polls']}, skipped (bus busy) {result['skipped']}")
    held, polled = results["health held"], results["health polling"]
    added = max(polled["loop_waits"], default=0.0) - max(held["loop_waits"], default=0.0)
    print(f"\nadded worst-case loop wait: {added * 1000:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
