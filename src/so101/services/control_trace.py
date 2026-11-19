"""Bounded copies of authoritative control ticks for passive recorders."""

from __future__ import annotations

import copy
import threading
import uuid
from collections import deque
from typing import Any


class ControlTrace:
    def __init__(self, capacity: int = 9000):
        self.stream_id = uuid.uuid4().hex
        self._ticks: deque[dict] = deque(maxlen=capacity)
        self._sequence = 0
        self._lock = threading.Lock()

    def append(
        self,
        step,
        *,
        backend: str,
        started: float,
        completed: float,
        fps: int,
        profile: dict[str, Any] | None = None,
        timing: dict[str, Any] | None = None,
    ):
        with self._lock:
            self._sequence += 1
            self._ticks.append(
                {
                    "schema_version": 1,
                    "stream_id": self.stream_id,
                    "seq": self._sequence,
                    "timestamp_ns": round(step.t * 10**9),
                    "backend": backend,
                    "step_started_monotonic": started,
                    "step_completed_monotonic": completed,
                    "requested_fps": fps,
                    "control_profile": copy.deepcopy(profile),
                    "timing": copy.deepcopy(timing),
                    "observation": dict(step.observation),
                    "action": dict(step.action),
                    "desired_action": dict(step.desired_action),
                    "joints": {
                        key.removesuffix(".pos"): {
                            "measured": value,
                            "commanded": step.action.get(key),
                            "desired": step.desired_action.get(key),
                            "unit": "percent" if key == "gripper.pos" else "degrees",
                        }
                        for key, value in step.observation.items()
                        if key.endswith(".pos")
                    },
                }
            )

    def snapshot(self, after_seq: int = 0, *, latest_only: bool = False) -> dict:
        with self._lock:
            first = self._ticks[0]["seq"] if self._ticks else self._sequence + 1
            result: dict[str, Any] = {
                "stream_id": self.stream_id,
                "last_seq": self._sequence,
                "first_available_seq": first,
                "missing_before_buffer": max(0, first - after_seq - 1),
            }
            selected = (
                [] if latest_only else [tick for tick in self._ticks if tick["seq"] > after_seq]
            )
        result["ticks"] = copy.deepcopy(selected)
        return result
