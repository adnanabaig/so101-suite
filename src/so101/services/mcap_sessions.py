"""Passive shared-feed status and durable episode annotations for MCAP logs."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..atomic_io import atomic_write_json

RELAY_PORTS = {"obsbot": 8765, "camera_1": 8768, "camera_2": 8769, "realsense": 8770}


def relay_status() -> dict[str, dict[str, Any]]:
    cameras = {}
    for name, port in RELAY_PORTS.items():
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/status.json", timeout=0.3
            ) as response:
                state = json.load(response)
            state["live"] = state.get("live") is True and 0 <= state.get("age_seconds", 999) < 2
            cameras[name] = state
        except (OSError, ValueError, TypeError) as exc:
            cameras[name] = {"live": False, "error": str(exc)}
    return cameras


def recorder_status(data_dir: Path) -> dict[str, Any]:
    try:
        state = json.loads((data_dir / "mcap/status.json").read_text())
        state["fresh"] = time.time() - state["updated_at"] < 5
        state["recording"] = bool(
            state.get("active") and state["fresh"] and state.get("bytes", 0) > 0
        )
        state["telemetry_fresh"] = time.time() - state.get("telemetry_received_at", 0) < 5
        state["service_ready"] = bool(state.get("ready") and state["fresh"])
        state["api_fresh"] = time.time() - state.get("api_received_at", 0) < 5
        return state
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"recording": False, "telemetry_fresh": False, "error": str(exc)}


class McapEpisodes:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "mcap/episodes.json"
        self._lock = threading.RLock()
        self.state: dict[str, Any] = {"schema_version": 1, "active": None, "episodes": []}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("active"):
                self._finish("server_restarted")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.state)

    def _finish(self, reason: str) -> dict[str, Any]:
        episode = dict(self.state["active"])
        episode.update(
            ended_at_ns=time.time_ns(),
            end_reason=reason,
            status="saved" if reason == "operator_saved" else "interrupted",
        )
        replacement = {**self.state, "active": None, "episodes": [*self.state["episodes"], episode]}
        atomic_write_json(self.path, replacement)
        self.state = replacement
        return self._public(episode)

    @staticmethod
    def _public(episode: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in episode.items() if key != "operator_id"}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._public(self.state["active"]) if self.state["active"] else None,
                "episodes": [self._public(item) for item in self.state["episodes"][-20:]],
            }

    def start(
        self, operator_id: str, task: str, source: str, segment: str, *, pending: bool = False
    ) -> dict[str, Any]:
        with self._lock:
            if self.state["active"] is not None:
                raise ValueError("An MCAP episode is already active")
            episode = {
                "id": uuid.uuid4().hex,
                "operator_id": operator_id,
                "task": task.strip(),
                "source": source,
                "started_at_ns": time.time_ns(),
                "status": "starting" if pending else "recording",
                "starting_segment": segment,
            }
            replacement = {**self.state, "active": episode}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, replacement)
            self.state = replacement
            return self._public(episode)

    def confirm_start(self, operator_id: str, episode_id: str, segment: str) -> dict[str, Any]:
        with self._lock:
            active = self.state["active"]
            if active is None or active["id"] != episode_id or active["operator_id"] != operator_id:
                raise ValueError("The pending episode was interrupted before capture became ready")
            confirmed = {
                **active,
                "status": "recording",
                "started_at_ns": time.time_ns(),
                "starting_segment": segment,
            }
            replacement = {**self.state, "active": confirmed}
            atomic_write_json(self.path, replacement)
            self.state = replacement
            return self._public(confirmed)

    def finish(self, operator_id: str, episode_id: str) -> dict[str, Any]:
        with self._lock:
            active = self.state["active"]
            if active is None:
                prior = next(
                    (item for item in self.state["episodes"] if item["id"] == episode_id), None
                )
                if prior and prior["operator_id"] == operator_id:
                    return self._public(prior)
                raise ValueError("No matching MCAP episode is active")
            if active["operator_id"] != operator_id or active["id"] != episode_id:
                raise ValueError("Only the episode's operator can save it")
            if active["status"] == "starting":
                raise ValueError("Capture is still starting; wait for the recorder acknowledgement")
            return self._finish("operator_saved")

    def interrupt(self, reason: str) -> None:
        with self._lock:
            if self.state["active"] is not None:
                self._finish(reason)
