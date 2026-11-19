"""Read existing relay feeds and mark MCAP episodes without creating motion authority."""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from mcap.exceptions import McapError
from pydantic import Field

from ...config import get_settings
from ..calibration_pose import cached_pose
from ..encoder_model import encoder_model
from ..mcap_review import McapReview
from ..mcap_sessions import RELAY_PORTS, McapEpisodes, recorder_status, relay_status
from ..motor_vision import vision
from ..robot_manager import RobotManager, get_manager
from ..schemas import RequestModel

router = APIRouter(prefix="/api/mcap", tags=["MCAP observation"])
_sessions: dict[str, McapEpisodes] = {}
_lock = threading.Lock()
_playbacks: dict[str, dict] = {}


@router.get("/cameras/{name}/controls")
def camera_controls(name: str) -> dict:
    if name not in RELAY_PORTS:
        raise HTTPException(404, "Unknown shared camera")
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{RELAY_PORTS[name]}/status.json", timeout=2
        ) as response:
            return json.load(response)
    except (OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


class CameraSettingsRequest(RequestModel):
    values: dict[str, int] = Field(min_length=1, max_length=14)


@router.post("/cameras/{name}/controls")
def set_camera_controls(name: str, request: CameraSettingsRequest) -> dict:
    if name not in RELAY_PORTS:
        raise HTTPException(404, "Unknown shared camera")
    manager = get_manager()
    if (
        manager.status()["mode"] != "idle"
        or sessions().snapshot()["active"]
        or recorder_status(manager.settings.data_dir)["recording"]
    ):
        raise HTTPException(409, "Stop teleop and recording before changing camera settings")
    outgoing = urllib.request.Request(
        f"http://127.0.0.1:{RELAY_PORTS[name]}/controls",
        data=json.dumps(request.values).encode(),
        headers={"Content-Type": "application/json", "X-SO101-Control": "camera-settings"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(outgoing, timeout=4) as response:
            return json.load(response)
    except OSError as exc:
        raise HTTPException(422, f"Camera settings were not acknowledged: {exc}") from exc


def review_store() -> McapReview:
    settings = get_settings()
    recording = recorder_status(settings.data_dir)
    raw = Path(recording.get("storage_path") or settings.data_dir / "mcap")
    return _review_store(settings.data_dir.resolve(), raw.resolve())


@lru_cache(maxsize=2)
def _review_store(data_dir: Path, raw_dir: Path) -> McapReview:
    return McapReview(data_dir, raw_dir)


def require_review_idle():
    if get_manager().status()["mode"] != "idle":
        raise HTTPException(409, "Stop motion before loading recorded playback")


@router.get("/reviews")
def review_catalog() -> dict:
    return {"recordings": review_store().catalog()}


@router.get("/evidence")
def observation_evidence() -> dict:
    import json

    directory = get_settings().data_dir
    result = {}
    for key, filename in (
        ("follower_calibration", "calibration_follower.json"),
        ("leader_calibration", "calibration_leader.json"),
        ("rest_pose", "rest_pose.json"),
    ):
        path = directory / filename
        try:
            result[key] = json.loads(path.read_text())
        except (OSError, ValueError):
            result[key] = None
    return result


@router.get("/reviews/{episode_id}/timeline")
def review_timeline(episode_id: str) -> dict:
    require_review_idle()
    try:
        result = review_store().timeline(episode_id)
        return {key: value for key, value in result.items() if key != "snapshots"}
    except (OSError, ValueError, McapError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/reviews/{episode_id}/open-foxglove")
def open_recorded_foxglove(episode_id: str) -> dict:
    require_review_idle()
    store = review_store()
    try:
        store.episode(episode_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    report = store.reports().get(episode_id)
    from pathlib import Path

    executable = shutil.which("foxglove-studio")
    if executable is None:
        raise HTTPException(503, "Foxglove desktop is not installed on this workstation")
    path = Path(report.get("path", "")) if report else None
    root = (store.raw_dir.parent / "episode-reviews").resolve()
    if (
        path is None
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
        or path.suffix != ".mcap"
    ):
        try:
            path = store.export(episode_id)
        except (OSError, ValueError, McapError) as exc:
            raise HTTPException(422, f"Could not prepare recorded review: {exc}") from exc
    require_review_idle()
    try:
        subprocess.Popen(
            [executable, str(path.resolve())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise HTTPException(503, f"Could not launch Foxglove desktop: {exc}") from exc
    return {"launch_requested": True, "episode_id": episode_id, "filename": path.name}


@router.get("/reviews/{episode_id}/snapshot")
def review_topic_snapshot(episode_id: str, topic: str, seconds: float = Query(ge=0)) -> dict:
    require_review_idle()
    try:
        return review_store().snapshot(episode_id, topic, seconds)
    except (OSError, ValueError, McapError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/reviews/{episode_id}/frames/{camera}")
def review_frame(episode_id: str, camera: str, seconds: float = Query(ge=0)) -> Response:
    require_review_idle()
    try:
        image, captured, target = review_store().frame(episode_id, camera, seconds)
    except (OSError, ValueError, McapError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(
        image,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-SO101-Captured-At-Ns": str(captured),
            "X-SO101-Frame-Offset-Ns": str(captured - target),
        },
    )


class PlaybackRequest(RequestModel):
    seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    speed: float = Field(default=1, ge=0.25, le=2, allow_inf_nan=False)


@router.post("/reviews/{episode_id}/playback")
def begin_review_playback(episode_id: str, request: PlaybackRequest) -> dict:
    require_review_idle()
    store = review_store()
    try:
        episode = store.episode(episode_id)
        store.sources(episode)
    except (OSError, ValueError, McapError) as exc:
        raise HTTPException(422, str(exc)) from exc
    duration = (episode["ended_at_ns"] - episode["started_at_ns"]) / 1e9
    if request.seconds > duration:
        raise HTTPException(422, "Playback time is outside the episode")
    token = secrets.token_urlsafe(18)
    state: dict = {
        "episode_id": episode_id,
        "seconds": request.seconds,
        "speed": request.speed,
        "starts_at": time.monotonic() + 0.3,
        "expires_at": time.monotonic() + 3600,
    }
    with _lock:
        expired = [
            key for key, value in _playbacks.items() if value["expires_at"] < time.monotonic()
        ]
        for key in expired:
            _playbacks.pop(key, None)
        if len(_playbacks) >= 16:
            _playbacks.pop(next(iter(_playbacks)))
        _playbacks[token] = state
    return {"playback_id": token, "delay_seconds": max(0, state["starts_at"] - time.monotonic())}


@router.get("/reviews/{episode_id}/streams/{camera}")
def review_stream(episode_id: str, camera: str, playback_id: str) -> StreamingResponse:
    require_review_idle()
    with _lock:
        playback = _playbacks.get(playback_id)
    if (
        not playback
        or playback["episode_id"] != episode_id
        or playback["expires_at"] < time.monotonic()
    ):
        raise HTTPException(404, "Recorded playback session expired")
    store = review_store()
    try:
        episode = store.episode(episode_id)
        store.sources(episode)
        if camera not in {"obsbot", "camera_1", "camera_2", "realsense"}:
            raise ValueError("Invalid recorded camera")
    except (OSError, ValueError, McapError) as exc:
        raise HTTPException(422, str(exc)) from exc

    async def frames():
        iterator = store.frames(episode_id, camera, playback["seconds"])
        from starlette.concurrency import run_in_threadpool

        def take():
            return next(iterator, None)

        try:
            while True:
                record = await run_in_threadpool(take)
                if record is None or get_manager().status()["mode"] != "idle":
                    return
                stamp, jpeg = record
                seconds = (stamp - episode["started_at_ns"]) / 1e9
                deadline = (
                    playback["starts_at"] + (seconds - playback["seconds"]) / playback["speed"]
                )
                await asyncio.sleep(max(0, deadline - time.monotonic()))
                if get_manager().status()["mode"] != "idle":
                    return
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\nX-SO101-Captured-At-Ns: "
                    + str(stamp).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
        finally:
            await run_in_threadpool(iterator.close)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/encoder-model")
def model_snapshot() -> dict:
    telemetry = get_manager().telemetry()
    pose = cached_pose("follower")
    if pose and pose["captured_at"] > telemetry.get("t", 0):
        age = max(0.0, (time.time() - pose["captured_at"]) * 1000)
        result = encoder_model(
            {
                "observation": pose["observation"] if pose["complete"] else {},
                "observation_live": True,
                "sample_age_ms": age,
            }
        )
        result.update(
            source="read_only_pose",
            arm="follower",
            captured_at=pose["captured_at"],
            calibration_sha256=pose["calibration_sha256"],
        )
        return result
    return encoder_model(telemetry)


@router.get("/control-ticks")
def control_ticks(after_seq: int = Query(default=0, ge=0), latest_only: bool = False) -> dict:
    return get_manager().control_trace.snapshot(after_seq, latest_only=latest_only)


def sessions() -> McapEpisodes:
    path = get_settings().data_dir.resolve()
    with _lock:
        key = str(path)
        if key not in _sessions:
            _sessions[key] = McapEpisodes(path)
        return _sessions[key]


def shutdown() -> None:
    vision.stop()
    sessions().interrupt("server_shutdown")


@router.post("/vision/start")
def start_vision() -> dict:
    try:
        vision.start()
    except (RuntimeError, ImportError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return vision.status()


@router.post("/vision/stop")
def stop_vision() -> dict:
    vision.stop()
    return vision.status()


@router.get("/vision/status")
@router.get("/vision/status.json")
def vision_status() -> dict:
    return vision.status()


@router.get("/vision/stream.mjpg")
def vision_stream() -> StreamingResponse:
    async def frames():
        while True:
            jpeg = vision.snapshot()
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/vision/snapshot.jpg")
def vision_snapshot() -> Response:
    jpeg = vision.snapshot()
    if jpeg is None:
        raise HTTPException(503, "No fresh computer-vision frame")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/vision/tags.png")
def vision_tags() -> Response:
    import cv2
    import numpy as np

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sheet = np.full((640, 960), 255, dtype=np.uint8)
    for motor_id, name in enumerate(
        ("pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper"), start=1
    ):
        left = ((motor_id - 1) % 3) * 320 + 60
        top = ((motor_id - 1) // 3) * 320 + 35
        sheet[top : top + 200, left : left + 200] = cv2.aruco.generateImageMarker(
            dictionary, motor_id, 200
        )
        cv2.putText(
            sheet,
            f"{motor_id}: {name}",
            (left - 15, top + 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            0,
            2,
        )
    ok, encoded = cv2.imencode(".png", sheet)
    if not ok:
        raise HTTPException(503, "Could not generate motor tags")
    return Response(encoded.tobytes(), media_type="image/png")


def start_supervisor(manager: RobotManager) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    episodes = sessions()

    def supervise() -> None:
        while not stop.wait(0.5):
            active = episodes.snapshot()["active"]
            if active is None:
                continue
            robot = manager.status()
            recording = recorder_status(manager.settings.data_dir)
            starting = active["status"] == "starting"
            pending_recent = starting and time.time_ns() - active["started_at_ns"] < 8 * 10**9
            if robot["mode"] != "teleop" or (
                not recording["recording"]
                and not (pending_recent and recording.get("service_ready"))
            ):
                episodes.interrupt("teleop_or_recording_stopped")
            elif (
                not starting
                and robot.get("backend") == "real"
                and (
                    recording.get("control_error")
                    or recording.get("control_gap_count", 0) > 0
                    or time.time() - recording.get("control_received_at", 0) > 1
                    or recording.get("activity_id") != active["id"]
                )
            ):
                episodes.interrupt("control_capture_unhealthy")

    thread = threading.Thread(target=supervise, name="so101-mcap-episodes", daemon=True)
    thread.start()
    return stop, thread


class StartEpisode(RequestModel):
    operator_id: str = Field(min_length=1, max_length=128)
    task: str = Field(default="", max_length=1000)


class SaveEpisode(RequestModel):
    operator_id: str = Field(min_length=1, max_length=128)
    episode_id: str = Field(min_length=1, max_length=80)


@router.get("/status")
def status() -> dict:
    manager = get_manager()
    state = sessions()
    recorder = recorder_status(manager.settings.data_dir)
    return {"recorder": recorder, "cameras": relay_status(), **state.snapshot()}


@router.post("/episodes/start")
def start(req: StartEpisode) -> dict:
    manager = get_manager()
    try:
        manager.require_operator_heartbeat(req.operator_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    robot = manager.status()
    recording = recorder_status(manager.settings.data_dir)
    if robot["mode"] != "teleop":
        raise HTTPException(409, "Start leader teleop before marking an MCAP episode")
    if not manager.owns_operation(req.operator_id):
        raise HTTPException(409, "Only the teleop operator can start its episode")
    running = recording["recording"] and recording["telemetry_fresh"]
    ready = recording.get("service_ready") and (
        recording["telemetry_fresh"] or recording.get("api_fresh")
    )
    if not (running or ready):
        raise HTTPException(409, "The local MCAP recorder and telemetry must be live")
    if any(not item.get("live") for item in relay_status().values()):
        raise HTTPException(409, "Restore the shared camera feeds before recording an episode")
    try:
        episode = sessions().start(
            req.operator_id,
            req.task,
            "leader_teleop",
            recording.get("path", ""),
            pending=bool(recording.get("service_ready")),
        )
        if episode["status"] != "starting":
            return episode
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            capture = recorder_status(manager.settings.data_dir)
            if (
                capture["recording"]
                and capture["telemetry_fresh"]
                and capture.get("activity_id") == episode["id"]
                and (
                    robot.get("backend") != "real"
                    or (
                        capture.get("control_ticks_recorded", 0) > 0
                        and not capture.get("control_error")
                        and not capture.get("control_gap_count")
                    )
                )
            ):
                manager.require_operator_heartbeat(req.operator_id)
                if (
                    not manager.owns_operation(req.operator_id)
                    or manager.status()["mode"] != "teleop"
                ):
                    raise ValueError("Teleop ended while the recorder was starting")
                return sessions().confirm_start(req.operator_id, episode["id"], capture["path"])
            time.sleep(0.05)
        raise ValueError("MCAP capture did not start; no episode is ready for demonstration")
    except (ValueError, RuntimeError) as exc:
        if "episode" in locals() and episode.get("status") == "starting":
            sessions().interrupt("capture_start_failed")
        raise HTTPException(409, str(exc)) from exc


@router.post("/episodes/save")
def save(req: SaveEpisode) -> dict:
    try:
        return sessions().finish(req.operator_id, req.episode_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/cameras/{name}/stream.mjpg")
def camera_stream(name: str) -> StreamingResponse:
    if name not in RELAY_PORTS:
        raise HTTPException(404, "Unknown shared camera")
    try:
        response = urllib.request.urlopen(
            f"http://127.0.0.1:{RELAY_PORTS[name]}/stream.mjpg", timeout=3
        )
    except OSError as exc:
        raise HTTPException(503, f"Shared camera unavailable: {exc}") from exc

    def chunks():
        try:
            while data := response.read1(65536):
                yield data
        finally:
            response.close()

    return StreamingResponse(
        chunks(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/cameras/{name}/snapshot.jpg")
def camera_snapshot(name: str) -> Response:
    if name not in RELAY_PORTS:
        raise HTTPException(404, "Unknown shared camera")
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{RELAY_PORTS[name]}/snapshot.jpg", timeout=2
        ) as response:
            jpeg = response.read()
    except OSError as exc:
        raise HTTPException(503, f"Shared camera unavailable: {exc}") from exc
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
