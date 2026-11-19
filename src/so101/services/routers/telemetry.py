"""Live telemetry (WebSocket) + camera video (MJPEG over HTTP)."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from ..imaging import encode
from ..robot_manager import get_manager

router = APIRouter(tags=["telemetry"])


@router.get("/api/cameras/preview/status")
def camera_preview_status() -> dict:
    """Cached camera-only preview state; does not open any hardware."""
    return get_manager().camera_preview_status()


@router.post("/api/cameras/preview/start")
def start_camera_preview() -> dict:
    """Open the configured cameras without connecting either motor bus."""
    manager = get_manager()
    try:
        manager.start_camera_preview()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return manager.camera_preview_status()


@router.post("/api/cameras/preview/stop")
def stop_camera_preview() -> dict:
    """Stop camera-only capture; this performs no robot stop action."""
    manager = get_manager()
    manager.stop_camera_preview()
    return manager.camera_preview_status()


@router.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket) -> None:
    """Stream the latest telemetry snapshot at ~20 Hz."""
    await ws.accept()
    mgr = get_manager()
    try:
        while True:
            await ws.send_json(mgr.telemetry())
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return
    except Exception:
        with suppress(RuntimeError):
            await ws.close()


async def _mjpeg(cam: str, fps: int = 20):
    mgr = get_manager()
    boundary = b"--frame"
    period = 1.0 / fps
    while True:
        frame = mgr.get_frame(cam)
        if frame is None:
            return
        payload, ctype = await asyncio.to_thread(encode, frame)
        yield (
            boundary
            + b"\r\nContent-Type: "
            + ctype.encode()
            + b"\r\nContent-Length: "
            + str(len(payload)).encode()
            + b"\r\n\r\n"
            + payload
            + b"\r\n"
        )
        await asyncio.sleep(period)


@router.get("/api/video")
def video(cam: str = Query(min_length=1, max_length=300)) -> StreamingResponse:
    """MJPEG for a currently live camera, addressed by opaque query value."""
    if get_manager().get_frame(cam) is None:
        raise HTTPException(503, f"camera {cam!r} has no fresh frame")
    return StreamingResponse(
        _mjpeg(cam),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/video/snapshot", response_class=Response)
def video_snapshot(cam: str = Query(min_length=1, max_length=300)) -> Response:
    """One fresh frame, never cached (C10).

    The UI polls this for secondary tiles instead of holding an MJPEG
    connection open for each (frontend F13, docs/audit/v0.1/frontend.md).
    JPEG whenever OpenCV or Pillow is installed; the dependency-free fallback
    is PNG, labelled as such.
    """
    no_store = {"Cache-Control": "no-store"}
    frame = get_manager().get_frame(cam)
    if frame is None:
        raise HTTPException(503, f"camera {cam!r} has no fresh frame", headers=no_store)
    payload, content_type = encode(frame)
    return Response(content=payload, media_type=content_type, headers=no_store)
