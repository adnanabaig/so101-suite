"""Offscreen camera frames, rendered on one dedicated thread.

An OpenGL context belongs to the thread that made it current, so a single
worker thread owns every context. Callers pass a
:class:`~so101.simulation.world.WorldSnapshot` (copied under their own lock in
microseconds); the worker poses a private ``MjData`` from it and renders. The
physics never waits for a frame and a frame never sees a half-finished step.
Each scene has one context, which renders every frame size (see
:class:`_SceneRenderer`), and the worker keeps at most ``MAX_RENDERERS``.

Fail-soft: if no GL context can be made (a headless box without EGL/OSMesa,
``MUJOCO_GL=disable``, a sandbox that blocks the window server),
:attr:`FrameRenderer.available` is ``False``, :attr:`FrameRenderer.error` says
why, and :meth:`FrameRenderer.render` raises :class:`RenderUnavailable`.
Everything else in the simulator keeps working; APIs report
``render_available: false``. A probe that hangs instead of failing counts as
failed after ``PROBE_TIMEOUT_S``, and no call waits past its own timeout (see
:class:`FrameRenderer`).

Frames are reproducible to within a couple of grey levels: the GL driver
renders a few edge pixels one or two levels differently depending on what the
context drew before (seen with and without multisampling). States, actions
and environment state are bit-exact; image bytes are not guaranteed to be.

On macOS MuJoCo uses CGL (no window); on Linux set ``MUJOCO_GL=egl`` (GPU) or
``MUJOCO_GL=osmesa`` (CPU) for headless rendering.

:func:`encode_frame` turns a frame into PNG bytes (pure Python) or JPEG bytes
(when Pillow or OpenCV is installed; otherwise it falls back to PNG).

Needs the ``[sim]`` extra.
"""

from __future__ import annotations

import copy
import queue
import struct
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .world import OFFSCREEN_SIZE, SceneSpec, WorldSnapshot, compiled_scene

MIN_SIZE = 16
MAX_RENDERERS = 2  # GL contexts kept alive on the worker, one per scene, any frame size
MAX_SCENE_STATES = 4  # per-scene (model copy, MjData) pairs kept on the worker, ~15 MB each
MAX_QUEUED = 8  # requests waiting on the worker at once; more are refused, not queued
DEFAULT_TIMEOUT_S = 10.0
PROBE_TIMEOUT_S = 10.0  # a GL probe still running this long after the start counts as hung
PROBE_RETRY_S = 60.0  # then calls answer at once, and one per period waits for it again


class RenderUnavailable(RuntimeError):
    """No OpenGL context is available in this process."""


@dataclass(frozen=True)
class _Request:
    snapshot: WorldSnapshot
    camera: str
    width: int
    height: int
    deadline: float  # time.perf_counter() after which the caller has given up


def check_size(width: int, height: int) -> tuple[int, int]:
    """Validate a frame size against the offscreen buffer (1280 x 960)."""

    max_width, max_height = OFFSCREEN_SIZE
    width, height = int(width), int(height)
    if not (MIN_SIZE <= width <= max_width and MIN_SIZE <= height <= max_height):
        bounds = f"{MIN_SIZE}x{MIN_SIZE}..{max_width}x{max_height}"
        raise ValueError(f"frame size {width}x{height} is outside {bounds}")
    return width, height


class _SceneRenderer:
    """One scene's GL context, rendering frames of any size up to the offscreen buffer.

    A ``mujoco.Renderer`` fixes its frame size at construction, and each one
    holds its own context: the uploaded meshes and a multisampled buffer of the
    model's full offscreen size, about 65 MB however small its frames. Here the
    frame size only sets the viewport, as in dm_control, so frames of every
    size share one context. The calls are those of ``mujoco.Renderer``.
    """

    def __init__(self, model: Any) -> None:
        self._gl = mujoco.GLContext(model.vis.global_.offwidth, model.vis.global_.offheight)
        self._gl.make_current()
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._context)
        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def render(self, model: Any, data: Any, camera_id: int, width: int, height: int) -> np.ndarray:
        """An HxWx3 uint8 frame from fixed camera ``camera_id`` of a posed ``data``."""

        self._camera.fixedcamid = camera_id
        mujoco.mjv_updateScene(
            model,
            data,
            self._option,
            None,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )
        self._gl.make_current()
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(rect, self._scene, self._context)
        frame = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(frame, None, rect, self._context)
        return np.ascontiguousarray(frame[::-1])  # GL rows run bottom to top

    def close(self) -> None:
        self._gl.free()  # in the order mujoco.Renderer frees them
        self._context.free()


class FrameRenderer:
    """Owns the render thread. Thread-safe; ``start``/``close`` are idempotent.

    No call waits past its own timeout. Callers wait for the GL probe at most
    until it counts as hung, ``PROBE_TIMEOUT_S`` after the worker started; from
    then on they answer at once with the remembered reason, except that one call
    per ``PROBE_RETRY_S`` waits for the probe again. A probe that returns late is
    used as soon as it does. At most ``MAX_QUEUED`` requests wait at once, and a
    request whose caller has given up is skipped, never rendered.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        # One queue per worker, unbounded so ``close`` can always post its
        # sentinel; ``_slots`` bounds the requests.
        self._queue: queue.Queue[tuple[_Request, Future[np.ndarray]] | None] = queue.Queue()
        self._slots = threading.BoundedSemaphore(MAX_QUEUED)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._available: bool | None = None if enabled else False
        self._error: str | None = None if enabled else "rendering disabled"
        self._probe_deadline = 0.0  # the pending probe counts as hung after this
        self._retry_at: float | None = None  # once it does: when a caller may wait again
        self._frames = 0
        self._render_s = 0.0

    # -- lifecycle -------------------------------------------------------------
    def start(self, timeout: float = DEFAULT_TIMEOUT_S) -> bool:
        """Start the worker (once) and wait at most ``timeout`` for its GL probe.

        Returns availability. Once the probe counts as hung this returns
        ``False`` at once, except for the one call per ``PROBE_RETRY_S`` that
        waits again (at most ``timeout`` and ``PROBE_TIMEOUT_S``).
        """

        now = time.perf_counter()
        with self._lock:
            self._spawn(now)
            if self._thread is None or self._ready.is_set():
                return bool(self._available)
            retry_at = self._hung_until(now)
            if retry_at is not None:
                if now < retry_at:
                    return False
                self._retry_at = now + PROBE_RETRY_S  # this call retries; the rest answer at once
                until = now + min(timeout, PROBE_TIMEOUT_S)
            else:
                until = min(now + timeout, self._probe_deadline)
        if self._ready.wait(max(0.0, until - now)):
            return bool(self._available)
        with self._lock:
            if self._ready.is_set():
                return bool(self._available)
            # Judge by the planned end as well: the wait's clock can round early.
            if (
                self._thread is not None
                and self._hung_until(max(time.perf_counter(), until)) is None
            ):
                self._error = f"render thread did not start within {timeout:.3g} s"
            return False

    def close(self, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        with self._lock:
            thread, self._thread = self._thread, None
            work = self._queue
        if thread is not None:
            work.put(None)
            thread.join(timeout)
            if thread.is_alive():  # stuck in GL: fail what it will never take
                self._fail_pending(work)
                work.put(None)  # in case it ever returns
            if self._available:
                self._available = None  # unknown until the next start probes again
                self._error = None

    @property
    def available(self) -> bool:
        """True once the worker has made a GL context (starts it and waits like ``start``)."""

        return self.start()

    @property
    def error(self) -> str | None:
        return self._error

    def status(self) -> tuple[bool | None, str | None]:
        """``(available, error)`` without waiting; starts the worker if needed.

        ``available`` is ``None`` while the GL probe runs and does not count as hung yet.
        """

        now = time.perf_counter()
        with self._lock:
            self._spawn(now)
            return self._report(now)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            available, error = self._report(time.perf_counter())
        mean_ms = 1000.0 * self._render_s / self._frames if self._frames else None
        return {
            "available": available,
            "error": error,
            "frames": self._frames,
            "mean_render_ms": None if mean_ms is None else round(mean_ms, 2),
            "fps": None if not mean_ms else round(1000.0 / mean_ms, 1),
        }

    def _spawn(self, now: float) -> None:
        """Start a worker unless one runs or rendering is disabled. Call under the lock."""

        if not self._enabled or self._thread is not None:
            return
        self._ready.clear()
        self._available, self._error = None, None
        self._probe_deadline, self._retry_at = now + PROBE_TIMEOUT_S, None
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, args=(self._queue,), name="so101-sim-render", daemon=True
        )
        self._thread.start()

    def _hung_until(self, now: float) -> float | None:
        """When a caller may wait again for the pending probe, if it counts as hung.

        Latches the hang once the probe's time is up. Call under the lock, with
        a worker whose probe has not returned.
        """

        if self._retry_at is None and now >= self._probe_deadline:
            self._retry_at = now + PROBE_RETRY_S
            self._error = (
                f"render thread did not start within {PROBE_TIMEOUT_S:g} s: "
                "the GL probe has not returned"
            )
        return self._retry_at

    def _report(self, now: float) -> tuple[bool | None, str | None]:
        """``(available, error)`` as of ``now``, without waiting. Call under the lock."""

        if self._thread is None or self._ready.is_set():
            return self._available, self._error
        if self._hung_until(now) is None:
            return None, None  # the probe is still running
        return False, self._error

    # -- rendering ------------------------------------------------------------------
    def submit(
        self,
        snapshot: WorldSnapshot,
        camera: str,
        width: int,
        height: int,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> Future[np.ndarray]:
        """Queue a frame; the future resolves to HxWx3 uint8 (or raises).

        ``timeout`` bounds the wait for the GL probe and the request's time in
        the queue: a request still queued after it is skipped, and its future
        raises ``TimeoutError``. Raises :class:`RenderUnavailable` when
        rendering is unavailable or ``MAX_QUEUED`` requests are already waiting.
        """

        width, height = check_size(width, height)
        deadline = time.perf_counter() + timeout
        if not self._slots.acquire(blocking=False):
            raise RenderUnavailable(
                f"the render queue is full ({MAX_QUEUED} requests are already waiting)"
            )
        try:
            if not self.start(max(0.0, deadline - time.perf_counter())):
                raise RenderUnavailable(self._error or "rendering is unavailable")
            future: Future[np.ndarray] = Future()
            with self._lock:  # never into the queue of a worker that is shutting down
                if self._thread is None:
                    raise RenderUnavailable("renderer closed")
                self._queue.put((_Request(snapshot, camera, width, height, deadline), future))
        except BaseException:
            self._slots.release()
            raise
        return future

    def render(
        self,
        snapshot: WorldSnapshot,
        camera: str,
        width: int,
        height: int,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> np.ndarray:
        """Render ``camera`` for ``snapshot``; blocks at most ``timeout`` in all.

        Raises :class:`RenderUnavailable` as :meth:`submit` does, or
        ``TimeoutError`` when the frame is not ready in time.
        """

        deadline = time.perf_counter() + timeout
        future = self.submit(snapshot, camera, width, height, timeout=timeout)
        try:
            return future.result(max(0.0, deadline - time.perf_counter()))
        except TimeoutError:
            future.cancel()  # the worker skips it unless it has already begun
            raise

    # -- worker ---------------------------------------------------------------------------
    def _run(self, work: queue.Queue[tuple[_Request, Future[np.ndarray]] | None]) -> None:
        renderers: OrderedDict[SceneSpec, _SceneRenderer] = OrderedDict()
        states: OrderedDict[SceneSpec, tuple[Any, Any]] = OrderedDict()
        available = False
        error: str | None = "render thread stopped"  # kept if the probe raises a BaseException
        try:
            self._probe()
            available, error = True, None
        except Exception as exc:  # any GL failure means "no rendering here"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                # A worker closed while it probed leaves the state to the next one.
                current = self._thread is threading.current_thread()
                if current:
                    self._available, self._error = available, error
                    self._ready.set()
        while current:
            item = work.get()
            if item is None:
                break
            request, future = item
            self._slots.release()
            if not future.set_running_or_notify_cancel():
                continue  # the caller timed out and cancelled it
            if time.perf_counter() > request.deadline:
                future.set_exception(TimeoutError("the frame request expired in the render queue"))
                continue
            if not self._available:
                future.set_exception(RenderUnavailable(self._error or "rendering is unavailable"))
                continue
            started = time.perf_counter()
            try:
                frame = self._render(request, renderers, states)
            except Exception as exc:
                future.set_exception(exc)
            else:
                self._frames += 1
                self._render_s += time.perf_counter() - started
                future.set_result(frame)
        for renderer in renderers.values():
            renderer.close()
        self._fail_pending(work)  # anything that raced the shutdown

    def _fail_pending(self, work: queue.Queue[tuple[_Request, Future[np.ndarray]] | None]) -> None:
        while True:
            try:
                item = work.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                self._slots.release()
                if item[1].set_running_or_notify_cancel():
                    item[1].set_exception(RenderUnavailable("renderer closed"))

    @staticmethod
    def _probe() -> None:
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><light pos='0 0 1'/><camera pos='0 0 1'/>"
            "<geom type='sphere' size='0.1'/></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = _SceneRenderer(model)
        try:
            renderer.render(model, data, 0, MIN_SIZE, MIN_SIZE)
        finally:
            renderer.close()

    def _render(
        self,
        request: _Request,
        renderers: OrderedDict[SceneSpec, _SceneRenderer],
        states: OrderedDict[SceneSpec, tuple[Any, Any]],
    ) -> np.ndarray:
        snapshot = request.snapshot
        compiled = compiled_scene(snapshot.scene)
        if request.camera not in compiled.index.cameras:
            raise ValueError(f"unknown camera {request.camera!r}")
        # Both caches are LRU-bounded: each randomized reset can be a new
        # SceneSpec, and a long-running server must not keep every one alive.
        # The frame size is not a key, so no request size can add a context.
        if snapshot.scene not in states:
            model = copy.copy(compiled.model)
            states[snapshot.scene] = (model, mujoco.MjData(model))
            while len(states) > MAX_SCENE_STATES:
                evicted, _ = states.popitem(last=False)
                stale = renderers.pop(evicted, None)
                if stale is not None:
                    stale.close()
        states.move_to_end(snapshot.scene)
        model, data = states[snapshot.scene]
        renderer = renderers.get(snapshot.scene)
        if renderer is None:
            renderer = _SceneRenderer(model)
            renderers[snapshot.scene] = renderer
            if len(renderers) > MAX_RENDERERS:
                renderers.popitem(last=False)[1].close()
        renderers.move_to_end(snapshot.scene)
        data.qpos[:] = snapshot.qpos
        data.qvel[:] = snapshot.qvel
        data.mocap_pos[:] = snapshot.mocap_pos
        data.mocap_quat[:] = snapshot.mocap_quat
        data.time = snapshot.time_s
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
        camera_id = compiled.index.cameras.index(request.camera)  # names are in id order
        return renderer.render(model, data, camera_id, request.width, request.height)


_shared: FrameRenderer | None = None
_shared_lock = threading.Lock()


def shared_renderer() -> FrameRenderer:
    """The process-wide render thread (started lazily on first use)."""

    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = FrameRenderer()
        return _shared


def frame_source(renderer: FrameRenderer | None = None) -> Callable[..., np.ndarray]:
    """A :data:`~so101.simulation.rollout.FrameSource` that renders a world snapshot."""

    def frames(world: Any, camera: str, width: int, height: int) -> np.ndarray:
        return (renderer or shared_renderer()).render(world.snapshot(), camera, width, height)

    return frames


# -- encoding ----------------------------------------------------------------------------
def encode_png(frame: np.ndarray, *, level: int = 6) -> bytes:
    """PNG bytes for an HxWx3 (or HxW) uint8 frame. Pure Python (zlib)."""

    image = np.ascontiguousarray(frame, dtype=np.uint8)
    if image.ndim == 2:
        color = 0
        image = image[:, :, None]
    elif image.ndim == 3 and image.shape[2] == 3:
        color = 2
    else:
        raise ValueError(f"expected HxWx3 or HxW uint8, got {frame.shape}")
    height, width = image.shape[:2]
    rows = np.zeros((height, 1 + width * image.shape[2]), dtype=np.uint8)
    rows[:, 1:] = image.reshape(height, -1)  # filter byte 0 (none) per scanline

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), level))
        + chunk(b"IEND", b"")
    )


def _jpeg_encoder() -> Callable[[np.ndarray, int], bytes] | None:
    try:
        from PIL import Image  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        pass
    else:
        import io

        def with_pillow(frame: np.ndarray, quality: int) -> bytes:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
            return buffer.getvalue()

        return with_pillow
    try:
        import cv2  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None

    def with_opencv(frame: np.ndarray, quality: int) -> bytes:
        ok, data = cv2.imencode(".jpg", frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("OpenCV could not encode the frame")
        return bytes(data)

    return with_opencv


def jpeg_supported() -> bool:
    return _jpeg_encoder() is not None


def encode_frame(frame: np.ndarray, fmt: str = "png", *, quality: int = 85) -> tuple[bytes, str]:
    """``(bytes, media_type)``. JPEG needs Pillow or OpenCV; without them it is PNG."""

    fmt = fmt.lower()
    if fmt not in ("png", "jpeg", "jpg"):
        raise ValueError(f"unsupported image format {fmt!r}; use png or jpeg")
    if fmt != "png":
        encoder = _jpeg_encoder()
        if encoder is not None:
            return encoder(np.ascontiguousarray(frame, dtype=np.uint8), quality), "image/jpeg"
    return encode_png(frame), "image/png"


__all__ = [
    "FrameRenderer",
    "RenderUnavailable",
    "check_size",
    "encode_frame",
    "encode_png",
    "frame_source",
    "jpeg_supported",
    "shared_renderer",
]
