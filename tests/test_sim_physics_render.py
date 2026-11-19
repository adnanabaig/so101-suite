"""Offscreen rendering: the render thread, fail-soft probing, concurrency and PNG."""

from __future__ import annotations

import struct
import threading
import time
import zlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from so101.simulation import render as render_module  # noqa: E402
from so101.simulation.render import (  # noqa: E402
    FrameRenderer,
    RenderUnavailable,
    check_size,
    encode_frame,
    encode_png,
    frame_source,
)
from so101.simulation.rollout import EpisodeRunner  # noqa: E402
from so101.simulation.tasks import make_task  # noqa: E402

CAMERAS = ("front", "wrist_cam")
SLACK_S = 0.5  # thread scheduling on a busy machine; the old code blocked the full 10 s


def assert_same_frame(a: np.ndarray, b: np.ndarray) -> None:
    """Equal up to the GL driver's edge-pixel jitter.

    Re-rendering a snapshot changes a few edge pixels, usually by 1-2 levels;
    with other GPU work running it reached 9 levels on 0.5 % of the values of a
    64x48 frame. Any real change of pose or camera moves more than 1 % of the
    values by more than 16 levels (``assert_different_frame``).
    """

    diff = np.abs(a.astype(int) - b.astype(int))
    big, changed = (diff > 16).mean(), (diff > 0).mean()
    assert big < 0.002 and changed < 0.05, (diff.max(), big, changed)


def assert_different_frame(a: np.ndarray, b: np.ndarray) -> None:
    assert (np.abs(a.astype(int) - b.astype(int)) > 16).mean() > 0.01


def decode_png(data: bytes) -> np.ndarray:
    """Minimal decoder for what encode_png writes (8-bit, filter 0)."""

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks = 8, {}
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        tag = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(tag + payload)
        chunks[tag] = payload
        offset += 12 + length
    width, height, depth, color = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert depth == 8
    channels = {0: 1, 2: 3}[color]
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), dtype=np.uint8)
    rows = rows.reshape(height, 1 + width * channels)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, channels).squeeze()


@pytest.fixture(scope="module")
def renderer():
    frames = FrameRenderer()
    if not frames.available:
        pytest.skip(f"no OpenGL here: {frames.error}")
    yield frames
    frames.close()


@pytest.fixture
def world():
    task = make_task("pick_lift")
    world = task.make_world()
    task.reset(world, seed=0)
    return world


def test_png_round_trip_and_the_jpeg_fallback():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(7, 5, 3), dtype=np.uint8)
    assert np.array_equal(decode_png(encode_png(frame)), frame)
    gray = frame[:, :, 0]
    assert np.array_equal(decode_png(encode_png(gray)), gray)
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 4), dtype=np.uint8))
    data, media = encode_frame(frame, "jpeg")
    assert media in ("image/jpeg", "image/png")
    if media == "image/png":  # neither Pillow nor OpenCV here: PNG instead, never an error
        assert np.array_equal(decode_png(data), frame)
    with pytest.raises(ValueError, match="unsupported image format"):
        encode_frame(frame, "gif")


def test_frame_size_is_checked():
    assert check_size(320, 240) == (320, 240)
    for size in ((8, 240), (320, 4), (1281, 240), (320, 961)):
        with pytest.raises(ValueError, match="outside"):
            check_size(*size)


def test_without_gl_rendering_fails_soft(monkeypatch, world):
    disabled = FrameRenderer(enabled=False)
    assert not disabled.available and disabled.error == "rendering disabled"
    with pytest.raises(RenderUnavailable):
        disabled.render(world.snapshot(), "front", 64, 48)

    def no_gl() -> None:
        raise RuntimeError("gladLoadGL error")

    monkeypatch.setattr(FrameRenderer, "_probe", staticmethod(no_gl))
    broken = FrameRenderer()
    try:
        assert broken.start() is False
        assert not broken.available and "gladLoadGL" in (broken.error or "")
        assert broken.stats()["available"] is False
        with pytest.raises(RenderUnavailable, match="gladLoadGL"):
            broken.render(world.snapshot(), "front", 64, 48)
    finally:
        broken.close()


@pytest.fixture
def hung_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """The GL probe hangs (a stuck driver or window server) until the event is set."""

    gate = threading.Event()

    def hang() -> None:
        gate.wait(30)

    monkeypatch.setattr(FrameRenderer, "_probe", staticmethod(hang))
    yield gate
    gate.set()


def test_a_hung_probe_never_holds_a_caller_past_its_timeout(hung_probe, world):
    frames = FrameRenderer()
    snapshot = world.snapshot()
    outcomes: list[tuple[float, BaseException | None]] = []

    def request() -> None:
        started = time.perf_counter()
        try:
            frames.render(snapshot, "front", 64, 48, timeout=1.0)
        except BaseException as exc:
            outcomes.append((time.perf_counter() - started, exc))
        else:  # pragma: no cover - reported below
            outcomes.append((time.perf_counter() - started, None))

    threads = [threading.Thread(target=request) for _ in range(render_module.MAX_QUEUED + 4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert len(outcomes) == len(threads)
        for elapsed, exc in outcomes:
            assert isinstance(exc, RenderUnavailable), exc
            assert elapsed < 1.0 + SLACK_S, elapsed
        messages = [str(exc) for _, exc in outcomes]
        # The probe still has most of its 10 s, so each waiter gave up at its own
        # timeout. Only MAX_QUEUED of them waited at all; the rest were refused at once.
        waited = [m for m in messages if "render thread did not start within" in m]
        assert waited and not any("has not returned" in m for m in waited)
        assert any("render queue is full" in m for m in messages)
        assert len(waited) <= render_module.MAX_QUEUED
        assert frames.status() == (None, None)  # still probing, not yet counted as hung
    finally:
        hung_probe.set()
        frames.close()


def test_a_probe_timeout_is_remembered_with_one_retry_per_cool_down(
    hung_probe, world, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(render_module, "PROBE_TIMEOUT_S", 0.4)
    monkeypatch.setattr(render_module, "PROBE_RETRY_S", 1.0)
    frames = FrameRenderer()
    snapshot = world.snapshot()
    try:
        assert frames.status() == (None, None)  # starts the probe and never waits for it
        started = time.perf_counter()
        assert frames.start(timeout=5.0) is False  # waits until the probe counts as hung
        assert time.perf_counter() - started < 0.4 + SLACK_S
        reason = frames.error
        assert reason == "render thread did not start within 0.4 s: the GL probe has not returned"

        # From now on every call answers at once with the remembered reason.
        started = time.perf_counter()
        assert frames.available is False
        assert frames.start(timeout=5.0) is False
        with pytest.raises(RenderUnavailable, match="the GL probe has not returned"):
            frames.render(snapshot, "front", 64, 48, timeout=5.0)
        assert frames.status() == (False, reason)
        assert frames.stats()["available"] is False
        assert time.perf_counter() - started < SLACK_S

        # After the cool-down one call waits again, bounded by its own timeout ...
        time.sleep(1.1)
        started = time.perf_counter()
        assert frames.start(timeout=0.3) is False
        assert 0.25 < time.perf_counter() - started < 0.3 + SLACK_S
        # ... and the calls after it answer at once until the next cool-down ends.
        started = time.perf_counter()
        assert frames.available is False
        assert time.perf_counter() - started < SLACK_S

        # A probe that returns late is used as soon as it does.
        hung_probe.set()
        for _ in range(100):
            if frames.status() == (True, None):
                break
            time.sleep(0.05)
        assert frames.status() == (True, None) and frames.available is True
    finally:
        hung_probe.set()
        frames.close()


def test_the_queue_is_bounded_and_abandoned_requests_are_skipped(
    monkeypatch: pytest.MonkeyPatch, world
):
    entered, release = threading.Event(), threading.Event()
    rendered: list[tuple[int, int]] = []

    def slow_render(self: FrameRenderer, request: Any, renderers: Any, states: Any) -> np.ndarray:
        entered.set()
        release.wait(10)
        rendered.append((request.width, request.height))
        return np.zeros((request.height, request.width, 3), dtype=np.uint8)

    monkeypatch.setattr(FrameRenderer, "_probe", staticmethod(lambda: None))
    monkeypatch.setattr(FrameRenderer, "_render", slow_render)
    frames = FrameRenderer()
    snapshot = world.snapshot()
    try:
        busy = frames.submit(snapshot, "front", 16, 16)
        assert entered.wait(5)  # the worker is stuck on it
        # Requests that give up after 0.2 s fill the queue ...
        expiring = [
            frames.submit(snapshot, "front", 32, 24, timeout=0.2)
            for _ in range(render_module.MAX_QUEUED)
        ]
        # ... and the next one is refused at once instead of joining the backlog.
        started = time.perf_counter()
        with pytest.raises(RenderUnavailable, match="render queue is full"):
            frames.render(snapshot, "front", 16, 16, timeout=5.0)
        assert time.perf_counter() - started < SLACK_S
        time.sleep(0.3)  # every queued request has now expired
        release.set()
        assert busy.result(5).shape == (16, 16, 3)
        for future in expiring:
            with pytest.raises(TimeoutError, match="expired in the render queue"):
                future.result(5)

        # render() cancels its request when it times out, and the worker skips it.
        release.clear()
        entered.clear()
        stuck = frames.submit(snapshot, "front", 16, 16)
        assert entered.wait(5)
        started = time.perf_counter()
        with pytest.raises(TimeoutError):
            frames.render(snapshot, "front", 48, 36, timeout=0.2)
        assert time.perf_counter() - started < 0.2 + SLACK_S
        release.set()
        assert stuck.result(5).shape == (16, 16, 3)

        # A fresh request is not stuck behind the abandoned ones, which were never drawn.
        assert frames.render(snapshot, "front", 20, 20, timeout=5.0).shape == (20, 20, 3)
        assert rendered == [(16, 16), (16, 16), (20, 20)]
    finally:
        release.set()
        frames.close()


def test_every_frame_size_shares_one_gl_context_per_scene(monkeypatch: pytest.MonkeyPatch, world):
    live: list[Any] = []

    class Counted(render_module._SceneRenderer):
        def __init__(self, model: Any) -> None:
            super().__init__(model)
            live.append(self)

        def close(self) -> None:
            live.remove(self)
            super().close()

    monkeypatch.setattr(render_module, "_SceneRenderer", Counted)
    frames = FrameRenderer()
    try:
        if not frames.available:
            pytest.skip(f"no OpenGL here: {frames.error}")
        assert live == []  # the probe closed its own
        snapshot = world.snapshot()
        first = frames.render(snapshot, "front", 64, 48)
        # More sizes than the old per-size cache held (8), on both cameras: one context
        # (each used to cost about 65 MB, and cycling sizes rebuilt them without end).
        sizes = [(16 + 37 * i, 16 + 23 * i) for i in range(12)]
        for camera in CAMERAS:
            for width, height in sizes:
                assert frames.render(snapshot, camera, width, height).shape == (height, width, 3)
        assert len(live) == 1
        assert_same_frame(frames.render(snapshot, "front", 64, 48), first)

        # Each scene has its own context, and at most MAX_RENDERERS stay alive.
        for name in ("reach", "pick_place"):
            task = make_task(name)
            other = task.make_world()
            task.reset(other, 0)
            frames.render(other.snapshot(), "front", 32, 24)
        assert len(live) == render_module.MAX_RENDERERS == 2
    finally:
        frames.close()
    assert live == []


def test_frames_follow_the_snapshot_not_the_live_world(renderer, world):
    snapshot = world.snapshot()
    qpos = world.data.qpos.copy()
    first = {camera: renderer.render(snapshot, camera, 96, 72) for camera in CAMERAS}
    for frame in first.values():
        assert frame.shape == (72, 96, 3) and frame.dtype == np.uint8
        assert frame.std() > 5.0  # a scene, not a blank buffer
    assert np.array_equal(world.data.qpos, qpos)  # rendering never touches the world
    assert_same_frame(renderer.render(snapshot, "front", 96, 72), first["front"])

    world.set_targets({"shoulder_pan": 40.0, "gripper": 5.0})
    world.step(world.steps_for(1.0))
    moved = renderer.render(world.snapshot(), "front", 96, 72)
    assert_different_frame(moved, first["front"])
    # The old snapshot still renders the old pose.
    assert_same_frame(renderer.render(snapshot, "front", 96, 72), first["front"])
    with pytest.raises(ValueError, match="unknown camera"):
        renderer.render(snapshot, "top", 96, 72)


def test_concurrent_render_requests_while_physics_runs(renderer, world):
    lock = threading.Lock()
    stop = threading.Event()
    errors: list[BaseException] = []
    rendered: list[tuple[object, str, np.ndarray]] = []

    def physics() -> None:
        try:
            sign = 1.0
            while not stop.is_set():
                with lock:
                    world.set_targets({"shoulder_pan": 30.0 * sign})
                    world.step(8)
                sign = -sign if world.time_s % 0.5 < 0.04 else sign
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def client(index: int) -> None:
        try:
            for frame_index in range(6):
                with lock:
                    snapshot = world.snapshot()
                camera = CAMERAS[(index + frame_index) % 2]
                frame = renderer.render(snapshot, camera, 64, 48)
                assert frame.shape == (48, 64, 3)
                rendered.append((snapshot, camera, frame))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    stepper = threading.Thread(target=physics)
    clients = [threading.Thread(target=client, args=(i,)) for i in range(4)]
    stepper.start()
    for thread in clients:
        thread.start()
    for thread in clients:
        thread.join(30)
    stop.set()
    stepper.join(5)
    assert not errors, errors
    assert len(rendered) == 24
    # Each frame is exactly what that snapshot renders on its own.
    for snapshot, camera, frame in rendered[::5]:
        assert_same_frame(renderer.render(snapshot, camera, 64, 48), frame)  # type: ignore[arg-type]
    stats = renderer.stats()
    assert stats["frames"] >= 24 and stats["fps"] > 0


def test_runner_observations_carry_rendered_images(renderer):
    task = make_task("reach")
    runner = EpisodeRunner(
        task, cameras=CAMERAS, image_size=(64, 48), frames=frame_source(renderer)
    )
    runner.reset(0)
    observation = runner.observe()
    for camera in CAMERAS:
        image = observation[f"observation.images.{camera}"]
        assert image.shape == (48, 64, 3) and image.dtype == np.uint8


def test_close_is_idempotent_and_restartable(world):
    frames = FrameRenderer()
    if not frames.available:
        pytest.skip(f"no OpenGL here: {frames.error}")
    frames.close()
    frames.close()
    assert frames.render(world.snapshot(), "front", 32, 24).shape == (24, 32, 3)
    frames.close()
