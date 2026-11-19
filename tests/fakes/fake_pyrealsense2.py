"""A small stand-in for ``pyrealsense2``: enough of it for ``so101.hardware.realsense``.

Build one with :func:`make_module` and install it with
``monkeypatch.setitem(sys.modules, "pyrealsense2", module)``. The module keeps
its devices and failure knobs on ``module.world`` (a :class:`World`).

Every class and attribute modelled here is listed in :data:`MODELED`;
``tests/test_realsense.py`` checks those names against the real package
whenever it is importable, so the fake cannot drift into an API that does
not exist.
"""

from __future__ import annotations

import time
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Real pyrealsense2 names this fake models: module attribute -> its attributes.
MODELED: dict[str, tuple[str, ...]] = {
    "camera_info": (
        "name",
        "serial_number",
        "firmware_version",
        "recommended_firmware_version",
        "usb_type_descriptor",
        "product_line",
        "product_id",
        "physical_port",
    ),
    "stream": ("color", "depth"),
    "format": ("rgb8", "z16", "bgr8"),
    "context": ("query_devices",),
    "device_list": ("__iter__", "__len__", "__getitem__"),
    "device": ("supports", "get_info", "query_sensors", "first_depth_sensor", "hardware_reset"),
    "sensor": ("get_stream_profiles",),
    "depth_sensor": ("get_stream_profiles", "get_depth_scale"),
    "stream_profile": (
        "stream_type",
        "format",
        "fps",
        "is_video_stream_profile",
        "as_video_stream_profile",
    ),
    "video_stream_profile": ("width", "height", "fps", "format"),
    "config": ("enable_device", "enable_stream"),
    "pipeline": ("start", "stop", "try_wait_for_frames"),
    "pipeline_profile": ("get_stream", "get_device"),
    "align": ("process",),
    "composite_frame": ("get_color_frame", "get_depth_frame"),
    "video_frame": ("get_data", "__bool__"),
    "depth_frame": ("get_data", "__bool__"),
}
#: Keyword arguments the code passes, as (attribute path, name in its docstring).
MODELED_KEYWORDS = (("pipeline.try_wait_for_frames", "timeout_ms"),)


class _Member:
    """An enum member: compared by identity, like pybind11 enum values."""

    def __init__(self, enum: str, name: str) -> None:
        self.name = name
        self._text = f"{enum}.{name}"

    def __repr__(self) -> str:
        return f"<{self._text}>"

    def __str__(self) -> str:
        return self._text


class camera_info:  # noqa: N801 - mirrors pyrealsense2
    name = _Member("camera_info", "name")
    serial_number = _Member("camera_info", "serial_number")
    firmware_version = _Member("camera_info", "firmware_version")
    recommended_firmware_version = _Member("camera_info", "recommended_firmware_version")
    usb_type_descriptor = _Member("camera_info", "usb_type_descriptor")
    product_line = _Member("camera_info", "product_line")
    product_id = _Member("camera_info", "product_id")
    physical_port = _Member("camera_info", "physical_port")


class stream:  # noqa: N801
    color = _Member("stream", "color")
    depth = _Member("stream", "depth")


class format:  # noqa: A001, N801
    rgb8 = _Member("format", "rgb8")
    z16 = _Member("format", "z16")
    bgr8 = _Member("format", "bgr8")


COLOR_MODES = ((640, 480, 30), (640, 480, 15), (1280, 720, 30), (424, 240, 60))
DEPTH_MODES = ((640, 480, 30), (848, 480, 30), (1280, 720, 30), (480, 270, 60))


class video_stream_profile:  # noqa: N801
    def __init__(self, kind: _Member, fmt: _Member, width: int, height: int, fps: int) -> None:
        self._kind, self._fmt, self._size, self._fps = kind, fmt, (width, height), fps

    def stream_type(self) -> _Member:
        return self._kind

    def format(self) -> _Member:
        return self._fmt

    def fps(self) -> int:
        return self._fps

    def is_video_stream_profile(self) -> bool:
        return True

    def as_video_stream_profile(self) -> video_stream_profile:
        return self

    def width(self) -> int:
        return self._size[0]

    def height(self) -> int:
        return self._size[1]


stream_profile = video_stream_profile


class _MotionProfile(video_stream_profile):
    """A non-video profile (like an IMU stream) that must be skipped."""

    def is_video_stream_profile(self) -> bool:
        return False


class sensor:  # noqa: N801
    def __init__(self, profiles: list[video_stream_profile]) -> None:
        self._profiles = profiles

    def get_stream_profiles(self) -> list[video_stream_profile]:
        return list(self._profiles)


class depth_sensor(sensor):  # noqa: N801
    def __init__(self, profiles: list[video_stream_profile], scale: float) -> None:
        super().__init__(profiles)
        self._scale = scale

    def get_depth_scale(self) -> float:
        return self._scale


@dataclass
class FakeDevice:
    """One attached camera. ``info`` maps camera_info names to values (absent = unsupported)."""

    serial: str = "123622270882"
    name: str = "Intel RealSense D435"
    firmware: str = "5.16.0.1"
    recommended_firmware: str = "5.16.0.1"
    usb_type: str = "3.2"
    color_modes: tuple[tuple[int, int, int], ...] = COLOR_MODES
    depth_modes: tuple[tuple[int, int, int], ...] = DEPTH_MODES
    depth_scale: float = 0.001
    unsupported: tuple[str, ...] = ()
    resets: int = 0
    world: World | None = None

    @property
    def info(self) -> dict[str, str]:
        values = {
            "name": self.name,
            "serial_number": self.serial,
            "firmware_version": self.firmware,
            "recommended_firmware_version": self.recommended_firmware,
            "usb_type_descriptor": self.usb_type,
            "product_line": "D400",
            "product_id": "0B07",
            "physical_port": f"/fake/usb/{self.serial}",
        }
        return {key: value for key, value in values.items() if key not in self.unsupported}


class device:  # noqa: N801
    def __init__(self, fake: FakeDevice) -> None:
        self.fake = fake

    def supports(self, info: _Member) -> bool:
        return info.name in self.fake.info

    def get_info(self, info: _Member) -> str:
        if info.name == "physical_port" and self.fake.world and self.fake.world.info_error:
            raise RuntimeError("device disconnected")
        return self.fake.info[info.name]

    def query_sensors(self) -> list[sensor]:
        depth = [video_stream_profile(stream.depth, format.z16, *m) for m in self.fake.depth_modes]
        depth.append(_MotionProfile(stream.depth, format.z16, 1, 1, 200))
        color = [video_stream_profile(stream.color, format.rgb8, *m) for m in self.fake.color_modes]
        color += [video_stream_profile(stream.color, format.bgr8, 320, 180, 30)]
        return [depth_sensor(depth, self.fake.depth_scale), sensor(color)]

    def first_depth_sensor(self) -> depth_sensor:
        return depth_sensor([], self.fake.depth_scale)

    def hardware_reset(self) -> None:
        self.fake.resets += 1
        if self.fake.world is not None:
            self.fake.world.on_reset(self.fake)


class device_list:  # noqa: N801
    def __init__(self, devices: list[device]) -> None:
        self._devices = devices

    def __iter__(self) -> Iterator[device]:
        return iter(self._devices)

    def __len__(self) -> int:
        return len(self._devices)

    def __getitem__(self, index: int) -> device:
        return self._devices[index]


class video_frame:  # noqa: N801
    def __init__(self, data: np.ndarray | None) -> None:
        self._data = data

    def __bool__(self) -> bool:
        return self._data is not None

    def get_data(self) -> np.ndarray:
        assert self._data is not None
        return self._data


class depth_frame(video_frame):  # noqa: N801
    pass


class composite_frame:  # noqa: N801
    def __init__(self, color: np.ndarray | None, depth: np.ndarray | None, aligned: bool = False):
        self.color, self.depth, self.aligned = color, depth, aligned

    def get_color_frame(self) -> video_frame:
        return video_frame(self.color)

    def get_depth_frame(self) -> depth_frame:
        return depth_frame(self.depth)


class align:  # noqa: N801
    def __init__(self, align_to: _Member) -> None:
        assert align_to is stream.color
        self.calls = 0

    def process(self, frames: composite_frame) -> composite_frame:
        """Resample depth onto the color grid (nearest neighbour), as rs.align does."""
        self.calls += 1
        if frames.color is None or frames.depth is None:
            return composite_frame(frames.color, frames.depth, aligned=True)
        h, w = frames.color.shape[:2]
        dh, dw = frames.depth.shape
        rows = np.arange(h) * dh // h
        cols = np.arange(w) * dw // w
        return composite_frame(frames.color, frames.depth[rows][:, cols], aligned=True)


@dataclass
class config:  # noqa: N801
    serial: str | None = None
    streams: dict[str, tuple[int, int, str, int]] = field(default_factory=dict)

    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, kind: _Member, width: int, height: int, fmt: _Member, fps: int):
        self.streams[kind.name] = (width, height, fmt.name, fps)


class pipeline_profile:  # noqa: N801
    def __init__(self, cfg: config, dev: device) -> None:
        self._cfg, self._device = cfg, dev

    def get_stream(self, kind: _Member) -> video_stream_profile:
        width, height, fmt, fps = self._cfg.streams[kind.name]
        return video_stream_profile(kind, getattr(format, fmt), width, height, fps)

    def get_device(self) -> device:
        return self._device


class context:  # noqa: N801
    def __init__(self) -> None:
        self.world = WORLD.current

    def query_devices(self) -> device_list:
        return device_list([device(fake) for fake in self.world.visible()])


class pipeline:  # noqa: N801
    def __init__(self, ctx: context | None = None) -> None:
        self.world = WORLD.current
        self.config: config | None = None
        self.started = self.stopped = False
        self.frames_served = 0
        self.world.pipelines.append(self)

    def start(self, cfg: config) -> pipeline_profile:
        world = self.world
        world.starts += 1
        if world.start_errors:
            raise RuntimeError(world.start_errors.pop(0))
        visible = world.visible()
        target = next(
            (fake for fake in visible if cfg.serial is None or fake.serial == cfg.serial), None
        )
        if target is None:
            raise RuntimeError("No device connected")
        self.config, self.started = cfg, True
        return pipeline_profile(cfg, device(target))

    def stop(self) -> None:
        if not self.started:
            raise RuntimeError("stop() cannot be called before start()")
        self.stopped = True

    def try_wait_for_frames(self, timeout_ms: int = 5000) -> tuple[bool, composite_frame]:
        world = self.world
        if world.wait_error is not None and self.frames_served >= world.frames_before_error:
            raise world.wait_error
        if world.stall or (
            world.stall_after is not None and self.frames_served >= world.stall_after
        ):
            time.sleep(timeout_ms / 1000)
            return False, composite_frame(None, None)
        time.sleep(world.frame_interval_s)
        assert self.config is not None
        self.frames_served += 1
        return True, world.frameset(self.config, self.frames_served)


@dataclass
class World:
    """Devices and failure knobs shared by every object of one fake module."""

    devices: list[FakeDevice] = field(default_factory=list)
    start_errors: list[str] = field(default_factory=list)
    stall: bool = False  # try_wait_for_frames always times out
    stall_after: int | None = None  # ...after this many framesets
    wait_error: Exception | None = None  # try_wait_for_frames raises this...
    frames_before_error: int = 0  # ...after this many framesets
    drop_depth: bool = False  # framesets arrive without a depth frame
    reset_hides_for: int | None = 0  # query_devices() calls a reset device stays away; None = gone
    info_error: bool = False  # get_info(physical_port) raises
    frame_interval_s: float = 0.002
    raw_depth: np.ndarray | None = None  # z16 units served (resized to the depth mode)
    depth_tracks_index: bool = False  # depth = the frameset's index, color = index % 256
    starts: int = 0
    pipelines: list[pipeline] = field(default_factory=list)
    _hidden: dict[str, int | None] = field(default_factory=dict)

    def add(self, **fields: Any) -> FakeDevice:
        fake = FakeDevice(world=self, **fields)
        self.devices.append(fake)
        return fake

    def on_reset(self, fake: FakeDevice) -> None:
        self._hidden[fake.serial] = self.reset_hides_for

    def visible(self) -> list[FakeDevice]:
        shown = []
        for fake in self.devices:
            remaining = self._hidden.get(fake.serial, 0)
            if remaining is None:
                continue
            if remaining > 0:
                self._hidden[fake.serial] = remaining - 1
                continue
            shown.append(fake)
        return shown

    def frameset(self, cfg: config, index: int) -> composite_frame:
        width, height, _, _ = cfg.streams["color"]
        color = np.full((height, width, 3), index % 256, dtype=np.uint8)
        depth = None
        if "depth" in cfg.streams and not self.drop_depth:
            dw, dh, _, _ = cfg.streams["depth"]
            if self.depth_tracks_index:
                depth = np.full((dh, dw), index, dtype=np.uint16)
            elif self.raw_depth is not None:
                rows = np.arange(dh) * self.raw_depth.shape[0] // dh
                cols = np.arange(dw) * self.raw_depth.shape[1] // dw
                depth = self.raw_depth[rows][:, cols].astype(np.uint16)
            else:
                depth = np.full((dh, dw), 1000, dtype=np.uint16)
        return composite_frame(color, depth)


class _Current:
    current: World


WORLD = _Current()


def make_module(world: World | None = None) -> types.ModuleType:
    """A fresh fake ``pyrealsense2`` module bound to ``world``."""
    WORLD.current = world or World()
    module = types.ModuleType("pyrealsense2")
    for name in MODELED:
        setattr(module, name, globals()[name])
    module.world = WORLD.current  # type: ignore[attr-defined]
    module.__file__ = __file__
    return module
