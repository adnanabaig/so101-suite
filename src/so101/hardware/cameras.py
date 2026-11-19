"""Camera capture (OpenCV and Intel RealSense) with a synthetic backend for mock mode only.

Entries come from SO101_CAMERAS; :mod:`so101.hardware.camera_spec` has the
grammar. ``CameraManager.read()`` returns one frame per stream key: every
color stream (HxWx3 uint8 RGB) in configured order, then every depth stream
(HxWx1 uint16 millimetres, 0 = no data) under ``<entry>#depth``.

Under SO101_BACKEND=sim the manager opens the simulated cameras
(SO101_SIM_CAMERAS, rendered by MuJoCo) instead, and never opens or probes a
SO101_CAMERAS entry: real entries are ignored with one log line.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import SIM_CAMERAS, get_settings
from ..logging import get_logger
from .backend import current_backend
from .camera_spec import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    CameraSpec,
    base_key,
    camera_feature_name,
    is_depth_key,
    parse_camera_entries,
    parse_camera_entry,
    redact_credentials,
    stream_feature_names,
    stream_keys,
    strip_credentials,
)

__all__ = [
    "CameraDevice",
    "CameraManager",
    "CameraUnavailableError",
    "DepthCameraDevice",
    "camera_feature_name",
    "configured_streams",
    "is_depth_key",
    "stream_feature_names",
]

log = get_logger(__name__)

#: Synthetic depth in mock mode: a tabletop from far (top row) to near (bottom row).
MOCK_DEPTH_FAR_MM = 1500
MOCK_DEPTH_NEAR_MM = 300
#: The moving box stands this far proud of the table.
MOCK_OBJECT_HEIGHT_MM = 150
#: Where the synthetic depth has dropouts (0 = no data), as fractions of (y, x).
MOCK_DEPTH_HOLES = ((0.15, 0.2), (0.4, 0.75), (0.8, 0.5))


def configured_streams(settings: Any = None) -> list[str]:
    """The stream keys ``CameraManager()`` reads under the current backend.

    SO101_CAMERAS streams (credentials stripped) on the real and mock
    backends; the simulated camera names under SO101_BACKEND=sim.
    """
    settings = settings or get_settings()
    if current_backend(settings) == "sim":
        return list(settings.sim_camera_names)
    return list(settings.camera_streams)


_IGNORED_UNDER_SIM: set[tuple[str, ...]] = set()


def _log_ignored_under_sim(entries: list[str]) -> None:
    """One log line per process and set of ignored entries, not one per manager."""
    shown = tuple(strip_credentials(entry) for entry in entries)
    if shown and shown not in _IGNORED_UNDER_SIM:
        _IGNORED_UNDER_SIM.add(shown)
        log.warning(
            "SO101_BACKEND=sim: ignoring real camera entries %s; the simulated cameras "
            "(SO101_SIM_CAMERAS) stand in for them and no real camera is opened",
            ", ".join(shown),
        )


class CameraUnavailableError(RuntimeError):
    """A configured real camera could not be imported, opened, or read.

    ``detail`` says what went wrong and ``remediation`` (possibly empty) what
    the operator can do about it. ``str()`` carries both, so logs and API
    errors show the fix as well.
    """

    def __init__(self, detail: str, *, remediation: str = "") -> None:
        super().__init__(f"{detail.rstrip('.')}. {remediation}" if remediation else detail)
        self.detail = detail
        self.remediation = remediation


@runtime_checkable
class CameraDevice(Protocol):
    """The camera object ``CameraManager`` drives; ``SimCamera`` satisfies it as is.

    ``read()`` returns an HxWx3 uint8 RGB frame. An optional ``info()`` returning
    a dict describes the device (``CameraManager.info`` reports it).
    """

    def read(self) -> np.ndarray: ...

    def release(self) -> None: ...


@runtime_checkable
class DepthCameraDevice(CameraDevice, Protocol):
    """Optional capability: color and depth from the same capture.

    ``read_rgbd()`` returns ``(HxWx3 uint8 RGB, HxWx1 uint16 millimetres)`` with
    depth aligned to the color frame and 0 meaning no data.
    """

    def read_rgbd(self) -> tuple[np.ndarray, np.ndarray]: ...


class _MockCamera:
    """Animated gradient + moving box so the video pane is obviously live."""

    def __init__(self, cam_id: str, width: int = 640, height: int = 480) -> None:
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self._t0 = time.monotonic()

    def read(self) -> np.ndarray:
        return self._render(time.monotonic() - self._t0)[0]

    def _render(self, t: float) -> tuple[np.ndarray, tuple[slice, slice]]:
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        xs = np.linspace(0, 255, w, dtype=np.uint8)
        frame[:, :, 0] = xs[None, :]
        frame[:, :, 1] = np.uint8((np.sin(t) * 0.5 + 0.5) * 255)
        cx = int((np.sin(t * 0.8) * 0.4 + 0.5) * w)
        cy = int((np.cos(t * 0.6) * 0.4 + 0.5) * h)
        box = (slice(max(0, cy - 20), cy + 20), slice(max(0, cx - 20), cx + 20))
        frame[box] = (0, 0, 255)
        return frame, box

    def info(self) -> dict[str, Any]:
        return {
            "synthetic": True,
            "streams": {"color": {"width": self.width, "height": self.height, "format": "rgb8"}},
        }

    def release(self) -> None:  # noqa: D401 - parity with cv2.VideoCapture
        pass


class _MockDepthCamera(_MockCamera):
    """Mock RealSense with ``+depth``: the color frame plus synthetic aligned depth."""

    def read_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        frame, box = self._render(time.monotonic() - self._t0)
        return frame, self._depth(box)

    def _depth(self, box: tuple[slice, slice]) -> np.ndarray:
        h, w = self.height, self.width
        rows = np.linspace(MOCK_DEPTH_FAR_MM, MOCK_DEPTH_NEAR_MM, h)
        depth = np.repeat(rows[:, None], w, axis=1)
        depth[box] = np.maximum(depth[box] - MOCK_OBJECT_HEIGHT_MM, MOCK_DEPTH_NEAR_MM)
        hole_h, hole_w = max(2, h // 24), max(2, w // 32)
        for fy, fx in MOCK_DEPTH_HOLES:
            y, x = int(fy * h), int(fx * w)
            depth[y : y + hole_h, x : x + hole_w] = 0
        return np.rint(depth).astype(np.uint16)[:, :, None]

    def info(self) -> dict[str, Any]:
        facts = super().info()
        facts["depth_scale"] = 0.001
        facts["streams"]["depth"] = {
            "width": self.width,
            "height": self.height,
            "format": "z16",
            "unit": "mm",
            "aligned_to": "color",
        }
        return facts


class _CvCamera:
    def __init__(self, cam_id: str) -> None:
        import cv2

        # Messages name the camera without its credentials; only the capture
        # sees the real source.
        self.cam_id = redact_credentials(cam_id)
        src: int | str = int(cam_id) if cam_id.isdigit() else cam_id
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.cam_id!r}")
        settings = get_settings()
        try:
            if settings.camera_fourcc is not None:
                self._cap.set(
                    cv2.CAP_PROP_FOURCC,
                    sum(ord(letter) << (8 * i) for i, letter in enumerate(settings.camera_fourcc)),
                )
            for requested, prop in (
                (settings.camera_width, "CAP_PROP_FRAME_WIDTH"),
                (settings.camera_height, "CAP_PROP_FRAME_HEIGHT"),
                (settings.camera_fps, "CAP_PROP_FPS"),
            ):
                if requested is not None:
                    self._cap.set(getattr(cv2, prop), requested)
        except BaseException:
            self._cap.release()
            raise

    def info(self) -> dict[str, Any]:
        import cv2

        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return {
            "synthetic": False,
            "streams": {
                "color": {
                    "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": self._cap.get(cv2.CAP_PROP_FPS),
                    "fourcc": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)),
                    "format": "rgb8",
                }
            },
        }

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"failed to read camera {self.cam_id!r}")
        # cv2 is BGR; convert to RGB for downstream consumers.
        return frame[:, :, ::-1].copy()

    def release(self) -> None:
        self._cap.release()


def _unavailable(entry: str, exc: Exception) -> CameraUnavailableError:
    detail = redact_credentials(getattr(exc, "detail", None) or str(exc))
    return CameraUnavailableError(
        f"real camera {entry!r} is unavailable: {detail}; no mock stream was substituted",
        remediation=getattr(exc, "remediation", ""),
    )


class CameraManager:
    """Open configured cameras; real mode never substitutes synthetic frames."""

    def __init__(self, cam_ids: list[str] | None = None) -> None:
        settings = get_settings()
        self.force_mock = settings.mock
        self.backend = current_backend(settings)
        if self.backend == "sim":
            # The simulator's cameras only. Real entries, whether configured
            # in SO101_CAMERAS or passed in, are ignored (and logged once).
            requested = list(cam_ids) if cam_ids is not None else settings.sim_camera_names
            self.cam_ids = [entry.strip() for entry in requested if entry.strip() in SIM_CAMERAS]
            real = [entry for entry in requested if entry.strip() not in SIM_CAMERAS]
            _log_ignored_under_sim(real if cam_ids is not None else settings.camera_ids)
        else:
            self.cam_ids = cam_ids if cam_ids is not None else settings.camera_ids
        # Raises ValueError naming a malformed entry (Settings validated its own).
        self.specs: list[CameraSpec] = parse_camera_entries(self.cam_ids)
        self._specs = {spec.entry: spec for spec in self.specs}
        self._cams: dict[str, CameraDevice] = {}

    @property
    def streams(self) -> list[str]:
        """The keys ``read()`` returns, in order: every color stream, then every depth stream."""
        return stream_keys(self.specs)

    @property
    def feature_names(self) -> dict[str, str]:
        """``observation.images.<name>`` per stream key, for recorders and policy runners.

        Simulated cameras keep their own names (``front``, ``wrist_cam``), as
        ``so101 sim demos`` records them; everything else follows
        :func:`stream_feature_names`.
        """
        if self.backend == "sim" and not self.force_mock:
            return {key: key for key in self.streams}
        return stream_feature_names(self.streams)

    def open(self) -> None:
        if self._cams:
            return
        try:
            for spec in self.specs:
                camera = self._open_one(spec.entry)
                self._cams[spec.entry] = camera
                if spec.depth and not isinstance(camera, DepthCameraDevice):
                    raise CameraUnavailableError(
                        f"camera {spec.entry!r} asks for +depth but its backend has no depth stream"
                    )
        except Exception:
            self.close()
            raise

    def _open_one(self, cid: str) -> CameraDevice:
        spec = self._specs.get(cid) or parse_camera_entry(cid)
        if self.force_mock:
            if spec.kind == "realsense":
                width, height = spec.width or DEFAULT_WIDTH, spec.height or DEFAULT_HEIGHT
                mock = _MockDepthCamera if spec.depth else _MockCamera
                return mock(cid, width, height)
            return _MockCamera(cid)
        if self.backend == "sim":
            if cid in SIM_CAMERAS:
                try:
                    from ..simulation.devices import make_sim_camera  # the [sim] extra
                except Exception as exc:  # a missing or a broken MuJoCo
                    raise CameraUnavailableError(
                        f"sim camera {cid!r} needs MuJoCo ({type(exc).__name__}: {exc})",
                        remediation='Install the sim extra: pip install "so101[sim]".',
                    ) from exc
                return make_sim_camera(cid)
            # Anything else must never open a real device behind the
            # simulator's back, so this runs before cv2 or pyrealsense2 is
            # imported.
            raise CameraUnavailableError(
                f"SO101_BACKEND=sim: {strip_credentials(cid)!r} is not a simulated camera "
                f"({', '.join(SIM_CAMERAS)}), so no real camera was opened",
                remediation="Use SO101_BACKEND=real for real cameras.",
            )
        if spec.kind == "realsense":
            try:
                from .realsense import open_realsense

                return open_realsense(spec)
            except Exception as exc:
                raise _unavailable(cid, exc) from exc
        try:
            import cv2  # noqa: F401

            return _CvCamera(spec.source)
        except Exception as exc:
            raise _unavailable(cid, exc) from exc

    def read(self) -> dict[str, np.ndarray]:
        if not self._cams:
            self.open()
        color: dict[str, np.ndarray] = {}
        depth: dict[str, np.ndarray] = {}
        for entry, camera in self._cams.items():
            depth_key = self._specs[entry].depth_key
            if depth_key is not None and isinstance(camera, DepthCameraDevice):
                # One capture, so color and depth show the same instant.
                color[entry], depth[depth_key] = camera.read_rgbd()
            else:
                color[entry] = camera.read()
        return {**color, **depth}

    def info(self, key: str) -> dict[str, Any]:
        """What is known about the camera behind a stream key (also before ``open()``)."""
        spec = self._specs.get(base_key(key))
        if spec is None:
            raise KeyError(key)
        camera = self._cams.get(spec.entry)
        describe = getattr(camera, "info", None)
        return {
            "entry": spec.entry,
            "kind": spec.kind,
            "name": spec.name,
            "streams": list(spec.keys),
            "backend": self.backend,
            "open": camera is not None,
            "device": dict(describe()) if callable(describe) else {},
        }

    def describe(self) -> dict[str, dict[str, Any]]:
        """``info()`` for every configured camera, keyed by entry."""
        return {spec.entry: self.info(spec.entry) for spec in self.specs}

    def close(self) -> None:
        cameras = list(self._cams.values())
        self._cams.clear()
        for camera in cameras:
            try:
                camera.release()
            except Exception as exc:  # one stuck camera must not keep the others open
                log.warning("camera release failed: %s", exc)

    def __enter__(self) -> CameraManager:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
