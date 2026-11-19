"""Intel RealSense capture on pyrealsense2: RGB color plus opt-in aligned depth.

Built on pyrealsense2 (librealsense) directly rather than wrapping LeRobot's
``RealSenseCamera``, which stores raw z16 units and documents them as
millimetres. Here depth is converted with the device's own depth scale, color
and depth come from one frameset, and every failure carries an OS-specific
fix.

- Color is rgb8, returned as HxWx3 uint8 RGB.
- Depth (``+depth``) is z16 at the color fps, aligned to the color frame with
  ``rs.align``, and returned as HxWx1 uint16 millimetres where 0 means no
  data: ``round(z * depth_scale * 1000)``, clipped to 0..65535. The scale
  (metres per unit) is read from the device, never assumed.
- A daemon thread keeps the latest frameset; ``read()`` waits only for the
  first one and refuses frames older than :data:`STALE_AFTER_S`.

pyrealsense2 is imported lazily (the ``realsense`` extra). See docs/realsense.md.
"""

from __future__ import annotations

import functools
import math
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from ..logging import get_logger
from .camera_spec import CameraSpec
from .cameras import CameraUnavailableError

log = get_logger(__name__)

#: Connect attempts; the last one follows a hardware reset.
MAX_CONNECT_ATTEMPTS = 3
#: After ``hardware_reset()`` the device drops off the bus; wait this long
#: before looking for it, then poll for its serial up to the timeout.
RESET_SETTLE_S = 2.0
RESET_POLL_S = 0.5
RESET_REENUMERATE_TIMEOUT_S = 10.0
#: How long connect() and the first read() wait for a frameset.
FIRST_FRAME_TIMEOUT_S = 5.0
#: Frames from roughly the first half second are dropped while exposure settles.
WARMUP_S = 0.5
#: Each wait inside the capture thread (try_wait_for_frames timeout).
FRAME_WAIT_MS = 1000
#: read() refuses a frameset older than this.
STALE_AFTER_S = 1.0
#: release() waits this long for the capture thread to stop.
JOIN_TIMEOUT_S = 2.0
#: Depth sizes tried at the color fps when color's own size has no z16 mode.
PREFERRED_DEPTH_SIZES = ((640, 480), (848, 480))
UDEV_RULES_NAME = "99-realsense-libusb.rules"
#: Linux lists every USB device here, readable without the udev rules.
USB_SYSFS = Path("/sys/bus/usb/devices")
INTEL_VENDOR_ID = "8086"
#: A negotiated link below this (Mbit/s; sysfs ``speed``) is USB 2 or slower.
USB3_MIN_MBPS = 5000.0

# RealSenseDevice field -> rs.camera_info member.
_CAMERA_INFO = {
    "name": "name",
    "serial": "serial_number",
    "firmware": "firmware_version",
    "recommended_firmware": "recommended_firmware_version",
    "usb_type": "usb_type_descriptor",
    "product_line": "product_line",
    "product_id": "product_id",
    "physical_port": "physical_port",
}
_LINUX_PERMISSION_MARKERS = (
    "rs2_usb_status_access",
    "permission denied",
    "failed to set power state",
    "set power state",
)
# "failed to set power state" (librealsense's Linux V4L2 backend, which the
# PyPI wheel uses) shows up both for permissions and for a camera another
# program holds, so it earns both hints.
_BUSY_MARKERS = ("busy", "in use", "access is denied", "access denied", "set power state")
_RESOLVE_MARKERS = ("couldn't resolve requests", "could not resolve requests")

# librealsense lets several programs share a camera, but each stream (color,
# depth) belongs to one of them at a time (doc/rs400_support.md, v2.56.5).
BUSY_HINT = (
    "Each camera stream can be used by only one program at a time: close RealSense Viewer, "
    "another so101 server, or any other program streaming from this camera."
)
MACOS_HINT = (
    "On macOS librealsense usually needs root to claim the camera: run so101 with sudo. "
    "Without depth, the color stream also works as an ordinary OpenCV camera: put its index "
    "from `make find-cameras` in SO101_CAMERAS instead."
)
GENERIC_HINT = (
    "Replug the camera into a USB 3 port with a USB 3 cable, close other programs using it, "
    "and check `make find-cameras`."
)
RESOLVE_HINT = (
    "The camera cannot stream the requested modes together; on a USB 2 link, or with depth "
    "on, try a smaller or slower color mode in the entry."
)


def distribution_name(platform: str | None = None) -> str:
    """The PyPI distribution that provides ``pyrealsense2`` on this OS."""
    return "pyrealsense2-macosx" if (platform or sys.platform) == "darwin" else "pyrealsense2"


def udev_rules_path() -> Path:
    """librealsense's udev rules, vendored for Linux (copy them to /etc/udev/rules.d/)."""
    return Path(str(resources.files("so101.hardware") / "resources" / UDEV_RULES_NAME))


def udev_install_command() -> str:
    return (
        f"sudo cp {udev_rules_path()} /etc/udev/rules.d/ && sudo udevadm control "
        "--reload-rules && sudo udevadm trigger"
    )


def udev_hint() -> str:
    """How to install the udev rules on Linux: the installer's step, or by hand."""
    return (
        "Install the RealSense udev rules so your user may open the camera: "
        "install/install.sh --realsense offers to, or by hand: "
        f"{udev_install_command()}. Then unplug and replug the camera."
    )


def install_hint(platform: str | None = None) -> str:
    platform = platform or sys.platform
    hint = (
        f'Install the RealSense extra: pip install "so101[realsense]" (it brings '
        f"{distribution_name(platform)})"
    )
    if platform.startswith("linux"):
        return f"{hint}; install/install.sh --realsense does both. {udev_hint()}"
    if platform == "darwin":
        return f"{hint}. On macOS it usually needs sudo to open a camera; see docs/realsense.md."
    return f"{hint}. See docs/realsense.md."


def usb2_hint(link: str) -> str:
    """``link`` is librealsense's USB type ("2.1") or a sysfs speed ("480 Mbit/s")."""
    return (
        f"The camera is on a USB 2 link ({link}): plug it straight into a USB 3 port with a "
        "USB 3 cable (USB 2 offers fewer and smaller stream modes)."
    )


def remediation_for(
    error: str,
    *,
    platform: str | None = None,
    usb_type: str | None = None,
    link_mbps: float | None = None,
) -> str:
    """What to try after a RealSense failed to start or stream, for this OS and error.

    ``usb_type`` is librealsense's USB descriptor type, ``link_mbps`` the link
    speed Linux negotiated (sysfs); either one showing USB 2 adds that hint.
    """

    platform = platform or sys.platform
    text = error.lower()
    hints: list[str] = []
    if usb_type and usb_type.startswith("2"):
        hints.append(usb2_hint(f"USB {usb_type}"))
    elif link_mbps is not None and link_mbps < USB3_MIN_MBPS:
        hints.append(usb2_hint(f"{link_mbps:g} Mbit/s"))
    if platform.startswith("linux") and any(m in text for m in _LINUX_PERMISSION_MARKERS):
        hints.append(udev_hint())
    if any(m in text for m in _RESOLVE_MARKERS):
        hints.append(RESOLVE_HINT)
    # Windows reports a device held by another process in many ways, so the
    # hint is always worth giving there.
    if platform == "win32" or any(m in text for m in _BUSY_MARKERS):
        hints.append(BUSY_HINT)
    if platform == "darwin":
        hints.append(MACOS_HINT)
    return " ".join(hints) or GENERIC_HINT


def not_found_remediation(platform: str | None = None) -> str:
    platform = platform or sys.platform
    hints = ["Check that it is plugged in (USB 3 port and cable), then run `make find-cameras`."]
    if platform.startswith("linux"):
        hints.append(f"If lsusb lists it (Intel, 8086:xxxx) but this does not: {udev_hint()}")
    elif platform == "darwin":
        hints.append(MACOS_HINT)
    return " ".join(hints)


# --- Linux: the USB bus as sysfs sees it ------------------------------------------------


@functools.cache
def realsense_usb_ids() -> dict[str, bool]:
    """RealSense USB product ids (vendor 8086) from the vendored udev rules.

    Maps each id to True for a recovery (DFU) id, False for a camera.
    """
    ids: dict[str, bool] = {}
    recovery = False
    for line in udev_rules_path().read_text(encoding="utf-8").splitlines():
        if line.startswith("# ") and not line.startswith("# ---"):
            recovery = "recovery" in line.lower()
        if not line.startswith('SUBSYSTEMS=="usb"'):
            continue
        if f'ATTRS{{idVendor}}=="{INTEL_VENDOR_ID}"' not in line:
            continue
        match = re.search(r'ATTRS\{idProduct\}=="([0-9a-fA-F|]+)"', line)
        for product in match[1].lower().split("|") if match else ():
            ids.setdefault(product, recovery)
    return ids


@dataclass(frozen=True)
class UsbRealSense:
    """A RealSense on the USB bus (Linux sysfs), whether or not librealsense can open it."""

    port: str  # sysfs name: bus-port[.port...], e.g. "2-1"
    product_id: str
    product: str | None = None
    speed_mbps: float | None = None  # the negotiated link: 480 = USB 2, 5000+ = USB 3
    recovery: bool = False  # enumerated in recovery (DFU) mode

    @property
    def usb2(self) -> bool:
        return self.speed_mbps is not None and self.speed_mbps < USB3_MIN_MBPS

    @property
    def link(self) -> str:
        return "unknown speed" if self.speed_mbps is None else f"{self.speed_mbps:g} Mbit/s"

    def describe(self) -> str:
        return (
            f"{self.product or 'Intel RealSense'} (USB {INTEL_VENDOR_ID}:{self.product_id}) "
            f"at USB port {self.port}, {self.link}"
        )


def _read_sysfs(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def scan_usb_realsense(sysfs: Path | None = None) -> list[UsbRealSense]:
    """RealSense devices on the USB bus with their link speed (Linux; elsewhere empty)."""
    try:
        entries = sorted((sysfs or USB_SYSFS).iterdir())
    except OSError:
        return []
    ids = realsense_usb_ids()
    found: list[UsbRealSense] = []
    for entry in entries:
        if _read_sysfs(entry / "idVendor") != INTEL_VENDOR_ID:
            continue
        product_id = (_read_sysfs(entry / "idProduct") or "").lower()
        if product_id not in ids:
            continue
        try:
            speed = float(_read_sysfs(entry / "speed") or "")
        except ValueError:
            speed = None
        found.append(
            UsbRealSense(
                port=entry.name,
                product_id=product_id,
                product=_read_sysfs(entry / "product"),
                speed_mbps=speed,
                recovery=ids[product_id],
            )
        )
    return found


def usb_link_mbps(sysfs: Path | None = None) -> float | None:
    """The link speed when exactly one RealSense camera is on the bus (Linux), else None."""
    cameras = [device for device in scan_usb_realsense(sysfs) if not device.recovery]
    return cameras[0].speed_mbps if len(cameras) == 1 else None


def bus_diagnosis(seen: int, sysfs: Path | None = None) -> tuple[str, str] | None:
    """(detail, remediation) when the USB bus explains why librealsense saw ``seen`` devices.

    Linux only: a camera on the bus that librealsense does not list usually
    means missing udev rules (else a uvcvideo driver problem); a recovery-mode
    id needs a firmware update.
    """
    devices = scan_usb_realsense(sysfs)
    recovery = [device for device in devices if device.recovery]
    cameras = [device for device in devices if not device.recovery]
    if recovery:
        return (
            "in recovery (DFU) mode on the USB bus: " + "; ".join(d.describe() for d in recovery),
            "Update the camera's firmware (docs/realsense.md, Firmware), then replug it.",
        )
    if len(cameras) > seen:
        return (
            "on the USB bus but not opened by librealsense: "
            + "; ".join(device.describe() for device in cameras),
            f"{udev_hint()} If the rules are already installed, check `sudo dmesg` for "
            "uvcvideo errors after replugging.",
        )
    return None


def import_pyrealsense2() -> Any:
    """``import pyrealsense2``, or a ``CameraUnavailableError`` saying how to install it."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraUnavailableError(
            f"pyrealsense2 is not importable ({exc})", remediation=install_hint()
        ) from exc
    if not hasattr(rs, "pipeline"):
        raise CameraUnavailableError(
            f"the pyrealsense2 at {getattr(rs, '__file__', None) or '?'} has no pipeline (a "
            "broken build, or a folder named pyrealsense2 shadows the package)",
            remediation=install_hint(),
        )
    return rs


def _version(text: str | None) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in text.strip().split(".")) if text else None
    except ValueError:
        return None


def firmware_older(firmware: str | None, recommended: str | None) -> bool:
    """True when both versions parse and the firmware is older than recommended."""
    current, wanted = _version(firmware), _version(recommended)
    return current is not None and wanted is not None and current < wanted


@dataclass(frozen=True)
class RealSenseDevice:
    """One attached RealSense, from ``camera_info``; any field the device lacks is None."""

    name: str | None = None
    serial: str | None = None
    firmware: str | None = None
    recommended_firmware: str | None = None
    usb_type: str | None = None
    product_line: str | None = None
    product_id: str | None = None
    physical_port: str | None = None

    @property
    def usb2(self) -> bool:
        return bool(self.usb_type) and str(self.usb_type).startswith("2")

    @property
    def firmware_outdated(self) -> bool:
        return firmware_older(self.firmware, self.recommended_firmware)

    @property
    def entries(self) -> list[str]:
        """Paste-ready SO101_CAMERAS entries: color only, then color plus depth."""
        if not self.serial:
            return []
        return [f"realsense:{self.serial}", f"realsense:{self.serial}+depth"]

    @property
    def label(self) -> str:
        return f"{self.name or 'RealSense'} (serial {self.serial or 'unknown'})"

    def to_dict(self) -> dict[str, Any]:
        facts = asdict(self)
        facts.update(usb2=self.usb2, firmware_outdated=self.firmware_outdated)
        return facts


def device_facts(rs: Any, device: Any) -> RealSenseDevice:
    """Read ``camera_info`` fields, each only if ``device.supports()`` it."""
    values: dict[str, str] = {}
    for field_name, info_name in _CAMERA_INFO.items():
        key = getattr(rs.camera_info, info_name, None)
        try:
            if key is not None and device.supports(key):
                values[field_name] = str(device.get_info(key))
        except Exception as exc:  # e.g. unplugged mid-query; report what we have
            log.debug("RealSense camera_info %s unreadable: %s", info_name, exc)
    return RealSenseDevice(**values)


def list_realsense_devices() -> list[RealSenseDevice]:
    """Attached RealSense devices. Enumerates only; nothing is streamed."""
    rs = import_pyrealsense2()
    return [device_facts(rs, device) for device in rs.context().query_devices()]


def _attached(devices: Sequence[RealSenseDevice]) -> str:
    if not devices:
        return "none attached"
    listed = [
        " or ".join(device.entries) + f" ({device.name or 'RealSense'}, USB {device.usb_type})"
        if device.entries
        else f"{device.label} without a readable serial"
        for device in devices
    ]
    return "attached: " + "; ".join(listed)


def _not_found(detail: str, seen: int) -> CameraUnavailableError:
    diagnosis = bus_diagnosis(seen)
    if diagnosis is not None:
        return CameraUnavailableError(
            f"{detail}; a RealSense is {diagnosis[0]}", remediation=diagnosis[1]
        )
    return CameraUnavailableError(detail, remediation=not_found_remediation())


def select_device(rs: Any, ctx: Any, serial: str | None) -> tuple[Any, RealSenseDevice]:
    """The device with ``serial``, or the only attached one when ``serial`` is None."""
    found = [(device, device_facts(rs, device)) for device in ctx.query_devices()]
    facts = [entry[1] for entry in found]
    if serial is not None:
        for device, info in found:
            if info.serial is not None and info.serial.casefold() == serial.casefold():
                return device, info
        raise _not_found(
            f"no RealSense with serial {serial} is attached ({_attached(facts)})", len(found)
        )
    if len(found) == 1:
        return found[0]
    if not found:
        raise _not_found("no RealSense camera is attached", 0)
    raise CameraUnavailableError(
        f"{len(found)} RealSense cameras are attached and the entry names none "
        f"({_attached(facts)})",
        remediation="Put one entry per camera, with its serial, in SO101_CAMERAS.",
    )


@dataclass(frozen=True, order=True)
class StreamMode:
    width: int
    height: int
    fps: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"

    @property
    def pixels(self) -> int:
        return self.width * self.height


def stream_modes(rs: Any, device: Any, stream: Any, fmt: Any) -> list[StreamMode]:
    """Every video mode the device's sensors offer for ``stream`` in ``fmt``."""
    modes: set[StreamMode] = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != stream or profile.format() != fmt:
                continue
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            modes.add(StreamMode(int(video.width()), int(video.height()), int(profile.fps())))
    return sorted(modes)


def choose_color_mode(spec: CameraSpec, modes: Sequence[StreamMode]) -> StreamMode:
    """The rgb8 mode to open: the configured one, or for the default the nearest offered."""
    wanted = StreamMode(spec.width or 0, spec.height or 0, spec.fps or 0)
    if wanted in modes:
        return wanted
    if not modes:
        raise CameraUnavailableError(f"{spec.entry!r}: the camera offers no rgb8 color stream")
    offered = ", ".join(str(mode) for mode in modes)
    if spec.mode_explicit:
        raise CameraUnavailableError(
            f"{spec.entry!r}: the camera has no rgb8 color mode {wanted}; it offers {offered}",
            remediation="Put one of those modes in the entry, or leave the mode out for "
            f"the default {wanted}.",
        )
    candidates = [mode for mode in modes if mode.fps == wanted.fps] or list(modes)
    chosen = min(
        candidates,
        key=lambda mode: (abs(mode.pixels - wanted.pixels), abs(mode.fps - wanted.fps), mode),
    )
    log.info("RealSense %s: no rgb8 %s, using %s", spec.entry, wanted, chosen)
    return chosen


def choose_depth_mode(color: StreamMode, modes: Sequence[StreamMode]) -> StreamMode:
    """z16 at the color fps: the color size first, then 640x480, 848x480, the nearest."""
    at_fps = [mode for mode in modes if mode.fps == color.fps]
    for width, height in ((color.width, color.height), *PREFERRED_DEPTH_SIZES):
        if StreamMode(width, height, color.fps) in at_fps:
            return StreamMode(width, height, color.fps)
    if at_fps:
        return min(at_fps, key=lambda mode: (abs(mode.pixels - color.pixels), mode))
    offered = ", ".join(str(mode) for mode in modes) or "none"
    raise CameraUnavailableError(
        f"the camera has no z16 depth mode at {color.fps} fps to pair with color {color} "
        f"(depth modes: {offered})",
        remediation="Pick a color mode whose fps a depth mode also offers.",
    )


def depth_to_mm(raw: np.ndarray, depth_scale: float) -> np.ndarray:
    """z16 units to HxWx1 uint16 millimetres: ``round(z * scale * 1000)``, clipped; 0 stays 0."""
    z = np.asarray(raw)
    if z.ndim == 3:
        z = z[:, :, 0]
    factor = float(depth_scale) * 1000.0
    if factor == 1.0:
        mm = z.astype(np.uint16, copy=True)
    else:
        mm = np.clip(np.rint(z.astype(np.float64) * factor), 0, 65535).astype(np.uint16)
    return mm[:, :, None]


def _format_name(value: Any) -> str:
    return str(getattr(value, "name", value)).removeprefix("format.")


def _stream_facts(profile: Any) -> dict[str, Any]:
    video = profile.as_video_stream_profile()
    return {
        "width": int(video.width()),
        "height": int(video.height()),
        "fps": int(profile.fps()),
        "format": _format_name(profile.format()),
    }


class RealSenseCamera:
    """One RealSense device behind ``read()``/``read_rgbd()``/``info()``/``release()``."""

    def __init__(
        self,
        spec: CameraSpec,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self._clock = clock
        self._sleep = sleep
        self._pipeline: Any = None
        self._align: Any = None
        self._device: RealSenseDevice | None = None
        self._depth_scale: float | None = None
        self._streams: dict[str, dict[str, Any]] = {}
        self._cond = threading.Condition()
        self._latest: tuple[np.ndarray, np.ndarray | None, float] | None = None
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._released = False

    @property
    def label(self) -> str:
        return self._device.label if self._device else repr(self.spec.entry)

    # --- connect ------------------------------------------------------------------

    def connect(self) -> None:
        rs = import_pyrealsense2()
        ctx = rs.context()
        device, facts = select_device(rs, ctx, self.spec.serial)
        self._device = facts
        color = choose_color_mode(
            self.spec, stream_modes(rs, device, rs.stream.color, rs.format.rgb8)
        )
        depth = (
            choose_depth_mode(color, stream_modes(rs, device, rs.stream.depth, rs.format.z16))
            if self.spec.depth
            else None
        )
        error: Exception | None = None
        for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
            if attempt == MAX_CONNECT_ATTEMPTS and facts.serial:
                device = self._hardware_reset(rs, ctx, device, facts.serial)
            try:
                self._start(rs, ctx, facts.serial, color, depth)
                return
            except Exception as exc:
                error = exc
                log.warning(
                    "RealSense %s: start attempt %d of %d failed: %s",
                    facts.label,
                    attempt,
                    MAX_CONNECT_ATTEMPTS,
                    exc,
                )
                self._stop_pipeline()
        raise CameraUnavailableError(
            f"RealSense {facts.label} did not start after {MAX_CONNECT_ATTEMPTS} attempts "
            f"({type(error).__name__}: {error})",
            remediation=remediation_for(
                str(error), usb_type=facts.usb_type, link_mbps=usb_link_mbps()
            ),
        )

    def _start(
        self,
        rs: Any,
        ctx: Any,
        serial: str | None,
        color: StreamMode,
        depth: StreamMode | None,
    ) -> None:
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, color.width, color.height, rs.format.rgb8, color.fps)
        if depth is not None:
            config.enable_stream(
                rs.stream.depth, depth.width, depth.height, rs.format.z16, depth.fps
            )
        pipeline = rs.pipeline(ctx)
        profile = pipeline.start(config)
        self._pipeline = pipeline
        self._streams = {"color": _stream_facts(profile.get_stream(rs.stream.color))}
        if depth is not None:
            scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            if not (math.isfinite(scale) and scale > 0):
                raise RuntimeError(f"the device reports a depth scale of {scale}")
            self._depth_scale = scale
            self._align = rs.align(rs.stream.color)
            self._streams["depth"] = {
                **_stream_facts(profile.get_stream(rs.stream.depth)),
                "aligned_to": "color",
                "unit": "mm",
            }
        self._warm_up(pipeline)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture,
            name=f"so101-realsense-{serial or 'device'}",
            daemon=True,
        )
        self._thread.start()

    def _warm_up(self, pipeline: Any) -> None:
        deadline = self._clock() + FIRST_FRAME_TIMEOUT_S
        while not pipeline.try_wait_for_frames(timeout_ms=FRAME_WAIT_MS)[0]:
            if self._clock() >= deadline:
                raise TimeoutError(f"no frameset within {FIRST_FRAME_TIMEOUT_S:g} s of starting")
        settled = self._clock() + WARMUP_S
        while self._clock() < settled:
            pipeline.try_wait_for_frames(timeout_ms=FRAME_WAIT_MS)

    def _hardware_reset(self, rs: Any, ctx: Any, device: Any, serial: str) -> Any:
        log.warning("RealSense %s: hardware reset before the last connect attempt", serial)
        try:
            device.hardware_reset()
        except Exception as exc:
            log.warning("RealSense %s: hardware_reset() failed: %s", serial, exc)
        deadline = self._clock() + RESET_REENUMERATE_TIMEOUT_S
        self._sleep(RESET_SETTLE_S)
        while True:
            try:
                for candidate in ctx.query_devices():
                    if device_facts(rs, candidate).serial == serial:
                        return candidate
            except Exception as exc:  # the bus is still settling
                log.debug("RealSense %s: enumeration after reset failed: %s", serial, exc)
            if self._clock() >= deadline:
                raise CameraUnavailableError(
                    f"RealSense {serial} did not come back within "
                    f"{RESET_REENUMERATE_TIMEOUT_S:g} s of a hardware reset",
                    remediation=not_found_remediation(),
                )
            self._sleep(RESET_POLL_S)

    # --- capture ------------------------------------------------------------------

    def _capture(self) -> None:
        pipeline = self._pipeline
        try:
            while not self._stop.is_set():
                ok, frames = pipeline.try_wait_for_frames(timeout_ms=FRAME_WAIT_MS)
                if not ok:
                    self._last_error = f"no frameset within {FRAME_WAIT_MS} ms"
                    continue
                converted = self._convert(frames)
                if converted is None:
                    continue
                with self._cond:
                    self._latest = (*converted, self._clock())
                    self._cond.notify_all()
        except Exception as exc:  # the thread ends here; read() reports why
            self._last_error = f"{type(exc).__name__}: {exc}"
            log.warning("RealSense %s: capture stopped: %s", self.label, self._last_error)
        finally:
            with self._cond:
                self._cond.notify_all()

    def _convert(self, frames: Any) -> tuple[np.ndarray, np.ndarray | None] | None:
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            self._last_error = "a frameset arrived without a color frame"
            return None
        color = np.asanyarray(color_frame.get_data()).copy()
        if self._align is None:
            return color, None
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            self._last_error = "a frameset arrived without a depth frame"
            return None
        assert self._depth_scale is not None
        return color, depth_to_mm(np.asanyarray(depth_frame.get_data()), self._depth_scale)

    # --- read ---------------------------------------------------------------------

    def _dead(self) -> CameraUnavailableError:
        return CameraUnavailableError(
            f"RealSense {self.label}: the capture thread stopped "
            f"({self._last_error or 'no error recorded'})",
            remediation=remediation_for(
                self._last_error or "", usb_type=self._usb_type, link_mbps=usb_link_mbps()
            ),
        )

    @property
    def _usb_type(self) -> str | None:
        return self._device.usb_type if self._device else None

    def _frames(self) -> tuple[np.ndarray, np.ndarray | None]:
        thread = self._thread
        if self._released or thread is None:
            raise CameraUnavailableError(f"RealSense {self.label} is not connected")
        deadline = self._clock() + FIRST_FRAME_TIMEOUT_S
        with self._cond:
            while self._latest is None:
                if not thread.is_alive():
                    raise self._dead()
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise CameraUnavailableError(
                        f"RealSense {self.label}: no frame within {FIRST_FRAME_TIMEOUT_S:g} s "
                        f"({self._last_error or 'no error recorded'})",
                        remediation=remediation_for(
                            self._last_error or "",
                            usb_type=self._usb_type,
                            link_mbps=usb_link_mbps(),
                        ),
                    )
                self._cond.wait(min(remaining, 0.1))
            color, depth, captured_at = self._latest
        if not thread.is_alive():
            raise self._dead()
        age = self._clock() - captured_at
        if age > STALE_AFTER_S:
            raise CameraUnavailableError(
                f"RealSense {self.label}: the newest frame is {age:.1f} s old "
                f"({self._last_error or 'no error recorded'})",
                remediation=remediation_for(
                    self._last_error or "", usb_type=self._usb_type, link_mbps=usb_link_mbps()
                ),
            )
        return color.copy(), None if depth is None else depth.copy()

    def read(self) -> np.ndarray:
        """The newest color frame, HxWx3 uint8 RGB."""
        return self._frames()[0]

    def read_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """Color and depth (HxWx1 uint16 mm, aligned to color) from the same frameset."""
        color, depth = self._frames()
        if depth is None:
            raise CameraUnavailableError(
                f"{self.spec.entry!r} streams color only; add +depth to the entry for depth"
            )
        return color, depth

    def info(self) -> dict[str, Any]:
        facts: dict[str, Any] = self._device.to_dict() if self._device else {}
        # Linux: the negotiated link speed, when one RealSense makes it unambiguous.
        link = usb_link_mbps()
        facts.update(
            usb_link_mbps=link,
            usb2=bool(facts.get("usb2")) or (link is not None and link < USB3_MIN_MBPS),
            depth_scale=self._depth_scale,
            streams={name: dict(values) for name, values in self._streams.items()},
            depth=self.spec.depth,
        )
        return facts

    # --- release ------------------------------------------------------------------

    def _stop_pipeline(self) -> None:
        pipeline, self._pipeline, self._align = self._pipeline, None, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:  # stop() before a successful start() raises
                log.debug("RealSense %s: pipeline.stop() failed: %s", self.label, exc)

    def release(self) -> None:
        """Stop the capture thread, then the pipeline. Safe to call more than once."""
        if self._released:
            return
        self._released = True
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT_S)
        self._stop_pipeline()


def open_realsense(spec: CameraSpec) -> RealSenseCamera:
    """Connect the camera for a ``realsense`` entry; raises ``CameraUnavailableError``."""
    camera = RealSenseCamera(spec)
    try:
        camera.connect()
    except Exception:
        camera.release()
        raise
    return camera
