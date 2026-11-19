"""Find cameras for SO101_CAMERAS: Intel RealSense devices and OpenCV indices.

:func:`discover_cameras` returns a structured :class:`CameraReport` and
:func:`render_report` the text ``make find-cameras`` prints, so a
``so101 find-cameras`` command can reuse both. A missing pyrealsense2 or
OpenCV becomes a hint in the report, never a traceback. RealSense devices are
only enumerated, not streamed.

Discovery is nevertheless a physical-device operation: callers must reject it
unless the real backend is selected.  :func:`camera_discovery_refusal` is the
shared guard used by both the installed CLI and ``scripts/find_cameras.py``.
"""

from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .cameras import CameraUnavailableError
from .realsense import (
    RealSenseDevice,
    UsbRealSense,
    bus_diagnosis,
    distribution_name,
    list_realsense_devices,
    scan_usb_realsense,
    usb2_hint,
)

#: OpenCV indices 0..MAX_OPENCV_INDEX-1 are probed.
MAX_OPENCV_INDEX = 10
#: Linux names each V4L2 node here (``video<i>/name``).
V4L2_SYSFS = Path("/sys/class/video4linux")
OPENCV_HINT = "OpenCV (cv2) not installed; install the robot extra: pip install -e '.[robot]'"
UVC_WARNING = (
    "A RealSense also shows up as ordinary (UVC) camera indices. Configure each camera once: "
    "as realsense:<serial> (color, optional depth) or as its OpenCV index (color only), "
    "not both."
)


def camera_discovery_refusal(settings: Any) -> str | None:
    """Why physical camera discovery is disabled, or ``None`` on the real backend.

    Mock and MuJoCo runs must be hermetic.  Importing or probing OpenCV and
    librealsense there could open an attached host camera even though the
    selected robot backend promises not to use physical devices.
    """

    from .backend import current_backend

    backend = current_backend(settings)
    if backend == "real":
        return None
    return (
        "camera discovery probes physical USB/video devices and is disabled while "
        f"the {backend} backend is selected; set SO101_BACKEND=real only on the hardware "
        "host when you intend to inspect attached cameras"
    )


@dataclass(frozen=True)
class OpenCVCamera:
    index: int
    width: int
    height: int
    name: str | None = None  # Linux only: the V4L2 device name

    @property
    def is_realsense(self) -> bool:
        return "realsense" in (self.name or "").lower()


@dataclass
class CameraReport:
    realsense: list[RealSenseDevice] = field(default_factory=list)
    realsense_version: str | None = None
    realsense_error: str | None = None  # why devices could not be listed, with the fix
    opencv: list[OpenCVCamera] = field(default_factory=list)
    opencv_error: str | None = None
    opencv_probed: int = 0
    # Linux: RealSense devices on the USB bus (sysfs), readable without udev rules.
    usb: list[UsbRealSense] = field(default_factory=list)
    # Linux: why the bus and librealsense disagree, as (detail, remediation).
    bus_problem: tuple[str, str] | None = None
    platform: str = sys.platform

    @property
    def warnings(self) -> list[str]:
        notes: list[str] = []
        for device in self.realsense:
            if device.firmware_outdated:
                notes.append(
                    f"{device.label}: firmware {device.firmware} is older than the recommended "
                    f"{device.recommended_firmware}; see docs/realsense.md (Firmware)."
                )
        if self.usb:
            # The negotiated link speed is the ground truth where Linux reports it.
            notes += [
                f"{device.describe()}: {usb2_hint(device.link)}"
                for device in self.usb
                if device.usb2 and not device.recovery
            ]
        else:
            notes += [
                f"{device.label}: {usb2_hint(f'USB {device.usb_type}')}"
                for device in self.realsense
                if device.usb2
            ]
        if self.bus_problem is not None:
            notes.append(f"A RealSense is {self.bus_problem[0]}. {self.bus_problem[1]}")
        if self.realsense and self.opencv:
            notes.append(UVC_WARNING)
        return notes

    @property
    def suggestion(self) -> str | None:
        """A paste-ready SO101_CAMERAS value: color for every camera found."""
        entries = [device.entries[0] for device in self.realsense if device.entries]
        # Leave out indices known to be a RealSense's own nodes (Linux names
        # them); elsewhere they cannot be told apart, which UVC_WARNING says.
        known = [camera for camera in self.opencv if not camera.is_realsense]
        if not self.realsense or self.platform.startswith("linux"):
            entries += [str(camera.index) for camera in known]
        return ",".join(entries) or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "realsense": [device.to_dict() for device in self.realsense],
            "realsense_version": self.realsense_version,
            "realsense_error": self.realsense_error,
            "opencv": [asdict(camera) for camera in self.opencv],
            "opencv_error": self.opencv_error,
            "opencv_probed": self.opencv_probed,
            "usb": [asdict(device) for device in self.usb],
            "warnings": self.warnings,
            "suggestion": self.suggestion,
        }


def _v4l2_name(index: int, sysfs: Path) -> str | None:
    try:
        return (sysfs / f"video{index}" / "name").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def probe_opencv(
    max_index: int = MAX_OPENCV_INDEX,
    *,
    platform: str | None = None,
    sysfs: Path = V4L2_SYSFS,
) -> list[OpenCVCamera]:
    """Open each index, read one frame, close it. Raises ImportError without cv2."""
    import cv2

    linux = (platform or sys.platform).startswith("linux")
    found: list[OpenCVCamera] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                name = _v4l2_name(index, sysfs) if linux else None
                found.append(OpenCVCamera(index, int(width), int(height), name))
        finally:
            capture.release()
    return found


def discover_cameras(
    *,
    max_index: int = MAX_OPENCV_INDEX,
    platform: str | None = None,
    sysfs: Path = V4L2_SYSFS,
    usb_sysfs: Path | None = None,
) -> CameraReport:
    platform = platform or sys.platform
    report = CameraReport(platform=platform, opencv_probed=max_index)
    report.usb = scan_usb_realsense(usb_sysfs)
    try:
        report.realsense = list_realsense_devices()
    except CameraUnavailableError as exc:
        report.realsense_error = str(exc)
    except Exception as exc:  # librealsense failing to enumerate is still just a note
        report.realsense_error = f"could not list devices ({type(exc).__name__}: {exc})"
    else:
        report.bus_problem = bus_diagnosis(len(report.realsense), usb_sysfs)
        try:
            report.realsense_version = importlib.metadata.version(distribution_name(platform))
        except importlib.metadata.PackageNotFoundError:
            report.realsense_version = None
    try:
        report.opencv = probe_opencv(max_index, platform=platform, sysfs=sysfs)
    except ImportError:
        report.opencv_error = OPENCV_HINT
    except Exception as exc:
        report.opencv_error = f"probing OpenCV indices failed ({type(exc).__name__}: {exc})"
    return report


def render_report(report: CameraReport) -> str:
    """The text ``make find-cameras`` prints."""

    version = f" ({distribution_name(report.platform)} {report.realsense_version})"
    lines = [f"Intel RealSense{version if report.realsense_version else ''}:"]
    if report.realsense_error:
        lines.append(f"  {report.realsense_error}")
    elif not report.realsense:
        lines.append("  no RealSense found")
    for device in report.realsense:
        lines.append(
            f"  {device.name or 'RealSense'}  serial {device.serial or 'unknown'}  "
            f"firmware {device.firmware or '?'} (recommended {device.recommended_firmware or '?'})"
            f"  USB {device.usb_type or '?'}"
        )
        if len(device.entries) == 2:
            lines.append(f"    color:          {device.entries[0]}")
            lines.append(f"    color + depth:  {device.entries[1]}")
    if report.usb:
        lines += ["", "USB bus (Linux sysfs):"]
        lines += [f"  {device.describe()}" for device in report.usb]
    lines += ["", f"OpenCV indices 0..{report.opencv_probed - 1}:"]
    if report.opencv_error:
        lines.append(f"  {report.opencv_error}")
    elif not report.opencv:
        lines.append("  no cameras found")
    for camera in report.opencv:
        name = f"  ({camera.name})" if camera.name else ""
        lines.append(f"  camera {camera.index}: OK  {camera.width}x{camera.height}{name}")
    if report.warnings:
        lines.append("")
        lines += [f"! {warning}" for warning in report.warnings]
    lines.append("")
    suggestion = report.suggestion
    lines.append(f"Set:  SO101_CAMERAS={suggestion}" if suggestion else "no cameras found")
    return "\n".join(lines)
