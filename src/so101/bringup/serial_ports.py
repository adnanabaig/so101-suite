"""Serial ports on Windows, Linux and macOS, through pyserial's ``comports()``.

Nothing here hardcodes or globs a device path. pyserial 3.5 picks the lister by
platform (``serial/tools/list_ports.py:28-31``, ``list_ports_posix.py:27-31``):

* Windows: ``COM3``-style names (``list_ports_windows.py``).
* Linux: ``/dev/ttyUSB*``, ``/dev/ttyACM*`` and other tty devices, read from
  sysfs (``list_ports_linux.py:91-103``). A ``/dev/serial/by-id/...`` link is
  not listed, so a configured link is matched through its real path.
* macOS: the callout device ``/dev/cu.*`` (``list_ports_osx.py:269-272``;
  checked on this Mac). LeRobot's ``lerobot-find-port`` prints ``/dev/tty*``
  names there (``lerobot_find_port.py:36-41``), so a configured
  ``/dev/tty.usbmodemX`` matches a listed ``/dev/cu.usbmodemX``.

Each adapter is identified by its USB serial number and VID:PID
(``list_ports_common.py:38-50``). pyserial ships in the ``robot`` extra; when it
is missing, :func:`list_serial_ports` reports why instead of raising.
"""

from __future__ import annotations

import os
import posixpath
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

#: pyserial's ``ListPortInfo`` fields that describe the adapter.
_INFO_FIELDS = ("description", "hwid", "manufacturer", "product", "location")
_WINDOWS_DEVICE_PREFIX = "\\\\.\\"


@dataclass(frozen=True)
class SerialPortInfo:
    """One entry of ``comports()``."""

    device: str
    vid: int | None = None
    pid: int | None = None
    serial_number: str | None = None
    description: str | None = None
    hwid: str | None = None
    manufacturer: str | None = None
    product: str | None = None
    location: str | None = None

    @classmethod
    def from_pyserial(cls, info: Any) -> SerialPortInfo:
        def text(name: str) -> str | None:
            value = getattr(info, name, None)
            return None if value in (None, "", "n/a") else str(value)

        def number(name: str) -> int | None:
            value = getattr(info, name, None)
            return value if isinstance(value, int) else None

        return cls(
            device=str(getattr(info, "device", "")),
            vid=number("vid"),
            pid=number("pid"),
            serial_number=text("serial_number"),
            **{name: text(name) for name in _INFO_FIELDS},
        )

    @property
    def vid_pid(self) -> str | None:
        if self.vid is None or self.pid is None:
            return None
        return f"{self.vid:04X}:{self.pid:04X}"

    def identity(self) -> str:
        """How to tell this adapter apart: serial number and VID:PID."""
        parts = []
        if self.serial_number:
            parts.append(f"serial {self.serial_number}")
        if self.vid_pid:
            parts.append(f"VID:PID {self.vid_pid}")
        return ", ".join(parts) or "no USB identity"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "vid_pid": self.vid_pid}


@dataclass(frozen=True)
class PortListing:
    """``comports()`` at one moment, or why it could not be read."""

    ports: list[SerialPortInfo] = field(default_factory=list)
    error: str | None = None
    platform: str = sys.platform

    def describe(self) -> str:
        if self.error:
            return f"(ports could not be listed: {self.error})"
        if not self.ports:
            return "none"
        return ", ".join(f"{info.device} ({info.identity()})" for info in self.ports)


Comports = Callable[[], Iterable[Any]]


def list_serial_ports(
    comports: Comports | None = None, *, platform: str | None = None
) -> PortListing:
    """Every serial port pyserial can see, sorted by device name."""
    where = platform or sys.platform
    if comports is None:
        try:
            from serial.tools import list_ports
        except ImportError as exc:
            return PortListing(error=f"pyserial is not installed ({exc})", platform=where)
        comports = list_ports.comports
    try:
        raw = list(comports())
    except Exception as exc:  # a lister failure must not take preflight down
        return PortListing(
            error=f"listing serial ports failed: {type(exc).__name__}: {exc}", platform=where
        )
    ports = sorted((SerialPortInfo.from_pyserial(item) for item in raw), key=lambda p: p.device)
    return PortListing(ports=ports, platform=where)


def port_key(device: str, *, platform: str | None = None) -> str:
    """A name two spellings of the same port share.

    Windows: ``COM3``, ``com3`` and ``\\\\.\\COM3`` are one port. POSIX: the real
    path, so a ``/dev/serial/by-id/`` link matches its tty. macOS: ``/dev/tty.X``
    and ``/dev/cu.X`` are the two device files of one port.
    """
    where = platform or sys.platform
    name = device.strip()
    if where == "win32":
        if name.startswith(_WINDOWS_DEVICE_PREFIX):
            name = name[len(_WINDOWS_DEVICE_PREFIX) :]
        return name.upper()
    # Resolve links only for this machine's ports; another platform's name
    # (tests, a report from elsewhere) is compared as written.
    path = os.path.realpath(name) if where == sys.platform else posixpath.normpath(name)
    if where == "darwin" and path.startswith("/dev/tty."):
        path = "/dev/cu." + path[len("/dev/tty.") :]
    return path


def match_port(
    device: str, ports: Sequence[SerialPortInfo], *, platform: str | None = None
) -> SerialPortInfo | None:
    """The listed port ``device`` names, if any."""
    key = port_key(device, platform=platform)
    for info in ports:
        if port_key(info.device, platform=platform) == key:
            return info
    return None


def shared_serial_number(
    first: Mapping[str, Any] | None, second: Mapping[str, Any] | None
) -> str | None:
    """The USB serial number two different listed adapters both report, if any.

    Cheap adapters sometimes ship with identical serial numbers; then neither a
    ``/dev/serial/by-id/`` link nor a lookup by serial number can tell them apart.
    """
    if not first or not second or first.get("device") == second.get("device"):
        return None
    number = first.get("serial_number")
    return str(number) if number and number == second.get("serial_number") else None


def permission_remediation(device: str, *, platform: str | None = None) -> str:
    """What to do when a port exists but this user may not open it."""
    where = platform or sys.platform
    if where.startswith("linux"):
        return (
            f"Add your user to the group that owns {device}, usually dialout (uucp on Arch): "
            "`sudo usermod -aG dialout $USER`, then log out and back in. "
            f"`ls -l {device}` shows the owning group."
        )
    if where == "win32":
        return (
            f"Close any other program that has {device} open (Windows gives a COM port to "
            "one program at a time), then re-run."
        )
    return f"Check the device permissions with `ls -l {device}`."


#: The arm adapters' names on Linux: the udev rules ``install/install.sh
#: --follower-serial S --leader-serial S`` writes create them, and they are the
#: suite's default ports there (so101.config.default_arm_port).
LINUX_ARM_LINKS = ("/dev/so101-follower", "/dev/so101-leader")


def port_naming_hint(platform: str | None = None) -> str:
    """How arm ports are named on ``platform`` (default: this OS), Linux's udev names first."""
    where = platform or sys.platform
    if where == "win32":
        return "On Windows the port is a name like COM3 (Device Manager, Ports (COM & LPT))."
    if where == "darwin":
        return "On macOS the port looks like /dev/tty.usbmodem... or /dev/cu.usbmodem...."
    follower, leader = LINUX_ARM_LINKS
    return (
        f"On Linux the suite expects {follower} and {leader}: re-run install/install.sh "
        "with --follower-serial and --leader-serial (each adapter's USB serial number is in "
        "`python -m so101.platform_info`) to create those names, then unplug and replug the "
        "arms. Without them the port is /dev/ttyACM0 or similar, which can change between "
        "boots; its /dev/serial/by-id/ link doesn't."
    )


def not_found_remediation(listing: PortListing) -> str:
    """What to do when the configured port is not there (Linux: the udev names first)."""
    naming = port_naming_hint(listing.platform)
    find = (
        "find its port with `lerobot-find-port` (unplug when it asks; LeRobot's connect "
        "error suggests it, motors_bus.py:535-539) and set SO101_FOLLOWER_PORT / "
        "SO101_LEADER_PORT."
    )
    present = f"Serial ports present now: {listing.describe()}."
    if listing.platform.startswith("linux"):
        return f"Plug in the arm's USB adapter and power the arm. {naming} Or {find} {present}"
    return f"Plug in the arm's USB adapter and power the arm, then {find} {naming} {present}"


#: When pyserial is missing nothing can be listed or opened.
PYSERIAL_REMEDIATION = (
    'Install the robot extra on the robot host (`pip install -e ".[robot]"`, or '
    "`make install-robot` where make is available); it brings pyserial>=3.5 with "
    "LeRobot's feetech extra."
)
