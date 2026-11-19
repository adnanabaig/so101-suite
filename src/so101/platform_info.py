"""What this machine is, and what the suite should do on it.

``python -m so101.platform_info`` prints a short report, ``--json`` prints
the same report as JSON, ``--data-dir`` prints only the data folder,
``--realsense-udev-rules`` prints only the path of the RealSense udev rules
file the suite ships, ``--init-env`` adds the absolute storage folders to the
data folder's ``.env`` and ``--set-env KEY=VALUE`` adds any setting there
(``--replace`` also changes ones the file already has). The
installers run it after installing, and :func:`checks` returns the same facts
in ``so101 doctor``'s pass/warn/fail shape.

Settings files (:func:`settings_env_files`): every ``so101`` command reads
``<data folder>/.env`` and then ``./.env`` in the folder it runs in, the later
one winning, and the process environment wins over both. So the installed
suite behaves the same from any folder, and a source checkout's own ``.env``
still applies when you run from the checkout.

It uses the standard library only. ``torch``, ``pyserial`` and
``pyrealsense2`` are optional: torch and pyserial are imported lazily inside
the probes, and pyrealsense2 only in a child process, so the report also works
in a bare venv and says what is missing instead of failing. Every probe goes
through :class:`Probes`, so tests can describe a Windows desktop, a Linux
laptop or a Mac without patching ``sys.platform``.

Sources for the platform facts used here (versions as of 2026-10):

- MuJoCo 3.14.0 ``mujoco/rendering/classic/gl_context.py:24-56``: which
  ``MUJOCO_GL`` values each OS accepts. Any other value makes ``import mujoco``
  raise RuntimeError.
- NVIDIA's CUDA Toolkit release notes
  (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html):
  Table 3 (CUDA 12.x runs on R525+ drivers under minor-version
  compatibility), Table 4 (the driver each CUDA 12 update ships with), and the
  CUDA 13.0 Update 2 notes (Linux 580.95.05). The torch 2.11 wheels on
  download.pytorch.org pin CUDA 12.6.3, 12.8.1 and 13.0.2 (their wheel
  metadata).
- pyserial 3.5 ``serial/tools/list_ports_osx.py:269-270``: on macOS,
  ``comports()`` lists only ``/dev/cu.*`` devices.
- PyPI's file lists for pyrealsense2 (https://pypi.org/pypi/pyrealsense2/json)
  and pyrealsense2-macosx (https://pypi.org/pypi/pyrealsense2-macosx/json):
  pyrealsense2 2.56.4 and 2.56.5 have cp312 and cp313 wheels for win_amd64
  and manylinux1_x86_64, and none for Linux aarch64 on Python 3.12+; cp314
  and Linux aarch64 cp312 wheels start at 2.57.7. pyrealsense2-macosx 2.56.5
  is built for macOS 15+ arm64 (cp39 to cp314), 2.54.2 for macOS 12+
  universal2 (cp38 to cp312). The ``pyrealsense2`` module has no
  ``__version__`` (checked on pyrealsense2-macosx 2.56.5), so the version comes
  from the package metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes.util
import importlib
import importlib.metadata
import importlib.util
import json
import locale
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import replace as replace_fields
from pathlib import Path, PurePath
from typing import Any

APP_DIR_NAME = "so101"

# The suite's requires-python is >=3.12,<3.14 (LeRobot 0.6.1 needs 3.12 or newer;
# every pinned dependency has cp312 and cp313 wheels). The installers prefer 3.13.
MIN_PYTHON = (3, 12)
MAX_PYTHON = (3, 13)

# USB-serial bridges seen on Feetech bus-servo boards and generic adapters,
# by (vendor id, product id). Used only to label ports in reports.
KNOWN_USB_SERIAL: dict[tuple[int, int], str] = {
    (0x1A86, 0x55D3): "WCH CH343",
    (0x1A86, 0x55D4): "WCH CH9102",
    (0x1A86, 0x7523): "WCH CH340",
    (0x1A86, 0x5523): "WCH CH341",
    (0x10C4, 0xEA60): "Silicon Labs CP210x",
    (0x0403, 0x6001): "FTDI FT232R",
    (0x0403, 0x6015): "FTDI FT-X",
}

# NVIDIA driver for each CUDA build of torch, by OS. The torch 2.11 wheels pin
# CUDA 12.6.3, 12.8.1 and 13.0.2 (their metadata on download.pytorch.org), and
# these are the drivers NVIDIA's CUDA Toolkit release notes list for those
# updates (Table 4, and the 13.0 Update 2 notes). The notes give no Windows
# driver for 13.0, so Windows uses the R580 floor from Table 3.
MIN_DRIVER_FOR_CUDA: dict[str, dict[str, str]] = {
    "linux": {"12.6": "560.35.05", "12.8": "570.124.06", "13.0": "580.95.05"},
    "windows": {"12.6": "561.17", "12.8": "572.61", "13.0": "580"},
}
# Older drivers still run a build of the same CUDA major version under
# minor-version compatibility (Table 3). That rules out only PTX JIT, which the
# wheels don't need: they carry native code for every GPU the suite targets.
MIN_DRIVER_FOR_CUDA_MAJOR: dict[str, str] = {"12": "525", "13": "580"}

# Intel RealSense cameras, through the suite's ``realsense`` extra (pinned
# below 2.57): pyrealsense2, or on macOS the community build
# pyrealsense2-macosx, which provides the same ``pyrealsense2`` module.
REALSENSE_DIST = "pyrealsense2"
REALSENSE_DIST_MACOS = "pyrealsense2-macosx"
# Newest Python with a pyrealsense2 2.56 wheel on Windows and Linux.
REALSENSE_PYTHON_MAX = (3, 13)
REALSENSE_UDEV_RULES = "/etc/udev/rules.d/99-realsense-libusb.rules"
# Where Intel's librealsense2-udev-rules apt package puts its copy. The name is
# not verified here; if it is wrong, doctor only suggests rules that are there.
REALSENSE_UDEV_RULES_SYSTEM = (
    "/lib/udev/rules.d/60-librealsense2-udev-rules.rules",
    "/usr/lib/udev/rules.d/60-librealsense2-udev-rules.rules",
)
# USB product ids for every RealSense model this suite claims to target:
# D415, D435, D435i, D405 and D455 respectively. A rules file must grant
# access to all of them; one surviving rule in a truncated/stale file is not
# evidence that camera access is configured.
REALSENSE_TARGET_PRODUCT_IDS = frozenset({"0ad3", "0b07", "0b3a", "0b5b", "0b5c"})
# The suite ships the udev rules file, and this function of the camera code
# returns its path: (module, function). See realsense_udev_rules_source().
REALSENSE_RULES_FUNCTION = ("so101.hardware.realsense", "udev_rules_path")
REALSENSE_PROBE_TIMEOUT = 20.0  # seconds to list the attached cameras

# MuJoCo 3.14 gl_context.py:25-32.
_MUJOCO_GL_OFF = frozenset({"disable", "disabled", "off", "false", "0"})

# Linux setup. install/install.sh writes the arm rules file.
ARM_UDEV_RULES = "/etc/udev/rules.d/99-so101-arms.rules"
ARM_LINKS = ("/dev/so101-follower", "/dev/so101-leader")
NVIDIA_PROC = "/proc/driver/nvidia/version"  # present while the NVIDIA kernel module is loaded
# brltty's udev rules claim CH340/CH341 USB-serial adapters on Ubuntu.
BRLTTY_PATHS = ("/usr/bin/brltty", "/bin/brltty")
CH34X_VENDOR = 0x1A86
GNOME_SUSPEND_ON_AC = ("org.gnome.settings-daemon.plugins.power", "sleep-inactive-ac-type")
DISK_WARN_GIB = 20.0  # datasets with video and checkpoints take tens of GB
DISK_FAIL_GIB = 2.0  # below this, recording an episode can run out of space
MUJOCO_RENDER_TIMEOUT = 30.0  # seconds for the child process to render one frame
_MUJOCO_GL_ANY_OS = frozenset({"enable", "enabled", "on", "true", "1", "glfw", ""})
_MUJOCO_GL_BY_OS: dict[str, frozenset[str]] = {
    "linux": frozenset({"glx", "egl", "osmesa"}),
    "windows": frozenset({"wgl"}),
    "macos": frozenset({"cgl"}),
}


# --- probe results -----------------------------------------------------------


@dataclass(frozen=True)
class TorchInfo:
    """What an imported ``torch`` reports. ``error`` is set if probing failed."""

    version: str
    cuda_build: str | None = None  # torch.version.cuda: None for CPU and MPS builds
    cuda_available: bool = False
    device_name: str | None = None
    vram_gib: float | None = None
    mps_available: bool = False
    error: str | None = None


@dataclass(frozen=True)
class NvidiaSmi:
    """The first GPU line of ``nvidia-smi --query-gpu=name,driver_version,memory.total``."""

    name: str
    driver: str
    memory_mib: int | None = None


@dataclass(frozen=True)
class SerialPort:
    device: str
    description: str = ""
    vid: int | None = None
    pid: int | None = None
    serial_number: str | None = None
    manufacturer: str | None = None

    @property
    def is_usb(self) -> bool:
        return self.vid is not None

    @property
    def chip(self) -> str | None:
        if self.vid is None or self.pid is None:
            return None
        return KNOWN_USB_SERIAL.get((self.vid, self.pid))


@dataclass(frozen=True)
class GroupMembership:
    """Linux groups of the current user: in this login session, and as configured.

    ``usermod -aG`` changes the configured set at once, but a session keeps the
    groups it started with until the user logs out and back in.
    """

    session: frozenset[str]
    configured: frozenset[str]


@dataclass(frozen=True)
class RealSenseDevice:
    name: str
    serial: str | None = None
    firmware: str | None = None
    usb_type: str | None = None  # USB descriptor version such as "3.2"; None if not USB

    @property
    def usb2(self) -> bool:
        """On a USB 1.x or 2.x link, which caps resolution and frame rate."""

        return self.usb_type is not None and self.usb_type.strip().startswith(("1.", "2."))


@dataclass(frozen=True)
class RealSenseListing:
    """The cameras pyrealsense2 lists. ``error`` is set if listing failed."""

    devices: tuple[RealSenseDevice, ...] = ()
    error: str | None = None


# --- probes ------------------------------------------------------------------


def _python_version() -> tuple[int, int, int]:
    return sys.version_info[0], sys.version_info[1], sys.version_info[2]


def _run(cmd: Sequence[str]) -> str | None:
    """stdout of a short read-only command, or None if it is missing or fails."""

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash on Windows
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def probe_torch() -> TorchInfo | None:
    """None if torch is not installed; otherwise what it reports."""

    try:
        import torch
    except ModuleNotFoundError:
        return None
    except Exception as exc:  # a broken install: missing DLL, wrong CUDA libraries
        return TorchInfo(version="unknown", error=f"{type(exc).__name__}: {exc}")
    version = str(getattr(torch, "__version__", "unknown"))
    try:
        cuda_build = getattr(torch.version, "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        device_name = vram = None
        if cuda_available:
            device_name = str(torch.cuda.get_device_name(0))
            vram = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        mps = getattr(torch.backends, "mps", None)
        mps_available = bool(mps is not None and mps.is_available())
    except Exception as exc:
        return TorchInfo(version=version, error=f"{type(exc).__name__}: {exc}")
    return TorchInfo(
        version=version,
        cuda_build=str(cuda_build) if cuda_build else None,
        cuda_available=cuda_available,
        device_name=device_name,
        vram_gib=vram,
        mps_available=mps_available,
    )


def probe_serial_ports() -> list[SerialPort] | None:
    """None if pyserial is not installed."""

    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    ports = []
    for info in list_ports.comports():
        ports.append(
            SerialPort(
                device=str(info.device),
                description="" if info.description in (None, "n/a") else str(info.description),
                vid=info.vid,
                pid=info.pid,
                serial_number=info.serial_number,
                manufacturer=info.manufacturer,
            )
        )
    return sorted(ports, key=lambda p: p.device)


def _access_rw(path: str) -> bool:
    return os.access(path, os.R_OK | os.W_OK)


def _device_group(path: str) -> str | None:
    try:
        import grp  # POSIX only

        return grp.getgrgid(os.stat(path).st_gid).gr_name
    except (ImportError, OSError, KeyError):
        return None


def _user_groups() -> GroupMembership | None:
    try:
        import grp  # POSIX only
        import pwd
    except ImportError:
        return None
    try:
        user = pwd.getpwuid(os.getuid())
        session = set()
        for gid in os.getgroups():
            try:
                session.add(grp.getgrgid(gid).gr_name)
            except KeyError:
                continue
        configured = {g.gr_name for g in grp.getgrall() if user.pw_name in g.gr_mem}
        try:
            configured.add(grp.getgrgid(user.pw_gid).gr_name)
        except KeyError:
            pass
    except (OSError, KeyError):
        return None
    return GroupMembership(session=frozenset(session), configured=frozenset(configured))


def _pci_display_vendors() -> frozenset[str]:
    """Vendors of PCI display controllers, from Linux sysfs (no lspci needed)."""

    names = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
    found = set()
    for device in Path("/sys/bus/pci/devices").glob("*"):
        try:
            if not (device / "class").read_text(encoding="ascii").startswith("0x03"):
                continue
            vendor = (device / "vendor").read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if vendor in names:
            found.add(names[vendor])
    return frozenset(found)


def _long_paths_enabled() -> bool | None:
    """Windows' LongPathsEnabled registry value; None elsewhere or if unreadable."""

    try:
        import winreg  # Windows only
    except ImportError:
        return None
    try:
        key = winreg.OpenKey(  # type: ignore[attr-defined,unused-ignore]
            winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined,unused-ignore]
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        )
        with key:
            value, _ = winreg.QueryValueEx(  # type: ignore[attr-defined,unused-ignore]
                key, "LongPathsEnabled"
            )
    except OSError:
        return None
    return bool(value)


def _preferred_encoding() -> str:
    return locale.getpreferredencoding(False)


def _utf8_mode() -> bool:
    return bool(sys.flags.utf8_mode)


def _usb_vendor_ids() -> frozenset[int]:
    """Linux: the vendor ids of the connected USB devices, from sysfs."""

    found = set()
    for path in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            found.add(int(path.read_text(encoding="ascii").strip(), 16))
        except (OSError, ValueError):
            continue
    return frozenset(found)


def _has_battery() -> bool:
    """Linux: true on a laptop (a battery in sysfs)."""

    return any(Path("/sys/class/power_supply").glob("BAT*"))


def _disk_free(path: str) -> int | None:
    """Free bytes on the filesystem that holds ``path`` (or its nearest existing parent)."""

    current = Path(path)
    while not current.exists() and current != current.parent:
        current = current.parent
    try:
        return shutil.disk_usage(current).free
    except OSError:
        return None


def _free_threaded() -> bool:
    """True for a free-threaded (no-GIL) build such as python3.13t."""

    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _dist_version(name: str) -> str | None:
    """The installed version of a distribution, or None if it isn't installed."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_present(name: str) -> bool:
    """True if a top-level module can be imported; finds it without importing it."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _read_dotenv() -> dict[str, str]:
    """The settings files Settings reads (:func:`current_env_files`), merged.

    Keys are lowercased, as pydantic-settings matches them, so a later file
    overrides an earlier one however each spells the name.
    """

    values: dict[str, str] = {}
    for path in current_env_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        values.update((key.lower(), value) for key, value in parse_dotenv(text).items())
    return values


# Runs in a child process: librealsense is native code that can hang or crash
# when a camera misbehaves or permissions are missing, and the report has to
# survive both. Prints one JSON object.
_REALSENSE_LIST = """\
import json
try:
    import pyrealsense2 as rs
    found = []
    for dev in rs.context().query_devices():
        info = {}
        for key in ("name", "serial_number", "firmware_version", "usb_type_descriptor"):
            field = getattr(rs.camera_info, key, None)
            try:
                ok = field is not None and dev.supports(field)
                info[key] = dev.get_info(field) if ok else None
            except Exception:
                info[key] = None
        found.append(info)
    print(json.dumps({"devices": found}))
except Exception as exc:
    print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)}))
"""


def probe_realsense_devices() -> RealSenseListing:
    """The RealSense cameras pyrealsense2 sees, listed in a child process."""

    if not sys.executable:
        return RealSenseListing(error="no Python executable to run pyrealsense2 in")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash on Windows
    try:
        proc = subprocess.run(
            [sys.executable, "-P", "-c", _REALSENSE_LIST],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=REALSENSE_PROBE_TIMEOUT,
            check=False,
            creationflags=flags,
        )
    except subprocess.TimeoutExpired:
        return RealSenseListing(
            error=f"pyrealsense2 didn't list the cameras within {REALSENSE_PROBE_TIMEOUT:g} s"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RealSenseListing(error=f"{type(exc).__name__}: {exc}")
    return parse_realsense_listing(proc.stdout, proc.returncode, proc.stderr)


_MUJOCO_RENDER = """\
import json
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><light pos='0 0 1'/><geom size='0.1'/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 64, 64)
    renderer.update_scene(data)
    pixels = renderer.render()
    getattr(renderer, "close", lambda: None)()
    result = {"ok": True, "detail": "rendered a %dx%d frame" % (pixels.shape[1], pixels.shape[0])}
except BaseException as exc:
    result = {"ok": False, "detail": "%s: %s" % (type(exc).__name__, exc)}
print(json.dumps(result))
"""


@dataclass(frozen=True)
class RenderCheck:
    gl: str | None  # the MUJOCO_GL it rendered with; None: MuJoCo's default
    ok: bool
    detail: str


def parse_render_result(gl: str | None, stdout: str, returncode: int, stderr: str) -> RenderCheck:
    """The render child's last JSON line, or why there is none (a crash, say)."""

    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return RenderCheck(gl, bool(payload["ok"]), str(payload.get("detail", "")))
    tail = stderr.strip().splitlines()[-1:] if stderr.strip() else []
    detail = f"MuJoCo exited with code {returncode}" + (f": {tail[0]}" if tail else "")
    return RenderCheck(gl, False, detail)


def probe_mujoco_render(gl: str | None) -> RenderCheck:
    """Render one frame in a child process with ``MUJOCO_GL=gl`` (unset if None)."""

    if not sys.executable:
        return RenderCheck(gl, False, "no Python executable to run MuJoCo in")
    env = {key: value for key, value in os.environ.items() if key != "MUJOCO_GL"}
    if gl is not None:
        env["MUJOCO_GL"] = gl
    try:
        proc = subprocess.run(
            [sys.executable, "-P", "-c", _MUJOCO_RENDER],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MUJOCO_RENDER_TIMEOUT,
            check=False,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return RenderCheck(gl, False, f"no frame within {MUJOCO_RENDER_TIMEOUT:g} s")
    except (OSError, subprocess.SubprocessError) as exc:
        return RenderCheck(gl, False, f"{type(exc).__name__}: {exc}")
    return parse_render_result(gl, proc.stdout, proc.returncode, proc.stderr)


def realsense_udev_rules_source() -> str | None:
    """The udev rules file the suite ships for RealSense cameras, or None.

    The one place that knows where that file is (through
    ``REALSENSE_RULES_FUNCTION``): the installers ask with
    ``--realsense-udev-rules``, and doctor names it in its remediation. None
    if this build has no such file.
    """

    module_name, function_name = REALSENSE_RULES_FUNCTION
    try:
        path = getattr(importlib.import_module(module_name), function_name)()
    except Exception:  # not in this build, or importing the camera code failed
        return None
    if path is None:
        return None
    try:
        found = Path(os.fspath(path))
    except TypeError:  # an importlib.resources Traversable that isn't a path
        found = Path(str(path))
    return str(found) if found.is_file() else None


def realsense_udev_rules_content_valid(content: str) -> bool:
    """Whether ``content`` grants USB access to every supported RealSense.

    Intel's upstream file uses ``SUBSYSTEMS``/``ATTRS`` with ``MODE`` and
    ``GROUP``. Distribution/vendor packages can use the singular keys and a
    ``uaccess`` tag. Udev keys are case-sensitive and every accepted rule must
    have a four-digit product id; prefixed, lower-case or incomplete tokens do
    not count.
    """

    covered: set[str] = set()
    for raw in content.splitlines():
        clauses = [re.sub(r"\s+", "", item) for item in raw.split("#", 1)[0].split(",")]
        subsystem = any(item in {'SUBSYSTEM=="usb"', 'SUBSYSTEMS=="usb"'} for item in clauses)
        vendor = any(
            item in {'ATTR{idVendor}=="8086"', 'ATTRS{idVendor}=="8086"'} for item in clauses
        )
        access = any(
            re.fullmatch(pattern, item)
            for item in clauses
            for pattern in (
                r'MODE:?="0666"',
                r'GROUP:?="plugdev"',
                r'TAG\+="uaccess"',
            )
        )
        products: set[str] = set()
        for item in clauses:
            match = re.fullmatch(
                r'ATTRS?\{idProduct\}=="([0-9A-Fa-f]{4}(?:\|[0-9A-Fa-f]{4})*)"', item
            )
            if match is not None:
                products.update(value.lower() for value in match.group(1).split("|"))
        if subsystem and vendor and access:
            covered.update(products)
    return REALSENSE_TARGET_PRODUCT_IDS <= covered


def realsense_udev_rules_valid(path: str) -> bool:
    """Whether ``path`` is a readable, meaningful supported RealSense rules file."""

    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return realsense_udev_rules_content_valid(content)


@dataclass(frozen=True)
class Probes:
    """Every way this module looks at the machine. Tests replace single fields."""

    system: Callable[[], str] = platform.system
    release: Callable[[], str] = platform.release
    machine: Callable[[], str] = platform.machine
    python_version: Callable[[], tuple[int, int, int]] = _python_version
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    home: Callable[[], PurePath] = Path.home
    which: Callable[[str], str | None] = shutil.which
    run: Callable[[Sequence[str]], str | None] = _run
    torch: Callable[[], TorchInfo | None] = probe_torch
    serial_ports: Callable[[], list[SerialPort] | None] = probe_serial_ports
    access_rw: Callable[[str], bool] = _access_rw
    device_group: Callable[[str], str | None] = _device_group
    user_groups: Callable[[], GroupMembership | None] = _user_groups
    find_library: Callable[[str], str | None] = ctypes.util.find_library
    pci_display_vendors: Callable[[], frozenset[str]] = _pci_display_vendors
    long_paths_enabled: Callable[[], bool | None] = _long_paths_enabled
    preferred_encoding: Callable[[], str] = _preferred_encoding
    utf8_mode: Callable[[], bool] = _utf8_mode
    free_threaded: Callable[[], bool] = _free_threaded
    usb_vendor_ids: Callable[[], frozenset[int]] = _usb_vendor_ids
    has_battery: Callable[[], bool] = _has_battery
    disk_free: Callable[[str], int | None] = _disk_free
    mujoco_render: Callable[[str | None], RenderCheck] = probe_mujoco_render
    dist_version: Callable[[str], str | None] = _dist_version
    module_present: Callable[[str], bool] = _module_present
    realsense_devices: Callable[[], RealSenseListing] = probe_realsense_devices
    realsense_rules: Callable[[], str | None] = realsense_udev_rules_source
    realsense_rules_valid: Callable[[str], bool] = realsense_udev_rules_valid
    exists: Callable[[str], bool] = os.path.exists
    dotenv: Callable[[], Mapping[str, str]] = _read_dotenv


# --- pure decisions ----------------------------------------------------------


def os_name(system: str) -> str:
    """``platform.system()`` -> ``windows`` | ``macos`` | ``linux`` | lowercased other."""

    lowered = system.strip().lower()
    return {"darwin": "macos", "windows": "windows", "linux": "linux"}.get(lowered, lowered)


def normalize_arch(machine: str) -> str:
    lowered = machine.strip().lower()
    if lowered in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    if lowered in {"arm64", "aarch64", "armv8l"}:
        return "arm64"
    return lowered


def data_home(os_kind: str, env: Mapping[str, str], home: PurePath) -> PurePath:
    """The per-user folder for settings (.env), data, datasets and outputs.

    ``SO101_HOME`` wins when set. Otherwise:

    - Windows: ``%LOCALAPPDATA%\\so101``
    - macOS: ``~/Library/Application Support/so101``
    - Linux and others: ``$XDG_DATA_HOME/so101``, else ``~/.local/share/so101``

    The path flavour follows ``home`` (pass a ``PureWindowsPath`` to model
    Windows on another OS).
    """

    flavour = type(home)
    override = env.get("SO101_HOME", "").strip()
    if override:
        if override == "~" or override.startswith(("~/", "~\\")):
            return flavour(str(home) + override[1:])
        return flavour(override)
    if os_kind == "windows":
        local = env.get("LOCALAPPDATA", "").strip()
        base = (
            flavour(local) if local and flavour(local).is_absolute() else home / "AppData" / "Local"
        )
        return base / APP_DIR_NAME
    if os_kind == "macos":
        return home / "Library" / "Application Support" / APP_DIR_NAME
    xdg = env.get("XDG_DATA_HOME", "").strip()
    # The XDG spec says relative values are invalid and must be ignored.
    base = flavour(xdg) if xdg and flavour(xdg).is_absolute() else home / ".local" / "share"
    return base / APP_DIR_NAME


# --- settings files ------------------------------------------------------------

ENV_FILE_NAME = ".env"
# Exit status of --set-env when the file already had another value for a key
# and --replace wasn't given (the installers then ask before replacing).
SET_ENV_KEPT = 3
_SETTING_NAME = re.compile(r"SO101_[A-Z0-9_]+")
_BARE_VALUE = re.compile(r"[A-Za-z0-9_./:@%+,=~-]*")


def settings_env_files(
    os_kind: str, env: Mapping[str, str], home: PurePath, cwd: Path | None
) -> tuple[Path, ...]:
    """The .env files the suite's settings come from, lowest priority first.

    1. ``<data folder>/.env`` (:func:`data_home`: ``SO101_HOME``, else the
       per-user folder, ``~/.local/share/so101`` on Linux): the installed
       suite's settings, which the installer writes.
    2. ``.env`` in ``cwd``: a source checkout's or a project's own. It wins.

    ``SO101_*`` variables in the environment win over both. When ``cwd`` is the
    data folder (the launcher runs there) the file is listed once. A relative
    ``SO101_HOME`` is taken from ``cwd``; ``cwd=None`` (it was deleted) lists
    only the data folder's file.
    """

    folder = Path(data_home(os_kind, env, home))
    if not folder.is_absolute() and cwd is not None:
        folder = cwd / folder
    files = [folder / ENV_FILE_NAME]
    if cwd is not None:
        local = cwd / ENV_FILE_NAME
        if os.path.realpath(local) != os.path.realpath(files[0]):
            files.append(local)
    return tuple(files)


def current_env_files() -> tuple[Path, ...]:
    """:func:`settings_env_files` for this process: its OS, environment and folder."""

    try:
        cwd: Path | None = Path.cwd()
    except OSError:  # the working directory was deleted
        cwd = None
    return settings_env_files(os_name(platform.system()), os.environ, Path.home(), cwd)


def current_data_home() -> Path:
    """The data folder as an absolute path (not created here)."""

    return Path(data_home(os_name(platform.system()), os.environ, Path.home())).absolute()


def storage_settings(home: Path) -> dict[str, str]:
    """The absolute storage folders inside the data folder ``home``, as settings.

    The installer and the launcher put these in ``<home>/.env`` so that
    recordings, datasets and checkpoints stay in the data folder whatever
    folder a command runs in (Settings' own defaults are relative to it).
    """

    return {
        "SO101_DATA_DIR": str(home / "data"),
        "SO101_DATASETS_DIR": str(home / "datasets"),
        "SO101_OUTPUTS_DIR": str(home / "outputs"),
    }


def _dotenv_value(value: str) -> str:
    """``value`` as a .env value that python-dotenv and :func:`parse_dotenv` read back."""

    if _BARE_VALUE.fullmatch(value):
        return value
    if "'" not in value and "\\" not in value:
        return f"'{value}'"  # spaces (macOS "Application Support") or '#'
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class EnvFileUpdate:
    """What :func:`update_env_text` did to a .env file's text."""

    text: str
    created: bool
    added: tuple[tuple[str, str], ...] = ()
    replaced: tuple[tuple[str, str, str], ...] = ()  # (key, old value, new value)
    kept: tuple[tuple[str, str, str], ...] = ()  # (key, value kept, value wanted)
    # The mode the file had, like "0644", when others could read it and
    # update_env_file made it owner-only (0600). Not a change to the text.
    tightened: str | None = None

    @property
    def changed(self) -> bool:
        return self.created or bool(self.added or self.replaced)


ENV_FILE_HEADER = (
    "# SO-101 suite settings (SO101_*). Every `so101` command and the launcher read\n"
    "# this file from any folder. A .env in the folder a command runs in, then the\n"
    "# environment, override it. .env.example in the source lists every setting.\n"
)


def update_env_text(
    text: str | None, values: Mapping[str, str], *, replace: bool = False
) -> EnvFileUpdate:
    """Add ``values`` to a .env file's ``text`` (``None``: the file doesn't exist yet).

    A key the text doesn't set is appended. A key it sets to the same value is
    left alone. A key it sets to another value is kept unless ``replace``, which
    rewrites every line that sets it (python-dotenv uses the last one). Keys
    match case-insensitively, as pydantic-settings matches them; nothing else in
    the text changes.
    """

    for key, value in values.items():
        if not _SETTING_NAME.fullmatch(key):
            raise ValueError(f"not a suite setting name: {key!r} (SO101_ and capitals)")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key}: a value can't span lines")
    created = text is None
    lines = (ENV_FILE_HEADER if text is None else text).splitlines()
    existing = {key.lower(): value for key, value in parse_dotenv("\n".join(lines)).items()}
    added: list[tuple[str, str]] = []
    replaced: list[tuple[str, str, str]] = []
    kept: list[tuple[str, str, str]] = []
    for key, value in values.items():
        old = existing.get(key.lower())
        if old is None:
            added.append((key, value))
        elif old == value:
            continue
        elif replace:
            replaced.append((key, old, value))
            for index, line in enumerate(lines):
                match = _DOTENV_LINE.fullmatch(line.strip())
                if match is not None and match.group(1).lower() == key.lower():
                    lines[index] = f"{key}={_dotenv_value(value)}"
        else:
            kept.append((key, old, value))
    if added:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(f"{key}={_dotenv_value(value)}" for key, value in added)
    new_text = "\n".join(lines) + "\n" if lines else ""
    return EnvFileUpdate(
        text=new_text if created or added or replaced else (text or ""),
        created=created,
        added=tuple(added),
        replaced=tuple(replaced),
        kept=tuple(kept),
    )


PRIVATE_FILE_MODE = 0o600  # .env files can hold SO101_HF_TOKEN and SO101_OPENAI_API_KEY


def write_private_text(path: Path, text: str) -> None:
    """Replace the file at ``path`` with ``text``, atomically and owner-only (0600).

    The text goes to a temporary file in the same folder, created with mode
    0600, which is flushed and fsynced and then renamed over ``path``
    (os.replace). A crash or a full disk leaves the old file or the new one,
    never a truncated one. On POSIX the folder is fsynced too, so the rename
    survives a power cut. A symlinked ``path`` stays a symlink: its target is
    replaced. Windows has no POSIX modes; the file gets the folder's ACL there.
    """

    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        if os.name == "posix":
            os.fchmod(fd, PRIVATE_FILE_MODE)  # mkstemp's 0600, whatever the umask
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    if os.name == "posix":
        # Best effort: the new file is already in place, and a few file
        # systems refuse fsync on a folder.
        with contextlib.suppress(OSError):
            folder = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)


def _readable_by_others(path: Path) -> int | None:
    """``path``'s permission bits if its group or others have any access (POSIX), else None."""

    if os.name != "posix":
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    return mode if mode & 0o077 else None


def update_env_file(
    path: Path, values: Mapping[str, str], *, replace: bool = False
) -> EnvFileUpdate:
    """:func:`update_env_text` on the file at ``path``; writes only if something changed.

    The file can hold tokens and keys, so it is written with
    :func:`write_private_text` (atomic, mode 0600), and an existing file that
    its group or others can read is made owner-only even when its text
    doesn't change (``tightened`` on the result says so).
    """

    try:
        text: str | None = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = None
    update = update_env_text(text, values, replace=replace)
    loose = None if text is None else _readable_by_others(path)
    if update.changed:
        write_private_text(path, update.text)
    elif loose is not None:
        os.chmod(path, PRIVATE_FILE_MODE)
    if loose is not None:
        update = replace_fields(update, tightened=f"{loose:04o}")
    return update


def describe_env_update(path: Path, update: EnvFileUpdate) -> list[str]:
    """The edit, line by line, as the installers show it."""

    state = "created" if update.created else "updated" if update.changed else "unchanged"
    out = [f"settings file: {path} ({state})"]
    out += [f"  + {key}={value}" for key, value in update.added]
    out += [f"  ~ {key}={new} (was {old})" for key, old, new in update.replaced]
    out += [
        f"  = {key}={old} kept; this install would set {new} (--replace changes it)"
        for key, old, new in update.kept
    ]
    if update.tightened is not None:
        out.append(
            f"  permissions: was {update.tightened} (others could read it); now "
            f"{PRIVATE_FILE_MODE:04o}, owner only"
        )
    return out


def parse_nvidia_smi(output: str | None) -> NvidiaSmi | None:
    if not output:
        return None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0] or not re.match(r"^\d+(\.\d+)*$", parts[1]):
            continue
        memory = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        return NvidiaSmi(name=parts[0], driver=parts[1], memory_mib=memory)
    return None


def driver_tuple(driver: str) -> tuple[int, int, int] | None:
    """``"570.124.06"`` -> ``(570, 124, 6)``; missing fields count as 0."""

    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", driver.strip())
    if not match:
        return None
    major, minor, patch = (int(part or 0) for part in match.groups())
    return major, minor, patch


def _floor(version: str) -> tuple[int, int, int]:
    parsed = driver_tuple(version)
    if parsed is None:
        raise ValueError(f"not a driver version: {version!r}")
    return parsed


def cuda_index_for_driver(driver: str | None, os_kind: str) -> str | None:
    """The PyTorch CUDA index (``cu128``/``cu126``) the installers pick, or None.

    cu128 when NVIDIA lists this driver for CUDA 12.8.1; otherwise cu126 on any
    R525+ driver (below 12.6.3's driver it runs in minor-version compatibility
    mode, and the installers say so). CUDA 13.0 is skipped on purpose: 12.8
    already covers every GPU the suite targets and runs on more drivers.
    """

    have = driver_tuple(driver) if driver else None
    floors = MIN_DRIVER_FOR_CUDA.get(os_kind)
    if have is None or floors is None:
        return None
    if have >= _floor(floors["12.8"]):
        return "cu128"
    if have >= _floor(MIN_DRIVER_FOR_CUDA_MAJOR["12"]):
        return "cu126"
    return None


def driver_problem(cuda_build: str | None, driver: str | None, os_kind: str) -> str | None:
    """Why this driver may be too old for this torch CUDA build, or None."""

    if not cuda_build or not driver:
        return None
    major_minor = ".".join(cuda_build.split(".")[:2])
    floor = MIN_DRIVER_FOR_CUDA.get(os_kind, {}).get(major_minor)
    have = driver_tuple(driver)
    if floor is None or have is None or have >= _floor(floor):
        return None
    return (
        f"torch is built for CUDA {major_minor}, and NVIDIA lists driver {floor} or "
        f"newer for it; this machine has {driver}."
    )


@dataclass(frozen=True)
class Accelerator:
    kind: str  # "cuda" | "mps" | "cpu"
    device: str | None = None
    vram_gib: float | None = None
    torch_version: str | None = None
    torch_cuda: str | None = None
    error: str | None = None


def accelerator_from(torch_info: TorchInfo | None) -> Accelerator:
    if torch_info is None:
        return Accelerator(kind="cpu", error="torch is not installed")
    if torch_info.error:
        return Accelerator(kind="cpu", torch_version=torch_info.version, error=torch_info.error)
    if torch_info.cuda_available:
        return Accelerator(
            kind="cuda",
            device=torch_info.device_name,
            vram_gib=torch_info.vram_gib,
            torch_version=torch_info.version,
            torch_cuda=torch_info.cuda_build,
        )
    kind = "mps" if torch_info.mps_available else "cpu"
    return Accelerator(
        kind=kind, torch_version=torch_info.version, torch_cuda=torch_info.cuda_build
    )


@dataclass(frozen=True)
class GpuInventory:
    nvidia_smi_installed: bool = False
    nvidia: NvidiaSmi | None = None  # None: nvidia-smi missing or failed
    vendors: frozenset[str] = frozenset()  # display controllers seen on the PCI bus
    gpu_mode: str | None = None  # supergfxctl mode on ASUS Linux laptops


def gpu_hint(os_kind: str, accel: Accelerator, gpus: GpuInventory) -> str | None:
    """Why a GPU that seems to be there isn't being used, and what to do."""

    if accel.kind in {"cuda", "mps"} or os_kind == "macos":
        return None
    if accel.error and "torch is not installed" not in accel.error:
        if os_kind == "windows" and ("DLL" in accel.error or "WinError 126" in accel.error):
            return (
                "torch failed to load a DLL. Install the latest Microsoft Visual C++ "
                "Redistributable (x64), then restart the suite."
            )
        return None
    if gpus.gpu_mode and gpus.gpu_mode.strip().lower() == "integrated":
        return (
            "supergfxctl is in Integrated mode, which turns the NVIDIA GPU off. Run "
            "`supergfxctl -m Hybrid`, log out and back in, then start the suite again."
        )
    nvidia_seen = gpus.nvidia is not None or "nvidia" in gpus.vendors
    cpu_build = accel.torch_version is not None and accel.torch_cuda is None
    if gpus.nvidia is not None:
        if cpu_build:
            return (
                f"{gpus.nvidia.name} is present, but torch {accel.torch_version} is a "
                "CPU-only build. Re-run the installer with `--profile cuda`."
            )
        problem = driver_problem(accel.torch_cuda, gpus.nvidia.driver, os_kind)
        if problem:
            return f"{problem} Update the NVIDIA driver, or re-run the installer."
        if accel.torch_version is not None:
            return (
                f"nvidia-smi sees {gpus.nvidia.name} (driver {gpus.nvidia.driver}), but "
                f"torch {accel.torch_version} cannot use it. Restart the computer after a "
                "driver update; if it persists, re-run the installer with `--profile cuda`."
            )
        return None
    if gpus.nvidia_smi_installed:
        if os_kind == "windows":
            return (
                "The NVIDIA driver is installed but no NVIDIA GPU answers (nvidia-smi failed). "
                "On an ASUS laptop, set the GPU mode in Armoury Crate (or G-Helper) to "
                "Standard or Ultimate, not Eco, then start the suite again."
            )
        return (
            "The NVIDIA driver is installed but not responding (nvidia-smi failed). Check "
            "that the nvidia kernel module loaded (Secure Boot can block it) and, on an "
            "ASUS laptop, that `supergfxctl -g` says Hybrid."
        )
    if nvidia_seen:
        return (
            "An NVIDIA GPU is present, but its driver isn't installed or loaded. Install "
            "the NVIDIA driver, reboot, then re-run the installer."
        )
    if accel.torch_cuda:
        return (
            f"torch is a CUDA {accel.torch_cuda} build, but no NVIDIA GPU is visible. If "
            "this machine has none, re-run the installer with `--profile cpu` (a much "
            "smaller download)."
        )
    return None


def gpu_note(os_kind: str, accel: Accelerator, gpus: GpuInventory) -> str | None:
    """Expected, nothing to fix: why torch uses the CPU on an AMD-only machine."""

    if os_kind == "macos" or accel.kind != "cpu" or accel.torch_cuda:
        return None
    if gpus.nvidia is not None or gpus.nvidia_smi_installed or "nvidia" in gpus.vendors:
        return None
    if "amd" not in gpus.vendors:
        return None
    return (
        "AMD GPU detected: PyTorch has no supported GPU path for it, so torch runs on the "
        "CPU. Operating the arm, recording and simulation are fine; train on a machine with "
        "an NVIDIA GPU (training on the CPU is slow). See docs/install.md."
    )


@dataclass(frozen=True)
class MujocoGL:
    current: str | None  # MUJOCO_GL as set, None if unset
    recommended: str | None  # None: leave it unset
    reason: str
    problem: str | None = None  # set when the current value breaks `import mujoco`


def mujoco_gl_advice(
    os_kind: str,
    env: Mapping[str, str],
    find_library: Callable[[str], str | None],
    nvidia: bool = False,
) -> MujocoGL:
    """What MUJOCO_GL should be. ``nvidia``: Linux with the NVIDIA kernel module loaded."""

    raw = env.get("MUJOCO_GL")
    current = raw.strip().lower() if raw is not None else None
    problem = None
    if current is not None and current not in _MUJOCO_GL_OFF:
        valid = _MUJOCO_GL_ANY_OS | _MUJOCO_GL_BY_OS.get(os_kind, frozenset())
        if current not in valid:
            allowed = ", ".join(sorted(v for v in valid if v)) or "none"
            problem = (
                f"MUJOCO_GL={raw} is not valid on {os_kind}: `import mujoco` raises "
                f"RuntimeError. Unset it (valid here: {allowed})."
            )
    if os_kind == "macos":
        return MujocoGL(current, None, "leave unset: MuJoCo uses CGL, no window needed", problem)
    if os_kind == "windows":
        return MujocoGL(
            current,
            None,
            "leave unset: MuJoCo renders through a hidden GLFW window (Windows has no "
            "EGL or OSMesa backend)",
            problem,
        )
    if nvidia and find_library("EGL"):
        return MujocoGL(
            current,
            "egl",
            "NVIDIA GPU: egl renders offscreen on it, with or without a display "
            "(MUJOCO_EGL_DEVICE_ID picks a GPU)",
            problem,
        )
    if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
        return MujocoGL(
            current, None, "leave unset: MuJoCo uses a hidden GLFW window on this display", problem
        )
    if find_library("EGL"):
        return MujocoGL(
            current,
            "egl",
            "no display: egl renders offscreen on the GPU (MUJOCO_EGL_DEVICE_ID picks one)",
            problem,
        )
    if find_library("OSMesa"):
        return MujocoGL(
            current, "osmesa", "no display and no EGL: osmesa renders on the CPU (slow)", problem
        )
    return MujocoGL(
        current,
        "disable",
        "no display and no EGL or OSMesa library, so camera rendering is off; install "
        "libegl1 (GPU) or libosmesa6 (CPU) to turn it on",
        problem,
    )


@dataclass(frozen=True)
class SerialAccess:
    ok: bool | None  # None: not applicable here, or no USB serial port to judge
    detail: str
    group: str | None = None
    remediation: str = ""


def serial_access(
    os_kind: str,
    ports: Sequence[SerialPort] | None,
    access_rw: Callable[[str], bool],
    device_group: Callable[[str], str | None],
    groups: GroupMembership | None,
) -> SerialAccess:
    """Can this user open the USB serial ports? Only Linux restricts it by group."""

    if os_kind != "linux":
        return SerialAccess(None, "no group needed to open serial ports on " + os_kind)
    usb = [p for p in ports or [] if p.is_usb]
    if not usb:
        return SerialAccess(None, "no USB serial port to check")
    blocked = [p for p in usb if not access_rw(p.device)]
    if not blocked:
        return SerialAccess(True, "this user can open " + ", ".join(p.device for p in usb))
    group = device_group(blocked[0].device) or "dialout"
    names = ", ".join(p.device for p in blocked)
    if groups is not None and group in groups.configured and group not in groups.session:
        return SerialAccess(
            False,
            f"you are in the '{group}' group, but this login session started before that",
            group,
            "Log out and back in (or reboot), then start the suite again.",
        )
    return SerialAccess(
        False,
        f"this user can't open {names} (owned by group '{group}')",
        group,
        f"Run `sudo usermod -aG {group} $USER`, then log out and back in.",
    )


_DOTENV_LINE = re.compile(r"(?:export[ \t]+)?([^=#\s]+)[ \t]*=[ \t]*(.*)")
_DOTENV_QUOTED = re.compile(r"""'((?:\\.|[^'\\])*)'|"((?:\\.|[^"\\])*)\"""")


def parse_dotenv(text: str) -> dict[str, str]:
    """The ``KEY=value`` lines of a .env file, read the way python-dotenv reads them.

    pydantic-settings reads .env through python-dotenv; this follows its rules
    for ``export``, comments and quotes (python-dotenv 1.x ``parser.py``), but
    doesn't decode escapes, join multi-line values or expand ``${VAR}``. A line
    it can't read is skipped, as python-dotenv skips it.
    """

    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _DOTENV_LINE.fullmatch(line.strip())
        if match is None:
            continue
        key, value = match.groups()
        if value[:1] in ("'", '"'):
            quoted = _DOTENV_QUOTED.match(value)
            if quoted is None:  # no closing quote
                continue
            value = quoted.group(1) if quoted.group(1) is not None else quoted.group(2)
        else:
            value = re.sub(r"\s+#.*", "", value).rstrip()
        values[key] = value
    return values


def _setting(values: Mapping[str, str], name: str) -> str | None:
    """A setting by name, matched case-insensitively as pydantic-settings does."""

    for key, value in values.items():
        if key.lower() == name:
            return value
    return None


def env_setting(
    env: Mapping[str, str], dotenv: Callable[[], Mapping[str, str]], name: str, default: str
) -> str:
    """A setting as Settings reads it: the environment, then .env, then ``default``."""

    found = _setting(env, name)
    if found is None:
        found = _setting(dotenv(), name)
    return default if found is None else found


def camera_setting(env: Mapping[str, str], dotenv: Callable[[], Mapping[str, str]]) -> str:
    """SO101_CAMERAS as Settings reads it: the environment, then .env, then "0"."""

    return env_setting(env, dotenv, "so101_cameras", "0")  # Settings.cameras defaults to "0"


POLICY_DEVICES = ("auto", "cpu", "cuda", "mps")


def policy_device_problem(setting: str, accel: Accelerator) -> str | None:
    """Why SO101_POLICY_DEVICE would stop a policy from loading, or None.

    Follows ``so101.policy.server.select_torch_device``: empty or ``auto`` picks
    cuda, then mps, then cpu; anything else must be a device torch can use here.
    """

    choice = setting.strip().lower() or "auto"
    base = choice.split(":", 1)[0]
    if base not in POLICY_DEVICES or (":" in choice and base != "cuda"):
        return (
            f"SO101_POLICY_DEVICE={setting.strip()!r} is not a device; use auto, cpu, cuda, "
            "cuda:<index> or mps"
        )
    if choice == "auto" or base == "cpu" or accel.torch_version is None or accel.error:
        return None  # nothing to check, or the torch check already says what's wrong
    if base != accel.kind:
        return (
            f"SO101_POLICY_DEVICE={choice}, but torch can only use {accel.kind} here, "
            "so policies fail to load"
        )
    return None


_CAMERA_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def realsense_configured(cameras: str) -> bool:
    """True if an SO101_CAMERAS entry's source starts with ``realsense``.

    Entries are comma-separated, and each may be ``name=source``. This is a
    plain string check, so the report doesn't depend on the camera code.
    """

    for entry in cameras.split(","):
        name, sep, source = entry.partition("=")
        if not (sep and _CAMERA_NAME.fullmatch(name.strip())):
            source = entry  # no name: "=" belongs to the source, as in a URL query
        if source.strip().lower().startswith("realsense"):
            return True
    return False


def realsense_distribution(os_kind: str) -> str:
    """The distribution that provides ``pyrealsense2`` on this OS."""

    return REALSENSE_DIST_MACOS if os_kind == "macos" else REALSENSE_DIST


def realsense_wheel_problem(
    os_kind: str, arch: str, python: tuple[int, int], darwin_release: str = ""
) -> str | None:
    """Why pip finds no pyrealsense2 wheel for this machine and Python, or None."""

    version = f"{python[0]}.{python[1]}"
    if os_kind == "macos":
        darwin = re.match(r"\d+", darwin_release.strip())
        # Darwin 24 is macOS 15. Older macOS has only 2.54.2, built up to Python 3.12.
        if darwin and int(darwin.group()) < 24 and python > (3, 12):
            return (
                "pyrealsense2-macosx 2.56.5 needs macOS 15 or newer, and the build for "
                f"older macOS (2.54.2) stops at Python 3.12; this is Python {version}. "
                "Re-run the installer with --fresh --realsense: it uses Python 3.12."
            )
        return None
    if os_kind not in {"windows", "linux"}:
        return None
    if arch != "x86_64":
        return (
            f"pyrealsense2 2.56 has no {os_kind} {arch} wheel for Python 3.12 or 3.13 (its "
            "wheels are for x86-64). Build librealsense with its Python bindings from source."
        )
    if python > REALSENSE_PYTHON_MAX:
        return (
            f"pyrealsense2 2.56 has wheels for Python 3.12 and 3.13, not {version}. Re-run "
            "the installer with --fresh --realsense (on Windows: -Fresh -RealSense); it "
            "uses Python 3.12."
        )
    return None


def _text(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


def parse_realsense_listing(stdout: str, returncode: int = 0, stderr: str = "") -> RealSenseListing:
    """What the child process printed: the last line that is a JSON object.

    The return code counts only when there is no such line: librealsense can
    crash while the interpreter exits, after the listing was printed.
    """

    data: dict[str, Any] | None = None
    for line in reversed(stdout.strip().splitlines()):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    if data is None:
        tail = (stderr.strip().splitlines() or [""])[-1].strip()
        reason = f"pyrealsense2 exited with code {returncode}"
        return RealSenseListing(error=f"{reason}: {tail}" if tail else reason)
    items = data.get("devices")
    devices = tuple(
        RealSenseDevice(
            name=_text(item.get("name")) or "RealSense camera",
            serial=_text(item.get("serial_number")),
            firmware=_text(item.get("firmware_version")),
            usb_type=_text(item.get("usb_type_descriptor")),
        )
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict)
    )
    return RealSenseListing(devices=devices, error=_text(data.get("error")))


@dataclass(frozen=True)
class RealSenseInfo:
    distribution: str  # pyrealsense2, or pyrealsense2-macosx on macOS
    installed: bool  # the pyrealsense2 module is there, with or without package metadata
    version: str | None  # the distribution's version; None if not installed or no metadata
    configured: bool  # an SO101_CAMERAS entry is a RealSense camera
    devices: tuple[RealSenseDevice, ...] = ()
    error: str | None = None  # listing the cameras failed
    udev_rules: str | None = None  # Linux: the installed rules file, None if there is none
    rules_source: str | None = None  # Linux, no rules installed: the file the suite ships
    wheel_problem: str | None = None  # not installed: why pip would find no wheel here


@dataclass(frozen=True)
class LinuxSetup:
    arm_rules: str | None  # the installed arm udev rules file, None if there is none
    arm_links: tuple[str, ...]  # the /dev/so101-* names present now
    brltty_conflict: bool  # brltty is installed and a CH34x adapter has no serial port
    laptop: bool  # a battery is present
    suspend_on_ac: str | None  # GNOME's sleep-inactive-ac-type ('nothing': never); None: unknown


def _linux_setup(p: Probes, ports: tuple[SerialPort, ...] | None) -> LinuxSetup:
    conflict = False
    if ports is not None and any(p.exists(path) for path in BRLTTY_PATHS):
        if CH34X_VENDOR in p.usb_vendor_ids():
            conflict = not any(port.vid == CH34X_VENDOR for port in ports)
    laptop = p.has_battery()
    suspend = None
    if laptop:
        value = p.run(["gsettings", "get", *GNOME_SUSPEND_ON_AC])
        suspend = value.strip().strip("'") or None if value else None
    return LinuxSetup(
        arm_rules=ARM_UDEV_RULES if p.exists(ARM_UDEV_RULES) else None,
        arm_links=tuple(link for link in ARM_LINKS if p.exists(link)),
        brltty_conflict=conflict,
        laptop=laptop,
        suspend_on_ac=suspend,
    )


def suite_mujoco_gl(gl: MujocoGL) -> str | None:
    """The MUJOCO_GL the suite runs with: a valid one that is set, else the launcher's pick."""

    if gl.current is not None and not gl.problem:
        return gl.current
    return gl.recommended


@dataclass(frozen=True)
class PlatformReport:
    os: str
    os_release: str
    arch: str
    machine: str
    python: str
    python_ok: bool
    free_threaded: bool
    rosetta: bool
    accelerator: Accelerator
    gpus: GpuInventory
    gpu_hint: str | None  # something to fix
    gpu_note: str | None  # expected, nothing to fix
    mujoco_gl: MujocoGL
    serial_ports: tuple[SerialPort, ...] | None  # None: pyserial is not installed
    serial_access: SerialAccess
    data_dir: str
    long_paths: bool | None
    encoding: str
    utf8_mode: bool
    realsense: RealSenseInfo
    policy_device: str = ""  # SO101_POLICY_DEVICE as set; empty when unset
    render: RenderCheck | None = None  # None: MuJoCo isn't installed, or rendering is off
    disk_free_gib: float | None = None  # free space for the data folder
    linux: LinuxSetup | None = None  # Linux only
    warnings: tuple[str, ...] = ()
    device_state_path: str = ""
    device_state_problem: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["gpus"]["vendors"] = sorted(self.gpus.vendors)
        if self.serial_ports is not None:
            out["serial_ports"] = [
                {**asdict(p), "chip": p.chip, "usb": p.is_usb} for p in self.serial_ports
            ]
        out["realsense"]["devices"] = [
            {**asdict(d), "usb2": d.usb2} for d in self.realsense.devices
        ]
        return out


def _gpu_inventory(os_kind: str, probes: Probes) -> GpuInventory:
    if os_kind == "macos":
        return GpuInventory(vendors=frozenset({"apple"}))
    smi = probes.which("nvidia-smi")
    nvidia = None
    if smi:
        nvidia = parse_nvidia_smi(
            probes.run(
                [
                    smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
    vendors: frozenset[str] = frozenset()
    mode = None
    if os_kind == "linux":
        vendors = probes.pci_display_vendors()
        supergfx = probes.which("supergfxctl")
        if supergfx:
            out = probes.run([supergfx, "-g"])
            mode = out.strip().splitlines()[0].strip() if out and out.strip() else None
    elif os_kind == "windows":
        out = probes.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ]
        )
        names = (out or "").lower()
        vendors = frozenset(
            name
            for name, needles in {
                "nvidia": ("nvidia", "geforce", "rtx"),
                "amd": ("amd", "radeon"),
                "intel": ("intel",),
            }.items()
            if any(n in names for n in needles)
        )
    return GpuInventory(
        nvidia_smi_installed=bool(smi), nvidia=nvidia, vendors=vendors, gpu_mode=mode
    )


def _realsense_info(
    os_kind: str, arch: str, python: tuple[int, int], darwin_release: str, probes: Probes
) -> RealSenseInfo:
    expected = realsense_distribution(os_kind)
    distribution, version = expected, None
    # Either distribution provides the module; macOS can have a self-built pyrealsense2.
    for candidate in dict.fromkeys((expected, REALSENSE_DIST)):
        found = probes.dist_version(candidate)
        if found:
            distribution, version = candidate, found
            break
    # Built from source (the only way on Linux arm64), the module has no metadata.
    installed = version is not None or probes.module_present("pyrealsense2")
    configured = realsense_configured(camera_setting(probes.env, probes.dotenv))
    listing = probes.realsense_devices() if installed else RealSenseListing()
    udev_rules = rules_source = None
    if os_kind == "linux" and (installed or configured):
        candidates = (REALSENSE_UDEV_RULES, *REALSENSE_UDEV_RULES_SYSTEM)
        udev_rules = next(
            (
                path
                for path in candidates
                if probes.exists(path) and probes.realsense_rules_valid(path)
            ),
            None,
        )
        if udev_rules is None:
            rules_source = probes.realsense_rules()
    return RealSenseInfo(
        distribution=distribution,
        installed=installed,
        version=version,
        configured=configured,
        devices=listing.devices,
        error=listing.error,
        udev_rules=udev_rules,
        rules_source=rules_source,
        wheel_problem=None
        if installed
        else realsense_wheel_problem(os_kind, arch, python, darwin_release),
    )


def _device_state_status(os_kind: str, env: Mapping[str, str]) -> tuple[str, str | None]:
    if os_kind == "windows":
        local = str(env.get("LOCALAPPDATA", "") or "").strip()
        if not local:
            return "", "LOCALAPPDATA is unavailable"
        root = Path(local) / "SO101" / "device-ownership-v2"
    elif os_kind == "macos":
        root = Path("/Library/Application Support/SO101/device-ownership-v2")
    elif os_kind == "linux":
        root = Path("/var/lib/so101/device-ownership-v2")
    else:
        return "", None
    required = (root, root / "locks", root / "intent-events", root / "intent-events/aliases")
    for path in required:
        try:
            info = os.lstat(path)
        except OSError as exc:
            return str(root), f"{path} is unavailable ({exc})"
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            return str(root), f"{path} is a symlink or reparse point"
        if not stat.S_ISDIR(info.st_mode):
            return str(root), f"{path} is not a directory"
        if os_kind != "windows":
            mode = stat.S_IMODE(info.st_mode)
            if info.st_uid != 0 or mode != 0o3770:
                return str(root), f"{path} is not root-owned mode 3770"
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            return str(root), f"this user cannot read/write {path}"
    return str(root), None


def _real_hardware_configured(
    env: Mapping[str, str], dotenv: Callable[[], Mapping[str, str]]
) -> bool:
    backend = env_setting(env, dotenv, "so101_backend", "").strip().lower()
    mock_value = env_setting(env, dotenv, "so101_mock", "").strip().lower()
    mock_enabled = mock_value in {"1", "true", "yes", "on"}
    if backend in {"mock", "sim"}:
        return False
    return bool(backend) or not mock_enabled


def collect(probes: Probes | None = None) -> PlatformReport:
    """Look at the machine once and return everything the suite needs to know."""

    live_machine = probes is None
    p = probes or Probes()
    kind = os_name(p.system())
    machine = p.machine()
    arch = normalize_arch(machine)
    release = p.release()
    major, minor, micro = p.python_version()
    free_threaded = p.free_threaded()
    python_ok = MIN_PYTHON <= (major, minor) <= MAX_PYTHON and not free_threaded
    warnings: list[str] = []

    rosetta = False
    if kind == "macos" and arch == "x86_64":
        rosetta = (p.run(["sysctl", "-n", "sysctl.proc_translated"]) or "").strip() == "1"
        if rosetta:
            warnings.append(
                "This Python is an x86_64 build running under Rosetta. Install an arm64 "
                "Python 3.13 so torch can use the Apple GPU (MPS)."
            )
        else:
            warnings.append(
                "Intel Macs are not supported: torch 2.7+ (LeRobot's minimum) and MuJoCo "
                "3.14 publish no macOS x86_64 wheels."
            )
    if (major, minor) < MIN_PYTHON:
        warnings.append(
            f"Python {major}.{minor} is too old: LeRobot 0.6.1 needs Python "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer."
        )
    elif (major, minor) > MAX_PYTHON:
        warnings.append(
            f"Python {major}.{minor} is newer than the suite supports: it needs Python "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} to {MAX_PYTHON[0]}.{MAX_PYTHON[1]}."
        )
    if free_threaded:
        warnings.append(
            f"This Python {major}.{minor} is a free-threaded build: torchcodec, PyAV, MuJoCo "
            "and OpenCV publish no free-threaded wheels. Use the standard build."
        )

    accel = accelerator_from(p.torch())
    gpus = _gpu_inventory(kind, p)
    hint = gpu_hint(kind, accel, gpus)
    note = None if hint else gpu_note(kind, accel, gpus)

    ports_list = p.serial_ports()
    ports = tuple(ports_list) if ports_list is not None else None
    access = serial_access(
        kind,
        ports,
        p.access_rw,
        p.device_group,
        p.user_groups() if kind == "linux" else None,
    )

    gl = mujoco_gl_advice(
        kind, p.env, p.find_library, nvidia=kind == "linux" and p.exists(NVIDIA_PROC)
    )
    render = None
    if p.dist_version("mujoco"):
        render_gl = suite_mujoco_gl(gl)
        if render_gl not in _MUJOCO_GL_OFF:
            render = p.mujoco_render(render_gl)
    data_dir = str(data_home(kind, p.env, p.home()))
    free = p.disk_free(data_dir)

    long_paths = p.long_paths_enabled() if kind == "windows" else None
    if long_paths is False:
        warnings.append(
            "Windows long paths are off, so paths over 260 characters (deep dataset and "
            "cache folders) fail. Turn them on: see docs/install.md."
        )
    encoding = p.preferred_encoding()
    utf8 = p.utf8_mode()
    if kind == "windows" and not utf8 and encoding.lower().replace("-", "") != "utf8":
        warnings.append(
            f"Python's default text encoding is {encoding}, not UTF-8. The suite's "
            "launcher turns UTF-8 mode on; when running `so101` by hand, set PYTHONUTF8=1."
        )

    state_path, state_problem = (
        _device_state_status(kind, p.env)
        if live_machine and _real_hardware_configured(p.env, p.dotenv)
        else ("", None)
    )
    return PlatformReport(
        os=kind,
        os_release=release,
        arch=arch,
        machine=machine,
        python=f"{major}.{minor}.{micro}",
        python_ok=python_ok,
        free_threaded=free_threaded,
        rosetta=rosetta,
        accelerator=accel,
        gpus=gpus,
        gpu_hint=hint,
        gpu_note=note,
        mujoco_gl=gl,
        serial_ports=ports,
        serial_access=access,
        data_dir=data_dir,
        long_paths=long_paths,
        encoding=encoding,
        utf8_mode=utf8,
        realsense=_realsense_info(kind, arch, (major, minor), release, p),
        policy_device=env_setting(p.env, p.dotenv, "so101_policy_device", ""),
        render=render,
        disk_free_gib=None if free is None else round(free / 2**30, 1),
        linux=_linux_setup(p, ports) if kind == "linux" else None,
        warnings=tuple(warnings),
        device_state_path=state_path,
        device_state_problem=state_problem,
    )


# --- presentation ------------------------------------------------------------


@dataclass(frozen=True)
class PlatformCheck:
    """Same fields as ``so101.services.schemas.DiagnosticCheck``, so doctor can show these."""

    id: str
    label: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    remediation: str = ""


def describe_accelerator(accel: Accelerator) -> str:
    if accel.kind == "cuda":
        vram = f", {accel.vram_gib:g} GiB" if accel.vram_gib is not None else ""
        return f"CUDA: {accel.device}{vram} (torch {accel.torch_version})"
    if accel.kind == "mps":
        return f"Apple GPU (MPS) (torch {accel.torch_version})"
    if accel.torch_version is None:
        return "CPU (torch is not installed)"
    return f"CPU (torch {accel.torch_version})"


def describe_port(port: SerialPort) -> str:
    bits = [port.device]
    if port.chip:
        bits.append(port.chip)
    elif port.description:
        bits.append(port.description)
    if port.serial_number:
        bits.append(f"serial {port.serial_number}")
    return "  ".join(bits)


def _gl_line(gl: MujocoGL) -> str:
    return (f"MUJOCO_GL={gl.recommended}: " if gl.recommended else "") + gl.reason


def describe_realsense_device(device: RealSenseDevice) -> str:
    bits = [device.name]
    if device.serial:
        bits.append(f"serial {device.serial}")
    if device.firmware:
        bits.append(f"firmware {device.firmware}")
    if device.usb_type:
        bits.append(f"USB {device.usb_type}")
    return "  ".join(bits)


_UDEV_RELOAD = "sudo udevadm control --reload-rules && sudo udevadm trigger"
REALSENSE_MACOS_NOTE = (
    "On macOS librealsense usually needs sudo to open the camera; color-only capture "
    "also works without it through the camera's UVC index (a plain camera number in "
    "SO101_CAMERAS)"
)


def realsense_udev_commands(source: str | None) -> str:
    """How to install the RealSense udev rules: printed, never run (they need sudo)."""

    if source:
        copy = f"Run `sudo cp {shlex.quote(source)} {REALSENSE_UDEV_RULES}`"
    else:
        copy = (
            "This build of the suite has no copy of the rules file: copy "
            f"config/99-realsense-libusb.rules from librealsense to {REALSENSE_UDEV_RULES} "
            "with sudo"
        )
    return f"{copy}, then `{_UDEV_RELOAD}`, then unplug and replug the camera."


def _realsense_checks(os_kind: str, rs: RealSenseInfo) -> list[PlatformCheck]:
    label = "RealSense"
    if not rs.installed:
        if not rs.configured:
            return [
                PlatformCheck(
                    "platform.realsense",
                    label,
                    "pass",
                    f"{rs.distribution} is not installed (optional: no RealSense camera "
                    "in SO101_CAMERAS)",
                )
            ]
        status, fix = (
            "fail",
            rs.wheel_problem or ("Re-run the installer with --realsense (on Windows: -RealSense)."),
        )
        detail = f"SO101_CAMERAS has a RealSense camera, but {rs.distribution} is not installed"
    else:
        package = (
            f"{rs.distribution} {rs.version}"
            if rs.version
            else "pyrealsense2 (no package metadata: built from source?)"
        )
        slow = [device.name for device in rs.devices if device.usb2]
        if rs.error:
            status = "warn"
            detail = f"{package}; listing the cameras failed: {rs.error}"
            fix = (
                "Unplug and replug the camera, close any other program using it (such as "
                "RealSense Viewer), and run doctor again."
            )
        elif not rs.devices:
            status = "warn" if rs.configured else "pass"
            detail = f"{package}; no RealSense camera found"
            fix = ""
            if rs.configured:
                fix = "Plug the camera into a USB 3 port, then run doctor again."
        else:
            status = "warn" if slow else "pass"
            cameras = "; ".join(describe_realsense_device(device) for device in rs.devices)
            detail = f"{package}; {cameras}"
            fix = (
                f"{', '.join(slow)} is on a USB 2 link, which limits depth and color "
                "resolution and frame rate. Plug it into a USB 3 port with a USB 3 cable."
                if slow
                else ""
            )
    if os_kind == "macos":
        detail = f"{detail}. {REALSENSE_MACOS_NOTE}"
    out = [PlatformCheck("platform.realsense", label, status, detail, fix)]
    if os_kind == "linux":
        if rs.udev_rules:
            out.append(
                PlatformCheck(
                    "platform.realsense.udev",
                    "RealSense udev",
                    "pass",
                    f"udev rules installed: {rs.udev_rules}",
                )
            )
        else:
            out.append(
                PlatformCheck(
                    "platform.realsense.udev",
                    "RealSense udev",
                    "warn",
                    "no usable RealSense udev rules installed, so librealsense usually can't open "
                    "the camera without root",
                    realsense_udev_commands(rs.rules_source),
                )
            )
    return out


SLEEP_FIX = (
    "While the arm is powered, keep the laptop awake with the lid open: Settings > Power > "
    "Automatic Suspend off, or run `gsettings set org.gnome.settings-daemon.plugins.power "
    "sleep-inactive-ac-type 'nothing'` (sleep-inactive-battery-type on battery)."
)


def _render_fix(os_kind: str) -> str:
    if os_kind == "linux":
        return (
            "Install the GL libraries (sudo apt install libegl1 libgl1). With the NVIDIA "
            "driver loaded set MUJOCO_GL=egl; without a GPU, MUJOCO_GL=osmesa (sudo apt "
            "install libosmesa6). The simulator still runs without camera images."
        )
    return "Update the graphics driver. The simulator still runs without camera images."


def _setup_checks(report: PlatformReport) -> list[PlatformCheck]:
    """Sim rendering, disk space and, on Linux, udev names, brltty and laptop sleep."""

    out = []
    if report.device_state_path:
        problem = report.device_state_problem
        if report.os == "windows":
            fix = "Re-run install/install.ps1 to create stable Local AppData device state."
        else:
            fix = (
                "Re-run install/install.sh and approve the shared motor-device state sudo "
                "step (root, hardware group, mode 3770)."
            )
        out.append(
            PlatformCheck(
                "platform.device_ownership",
                "Motor-device ownership",
                "fail" if problem else "pass",
                problem or f"stable shared state ready: {report.device_state_path}",
                fix if problem else "",
            )
        )
    render = report.render
    if render is not None:
        gl = f"MUJOCO_GL={render.gl}" if render.gl else "MUJOCO_GL unset"
        out.append(
            PlatformCheck(
                "platform.mujoco_render",
                "Sim render",
                "pass" if render.ok else "warn",
                f"{render.detail} ({gl})"
                if render.ok
                else f"rendering failed ({gl}): {render.detail}",
                "" if render.ok else _render_fix(report.os),
            )
        )
    free = report.disk_free_gib
    if free is not None:
        status = "fail" if free < DISK_FAIL_GIB else "warn" if free < DISK_WARN_GIB else "pass"
        out.append(
            PlatformCheck(
                "platform.disk",
                "Disk space",
                status,
                f"{free:g} GB free for the data folder ({report.data_dir})",
                ""
                if status == "pass"
                else "Free some space, or move the data folder to a bigger disk (SO101_HOME).",
            )
        )
    linux = report.linux
    if linux is None:
        return out
    if report.serial_ports is not None:  # the robot extra is installed
        if linux.arm_rules:
            present = ", ".join(linux.arm_links) or "neither name yet (plug the arms in)"
            out.append(
                PlatformCheck(
                    "platform.udev.arms",
                    "Arm names",
                    "pass",
                    f"udev rules installed ({linux.arm_rules}); present: {present}",
                )
            )
        else:
            out.append(
                PlatformCheck(
                    "platform.udev.arms",
                    "Arm names",
                    "warn",
                    "no udev rules name the arm adapters, so /dev/ttyACM0 and /dev/ttyACM1 "
                    "can swap after a replug",
                    "Re-run install/install.sh with --follower-serial and --leader-serial "
                    "(each adapter's serial number is in the serial ports row).",
                )
            )
    if linux.brltty_conflict:
        out.append(
            PlatformCheck(
                "platform.brltty",
                "brltty",
                "warn",
                "brltty is installed and a CH34x USB-serial adapter (vendor 1a86) is plugged "
                "in but has no serial port: on Ubuntu, brltty claims these adapters",
                "Remove it (sudo apt remove brltty), then unplug and replug the adapter.",
            )
        )
    if linux.laptop:
        detail = (
            "A sleeping or suspended laptop can't stop the arm: the servos keep holding their "
            "last goal."
        )
        if linux.suspend_on_ac == "nothing":
            detail += " Automatic suspend on AC power is off; keep the lid open too."
        elif linux.suspend_on_ac:
            detail += f" GNOME suspends this laptop when idle on AC power ({linux.suspend_on_ac})."
        else:
            detail += " (GNOME's power settings couldn't be read.)"
        ok = linux.suspend_on_ac == "nothing"
        out.append(
            PlatformCheck(
                "platform.sleep",
                "Laptop sleep",
                "pass" if ok else "warn",
                detail,
                "" if ok else SLEEP_FIX,
            )
        )
    return out


def checks(report: PlatformReport) -> list[PlatformCheck]:
    """The report as doctor-style checks: pass, warn (works, worth a look) or fail."""

    python_fix = ""
    if report.rosetta:
        python_fix = "Install an arm64 Python 3.13, then re-run the installer."
    elif report.free_threaded:
        python_fix = (
            "Install the standard (not free-threaded) Python 3.13, then re-run the installer."
        )
    elif not report.python_ok:
        python_fix = "Install Python 3.13 (3.12 also works), then re-run the installer."
    out = [
        PlatformCheck(
            "platform.python",
            "Python",
            "fail" if python_fix else "pass",
            f"Python {report.python} on {report.os} {report.arch}",
            python_fix,
        )
    ]

    accel = report.accelerator
    if accel.torch_version is None:
        out.append(
            PlatformCheck(
                "platform.torch",
                "PyTorch",
                "warn",
                "torch is not installed, so training and policies can't run",
                "Run the installer; it installs the right torch build for this machine.",
            )
        )
    elif accel.error:
        out.append(
            PlatformCheck(
                "platform.torch",
                "PyTorch",
                "fail",
                f"torch is installed but failed to load: {accel.error}",
                report.gpu_hint or "Re-run the installer with `--fresh`.",
            )
        )
    else:
        detail = describe_accelerator(accel)
        if report.gpu_note:
            detail = f"{detail}. {report.gpu_note}"
        out.append(
            PlatformCheck(
                "platform.accelerator",
                "Accelerator",
                "warn" if report.gpu_hint else "pass",
                detail,
                report.gpu_hint or "",
            )
        )

    device = report.policy_device.strip()
    if device:  # unset is the default (automatic) and needs no row
        problem = policy_device_problem(device, accel)
        out.append(
            PlatformCheck(
                "platform.policy_device",
                "Policy device",
                "fail" if problem else "pass",
                problem or f"SO101_POLICY_DEVICE={device}",
                "Unset SO101_POLICY_DEVICE, or set it to auto, to use cuda, then mps, then cpu."
                if problem
                else "",
            )
        )

    gl = report.mujoco_gl
    gl_status = "fail" if gl.problem else ("warn" if gl.recommended == "disable" else "pass")
    out.append(
        PlatformCheck(
            "platform.mujoco_gl", "Sim rendering", gl_status, _gl_line(gl), gl.problem or ""
        )
    )

    if report.serial_ports is None:
        out.append(
            PlatformCheck(
                "platform.serial",
                "Serial ports",
                "warn",
                "pyserial is not installed, so the arm can't be found",
                "Run the installer; it installs the robot extra.",
            )
        )
    else:
        usb = [port for port in report.serial_ports if port.is_usb]
        detail = "; ".join(describe_port(port) for port in usb) or "no USB serial port found"
        access = report.serial_access
        if access.ok is False:
            detail = f"{detail}. {access.detail}"
        out.append(
            PlatformCheck(
                "platform.serial",
                "Serial ports",
                "fail" if access.ok is False else "pass",
                detail,
                access.remediation,
            )
        )

    out.extend(_realsense_checks(report.os, report.realsense))
    out.extend(_setup_checks(report))

    for i, warning in enumerate(report.warnings):
        out.append(PlatformCheck(f"platform.warning.{i}", "Platform", "warn", warning))
    out.append(PlatformCheck("platform.data_dir", "Data folder", "pass", report.data_dir))
    return out


def format_text(report: PlatformReport) -> str:
    rows = [
        ("OS", f"{report.os} {report.os_release} ({report.machine})"),
        ("Python", report.python),
        ("Accelerator", describe_accelerator(report.accelerator)),
    ]
    if report.accelerator.error and report.accelerator.torch_version is not None:
        rows.append(("torch error", report.accelerator.error))
    if report.gpu_hint:
        rows.append(("GPU hint", report.gpu_hint))
    if report.gpu_note:
        rows.append(("GPU note", report.gpu_note))
    if report.policy_device.strip():
        problem = policy_device_problem(report.policy_device, report.accelerator)
        rows.append(("Policy device", f"PROBLEM: {problem}" if problem else report.policy_device))
    rows.append(("MuJoCo GL", _gl_line(report.mujoco_gl)))
    if report.mujoco_gl.problem:
        rows.append(("MuJoCo GL", "PROBLEM: " + report.mujoco_gl.problem))
    if report.serial_ports is None:
        rows.append(("Serial ports", "pyserial is not installed"))
    else:
        usb = [port for port in report.serial_ports if port.is_usb]
        rows.append(
            ("Serial ports", "; ".join(describe_port(port) for port in usb) or "no USB port")
        )
        access = report.serial_access
        if access.ok is False:
            rows.append(("Serial access", f"{access.detail}. {access.remediation}"))
    rs = report.realsense
    if rs.installed or rs.configured:  # otherwise it's optional and absent: nothing to say
        for check in _realsense_checks(report.os, rs):
            prefix = "PROBLEM: " if check.status == "fail" else ""
            fix = f". {check.remediation}" if check.remediation else ""
            rows.append((check.label, f"{prefix}{check.detail}{fix}"))
    for check in _setup_checks(report):
        prefix = "PROBLEM: " if check.status == "fail" else ""
        fix = f". {check.remediation}" if check.remediation else ""
        rows.append((check.label, f"{prefix}{check.detail}{fix}"))
    rows.append(("Data folder", report.data_dir))
    rows.extend(("Warning", warning) for warning in report.warnings)
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label.ljust(width)}  {value}" for label, value in rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m so101.platform_info",
        description="Report this machine's OS, accelerator, sim rendering, serial ports "
        "and data folder for the SO-101 suite.",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--data-dir",
        action="store_true",
        help="print only the data folder's absolute path (for installers and scripts)",
    )
    parser.add_argument(
        "--realsense-udev-rules",
        action="store_true",
        help="print only the path of the RealSense udev rules file the suite ships "
        "(exit status 1 if this build has none)",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        metavar="KEY=VALUE",
        help="add this setting to the data folder's .env (creating it), unless the file "
        f"already sets it; repeatable. Exit status {SET_ENV_KEPT} if a different value was kept",
    )
    parser.add_argument(
        "--init-env",
        action="store_true",
        help="add the data folder's absolute data, datasets and outputs folders to its .env "
        "(creating it), unless the file already sets them",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="with --set-env or --init-env: also change settings the file already has",
    )
    args = parser.parse_args(argv)
    if args.data_dir:
        # Cheap on purpose: no probes, no torch import.
        print(current_data_home())
        return 0
    if args.set_env or args.init_env:
        home = current_data_home()
        values: dict[str, str] = storage_settings(home) if args.init_env else {}
        for item in args.set_env or ():
            key, separator, value = item.partition("=")
            if not separator:
                parser.error(f"--set-env needs KEY=VALUE, not {item!r}")
            values[key] = value
        path = home / ENV_FILE_NAME
        try:
            update = update_env_file(path, values, replace=args.replace)
        except (OSError, ValueError) as exc:
            print(f"can't update {path}: {exc}", file=sys.stderr)
            return 1
        print("\n".join(describe_env_update(path, update)))
        return SET_ENV_KEPT if update.kept else 0
    if args.realsense_udev_rules:
        source = realsense_udev_rules_source()
        if source is None:
            print("this build of the suite has no RealSense udev rules file", file=sys.stderr)
            return 1
        print(source)
        return 0
    report = collect()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
