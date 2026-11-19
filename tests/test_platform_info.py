"""so101.platform_info, with each target machine described through Probes.

Nothing here patches sys.platform or depends on the machine running the tests:
every OS fact comes from a fake probe, and paths use PureWindowsPath or
PurePosixPath so a Windows layout can be checked on a Mac and vice versa.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from so101 import platform_info as pi
from so101.platform_info import (
    Accelerator,
    GpuInventory,
    GroupMembership,
    NvidiaSmi,
    Probes,
    RealSenseDevice,
    RealSenseListing,
    RenderCheck,
    SerialPort,
    TorchInfo,
)

SMI_3070 = "NVIDIA GeForce RTX 3070 Laptop GPU, 580.95.05, 8192\n"
CH343 = SerialPort("/dev/ttyACM0", "USB Single Serial", 0x1A86, 0x55D3, "585A0077531", "WCH")


def runner(outputs: dict[str, str | None]):
    """A fake Probes.run: maps the command's first word (basename) to stdout."""

    def run(cmd: Sequence[str]) -> str | None:
        name = PurePosixPath(str(cmd[0]).replace("\\", "/")).name
        return outputs.get(name)

    return run


def fake_probes(**overrides) -> Probes:
    """A machine with nothing interesting on it; tests override what they need."""

    base = dict(
        system=lambda: "Linux",
        release=lambda: "6.8.0",
        machine=lambda: "x86_64",
        python_version=lambda: (3, 12, 9),
        env={},
        home=lambda: PurePosixPath("/home/ada"),
        which=lambda name: None,
        run=runner({}),
        torch=lambda: None,
        serial_ports=lambda: [],
        access_rw=lambda path: True,
        device_group=lambda path: "dialout",
        user_groups=lambda: GroupMembership(frozenset({"ada"}), frozenset({"ada"})),
        find_library=lambda name: None,
        pci_display_vendors=lambda: frozenset(),
        long_paths_enabled=lambda: True,
        preferred_encoding=lambda: "UTF-8",
        utf8_mode=lambda: False,
        free_threaded=lambda: False,
        usb_vendor_ids=lambda: frozenset(),
        has_battery=lambda: False,
        disk_free=lambda path: 100 * 2**30,
        mujoco_render=lambda gl: RenderCheck(gl, True, "rendered a 64x64 frame"),
        dist_version=lambda name: None,
        module_present=lambda name: False,
        realsense_devices=lambda: RealSenseListing(),
        realsense_rules=lambda: None,
        exists=lambda path: False,
        dotenv=lambda: {},
    )
    base.update(overrides)
    return Probes(**base)


def statuses(report: pi.PlatformReport) -> dict[str, str]:
    return {check.id: check.status for check in pi.checks(report)}


# --- the target machines -------------------------------------------------------


def windows_rx5700xt(**overrides) -> Probes:
    """The Ryzen 7 desktop: Windows, Radeon RX 5700 XT, CPU torch."""

    values = dict(
        system=lambda: "Windows",
        release=lambda: "11",
        machine=lambda: "AMD64",
        env={"LOCALAPPDATA": r"C:\Users\ada\AppData\Local"},
        home=lambda: PureWindowsPath(r"C:\Users\ada"),
        run=runner({"powershell": "AMD Radeon RX 5700 XT\r\n"}),
        torch=lambda: TorchInfo(version="2.11.0+cpu"),
        serial_ports=lambda: [
            SerialPort("COM3", "USB-Enhanced-SERIAL CH343 (COM3)", 0x1A86, 0x55D3)
        ],
        preferred_encoding=lambda: "cp1252",
    )
    values.update(overrides)
    return fake_probes(**values)


def zephyrus_windows(**overrides) -> Probes:
    """The ASUS Zephyrus G15 on Windows: RTX 3070 Laptop beside the Ryzen iGPU."""

    values = dict(
        system=lambda: "Windows",
        release=lambda: "11",
        machine=lambda: "AMD64",
        env={"LOCALAPPDATA": r"C:\Users\ada\AppData\Local"},
        home=lambda: PureWindowsPath(r"C:\Users\ada"),
        which=lambda name: r"C:\Windows\System32\nvidia-smi.exe" if name == "nvidia-smi" else None,
        run=runner(
            {
                "nvidia-smi.exe": SMI_3070,
                "powershell": "AMD Radeon(TM) Graphics\r\nNVIDIA GeForce RTX 3070 Laptop GPU\r\n",
            }
        ),
        torch=lambda: TorchInfo(
            version="2.11.0+cu128",
            cuda_build="12.8",
            cuda_available=True,
            device_name="NVIDIA GeForce RTX 3070 Laptop GPU",
            vram_gib=8.0,
        ),
        utf8_mode=lambda: True,
    )
    values.update(overrides)
    return fake_probes(**values)


def zephyrus_linux(**overrides) -> Probes:
    """The same laptop on Linux, with supergfxctl and a user not yet in dialout."""

    tools = {"nvidia-smi": "/usr/bin/nvidia-smi", "supergfxctl": "/usr/bin/supergfxctl"}
    values = dict(
        env={"DISPLAY": ":0", "XDG_DATA_HOME": "/home/ada/.data"},
        which=tools.get,
        run=runner({"nvidia-smi": SMI_3070, "supergfxctl": "Hybrid\n"}),
        torch=lambda: TorchInfo(
            version="2.11.0+cu128",
            cuda_build="12.8",
            cuda_available=True,
            device_name="NVIDIA GeForce RTX 3070 Laptop GPU",
            vram_gib=8.0,
        ),
        serial_ports=lambda: [CH343, SerialPort("/dev/ttyS0", "ttyS0")],
        pci_display_vendors=lambda: frozenset({"amd", "nvidia"}),
        find_library=lambda name: "libEGL.so.1" if name == "EGL" else None,
    )
    values.update(overrides)
    return fake_probes(**values)


def apple_silicon(**overrides) -> Probes:
    values = dict(
        system=lambda: "Darwin",
        release=lambda: "25.6.0",
        machine=lambda: "arm64",
        home=lambda: PurePosixPath("/Users/ada"),
        torch=lambda: TorchInfo(version="2.11.0", mps_available=True),
        serial_ports=lambda: [
            SerialPort("/dev/cu.usbmodem585A0077531", "USB Single Serial", 0x1A86, 0x55D3),
            SerialPort("/dev/cu.Bluetooth-Incoming-Port"),
        ],
    )
    values.update(overrides)
    return fake_probes(**values)


def test_windows_desktop_runs_torch_on_cpu_and_says_why_without_warning():
    report = pi.collect(windows_rx5700xt())

    assert report.os == "windows"
    assert report.arch == "x86_64"
    assert report.accelerator.kind == "cpu"
    assert report.gpus.vendors == frozenset({"amd"})
    assert report.gpu_hint is None
    assert report.gpu_note is not None and "AMD GPU" in report.gpu_note
    assert report.data_dir == r"C:\Users\ada\AppData\Local\so101"
    assert report.mujoco_gl.recommended is None
    assert report.serial_access.ok is None
    assert any("UTF-8" in warning for warning in report.warnings)
    result = statuses(report)
    assert result["platform.accelerator"] == "pass"
    assert result["platform.serial"] == "pass"
    assert result["platform.python"] == "pass"


def test_zephyrus_windows_uses_cuda_with_no_hints():
    report = pi.collect(zephyrus_windows())

    assert report.accelerator == Accelerator(
        kind="cuda",
        device="NVIDIA GeForce RTX 3070 Laptop GPU",
        vram_gib=8.0,
        torch_version="2.11.0+cu128",
        torch_cuda="12.8",
    )
    assert report.gpus.nvidia == NvidiaSmi("NVIDIA GeForce RTX 3070 Laptop GPU", "580.95.05", 8192)
    assert report.gpus.vendors == frozenset({"amd", "nvidia"})
    assert report.gpu_hint is None and report.gpu_note is None
    assert report.warnings == ()
    assert "CUDA: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB" in pi.format_text(report)


def test_zephyrus_windows_in_eco_mode_points_at_armoury_crate():
    probes = zephyrus_windows(
        run=runner({"nvidia-smi.exe": None, "powershell": "AMD Radeon(TM) Graphics\r\n"}),
        torch=lambda: TorchInfo(version="2.11.0+cu128", cuda_build="12.8"),
    )
    report = pi.collect(probes)

    assert report.accelerator.kind == "cpu"
    assert report.gpus.nvidia_smi_installed and report.gpus.nvidia is None
    assert report.gpu_hint is not None
    assert "Armoury Crate" in report.gpu_hint and "Eco" in report.gpu_hint
    assert report.gpu_note is None
    assert statuses(report)["platform.accelerator"] == "warn"


def test_zephyrus_linux_reports_cuda_egl_free_display_and_dialout_fix():
    probes = zephyrus_linux(access_rw=lambda path: False)
    report = pi.collect(probes)

    assert report.accelerator.kind == "cuda"
    assert report.gpus.gpu_mode == "Hybrid"
    assert report.gpu_hint is None
    assert report.data_dir == "/home/ada/.data/so101"
    assert report.mujoco_gl.recommended is None  # a display is present
    assert report.serial_access.ok is False
    assert report.serial_access.group == "dialout"
    assert "usermod -aG dialout" in report.serial_access.remediation
    # Only the USB port is judged; /dev/ttyS0 is a legacy UART.
    assert "/dev/ttyACM0" in report.serial_access.detail
    assert "/dev/ttyS0" not in report.serial_access.detail
    result = statuses(report)
    assert result["platform.serial"] == "fail"
    assert result["platform.accelerator"] == "pass"
    assert "WCH CH343" in pi.format_text(report)


def test_zephyrus_linux_in_integrated_mode_says_switch_to_hybrid():
    probes = zephyrus_linux(
        run=runner({"nvidia-smi": None, "supergfxctl": "Integrated\n"}),
        pci_display_vendors=lambda: frozenset({"amd"}),
        torch=lambda: TorchInfo(version="2.11.0+cu128", cuda_build="12.8"),
    )
    report = pi.collect(probes)

    assert report.gpus.gpu_mode == "Integrated"
    assert report.gpu_hint is not None and "supergfxctl -m Hybrid" in report.gpu_hint
    assert report.gpu_note is None  # an AMD iGPU here is not the RX 5700 XT story


def test_apple_silicon_uses_mps_and_lists_only_usb_ports():
    report = pi.collect(apple_silicon())

    assert report.os == "macos" and report.arch == "arm64"
    assert report.accelerator.kind == "mps"
    assert report.gpu_hint is None and report.gpu_note is None
    assert report.data_dir == "/Users/ada/Library/Application Support/so101"
    assert report.mujoco_gl.recommended is None
    text = pi.format_text(report)
    assert "/dev/cu.usbmodem585A0077531" in text
    assert "Bluetooth" not in text
    assert all(check.status == "pass" for check in pi.checks(report))


def test_intel_mac_is_warned_and_rosetta_python_fails():
    intel = pi.collect(apple_silicon(machine=lambda: "x86_64", run=runner({"sysctl": None})))
    assert intel.rosetta is False
    assert any("Intel Macs are not supported" in warning for warning in intel.warnings)

    rosetta = pi.collect(apple_silicon(machine=lambda: "x86_64", run=runner({"sysctl": "1\n"})))
    assert rosetta.rosetta is True
    check = next(c for c in pi.checks(rosetta) if c.id == "platform.python")
    assert check.status == "fail"
    assert "arm64" in check.remediation


# --- GPU hints -------------------------------------------------------------------


def nvidia_inventory(driver: str = "580.95.05") -> GpuInventory:
    return GpuInventory(
        nvidia_smi_installed=True,
        nvidia=NvidiaSmi("NVIDIA GeForce RTX 3070 Laptop GPU", driver, 8192),
        vendors=frozenset({"nvidia"}),
    )


def test_cpu_torch_on_an_nvidia_machine_says_reinstall_with_cuda():
    accel = pi.accelerator_from(TorchInfo(version="2.11.0+cpu"))
    hint = pi.gpu_hint("linux", accel, nvidia_inventory())
    assert hint is not None and "--profile cuda" in hint and "CPU-only" in hint


def test_old_driver_for_the_torch_cuda_build_is_named():
    accel = pi.accelerator_from(TorchInfo(version="2.11.0+cu128", cuda_build="12.8"))
    hint = pi.gpu_hint("linux", accel, nvidia_inventory("550.120"))
    assert hint is not None
    assert "570.124.06" in hint and "550.120" in hint
    hint = pi.gpu_hint("windows", accel, nvidia_inventory("566.36"))
    assert hint is not None
    assert "572.61" in hint and "566.36" in hint


def test_driver_new_enough_but_cuda_unusable_suggests_restart():
    accel = pi.accelerator_from(TorchInfo(version="2.11.0+cu128", cuda_build="12.8"))
    hint = pi.gpu_hint("windows", accel, nvidia_inventory())
    assert hint is not None and "Restart the computer" in hint


def test_nvidia_on_the_bus_without_a_driver():
    accel = pi.accelerator_from(TorchInfo(version="2.11.0+cpu"))
    hint = pi.gpu_hint("linux", accel, GpuInventory(vendors=frozenset({"nvidia", "intel"})))
    assert hint is not None and "driver isn't installed" in hint


def test_nvidia_driver_installed_but_not_answering_on_linux():
    accel = pi.accelerator_from(TorchInfo(version="2.11.0+cu128", cuda_build="12.8"))
    hint = pi.gpu_hint("linux", accel, GpuInventory(nvidia_smi_installed=True))
    assert hint is not None and "Secure Boot" in hint


def test_cuda_torch_on_a_machine_without_nvidia_suggests_cpu_profile():
    accel = pi.accelerator_from(TorchInfo(version="2.11.0+cu128", cuda_build="12.8"))
    hint = pi.gpu_hint("windows", accel, GpuInventory(vendors=frozenset({"amd"})))
    assert hint is not None and "--profile cpu" in hint
    assert pi.gpu_note("windows", accel, GpuInventory(vendors=frozenset({"amd"}))) is None


def test_windows_dll_failure_points_at_the_vc_redistributable():
    accel = pi.accelerator_from(
        TorchInfo(version="unknown", error="OSError: [WinError 126] ... c10.dll")
    )
    hint = pi.gpu_hint("windows", accel, GpuInventory())
    assert hint is not None and "Visual C++" in hint
    assert pi.gpu_hint("linux", accel, GpuInventory()) is None


def test_no_hint_when_the_gpu_is_in_use_or_on_a_mac():
    cuda = pi.accelerator_from(
        TorchInfo(version="2.11.0+cu128", cuda_build="12.8", cuda_available=True)
    )
    assert pi.gpu_hint("linux", cuda, nvidia_inventory()) is None
    cpu = pi.accelerator_from(TorchInfo(version="2.11.0"))
    assert pi.gpu_hint("macos", cpu, GpuInventory(vendors=frozenset({"apple"}))) is None
    assert pi.gpu_hint("linux", cpu, GpuInventory()) is None


def test_accelerator_from_covers_missing_broken_and_mps_torch():
    assert pi.accelerator_from(None) == Accelerator(kind="cpu", error="torch is not installed")
    broken = pi.accelerator_from(TorchInfo(version="2.11.0", error="RuntimeError: boom"))
    assert broken.kind == "cpu" and broken.error == "RuntimeError: boom"
    assert pi.accelerator_from(TorchInfo(version="2.11.0", mps_available=True)).kind == "mps"


# --- drivers and nvidia-smi ------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (SMI_3070, NvidiaSmi("NVIDIA GeForce RTX 3070 Laptop GPU", "580.95.05", 8192)),
        (
            "NVIDIA GeForce RTX 3070, 576.88, [N/A]\n",
            NvidiaSmi("NVIDIA GeForce RTX 3070", "576.88"),
        ),
        ("No devices were found\n", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_nvidia_smi(output, expected):
    assert pi.parse_nvidia_smi(output) == expected


@pytest.mark.parametrize(
    ("driver", "expected"),
    [
        ("570.124.06", (570, 124, 6)),
        ("572.61", (572, 61, 0)),
        ("580", (580, 0, 0)),
        (" 575.64.03\n", (575, 64, 3)),
        ("garbage", None),
    ],
)
def test_driver_tuple(driver, expected):
    assert pi.driver_tuple(driver) == expected


@pytest.mark.parametrize(
    ("os_kind", "driver", "index"),
    [
        ("linux", "580.95.05", "cu128"),
        ("linux", "570.124.06", "cu128"),
        ("linux", "570.124.04", "cu126"),  # older than NVIDIA lists for CUDA 12.8.1
        ("linux", "570.86.10", "cu126"),
        ("linux", "560.35.03", "cu126"),
        ("linux", "550.120", "cu126"),  # minor-version compatibility
        ("linux", "525.60.13", "cu126"),
        ("linux", "520.61.05", None),
        ("linux", "garbage", None),
        ("linux", None, None),
        ("windows", "581.29", "cu128"),
        ("windows", "572.61", "cu128"),
        ("windows", "572.16", "cu126"),
        ("windows", "527.41", "cu126"),
        ("windows", "516.94", None),
        ("macos", "580.95.05", None),
    ],
)
def test_cuda_index_for_driver(os_kind, driver, index):
    assert pi.cuda_index_for_driver(driver, os_kind) == index


def test_driver_floors_parse():
    for floors in pi.MIN_DRIVER_FOR_CUDA.values():
        for floor in floors.values():
            assert pi.driver_tuple(floor) is not None
    with pytest.raises(ValueError):
        pi._floor("n/a")


def test_driver_problem_only_for_known_builds_and_old_drivers():
    assert pi.driver_problem("12.8", "570.124.06", "linux") is None
    assert pi.driver_problem("12.8", "570.86.10", "linux") is not None
    assert pi.driver_problem("13.0", "575.51", "linux") is not None
    assert pi.driver_problem("12.8.1", "565.90", "linux") is not None  # patch level ignored
    assert pi.driver_problem("12.8", "572.61", "windows") is None
    assert pi.driver_problem("12.8", "566.36", "windows") is not None
    assert pi.driver_problem("13.0", "581.29", "windows") is None
    assert pi.driver_problem("12.8", "500.0", "macos") is None  # no CUDA there
    assert pi.driver_problem("11.8", "400.0", "linux") is None  # no floor recorded
    assert pi.driver_problem(None, "400.0", "linux") is None
    assert pi.driver_problem("12.8", None, "linux") is None


# --- MuJoCo GL -------------------------------------------------------------------


def egl_only(name: str) -> str | None:
    return "libEGL.so.1" if name == "EGL" else None


def osmesa_only(name: str) -> str | None:
    return "libOSMesa.so.8" if name == "OSMesa" else None


@pytest.mark.parametrize(
    ("os_kind", "env", "find_library", "recommended"),
    [
        ("macos", {}, egl_only, None),
        ("windows", {}, egl_only, None),
        ("linux", {"DISPLAY": ":0"}, egl_only, None),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, egl_only, None),
        ("linux", {}, egl_only, "egl"),
        ("linux", {}, osmesa_only, "osmesa"),
        ("linux", {}, lambda name: None, "disable"),
    ],
)
def test_mujoco_gl_recommendation(os_kind, env, find_library, recommended):
    advice = pi.mujoco_gl_advice(os_kind, env, find_library)
    assert advice.recommended == recommended
    assert advice.problem is None


@pytest.mark.parametrize(
    ("os_kind", "value"),
    [("macos", "egl"), ("windows", "egl"), ("windows", "osmesa"), ("linux", "cgl"), ("macos", "x")],
)
def test_mujoco_gl_values_that_break_import_are_flagged(os_kind, value):
    advice = pi.mujoco_gl_advice(os_kind, {"MUJOCO_GL": value}, egl_only)
    assert advice.current == value
    assert advice.problem is not None and "RuntimeError" in advice.problem


@pytest.mark.parametrize(
    ("os_kind", "value"),
    [
        ("windows", "wgl"),
        ("windows", "glfw"),
        ("macos", "cgl"),
        ("linux", "EGL"),
        ("linux", "osmesa"),
        ("windows", "disable"),
        ("macos", "0"),
    ],
)
def test_mujoco_gl_values_valid_on_the_os(os_kind, value):
    assert pi.mujoco_gl_advice(os_kind, {"MUJOCO_GL": value}, egl_only).problem is None


def test_an_invalid_mujoco_gl_fails_the_sim_check():
    report = pi.collect(windows_rx5700xt(env={"MUJOCO_GL": "egl"}))
    check = next(c for c in pi.checks(report) if c.id == "platform.mujoco_gl")
    assert check.status == "fail" and "Unset it" in check.remediation


def test_headless_linux_without_gl_libraries_warns():
    report = pi.collect(fake_probes())
    assert report.mujoco_gl.recommended == "disable"
    assert statuses(report)["platform.mujoco_gl"] == "warn"


# --- serial ports ------------------------------------------------------------------


def test_serial_access_is_only_judged_on_linux():
    for os_kind in ("windows", "macos"):
        access = pi.serial_access(os_kind, [CH343], lambda p: False, lambda p: "x", None)
        assert access.ok is None


def test_serial_access_with_no_usb_ports():
    access = pi.serial_access(
        "linux", [SerialPort("/dev/ttyS0")], lambda p: False, lambda p: "dialout", None
    )
    assert access.ok is None


def test_serial_access_granted():
    access = pi.serial_access("linux", [CH343], lambda p: True, lambda p: "dialout", None)
    assert access.ok is True and "/dev/ttyACM0" in access.detail


def test_serial_access_group_added_but_session_is_stale():
    groups = GroupMembership(session=frozenset({"ada"}), configured=frozenset({"ada", "dialout"}))
    access = pi.serial_access("linux", [CH343], lambda p: False, lambda p: "dialout", groups)
    assert access.ok is False
    assert "Log out and back in" in access.remediation
    assert "usermod" not in access.remediation


def test_serial_access_uses_the_device_group_name():
    access = pi.serial_access("linux", [CH343], lambda p: False, lambda p: "uucp", None)
    assert access.group == "uucp" and "usermod -aG uucp" in access.remediation
    fallback = pi.serial_access("linux", [CH343], lambda p: False, lambda p: None, None)
    assert fallback.group == "dialout"


def test_missing_pyserial_is_a_warning():
    report = pi.collect(fake_probes(serial_ports=lambda: None))
    assert report.serial_ports is None
    assert statuses(report)["platform.serial"] == "warn"
    assert "pyserial is not installed" in pi.format_text(report)


def test_serial_port_labels():
    assert CH343.chip == "WCH CH343" and CH343.is_usb
    assert pi.describe_port(CH343) == "/dev/ttyACM0  WCH CH343  serial 585A0077531"
    unknown = SerialPort("COM7", "USB Serial Device (COM7)", 0x1234, 0x5678)
    assert unknown.chip is None
    assert pi.describe_port(unknown) == "COM7  USB Serial Device (COM7)"
    assert SerialPort("/dev/ttyS0").chip is None and not SerialPort("/dev/ttyS0").is_usb


# --- data folder ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("os_kind", "env", "home", "expected"),
    [
        (
            "windows",
            {"LOCALAPPDATA": r"D:\Profiles\ada\AppData\Local"},
            PureWindowsPath(r"C:\Users\ada"),
            PureWindowsPath(r"D:\Profiles\ada\AppData\Local\so101"),
        ),
        (
            "windows",
            {},
            PureWindowsPath(r"C:\Users\ada"),
            PureWindowsPath(r"C:\Users\ada\AppData\Local\so101"),
        ),
        (
            "windows",
            {"LOCALAPPDATA": "relative"},
            PureWindowsPath(r"C:\Users\ada"),
            PureWindowsPath(r"C:\Users\ada\AppData\Local\so101"),
        ),
        (
            "macos",
            {"XDG_DATA_HOME": "/ignored"},
            PurePosixPath("/Users/ada"),
            PurePosixPath("/Users/ada/Library/Application Support/so101"),
        ),
        (
            "linux",
            {"XDG_DATA_HOME": "/data/xdg"},
            PurePosixPath("/home/ada"),
            PurePosixPath("/data/xdg/so101"),
        ),
        (
            "linux",
            {"XDG_DATA_HOME": "relative/xdg"},
            PurePosixPath("/home/ada"),
            PurePosixPath("/home/ada/.local/share/so101"),
        ),
        ("linux", {}, PurePosixPath("/home/ada"), PurePosixPath("/home/ada/.local/share/so101")),
        (
            "linux",
            {"SO101_HOME": "~/robot"},
            PurePosixPath("/home/ada"),
            PurePosixPath("/home/ada/robot"),
        ),
        (
            "windows",
            {"SO101_HOME": r"E:\so101", "LOCALAPPDATA": r"C:\x"},
            PureWindowsPath(r"C:\Users\ada"),
            PureWindowsPath(r"E:\so101"),
        ),
        (
            "windows",
            {"SO101_HOME": "~\\so101"},
            PureWindowsPath(r"C:\Users\ada"),
            PureWindowsPath(r"C:\Users\ada\so101"),
        ),
    ],
)
def test_data_home(os_kind, env, home, expected):
    result = pi.data_home(os_kind, env, home)
    assert result == expected
    assert type(result) is type(home)


# --- Python, Windows settings, names ------------------------------------------------


def test_python_versions():
    old = pi.collect(fake_probes(python_version=lambda: (3, 11, 9)))
    assert old.python_ok is False
    assert statuses(old)["platform.python"] == "fail"
    assert any("too old" in warning for warning in old.warnings)
    for supported in ((3, 12, 9), (3, 13, 1)):
        report = pi.collect(fake_probes(python_version=lambda v=supported: v))
        assert report.python_ok is True
        assert report.warnings == ()
        assert statuses(report)["platform.python"] == "pass"
    newest = pi.collect(fake_probes(python_version=lambda: (3, 14, 0)))
    assert newest.python_ok is False
    assert any("needs Python 3.12 to 3.13" in warning for warning in newest.warnings)
    check = next(c for c in pi.checks(newest) if c.id == "platform.python")
    assert check.status == "fail" and "Python 3.13" in check.remediation


def test_a_free_threaded_python_is_refused():
    report = pi.collect(fake_probes(python_version=lambda: (3, 13, 1), free_threaded=lambda: True))
    assert report.python_ok is False and report.free_threaded is True
    assert any("free-threaded build" in warning for warning in report.warnings)
    check = next(c for c in pi.checks(report) if c.id == "platform.python")
    assert check.status == "fail" and "not free-threaded" in check.remediation
    assert report.to_dict()["free_threaded"] is True


def test_free_threaded_probe_reads_sysconfig(monkeypatch):
    monkeypatch.setattr(pi.sysconfig, "get_config_var", lambda name: 1)
    assert pi._free_threaded() is True
    monkeypatch.setattr(pi.sysconfig, "get_config_var", lambda name: None)
    assert pi._free_threaded() is False


@pytest.mark.parametrize(
    ("setting", "torch", "problem"),
    [
        ("auto", TorchInfo(version="2.11.0"), None),
        ("cpu", TorchInfo(version="2.11.0"), None),
        ("mps", TorchInfo(version="2.11.0", mps_available=True), None),
        ("CUDA:1", TorchInfo(version="2.11.0+cu128", cuda_build="12.8", cuda_available=True), None),
        ("cuda", TorchInfo(version="2.11.0", mps_available=True), "torch can only use mps here"),
        ("mps", TorchInfo(version="2.11.0+cpu"), "torch can only use cpu here"),
        ("gpu", TorchInfo(version="2.11.0"), "'gpu' is not a device"),
        ("mps:0", TorchInfo(version="2.11.0", mps_available=True), "'mps:0' is not a device"),
        ("cuda", None, None),  # no torch: the torch check already says so
        ("cuda", TorchInfo(version="2.11.0", error="OSError: c10.dll"), None),
    ],
)
def test_policy_device_problem_follows_the_policy_server(setting, torch, problem):
    found = pi.policy_device_problem(setting, pi.accelerator_from(torch))
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


def test_policy_device_is_checked_only_when_set():
    mac = pi.collect(apple_silicon())
    assert mac.policy_device == ""
    assert "platform.policy_device" not in statuses(mac)
    assert "Policy device" not in pi.format_text(mac)

    wrong = pi.collect(apple_silicon(env={"SO101_POLICY_DEVICE": "cuda"}))
    check = next(c for c in pi.checks(wrong) if c.id == "platform.policy_device")
    assert check.status == "fail"
    assert "torch can only use mps here" in check.detail
    assert "Unset SO101_POLICY_DEVICE" in check.remediation
    assert "Policy device  PROBLEM: SO101_POLICY_DEVICE=cuda" in pi.format_text(wrong)

    from_dotenv = pi.collect(apple_silicon(dotenv=lambda: {"so101_policy_device": "mps"}))
    check = next(c for c in pi.checks(from_dotenv) if c.id == "platform.policy_device")
    assert (check.status, check.detail) == ("pass", "SO101_POLICY_DEVICE=mps")
    assert pi.env_setting({"SO101_POLICY_DEVICE": " "}, dict, "so101_policy_device", "") == " "


def test_windows_long_paths_off_is_warned():
    report = pi.collect(windows_rx5700xt(long_paths_enabled=lambda: False, utf8_mode=lambda: True))
    assert report.long_paths is False
    assert any("long paths" in warning for warning in report.warnings)


def test_windows_utf8_locale_needs_no_warning():
    report = pi.collect(windows_rx5700xt(preferred_encoding=lambda: "UTF-8"))
    assert not any("UTF-8" in warning for warning in report.warnings)


def test_long_paths_are_only_read_on_windows():
    def boom() -> bool:
        raise AssertionError("long_paths_enabled probed off Windows")

    assert pi.collect(fake_probes(long_paths_enabled=boom)).long_paths is None


@pytest.mark.parametrize(
    ("system", "expected"),
    [("Darwin", "macos"), ("Windows", "windows"), ("Linux", "linux"), ("FreeBSD", "freebsd")],
)
def test_os_name(system, expected):
    assert pi.os_name(system) == expected


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("AMD64", "x86_64"), ("x86_64", "x86_64"), ("arm64", "arm64"), ("aarch64", "arm64")],
)
def test_normalize_arch(machine, expected):
    assert pi.normalize_arch(machine) == expected


# --- RealSense -------------------------------------------------------------------------

D435 = RealSenseDevice("Intel RealSense D435", "123456789012", "5.16.0.1", "3.2")
D405_ON_USB2 = RealSenseDevice("Intel RealSense D405", "234567890123", "5.16.0.1", "2.1")
REALSENSE_TARGET_IDS = ("0ad3", "0b07", "0b3a", "0b5b", "0b5c")


def realsense_rules_fixture(*, vendor: bool = False, ids=REALSENSE_TARGET_IDS) -> str:
    if vendor:
        return "".join(
            f'SUBSYSTEM=="usb", ATTR{{idVendor}}=="8086", ATTR{{idProduct}}=="{product}", '
            'TAG+="uaccess"\n'
            for product in ids
        )
    return "".join(
        f'SUBSYSTEMS=="usb", ATTRS{{idVendor}}=="8086", '
        f'ATTRS{{idProduct}}=="{product}", MODE:="0666", GROUP:="plugdev"\n'
        for product in ids
    )


def with_realsense(
    builder=fake_probes,
    *,
    version: str | None = "2.56.5.9235",
    dist: str = "pyrealsense2",
    cameras: str | None = None,
    devices: Sequence[RealSenseDevice] = (),
    error: str | None = None,
    **overrides,
) -> Probes:
    """A machine with pyrealsense2 installed (``version=None``: not installed)."""

    def no_listing() -> RealSenseListing:
        raise AssertionError("cameras must not be listed without pyrealsense2")

    values: dict[str, object] = dict(
        env={} if cameras is None else {"SO101_CAMERAS": cameras},
        dist_version=lambda name: version if name == dist else None,
        module_present=lambda name: version is not None,
        realsense_devices=(
            (lambda: RealSenseListing(tuple(devices), error)) if version else no_listing
        ),
    )
    values.update(overrides)
    return builder(**values)


def realsense_checks(report: pi.PlatformReport) -> dict[str, pi.PlatformCheck]:
    return {c.id: c for c in pi.checks(report) if c.id.startswith("platform.realsense")}


@pytest.mark.parametrize(
    ("cameras", "configured"),
    [
        ("0", False),
        ("", False),
        ("0,1,/dev/video2", False),
        ("realsense", True),
        ("realsense:123456789012", True),
        ("0,front=realsense:123456789012", True),
        (" wrist = RealSense ", True),
        ("realsense=0", False),  # a camera named "realsense" on OpenCV index 0
        ("rtsp://host/realsense?a=b", False),
        ("v4l2src device=/dev/video0 ! appsink", False),
    ],
)
def test_realsense_configured(cameras, configured):
    assert pi.realsense_configured(cameras) is configured


def test_parse_dotenv_reads_what_python_dotenv_reads():
    text = "\n".join(
        [
            "# comment",
            "SO101_MOCK=1",
            "export SO101_CAMERAS = 'front=realsense:1, wrist=0'  # two cameras",
            'SO101_PORT="8000" # quoted',
            "SO101_HASH=a#b",
            "SO101_TAIL=value   # comment",
            "SO101_TAB=value\t# comment",
            "not a binding",
            "SO101_EMPTY=",
            "KEY WITH SPACE=1",
            "  SO101_INDENTED=yes",
            "SO101_DQ=\"a 'b' c\"",
            "export\tSO101_EXPORT_TAB=1",
            "SO101_NO_VALUE",
            "SO101_CRLF=1\r",
        ]
    )
    expected = {
        "SO101_MOCK": "1",
        "SO101_CAMERAS": "front=realsense:1, wrist=0",
        "SO101_PORT": "8000",
        "SO101_HASH": "a#b",
        "SO101_TAIL": "value",
        "SO101_TAB": "value",
        "SO101_EMPTY": "",
        "SO101_INDENTED": "yes",
        "SO101_DQ": "a 'b' c",
        "SO101_EXPORT_TAB": "1",
        "SO101_CRLF": "1",
    }
    assert pi.parse_dotenv(text) == expected
    assert pi.parse_dotenv("SO101_CAMERAS='realsense") == {}  # no closing quote
    dotenv = pytest.importorskip("dotenv")  # what pydantic-settings reads .env with
    values = dotenv.dotenv_values(stream=io.StringIO(text), interpolate=False)
    assert {key: value for key, value in values.items() if value is not None} == expected


def test_camera_setting_reads_the_environment_then_dotenv_like_settings():
    def no_dotenv() -> dict[str, str]:
        raise AssertionError("the environment wins; .env isn't needed")

    assert pi.camera_setting({"so101_cameras": "realsense"}, no_dotenv) == "realsense"
    assert pi.camera_setting({"PATH": "/bin"}, lambda: {"So101_Cameras": "1"}) == "1"
    assert pi.camera_setting({}, dict) == "0"  # Settings.cameras' default


def test_realsense_not_installed_and_not_configured_is_a_silent_pass():
    report = pi.collect(with_realsense(version=None))

    assert report.realsense.installed is False and report.realsense.configured is False
    found = realsense_checks(report)
    assert set(found) == {"platform.realsense"}  # no udev check either
    assert found["platform.realsense"].status == "pass"
    assert "optional" in found["platform.realsense"].detail
    assert "RealSense" not in pi.format_text(report)


def test_a_configured_realsense_without_pyrealsense2_fails_with_the_installer_hint():
    report = pi.collect(with_realsense(version=None, dotenv=lambda: {"SO101_CAMERAS": "realsense"}))

    found = realsense_checks(report)
    assert found["platform.realsense"].status == "fail"
    assert "--realsense" in found["platform.realsense"].remediation
    # Linux: the rules are reported for a configured camera, even before installing.
    assert found["platform.realsense.udev"].status == "warn"
    assert "PROBLEM: SO101_CAMERAS has a RealSense camera" in pi.format_text(report)

    windows = pi.collect(with_realsense(windows_rx5700xt, version=None, cameras="realsense"))
    found = realsense_checks(windows)
    assert "-RealSense" in found["platform.realsense"].remediation
    assert set(found) == {"platform.realsense"}


@pytest.mark.parametrize(
    ("system", "release", "machine", "python", "expected"),
    [
        ("Linux", "6.8.0", "x86_64", (3, 14, 0), "Python 3.12 and 3.13, not 3.14"),
        ("Linux", "6.8.0", "aarch64", (3, 12, 9), "no linux arm64 wheel"),
        ("Darwin", "23.6.0", "arm64", (3, 13, 1), "needs macOS 15"),
    ],
)
def test_the_wheel_problem_replaces_the_installer_hint(system, release, machine, python, expected):
    report = pi.collect(
        with_realsense(
            version=None,
            cameras="realsense",
            system=lambda: system,
            release=lambda: release,
            machine=lambda: machine,
            python_version=lambda: python,
        )
    )
    check = realsense_checks(report)["platform.realsense"]
    assert check.status == "fail"
    assert expected in check.remediation
    assert check.remediation == report.realsense.wheel_problem


def test_realsense_wheel_problem_cases():
    problem = pi.realsense_wheel_problem
    assert problem("linux", "x86_64", (3, 12)) is None
    assert problem("windows", "x86_64", (3, 13)) is None
    assert problem("macos", "arm64", (3, 14), "25.6.0") is None  # 2.56.5 has cp314
    assert problem("macos", "arm64", (3, 12), "23.6.0") is None  # 2.54.2 has cp312
    assert problem("freebsd", "x86_64", (3, 12)) is None
    assert "-Fresh -RealSense" in (problem("windows", "x86_64", (3, 14)) or "")


def test_realsense_cameras_are_listed_with_serial_firmware_and_usb():
    report = pi.collect(with_realsense(devices=[D435], exists=lambda p: True))

    check = realsense_checks(report)["platform.realsense"]
    assert check.status == "pass"
    assert check.detail == (
        "pyrealsense2 2.56.5.9235; Intel RealSense D435  serial 123456789012  "
        "firmware 5.16.0.1  USB 3.2"
    )
    assert pi.describe_realsense_device(RealSenseDevice("D415")) == "D415"


def test_a_realsense_camera_on_usb2_is_a_warning():
    report = pi.collect(with_realsense(devices=[D435, D405_ON_USB2], exists=lambda p: True))

    check = realsense_checks(report)["platform.realsense"]
    assert check.status == "warn"
    assert "Intel RealSense D405 is on a USB 2 link" in check.remediation
    assert "Intel RealSense D435" not in check.remediation
    assert [d.usb2 for d in report.realsense.devices] == [False, True]
    assert RealSenseDevice("x", usb_type=None).usb2 is False
    assert RealSenseDevice("x", usb_type="1.1").usb2 is True


def test_a_failed_listing_or_no_camera():
    failed = pi.collect(with_realsense(error="RuntimeError: busy", exists=lambda p: True))
    check = realsense_checks(failed)["platform.realsense"]
    assert check.status == "warn"
    assert "RuntimeError: busy" in check.detail and "replug" in check.remediation

    absent = pi.collect(with_realsense(exists=lambda p: True))
    assert realsense_checks(absent)["platform.realsense"].status == "pass"
    expected = pi.collect(with_realsense(cameras="realsense", exists=lambda p: True))
    check = realsense_checks(expected)["platform.realsense"]
    assert check.status == "warn" and "USB 3 port" in check.remediation


def test_pyrealsense2_built_from_source_counts_as_installed():
    probes = with_realsense(version=None, module_present=lambda name: name == "pyrealsense2")
    report = pi.collect(dataclasses.replace(probes, realsense_devices=lambda: RealSenseListing()))

    assert report.realsense.installed and report.realsense.version is None
    assert report.realsense.wheel_problem is None
    assert "built from source" in realsense_checks(report)["platform.realsense"].detail


def test_linux_udev_rules_installed_by_us_or_by_intels_package():
    ours = pi.collect(
        with_realsense(
            exists=lambda p: p == pi.REALSENSE_UDEV_RULES,
            realsense_rules_valid=lambda p: p == pi.REALSENSE_UDEV_RULES,
        )
    )
    check = realsense_checks(ours)["platform.realsense.udev"]
    assert check.status == "pass" and pi.REALSENSE_UDEV_RULES in check.detail

    system = pi.REALSENSE_UDEV_RULES_SYSTEM[0]

    def no_source() -> str | None:
        raise AssertionError("the source is only needed when no rules are installed")

    intel = pi.collect(
        with_realsense(
            exists=lambda p: p == system,
            realsense_rules_valid=lambda p: p == system,
            realsense_rules=no_source,
        )
    )
    assert intel.realsense.udev_rules == system
    assert realsense_checks(intel)["platform.realsense.udev"].status == "pass"


@pytest.mark.parametrize(
    "rule",
    [
        realsense_rules_fixture(),
        realsense_rules_fixture(vendor=True),
    ],
    ids=["upstream-suite", "system-vendor"],
)
def test_meaningful_realsense_udev_rule_variants_are_accepted(rule):
    assert pi.realsense_udev_rules_content_valid(f"# vendor rules\n{rule}\n")


@pytest.mark.parametrize(
    "content",
    [
        "",
        "# placeholder\n",
        realsense_rules_fixture(ids=REALSENSE_TARGET_IDS[:1]),
        realsense_rules_fixture(ids=REALSENSE_TARGET_IDS[:-1]),
        'SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", MODE:="0666"\n',
        realsense_rules_fixture().replace("SUBSYSTEMS", "XSUBSYSTEMS"),
        realsense_rules_fixture().lower(),
        (
            'not-a-rule # SUBSYSTEM=="usb", ATTR{idVendor}=="8086", '
            'ATTR{idProduct}=="0b5c", TAG+="uaccess"\n'
        ),
    ],
)
def test_truncated_stale_or_malformed_realsense_udev_rules_are_rejected(content):
    assert not pi.realsense_udev_rules_content_valid(content)


def test_actual_vendored_realsense_rules_are_complete_but_truncated_copies_are_not():
    assert pi.REALSENSE_TARGET_PRODUCT_IDS == frozenset(REALSENSE_TARGET_IDS)
    source = pi.realsense_udev_rules_source()
    assert source is not None
    content = Path(source).read_text(encoding="utf-8")

    assert pi.realsense_udev_rules_content_valid(content)
    first_rule = next(line for line in content.splitlines() if line.startswith('SUBSYSTEMS=="usb"'))
    assert not pi.realsense_udev_rules_content_valid(first_rule)
    missing_d455 = "\n".join(line for line in content.splitlines() if '0b5c"' not in line)
    assert not pi.realsense_udev_rules_content_valid(missing_d455)


def test_doctor_skips_an_invalid_realsense_rules_file_before_a_valid_system_one():
    system = pi.REALSENSE_UDEV_RULES_SYSTEM[0]
    contents = {
        pi.REALSENSE_UDEV_RULES: realsense_rules_fixture(ids=REALSENSE_TARGET_IDS[:1]),
        system: realsense_rules_fixture(vendor=True),
    }
    report = pi.collect(
        with_realsense(
            exists=lambda p: p in contents,
            realsense_rules_valid=lambda p: pi.realsense_udev_rules_content_valid(contents[p]),
        )
    )

    assert report.realsense.udev_rules == system
    assert realsense_checks(report)["platform.realsense.udev"].status == "pass"


def test_doctor_does_not_pass_when_every_known_realsense_rules_file_is_invalid():
    source = "/opt/so101/99-realsense-libusb.rules"
    report = pi.collect(
        with_realsense(
            exists=lambda p: p in {pi.REALSENSE_UDEV_RULES, *pi.REALSENSE_UDEV_RULES_SYSTEM},
            realsense_rules_valid=lambda p: False,
            realsense_rules=lambda: source,
        )
    )

    check = realsense_checks(report)["platform.realsense.udev"]
    assert report.realsense.udev_rules is None
    assert report.realsense.rules_source == source
    assert check.status == "warn" and "no usable RealSense udev rules" in check.detail


def test_missing_udev_rules_print_the_commands_with_the_shipped_file():
    source = "/opt/so 101/realsense/99-realsense-libusb.rules"
    report = pi.collect(with_realsense(realsense_rules=lambda: source))

    check = realsense_checks(report)["platform.realsense.udev"]
    assert check.status == "warn"
    assert check.remediation == (
        "Run `sudo cp '/opt/so 101/realsense/99-realsense-libusb.rules' "
        "/etc/udev/rules.d/99-realsense-libusb.rules`, then `sudo udevadm control "
        "--reload-rules && sudo udevadm trigger`, then unplug and replug the camera."
    )
    assert report.realsense.rules_source == source
    text = pi.format_text(report)
    assert "RealSense udev" in text and "sudo udevadm trigger" in text

    unknown = pi.realsense_udev_commands(None)
    assert "config/99-realsense-libusb.rules from librealsense" in unknown
    assert unknown.endswith("then unplug and replug the camera.")


def test_macos_realsense_uses_pyrealsense2_macosx_and_notes_sudo():
    probes = with_realsense(apple_silicon, dist="pyrealsense2-macosx", version="2.56.5")
    report = pi.collect(probes)

    assert report.realsense.distribution == "pyrealsense2-macosx"
    found = realsense_checks(report)
    assert set(found) == {"platform.realsense"}  # no udev on macOS
    assert found["platform.realsense"].detail.startswith("pyrealsense2-macosx 2.56.5;")
    assert "needs sudo" in found["platform.realsense"].detail
    assert "UVC index" in found["platform.realsense"].detail
    # A self-built pyrealsense2 on a Mac is found under its own name.
    other = pi.collect(with_realsense(apple_silicon, dist="pyrealsense2", version="2.57.7"))
    assert other.realsense.distribution == "pyrealsense2"
    assert other.realsense.version == "2.57.7"


def test_realsense_in_json():
    report = pi.collect(with_realsense(devices=[D405_ON_USB2]))
    data = json.loads(json.dumps(report.to_dict()))
    assert data["realsense"]["devices"] == [
        {
            "name": "Intel RealSense D405",
            "serial": "234567890123",
            "firmware": "5.16.0.1",
            "usb_type": "2.1",
            "usb2": True,
        }
    ]
    assert data["realsense"]["installed"] is True
    assert data["realsense"]["udev_rules"] is None


def test_parse_realsense_listing():
    parse = pi.parse_realsense_listing
    line = json.dumps(
        {
            "devices": [
                {"name": "D435", "serial_number": "1", "usb_type_descriptor": "3.2"},
                {"name": "", "firmware_version": None},
                "not a device",
            ]
        }
    )
    listing = parse(f"librealsense says hello\n{line}\n[1, 2]\n")
    assert listing == RealSenseListing(
        devices=(
            RealSenseDevice("D435", serial="1", usb_type="3.2"),
            RealSenseDevice("RealSense camera"),
        )
    )
    assert parse('{"error": "RuntimeError: x"}') == RealSenseListing(error="RuntimeError: x")
    assert parse('{"devices": {"a": 1}}') == RealSenseListing()
    # A crash after printing doesn't lose the listing; a crash before it is reported.
    assert parse('{"devices": []}', -11) == RealSenseListing()
    assert parse("", -11, "boom\nSegmentation fault\n") == RealSenseListing(
        error="pyrealsense2 exited with code -11: Segmentation fault"
    )
    assert parse("", 1) == RealSenseListing(error="pyrealsense2 exited with code 1")


# --- output ----------------------------------------------------------------------------


def test_report_is_json_serializable():
    report = pi.collect(zephyrus_linux())
    data = json.loads(json.dumps(report.to_dict()))
    assert data["gpus"]["vendors"] == ["amd", "nvidia"]
    assert data["serial_ports"][0]["chip"] == "WCH CH343"
    assert data["accelerator"]["kind"] == "cuda"
    missing = pi.collect(fake_probes(serial_ports=lambda: None)).to_dict()
    assert missing["serial_ports"] is None


def test_checks_match_the_diagnostic_check_fields():
    from so101.services.schemas import DiagnosticCheck

    for check in pi.checks(pi.collect(zephyrus_linux(access_rw=lambda p: False))):
        DiagnosticCheck(**dataclasses.asdict(check))
        assert check.status in {"pass", "warn", "fail"}


def test_torch_missing_and_torch_broken_checks():
    missing = pi.collect(fake_probes())
    assert statuses(missing)["platform.torch"] == "warn"
    assert "torch is not installed" in pi.format_text(missing)
    broken = pi.collect(fake_probes(torch=lambda: TorchInfo(version="2.11.0", error="E: x")))
    assert statuses(broken)["platform.torch"] == "fail"
    assert "torch error" in pi.format_text(broken)


def test_main_prints_text_or_json(monkeypatch, capsys):
    report = pi.collect(zephyrus_linux())
    monkeypatch.setattr(pi, "collect", lambda probes=None: report)
    assert pi.main([]) == 0
    assert "Data folder" in capsys.readouterr().out
    assert pi.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["os"] == "linux"


def test_main_data_dir_prints_only_the_folder_without_probing(monkeypatch, capsys, tmp_path):
    def no_probing(probes=None):
        raise AssertionError("--data-dir must not run the probes")

    monkeypatch.setattr(pi, "collect", no_probing)
    monkeypatch.setenv("SO101_HOME", str(tmp_path / "suite"))
    assert pi.main(["--data-dir"]) == 0
    assert capsys.readouterr().out == f"{tmp_path / 'suite'}\n"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SO101_HOME", "relative")
    assert pi.main(["--data-dir"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "relative")


def test_main_realsense_udev_rules_prints_the_shipped_file(monkeypatch, capsys):
    def no_probing(probes=None):
        raise AssertionError("--realsense-udev-rules must not run the probes")

    monkeypatch.setattr(pi, "collect", no_probing)
    monkeypatch.setattr(pi, "realsense_udev_rules_source", lambda: "/opt/so101/rs.rules")
    assert pi.main(["--realsense-udev-rules"]) == 0
    assert capsys.readouterr().out == "/opt/so101/rs.rules\n"
    monkeypatch.setattr(pi, "realsense_udev_rules_source", lambda: None)
    assert pi.main(["--realsense-udev-rules"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "no RealSense udev rules file" in captured.err


# --- the real probes, against fakes or this machine --------------------------------------


def test_run_returns_none_for_missing_or_failing_commands():
    assert pi._run(["so101-no-such-command-xyz"]) is None
    assert pi._run([sys.executable, "-c", "raise SystemExit(3)"]) is None
    assert pi._run([sys.executable, "-c", "print('ok')"]) == "ok\n"


def test_probe_torch_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert pi.probe_torch() is None


def fake_torch(*, cuda: bool, mps: bool, fail: bool = False) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.__version__ = "2.11.0+cu128" if cuda else "2.11.0"
    module.version = types.SimpleNamespace(cuda="12.8" if cuda else None)

    def is_available() -> bool:
        if fail:
            raise RuntimeError("CUDA driver initialization failed")
        return cuda

    module.cuda = types.SimpleNamespace(
        is_available=is_available,
        get_device_name=lambda index: "NVIDIA GeForce RTX 3070 Laptop GPU",
        get_device_properties=lambda index: types.SimpleNamespace(total_memory=8 * 2**30),
    )
    module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return module


def test_probe_torch_reads_cuda_and_mps(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=True, mps=False))
    assert pi.probe_torch() == TorchInfo(
        version="2.11.0+cu128",
        cuda_build="12.8",
        cuda_available=True,
        device_name="NVIDIA GeForce RTX 3070 Laptop GPU",
        vram_gib=8.0,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=False, mps=True))
    assert pi.probe_torch() == TorchInfo(version="2.11.0", mps_available=True)


def test_probe_torch_reports_a_failing_probe(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=True, mps=False, fail=True))
    info = pi.probe_torch()
    assert info is not None and info.version == "2.11.0+cu128"
    assert info.error is not None and "CUDA driver initialization failed" in info.error


def test_probe_torch_reports_a_broken_import(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("[WinError 126] The specified module could not be found")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    info = pi.probe_torch()
    assert info is not None and info.error is not None and "WinError 126" in info.error


def test_probe_serial_ports_without_pyserial(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)
    monkeypatch.setitem(sys.modules, "serial.tools", None)
    assert pi.probe_serial_ports() is None


def test_probe_serial_ports_with_fake_pyserial(monkeypatch):
    infos = [
        types.SimpleNamespace(
            device="/dev/ttyACM1",
            description="n/a",
            vid=None,
            pid=None,
            serial_number=None,
            manufacturer=None,
        ),
        types.SimpleNamespace(
            device="/dev/ttyACM0",
            description="USB Single Serial",
            vid=0x1A86,
            pid=0x55D3,
            serial_number="585A0077531",
            manufacturer="WCH",
        ),
    ]
    list_ports = types.ModuleType("serial.tools.list_ports")
    list_ports.comports = lambda: infos
    tools = types.ModuleType("serial.tools")
    tools.list_ports = list_ports
    serial = types.ModuleType("serial")
    serial.tools = tools
    monkeypatch.setitem(sys.modules, "serial", serial)
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports)

    ports = pi.probe_serial_ports()
    assert ports is not None
    assert [p.device for p in ports] == ["/dev/ttyACM0", "/dev/ttyACM1"]
    assert ports[0].chip == "WCH CH343"
    assert ports[1].description == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX group probes")
def test_posix_probes_on_this_machine(tmp_path):
    groups = pi._user_groups()
    assert groups is not None and groups.session
    target = tmp_path / "dev"
    target.write_text("", encoding="utf-8")
    assert pi._access_rw(str(target)) is True
    assert pi._device_group(str(target)) is not None
    assert pi._device_group(str(tmp_path / "missing")) is None
    assert isinstance(pi._pci_display_vendors(), frozenset)
    assert pi._long_paths_enabled() is None
    assert isinstance(pi._preferred_encoding(), str)
    assert isinstance(pi._utf8_mode(), bool)
    assert pi._python_version()[:2] == sys.version_info[:2]


def test_default_probes_collect_on_this_machine():
    report = pi.collect()
    assert report.os in {"macos", "linux", "windows"}
    assert report.data_dir
    assert pi.format_text(report)


FAKE_PYREALSENSE2 = """\
class camera_info:
    name = "name"
    serial_number = "serial_number"
    firmware_version = "firmware_version"
    usb_type_descriptor = "usb_type_descriptor"


class _Device:
    def __init__(self, info):
        self._info = info

    def supports(self, field):
        return field in self._info

    def get_info(self, field):
        value = self._info[field]
        if isinstance(value, Exception):
            raise value
        return value


class context:
    def query_devices(self):
        return [
            _Device({"name": "Intel RealSense D435", "serial_number": "1234",
                     "usb_type_descriptor": "3.2"}),
            _Device({"name": "Intel RealSense D405", "firmware_version": RuntimeError("busy")}),
        ]
"""


def fake_pyrealsense2(monkeypatch, tmp_path, source: str) -> None:
    package = tmp_path / "fake_site" / "pyrealsense2"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(package.parent))  # the child process inherits it


def test_probe_realsense_devices_lists_cameras_in_a_child_process(monkeypatch, tmp_path):
    fake_pyrealsense2(monkeypatch, tmp_path, FAKE_PYREALSENSE2)
    assert pi.probe_realsense_devices() == RealSenseListing(
        devices=(
            RealSenseDevice("Intel RealSense D435", serial="1234", usb_type="3.2"),
            RealSenseDevice("Intel RealSense D405"),
        )
    )


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (
            "class context:\n"
            "    def query_devices(self):\n"
            "        raise RuntimeError('no access')\n",
            "RuntimeError: no access",
        ),
        ("raise ImportError('libusb missing')\n", "ImportError: libusb missing"),
        ("import os\nos._exit(3)\n", "pyrealsense2 exited with code 3"),
    ],
)
def test_probe_realsense_devices_survives_failures(monkeypatch, tmp_path, source, error):
    fake_pyrealsense2(monkeypatch, tmp_path, source)
    assert pi.probe_realsense_devices() == RealSenseListing(error=error)


def test_probe_realsense_devices_gives_up_on_a_hung_camera(monkeypatch, tmp_path):
    fake_pyrealsense2(monkeypatch, tmp_path, "import time\ntime.sleep(60)\n")
    monkeypatch.setattr(pi, "REALSENSE_PROBE_TIMEOUT", 1.0)
    listing = pi.probe_realsense_devices()
    assert listing == RealSenseListing(error="pyrealsense2 didn't list the cameras within 1 s")


def test_probe_realsense_devices_without_a_python(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "executable", "")
    assert pi.probe_realsense_devices().error == "no Python executable to run pyrealsense2 in"
    monkeypatch.setattr(sys, "executable", str(tmp_path / "no-such-python"))
    error = pi.probe_realsense_devices().error
    assert error is not None and error.startswith(("FileNotFoundError", "PermissionError"))


def test_realsense_udev_rules_source_asks_the_camera_code(monkeypatch, tmp_path):
    rules = tmp_path / "99-realsense-libusb.rules"
    rules.write_text("# rules\n", encoding="utf-8")
    module = types.ModuleType("fake_realsense")
    monkeypatch.setattr(pi, "REALSENSE_RULES_FUNCTION", ("fake_realsense", "udev_rules_path"))
    monkeypatch.setitem(sys.modules, "fake_realsense", module)

    class Traversable:  # what importlib.resources.files() can return: not os.PathLike
        def __str__(self) -> str:
            return str(rules)

    def broken() -> str:
        raise RuntimeError("camera code failed")

    for answer, expected in [
        (lambda: rules, str(rules)),
        (lambda: str(rules), str(rules)),
        (Traversable, str(rules)),
        (lambda: tmp_path / "missing.rules", None),
        (lambda: None, None),
        (broken, None),
    ]:
        module.udev_rules_path = answer  # type: ignore[attr-defined]
        assert pi.realsense_udev_rules_source() == expected
    monkeypatch.setattr(pi, "REALSENSE_RULES_FUNCTION", ("so101_no_such_module", "f"))
    assert pi.realsense_udev_rules_source() is None


def test_distribution_module_and_dotenv_probes(monkeypatch, tmp_path):
    assert pi._dist_version("pytest") == pytest.__version__
    assert pi._dist_version("so101-no-such-distribution") is None
    assert pi._module_present("json") is True
    assert pi._module_present("so101_no_such_module") is False
    assert pi._module_present("") is False
    monkeypatch.chdir(tmp_path)
    assert pi._read_dotenv() == {}
    (tmp_path / ".env").write_text("SO101_CAMERAS=front=realsense\n", encoding="utf-8")
    # Keys come back lowercased (pydantic-settings matches them case-insensitively).
    assert pi._read_dotenv() == {"so101_cameras": "front=realsense"}
    (tmp_path / ".env").write_bytes(b"SO101_CAMERAS=\xff\n")
    assert pi._read_dotenv() == {}


# --- Linux setup, sim render, disk space ----------------------------------------------


def linux_box(**overrides) -> Probes:
    """The Zephyrus on Ubuntu: NVIDIA driver loaded, a CH343 arm adapter, on AC power."""

    values = dict(
        env={"DISPLAY": ":0"},
        exists=lambda path: path == pi.NVIDIA_PROC,
        find_library=lambda name: "libEGL.so.1" if name == "EGL" else None,
        serial_ports=lambda: [CH343],
    )
    values.update(overrides)
    return fake_probes(**values)


def check_of(report: pi.PlatformReport, check_id: str) -> pi.PlatformCheck:
    return next(check for check in pi.checks(report) if check.id == check_id)


def test_mujoco_gl_is_egl_on_linux_with_the_nvidia_driver_even_with_a_display():
    egl = lambda name: "libEGL.so.1" if name == "EGL" else None  # noqa: E731
    with_gpu = pi.mujoco_gl_advice("linux", {"DISPLAY": ":0"}, egl, nvidia=True)
    assert with_gpu.recommended == "egl" and "NVIDIA" in with_gpu.reason
    assert pi.mujoco_gl_advice("linux", {"DISPLAY": ":0"}, egl).recommended is None
    no_egl = pi.mujoco_gl_advice("linux", {"DISPLAY": ":0"}, lambda name: None, nvidia=True)
    assert no_egl.recommended is None
    assert pi.collect(linux_box()).mujoco_gl.recommended == "egl"


def test_render_probe_runs_with_the_mujoco_gl_the_suite_uses():
    seen: list[str | None] = []

    def render(gl):
        seen.append(gl)
        return RenderCheck(gl, True, "rendered a 64x64 frame")

    with_mujoco = {"dist_version": lambda name: "3.3.7" if name == "mujoco" else None}
    report = pi.collect(linux_box(mujoco_render=render, **with_mujoco))
    assert seen == ["egl"]
    check = check_of(report, "platform.mujoco_render")
    assert (check.status, check.detail) == ("pass", "rendered a 64x64 frame (MUJOCO_GL=egl)")

    seen.clear()
    pi.collect(linux_box(mujoco_render=render, env={"MUJOCO_GL": "osmesa"}, **with_mujoco))
    assert seen == ["osmesa"]
    seen.clear()
    off = pi.collect(linux_box(mujoco_render=render, env={"MUJOCO_GL": "disable"}, **with_mujoco))
    assert seen == [] and off.render is None
    assert pi.collect(linux_box(mujoco_render=render)).render is None  # no mujoco installed
    assert seen == []


def test_a_failed_render_is_a_warning_with_the_fix_for_the_os():
    failed = lambda gl: RenderCheck(gl, False, "FatalError: gladLoadGL error")  # noqa: E731
    with_mujoco = {"dist_version": lambda name: "3.3.7" if name == "mujoco" else None}
    report = pi.collect(linux_box(mujoco_render=failed, **with_mujoco))
    check = check_of(report, "platform.mujoco_render")
    assert check.status == "warn"
    assert check.detail == "rendering failed (MUJOCO_GL=egl): FatalError: gladLoadGL error"
    assert "sudo apt install libegl1 libgl1" in check.remediation
    assert "Sim render" in pi.format_text(report)
    mac = pi.collect(apple_silicon(mujoco_render=failed, **with_mujoco))
    assert "graphics driver" in check_of(mac, "platform.mujoco_render").remediation
    assert "(MUJOCO_GL unset)" in check_of(mac, "platform.mujoco_render").detail


@pytest.mark.parametrize(
    ("stdout", "returncode", "stderr", "expected"),
    [
        (
            'noise\n{"ok": true, "detail": "rendered a 64x64 frame"}\n',
            0,
            "",
            (True, "rendered a 64x64 frame"),
        ),
        (
            '{"ok": false, "detail": "RuntimeError: no EGL"}\n',
            0,
            "",
            (False, "RuntimeError: no EGL"),
        ),
        (
            "",
            -11,
            "libEGL warning\nSegmentation fault\n",
            (False, "MuJoCo exited with code -11: Segmentation fault"),
        ),
        ("not json\n", 1, "", (False, "MuJoCo exited with code 1")),
    ],
)
def test_parse_render_result(stdout, returncode, stderr, expected):
    result = pi.parse_render_result("egl", stdout, returncode, stderr)
    assert (result.ok, result.detail) == expected and result.gl == "egl"


def fake_mujoco(monkeypatch, tmp_path, source: str) -> None:
    package = tmp_path / "fake_site" / "mujoco"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(package.parent))


FAKE_MUJOCO = """
import os

class MjModel:
    @staticmethod
    def from_xml_string(xml):
        return MjModel()

class MjData:
    def __init__(self, model):
        pass

def mj_forward(model, data):
    pass

class _Pixels:
    shape = (64, 64, 3)

class Renderer:
    def __init__(self, model, height, width):
        if os.environ.get("MUJOCO_GL") == "broken":
            raise RuntimeError("no OpenGL context")
    def update_scene(self, data):
        pass
    def render(self):
        return _Pixels()
"""


def test_probe_mujoco_render_in_a_child_process(monkeypatch, tmp_path):
    fake_mujoco(monkeypatch, tmp_path, FAKE_MUJOCO)
    monkeypatch.setenv("MUJOCO_GL", "broken")  # the probe replaces it with the one asked for
    assert pi.probe_mujoco_render("egl") == RenderCheck("egl", True, "rendered a 64x64 frame")
    assert pi.probe_mujoco_render(None) == RenderCheck(None, True, "rendered a 64x64 frame")
    broken = pi.probe_mujoco_render("broken")
    assert broken == RenderCheck("broken", False, "RuntimeError: no OpenGL context")


def test_probe_mujoco_render_gives_up_on_a_hang(monkeypatch, tmp_path):
    fake_mujoco(monkeypatch, tmp_path, "import time\ntime.sleep(30)\n")
    monkeypatch.setattr(pi, "MUJOCO_RENDER_TIMEOUT", 1.0)
    assert pi.probe_mujoco_render("egl") == RenderCheck("egl", False, "no frame within 1 s")
    monkeypatch.setattr(pi.sys, "executable", "")
    assert pi.probe_mujoco_render("egl").detail == "no Python executable to run MuJoCo in"


@pytest.mark.parametrize(("free_gib", "status"), [(250.0, "pass"), (12.0, "warn"), (1.5, "fail")])
def test_disk_space(free_gib, status):
    report = pi.collect(fake_probes(disk_free=lambda path: int(free_gib * 2**30)))
    check = check_of(report, "platform.disk")
    assert check.status == status
    assert check.detail == f"{free_gib:g} GB free for the data folder ({report.data_dir})"
    assert ("SO101_HOME" in check.remediation) is (status != "pass")
    assert "platform.disk" not in statuses(pi.collect(fake_probes(disk_free=lambda path: None)))


def test_real_device_ownership_state_is_a_doctor_gate() -> None:
    base = pi.collect(fake_probes())
    missing = dataclasses.replace(
        base,
        device_state_path="/var/lib/so101/device-ownership-v2",
        device_state_problem="locks directory is unavailable",
    )
    failed = check_of(missing, "platform.device_ownership")
    assert failed.status == "fail"
    assert "re-run install/install.sh" in failed.remediation.lower()

    ready = dataclasses.replace(missing, device_state_problem=None)
    assert check_of(ready, "platform.device_ownership").status == "pass"


def test_missing_real_device_state_has_installer_remediation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(path: object) -> None:
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(pi.os, "lstat", missing)
    path, problem = pi._device_state_status("linux", {})
    assert path == "/var/lib/so101/device-ownership-v2"
    assert problem is not None and "unavailable" in problem


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_rehearsal_does_not_require_real_device_ownership(backend: str) -> None:
    assert pi._real_hardware_configured({"SO101_BACKEND": backend}, lambda: {}) is False
    report = pi.collect(fake_probes(env={"SO101_BACKEND": backend}))
    assert report.device_state_path == ""


def test_disk_free_probe_walks_up_to_an_existing_folder(tmp_path, monkeypatch):
    # Free space moves while other processes write, so compare the folder the
    # probe asks about, not two live readings.
    asked: list[str] = []

    def disk_usage(path):
        asked.append(str(path))
        return SimpleNamespace(total=100, used=40, free=60)

    monkeypatch.setattr(pi.shutil, "disk_usage", disk_usage)
    assert pi._disk_free(str(tmp_path / "not" / "yet")) == 60
    assert asked == [str(tmp_path)]


def test_arm_udev_rules_are_checked_when_the_robot_extra_is_installed():
    missing = check_of(pi.collect(linux_box()), "platform.udev.arms")
    assert missing.status == "warn" and "--follower-serial" in missing.remediation

    installed = {pi.NVIDIA_PROC, pi.ARM_UDEV_RULES, "/dev/so101-follower"}
    report = pi.collect(linux_box(exists=lambda path: path in installed))
    check = check_of(report, "platform.udev.arms")
    assert check.status == "pass"
    assert check.detail == (
        f"udev rules installed ({pi.ARM_UDEV_RULES}); present: /dev/so101-follower"
    )
    no_pyserial = pi.collect(linux_box(serial_ports=lambda: None))
    assert "platform.udev.arms" not in statuses(no_pyserial)
    assert pi.collect(apple_silicon()).linux is None


def test_brltty_is_flagged_only_when_it_hides_a_ch34x_adapter():
    brltty = {pi.NVIDIA_PROC, "/usr/bin/brltty"}
    hidden = pi.collect(
        linux_box(
            exists=lambda path: path in brltty,
            usb_vendor_ids=lambda: frozenset({0x1A86, 0x8086}),
            serial_ports=lambda: [],
        )
    )
    check = check_of(hidden, "platform.brltty")
    assert check.status == "warn" and "sudo apt remove brltty" in check.remediation

    visible = pi.collect(
        linux_box(exists=lambda path: path in brltty, usb_vendor_ids=lambda: frozenset({0x1A86}))
    )
    assert "platform.brltty" not in statuses(visible)  # the CH343 has its port
    no_adapter = pi.collect(linux_box(exists=lambda path: path in brltty, serial_ports=lambda: []))
    assert "platform.brltty" not in statuses(no_adapter)


@pytest.mark.parametrize(
    ("gsettings", "status", "phrase"),
    [
        ("'nothing'\n", "pass", "Automatic suspend on AC power is off"),
        ("'suspend'\n", "warn", "GNOME suspends this laptop when idle on AC power (suspend)"),
        (None, "warn", "couldn't be read"),
    ],
)
def test_a_laptop_gets_the_sleep_warning(gsettings, status, phrase):
    report = pi.collect(linux_box(has_battery=lambda: True, run=runner({"gsettings": gsettings})))
    check = check_of(report, "platform.sleep")
    assert check.status == status
    assert check.detail.startswith("A sleeping or suspended laptop can't stop the arm")
    assert phrase in check.detail
    assert ("sleep-inactive-ac-type 'nothing'" in check.remediation) is (status == "warn")
    desktop = pi.collect(linux_box())
    assert "platform.sleep" not in statuses(desktop)


# --- settings files (the data folder's .env, then ./.env) ------------------------------------


def test_settings_env_files_are_the_data_folder_then_the_current_folder(tmp_path):
    from pathlib import Path

    home, work = tmp_path / "home", tmp_path / "work"
    linux = pi.settings_env_files("linux", {}, home, work)
    assert linux == (home / ".local/share/so101/.env", work / ".env")
    xdg = pi.settings_env_files("linux", {"XDG_DATA_HOME": str(tmp_path / "x")}, home, work)
    assert xdg[0] == tmp_path / "x" / "so101" / ".env"
    mac = pi.settings_env_files("macos", {}, home, work)
    assert mac[0] == home / "Library/Application Support/so101/.env"
    # SO101_HOME wins; a relative one is taken from the current folder.
    assert pi.settings_env_files("linux", {"SO101_HOME": "rel"}, home, work)[0] == (
        work / "rel" / ".env"
    )
    # Run from the data folder (as the launcher does): one file, listed once,
    # also when the two spellings differ only by a symlink.
    folder = tmp_path / "suite"
    folder.mkdir()
    assert pi.settings_env_files("linux", {"SO101_HOME": str(folder)}, home, folder) == (
        folder / ".env",
    )
    link = tmp_path / "link"
    link.symlink_to(folder)
    assert len(pi.settings_env_files("linux", {"SO101_HOME": str(link)}, home, folder)) == 1
    # A deleted working directory: only the data folder's file.
    assert pi.settings_env_files("linux", {"SO101_HOME": str(folder)}, home, None) == (
        Path(folder) / ".env",
    )


def test_current_env_files_follow_so101_home_and_the_working_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("SO101_HOME", str(tmp_path / "suite"))
    monkeypatch.chdir(tmp_path)
    assert pi.current_env_files() == (tmp_path / "suite" / ".env", tmp_path / ".env")
    assert pi.current_data_home() == tmp_path / "suite"

    def gone():
        raise FileNotFoundError("the working directory was deleted")

    monkeypatch.setattr(pi.Path, "cwd", staticmethod(gone))
    assert pi.current_env_files() == (tmp_path / "suite" / ".env",)


def test_read_dotenv_merges_the_files_and_the_current_folder_wins(tmp_path, monkeypatch):
    suite, work = tmp_path / "suite", tmp_path / "work"
    suite.mkdir()
    work.mkdir()
    (suite / ".env").write_text(
        "SO101_CAMERAS=realsense\nSO101_POLICY_DEVICE=cuda\n", encoding="utf-8"
    )
    (work / ".env").write_text("so101_cameras=0\n", encoding="utf-8")
    monkeypatch.setenv("SO101_HOME", str(suite))
    monkeypatch.chdir(work)
    merged = pi._read_dotenv()
    assert merged == {"so101_cameras": "0", "so101_policy_device": "cuda"}
    assert pi.camera_setting({}, pi._read_dotenv) == "0"
    assert pi.env_setting({}, pi._read_dotenv, "so101_policy_device", "") == "cuda"
    (work / ".env").write_bytes(b"\xff\xfe not utf-8")  # unreadable: skipped
    assert pi._read_dotenv()["so101_cameras"] == "realsense"


def test_update_env_text_creates_adds_keeps_and_replaces():
    created = pi.update_env_text(None, {"SO101_DATA_DIR": "/srv/so101/data"})
    assert created.created and created.changed
    assert created.added == (("SO101_DATA_DIR", "/srv/so101/data"),)
    assert created.text.startswith("# SO-101 suite settings")
    assert created.text.endswith("\nSO101_DATA_DIR=/srv/so101/data\n")

    text = "# mine\nso101_follower_port=/dev/ttyACM0  # keep\nSO101_MOCK=0"
    kept = pi.update_env_text(
        text, {"SO101_FOLLOWER_PORT": "/dev/so101-follower", "SO101_MOCK": "0"}
    )
    assert kept.kept == (("SO101_FOLLOWER_PORT", "/dev/ttyACM0", "/dev/so101-follower"),)
    assert not kept.changed and kept.text == text  # an equal value is left alone too

    replaced = pi.update_env_text(
        text, {"SO101_FOLLOWER_PORT": "/dev/so101-follower"}, replace=True
    )
    assert replaced.replaced == (("SO101_FOLLOWER_PORT", "/dev/ttyACM0", "/dev/so101-follower"),)
    assert replaced.text == "# mine\nSO101_FOLLOWER_PORT=/dev/so101-follower\nSO101_MOCK=0\n"

    added = pi.update_env_text("SO101_MOCK=0\n", {"SO101_LEADER_PORT": "/dev/so101-leader"})
    assert added.text == "SO101_MOCK=0\n\nSO101_LEADER_PORT=/dev/so101-leader\n"


@pytest.mark.parametrize(
    ("value", "line"),
    [
        ("/home/ada/.local/share/so101/data", "SO101_DATA_DIR=/home/ada/.local/share/so101/data"),
        (
            "/Users/ada/Library/Application Support/so101/data",
            "SO101_DATA_DIR='/Users/ada/Library/Application Support/so101/data'",
        ),
        ("/data/#1", "SO101_DATA_DIR='/data/#1'"),
        ("it's", 'SO101_DATA_DIR="it\'s"'),
        ("", "SO101_DATA_DIR="),
    ],
)
def test_update_env_text_quotes_values_so_they_read_back(value, line):
    text = pi.update_env_text("", {"SO101_DATA_DIR": value}).text
    assert line in text.splitlines()
    assert pi.parse_dotenv(text)["SO101_DATA_DIR"] == value


def test_update_env_text_refuses_names_and_values_it_cant_write():
    with pytest.raises(ValueError, match="not a suite setting name"):
        pi.update_env_text(None, {"PATH": "/x"})
    with pytest.raises(ValueError, match="span lines"):
        pi.update_env_text(None, {"SO101_DATA_DIR": "a\nSO101_MOCK=1"})


def test_update_env_file_creates_a_private_file_and_writes_only_changes(tmp_path):
    import os
    import stat

    path = tmp_path / "suite" / ".env"
    first = pi.update_env_file(path, pi.storage_settings(tmp_path / "suite"))
    assert first.created and path.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600  # may hold tokens
    assert pi.parse_dotenv(path.read_text(encoding="utf-8")) == {
        "SO101_DATA_DIR": str(tmp_path / "suite" / "data"),
        "SO101_DATASETS_DIR": str(tmp_path / "suite" / "datasets"),
        "SO101_OUTPUTS_DIR": str(tmp_path / "suite" / "outputs"),
    }
    before = path.stat().st_mtime_ns
    again = pi.update_env_file(path, pi.storage_settings(tmp_path / "suite"))
    assert not again.changed and path.stat().st_mtime_ns == before
    assert pi.describe_env_update(path, again) == [f"settings file: {path} (unchanged)"]


def test_main_set_env_and_init_env_edit_the_data_folders_env(tmp_path, monkeypatch, capsys):
    def no_probing(probes=None):
        raise AssertionError("--set-env must not run the probes")

    monkeypatch.setattr(pi, "collect", no_probing)
    suite = tmp_path / "suite"
    monkeypatch.setenv("SO101_HOME", str(suite))
    env = suite / ".env"

    assert pi.main(["--init-env", "--set-env", "SO101_FOLLOWER_PORT=/dev/so101-follower"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"settings file: {env} (created)"
    assert f"  + SO101_DATA_DIR={suite / 'data'}" in out
    assert "  + SO101_FOLLOWER_PORT=/dev/so101-follower" in out

    env.write_text(
        env.read_text(encoding="utf-8").replace("/dev/so101-follower", "/dev/ttyACM0"), "utf-8"
    )
    assert pi.main(["--set-env", "SO101_FOLLOWER_PORT=/dev/so101-follower"]) == pi.SET_ENV_KEPT
    out = capsys.readouterr().out
    assert (
        "SO101_FOLLOWER_PORT=/dev/ttyACM0 kept; this install would set /dev/so101-follower" in out
    )
    assert pi.parse_dotenv(env.read_text("utf-8"))["SO101_FOLLOWER_PORT"] == "/dev/ttyACM0"

    replace = ["--set-env", "SO101_FOLLOWER_PORT=/dev/so101-follower", "--replace"]
    assert pi.main(replace) == 0
    assert "~ SO101_FOLLOWER_PORT=/dev/so101-follower (was /dev/ttyACM0)" in capsys.readouterr().out
    assert pi.parse_dotenv(env.read_text("utf-8"))["SO101_FOLLOWER_PORT"] == "/dev/so101-follower"

    assert pi.main(["--set-env", "PATH=/x"]) == 1
    assert "not a suite setting name" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pi.main(["--set-env", "SO101_MOCK"])


# --- the .env writer: private, atomic ---------------------------------------------------

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


def _mode(path) -> int:
    import stat

    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_update_env_file_is_owner_only_after_create_and_after_update(tmp_path):
    import os

    old_umask = os.umask(0)  # the file's mode must not depend on the umask
    try:
        path = tmp_path / "suite" / ".env"
        pi.update_env_file(path, {"SO101_DATA_DIR": "/srv/so101/data"})
        assert _mode(path) == 0o600
        pi.update_env_file(path, {"SO101_HF_TOKEN": "hf_secret"})
        assert _mode(path) == 0o600
        pi.update_env_file(path, {"SO101_HF_TOKEN": "hf_other"}, replace=True)
        assert _mode(path) == 0o600
        assert pi.parse_dotenv(path.read_text("utf-8"))["SO101_HF_TOKEN"] == "hf_other"
    finally:
        os.umask(old_umask)


@posix_only
def test_update_env_file_tightens_a_file_others_can_read_and_says_so(tmp_path):
    path = tmp_path / ".env"
    path.write_text("SO101_HF_TOKEN=hf_secret\n", encoding="utf-8")
    path.chmod(0o644)

    update = pi.update_env_file(path, {"SO101_DATA_DIR": "/srv/so101/data"})
    assert _mode(path) == 0o600
    assert update.tightened == "0644"
    lines = pi.describe_env_update(path, update)
    assert lines[0] == f"settings file: {path} (updated)"
    assert "  permissions: was 0644 (others could read it); now 0600, owner only" in lines

    # Nothing to add: the file isn't rewritten, but its mode is still fixed.
    path.chmod(0o640)
    before = path.read_text("utf-8")
    again = pi.update_env_file(path, {"SO101_DATA_DIR": "/srv/so101/data"})
    assert not again.changed and again.tightened == "0640"
    assert _mode(path) == 0o600 and path.read_text("utf-8") == before
    assert pi.describe_env_update(path, again)[0] == f"settings file: {path} (unchanged)"

    private = pi.update_env_file(path, {"SO101_DATA_DIR": "/srv/so101/data"})
    assert private.tightened is None
    assert pi.describe_env_update(path, private) == [f"settings file: {path} (unchanged)"]


def test_update_env_file_crash_mid_write_keeps_the_old_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    old = "SO101_HF_TOKEN=hf_secret\nSO101_DATA_DIR=/old\n"
    path.write_text(old, encoding="utf-8")
    real_open = io.open

    class CrashMidWrite:
        """A writable file that writes half of what it's given, then fails (disk full)."""

        def __init__(self, handle):
            self.handle = handle

        def write(self, data):
            self.handle.write(data[: len(data) // 2])
            self.handle.flush()
            raise OSError(28, "No space left on device (simulated)")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()
            return False

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def crashing_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        return CrashMidWrite(handle) if "w" in mode else handle

    monkeypatch.setattr(io, "open", crashing_open)
    with pytest.raises(OSError, match="simulated"):
        pi.update_env_file(path, {"SO101_DATA_DIR": "/new"}, replace=True)
    monkeypatch.undo()
    assert path.read_text("utf-8") == old  # never truncated or half-written
    assert [p.name for p in tmp_path.iterdir()] == [".env"]  # no temporary file left


def test_update_env_file_replace_failure_keeps_the_old_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("SO101_DATA_DIR=/old\n", encoding="utf-8")

    def no_replace(src, dst):
        raise OSError(13, "Permission denied (simulated)")

    monkeypatch.setattr(pi.os, "replace", no_replace)
    with pytest.raises(OSError, match="simulated"):
        pi.update_env_file(path, {"SO101_DATA_DIR": "/new"}, replace=True)
    assert path.read_text("utf-8") == "SO101_DATA_DIR=/old\n"
    assert [p.name for p in tmp_path.iterdir()] == [".env"]


@posix_only
def test_update_env_file_fsyncs_the_file_and_then_its_folder(tmp_path, monkeypatch):
    import os
    import stat

    synced: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        synced.append("folder" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(pi.os, "fsync", recording_fsync)
    pi.update_env_file(tmp_path / ".env", {"SO101_DATA_DIR": "/srv/so101/data"})
    assert synced == ["file", "folder"]


@posix_only
def test_update_env_file_keeps_a_symlinked_env_a_symlink(tmp_path):
    target = tmp_path / "dotfiles" / "so101.env"
    target.parent.mkdir()
    target.write_text("SO101_MOCK=0\n", encoding="utf-8")
    target.chmod(0o644)
    link = tmp_path / "suite" / ".env"
    link.parent.mkdir()
    link.symlink_to(target)

    update = pi.update_env_file(link, {"SO101_DATA_DIR": "/srv/so101/data"})
    assert link.is_symlink()
    assert pi.parse_dotenv(target.read_text("utf-8")) == {
        "SO101_MOCK": "0",
        "SO101_DATA_DIR": "/srv/so101/data",
    }
    assert _mode(target) == 0o600 and update.tightened == "0644"
    assert sorted(p.name for p in link.parent.iterdir()) == [".env"]
