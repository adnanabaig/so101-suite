"""Serial ports through pyserial's comports(), as Windows, Linux and macOS name them."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from so101.bringup import serial_ports as sp


def pyserial_port(device: str, **fields: object) -> SimpleNamespace:
    """The attributes of pyserial 3.5's ListPortInfo (list_ports_common.py:38-50)."""
    base: dict[str, object] = {
        "device": device,
        "name": device.rsplit("/", 1)[-1],
        "description": "n/a",
        "hwid": "n/a",
        "vid": None,
        "pid": None,
        "serial_number": None,
        "location": None,
        "manufacturer": None,
        "product": None,
        "interface": None,
    }
    base.update(fields)
    return SimpleNamespace(**base)


def test_list_serial_ports_maps_and_sorts_comports() -> None:
    listing = sp.list_serial_ports(
        lambda: [
            pyserial_port(
                "COM4",
                vid=0x1A86,
                pid=0x55D3,
                serial_number="5A46",
                description="USB-Enhanced-SERIAL CH343 (COM4)",
                hwid="USB VID:PID=1A86:55D3 SER=5A46",
            ),
            pyserial_port("COM1"),
        ],
        platform="win32",
    )
    assert listing.error is None and listing.platform == "win32"
    assert [info.device for info in listing.ports] == ["COM1", "COM4"]
    plain, usb = listing.ports
    assert plain.description is None and plain.hwid is None and plain.vid_pid is None
    assert plain.identity() == "no USB identity"
    assert usb.vid_pid == "1A86:55D3" and usb.identity() == "serial 5A46, VID:PID 1A86:55D3"
    assert usb.to_dict()["vid_pid"] == "1A86:55D3"
    assert listing.describe() == "COM1 (no USB identity), COM4 (serial 5A46, VID:PID 1A86:55D3)"
    assert sp.PortListing(platform="linux").describe() == "none"


def test_list_serial_ports_reports_failures_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> list[object]:
        raise OSError("IOKit went away")

    failed = sp.list_serial_ports(broken, platform="darwin")
    assert (
        failed.ports == []
        and failed.error == "listing serial ports failed: OSError: IOKit went away"
    )
    assert "could not be listed" in failed.describe()

    monkeypatch.setitem(sys.modules, "serial", None)  # import serial -> ImportError
    missing = sp.list_serial_ports()
    assert missing.error is not None and missing.error.startswith("pyserial is not installed")


@pytest.mark.parametrize(
    ("platform", "first", "second", "same"),
    [
        ("win32", "COM3", "com3", True),
        ("win32", "\\\\.\\COM10", "COM10", True),
        ("win32", "COM3", "COM4", False),
        ("darwin", "/dev/tty.usbmodem5A46", "/dev/cu.usbmodem5A46", True),
        ("darwin", "/dev/tty.usbmodem5A46", "/dev/cu.usbmodem9999", False),
        ("linux", "/dev/tty.usbmodem5A46", "/dev/cu.usbmodem5A46", False),
        ("linux", "/dev/ttyACM0", "/dev/ttyACM0", True),
    ],
)
def test_port_key_joins_spellings_of_one_port(
    platform: str, first: str, second: str, same: bool
) -> None:
    assert (sp.port_key(first, platform=platform) == sp.port_key(second, platform=platform)) is same


def test_match_port_finds_the_macos_callout_twin() -> None:
    listed = [sp.SerialPortInfo(device="/dev/cu.usbmodem5A46", serial_number="5A46")]
    found = sp.match_port("/dev/tty.usbmodem5A46", listed, platform="darwin")
    assert found is not None and found.serial_number == "5A46"
    assert sp.match_port("/dev/tty.usbmodem0000", listed, platform="darwin") is None


def test_shared_serial_number() -> None:
    first = {"device": "/dev/ttyACM0", "serial_number": "X1"}
    assert sp.shared_serial_number(first, {"device": "/dev/ttyACM1", "serial_number": "X1"}) == "X1"
    assert sp.shared_serial_number(first, {"device": "/dev/ttyACM1", "serial_number": "X2"}) is None
    assert sp.shared_serial_number(first, dict(first)) is None  # one port, not two adapters
    assert sp.shared_serial_number(first, None) is None
    no_serial = {"device": "/dev/ttyACM0", "serial_number": None}
    assert sp.shared_serial_number(no_serial, {"device": "b", "serial_number": None}) is None


def test_remediation_names_the_platform_fix() -> None:
    linux = sp.permission_remediation("/dev/ttyACM0", platform="linux")
    assert "sudo usermod -aG dialout $USER" in linux and "ls -l /dev/ttyACM0" in linux
    assert "one program at a time" in sp.permission_remediation("COM3", platform="win32")
    assert sp.permission_remediation("/dev/cu.x", platform="darwin").startswith("Check the")

    windows = sp.not_found_remediation(
        sp.PortListing(ports=[sp.SerialPortInfo(device="COM3")], platform="win32")
    )
    assert "COM3 (no USB identity)" in windows and "Device Manager" in windows
    assert "lerobot-find-port" in windows and "SO101_FOLLOWER_PORT" in windows
    assert "/dev/cu.usbmodem" in sp.not_found_remediation(sp.PortListing(platform="darwin"))
    assert "/dev/serial/by-id/" in sp.not_found_remediation(sp.PortListing(platform="linux"))


def test_linux_hints_lead_with_the_udev_names_install_sh_creates() -> None:
    linux = sp.not_found_remediation(sp.PortListing(platform="linux"))
    assert linux.index("/dev/so101-follower") < linux.index("lerobot-find-port")
    assert linux.index("/dev/so101-follower") < linux.index("/dev/ttyACM0")
    assert "--follower-serial and --leader-serial" in linux
    assert linux.endswith("Serial ports present now: none.")
    hint = sp.port_naming_hint("linux")
    assert hint.startswith("On Linux the suite expects /dev/so101-follower and /dev/so101-leader")
    assert "COM3" in sp.port_naming_hint("win32")
    assert "/dev/tty.usbmodem" in sp.port_naming_hint("darwin")
    assert sp.port_naming_hint() == sp.port_naming_hint(sys.platform)
    mac = sp.not_found_remediation(sp.PortListing(platform="darwin"))
    assert mac.index("lerobot-find-port") < mac.index("/dev/tty.usbmodem")  # unchanged order
