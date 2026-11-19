"""Capability-detected image controls, owned by the relay capture thread."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path

from ..atomic_io import atomic_write_json

CONTROLS = {
    "brightness": 9963776,
    "contrast": 9963777,
    "saturation": 9963778,
    "white_balance_auto": 9963788,
    "gain": 9963795,
    "power_line_frequency": 9963800,
    "white_balance_temperature": 9963802,
    "sharpness": 9963803,
    "backlight_compensation": 9963804,
    "gamma": 9963792,
    "exposure_auto": 10094849,
    "exposure_absolute": 10094850,
    "exposure_auto_priority": 10094851,
}


class Query(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32),
        ("step", ctypes.c_int32),
        ("default", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class Value(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


class Menu(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("reserved", ctypes.c_uint32),
    ]


def ioctl(descriptor, number, structure):
    import fcntl

    operation = 0xC0000000 | (ctypes.sizeof(structure) << 16) | (ord("V") << 8) | number
    fcntl.ioctl(descriptor, operation, structure)
    return structure


class V4L2Controls:
    def __init__(self, device):
        self.descriptor = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        self.identity = str(Path(device))

    def snapshot(self):
        result = {}
        for name, identifier in CONTROLS.items():
            try:
                query = ioctl(self.descriptor, 36, Query(id=identifier))
                if query.flags & 1:
                    continue
                value = ioctl(self.descriptor, 27, Value(id=identifier)).value
                record = {
                    "value": value,
                    "minimum": query.minimum,
                    "maximum": query.maximum,
                    "step": query.step,
                    "default": query.default,
                    "writable": not bool(query.flags & (2 | 4 | 16)),
                    "inactive": bool(query.flags & 16),
                    "label": query.name.decode(errors="replace"),
                }
                if query.type == 3:
                    choices = {}
                    for index in range(query.minimum, min(query.maximum, 255) + 1):
                        try:
                            item = ioctl(self.descriptor, 37, Menu(id=identifier, index=index))
                            choices[str(index)] = item.name.decode(errors="replace")
                        except OSError:
                            continue
                    record["choices"] = choices
                result[name] = record
            except OSError:
                continue
        return result

    def write(self, name, value):
        ioctl(self.descriptor, 28, Value(id=CONTROLS[name], value=value))

    def close(self):
        os.close(self.descriptor)


class RealSenseControls:
    def __init__(self, capture):
        import pyrealsense2 as rs

        device = capture.pipeline.get_active_profile().get_device()
        self.sensor = device.first_color_sensor()
        self.identity = "realsense:" + device.get_info(rs.camera_info.serial_number)
        self.options = {
            name: getattr(rs.option, option)
            for name, option in {
                "exposure_auto": "enable_auto_exposure",
                "exposure_absolute": "exposure",
                "white_balance_auto": "enable_auto_white_balance",
                "white_balance_temperature": "white_balance",
                "brightness": "brightness",
                "contrast": "contrast",
                "saturation": "saturation",
                "gamma": "gamma",
                "sharpness": "sharpness",
                "gain": "gain",
                "backlight_compensation": "backlight_compensation",
                "power_line_frequency": "power_line_frequency",
                "exposure_auto_priority": "auto_exposure_priority",
            }.items()
        }

    def snapshot(self):
        result = {}
        for name, option in self.options.items():
            if not self.sensor.supports(option):
                continue
            limits = self.sensor.get_option_range(option)
            result[name] = {
                "value": self.sensor.get_option(option),
                "minimum": limits.min,
                "maximum": limits.max,
                "step": limits.step,
                "default": limits.default,
                "writable": not self.sensor.is_option_read_only(option),
                "inactive": False,
                "label": name,
            }
        for mode, names in (
            ("exposure_auto", ("exposure_absolute", "gain")),
            ("white_balance_auto", ("white_balance_temperature",)),
        ):
            if result.get(mode, {}).get("value") == 1:
                for name in names:
                    if name in result:
                        result[name].update(inactive=True, writable=False)
        return result

    def write(self, name, value):
        self.sensor.set_option(self.options[name], float(value))

    def close(self):
        pass


class CameraProfile:
    def __init__(self, driver, directory: Path):
        self.driver = driver
        self.path = directory / (
            hashlib.sha256(driver.identity.encode()).hexdigest()[:24] + ".json"
        )
        self.saved = {}
        self.error = None
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text())
                if saved["identity"] != driver.identity:
                    raise ValueError("Camera profile identity mismatch")
                self.apply(saved["values"], persist=False)
                self.saved = saved["values"]
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                self.error = str(exc)

    def snapshot(self):
        return {
            "identity": self.driver.identity,
            "controls": self.driver.snapshot(),
            "saved_values": self.saved,
            "profile_error": self.error,
            "profile_sha256": hashlib.sha256(
                json.dumps(
                    {"identity": self.driver.identity, "values": self.saved}, sort_keys=True
                ).encode()
            ).hexdigest(),
            "exposure_unit": "100 microseconds (RGB/UVC exposure convention)",
        }

    def apply(self, changes, *, persist=True):
        if not isinstance(changes, dict) or not changes or len(changes) > len(CONTROLS):
            raise ValueError("Provide a nonempty control map")
        original = self.driver.snapshot()
        for name, value in changes.items():
            control = original.get(name)
            if control is None or name not in CONTROLS:
                raise ValueError(f"Unsupported camera control: {name}")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not control["minimum"] <= value <= control["maximum"]
            ):
                raise ValueError(f"Value outside camera range: {name}")
            step = control["step"] or 1
            if (
                abs(
                    (value - control["minimum"]) / step - round((value - control["minimum"]) / step)
                )
                > 1e-6
            ):
                raise ValueError(f"Value does not match camera step: {name}")
            if "choices" in control and str(value) not in control["choices"]:
                raise ValueError(f"Unsupported camera menu value: {name}")
        dependencies = {
            "exposure_auto": (
                ("exposure_absolute", "gain")
                if isinstance(self.driver, RealSenseControls)
                or original.get("gain", {}).get("inactive")
                else ("exposure_absolute",),
                0 if isinstance(self.driver, RealSenseControls) else 1,
            ),
            "white_balance_auto": (("white_balance_temperature",), 0),
        }
        ordered = sorted(
            changes, key=lambda name: (name not in {"exposure_auto", "white_balance_auto"}, name)
        )
        written = []
        try:
            for name in ordered:
                current = self.driver.snapshot()[name]
                if not current["writable"]:
                    raise ValueError(
                        f"Control is inactive/read-only: {name}; disable its automatic mode first"
                    )
                written.append(name)
                self.driver.write(name, changes[name])
                if abs(self.driver.snapshot()[name]["value"] - changes[name]) > 1e-6:
                    raise ValueError(f"Camera did not retain requested {name}")
            saved = {**self.saved, **changes}
            for mode, (manuals, manual_value) in dependencies.items():
                if mode in saved and saved[mode] != manual_value:
                    for manual in manuals:
                        saved.pop(manual, None)
            if self.driver.snapshot().get("gain", {}).get("inactive"):
                saved.pop("gain", None)
            if persist:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    self.path,
                    {"schema_version": 1, "identity": self.driver.identity, "values": saved},
                )
                self.saved, self.error = saved, None
        except Exception as failure:
            errors = []
            modes = {
                mode: manual_value
                for mode, (manuals, manual_value) in dependencies.items()
                if mode in original and any(manual in written for manual in manuals)
            }
            rollback = (
                [(mode, value) for mode, value in modes.items()]
                + [
                    (name, original[name]["value"])
                    for name in reversed(written)
                    if name not in modes
                ]
                + [(mode, original[mode]["value"]) for mode in modes]
            )
            for name, value in rollback:
                try:
                    self.driver.write(name, value)
                    if abs(self.driver.snapshot()[name]["value"] - value) > 1e-6:
                        raise ValueError(f"{name} rollback readback mismatch")
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                self.error = "Camera rollback unconfirmed: " + "; ".join(errors)
                raise ValueError(f"{failure}; {self.error}") from failure
            raise
        return self.snapshot()
