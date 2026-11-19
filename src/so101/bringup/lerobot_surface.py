"""The LeRobot 0.6.1 surface the Suite relies on, pinned verbatim, plus a probe.

Provenance: the ``lerobot-0.6.1`` sdist (sha256 in :data:`LEROBOT_SDIST_SHA256`),
read file by file; paths below are relative to ``src/lerobot`` in that sdist and
line numbers refer to it. ``tests/test_bringup_lerobot061_pin.py`` holds an
independent verbatim copy of the register table and re-checks the installed
package whenever LeRobot is importable.

Nothing here talks to hardware. :func:`probe_lerobot_surface` only imports
modules and inspects signatures and tables.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

LEROBOT_PINNED_VERSION = "0.6.1"
LEROBOT_SDIST_SHA256 = "2111d67d72d9f6c2379e3b8d3ea41b58855066314d88f9c0893579b7707f04b5"
#: sha256 of the 0.6.1 source files the Suite depends on.
LEROBOT_SOURCE_SHA256: dict[str, str] = {
    "motors/feetech/tables.py": "71f7f7beb17169781bd26a33b165ed4c5c3df7b9c477d36860f9d6011eb00428",
    "motors/feetech/feetech.py": "0460413cd6a641cede015ac19ac53d2cfe9c343af1bd18ed74e1951229f64ff2",
    "motors/motors_bus.py": "6ce6109e32590ac2830f6995d13bb478b1a984695cc9ca59c0763ff091666393",
    "robots/so_follower/so_follower.py": (
        "debe2661e7e44761a74f7cb6aaa83838a0771f91d86f80f42fa822ebb9cdd74b"
    ),
    "teleoperators/so_leader/so_leader.py": (
        "7f16325a61e0fa27e8f7a0ea3ef282a952d6c63bcc1fb589dfa458d7705b799d"
    ),
    "utils/constants.py": "966887445ea327a72724602b625650212a4c85148ab516e0784a41f167687e73",
}

# motors/feetech/tables.py:15-17
FIRMWARE_MAJOR_VERSION = (0, 1)
FIRMWARE_MINOR_VERSION = (1, 1)
MODEL_NUMBER = (3, 2)

#: motors/feetech/tables.py:41-101, verbatim (data_name: (address, size_byte)),
#: including upstream's trailing space in "Acceleration_Multiplier " and the
#: shared address 80 of "Moving_Velocity" and "Moving_Velocity_Threshold".
STS_SMS_SERIES_CONTROL_TABLE: dict[str, tuple[int, int]] = {
    # EPROM
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,  # read-only
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,  # read-only
    "Model_Number": MODEL_NUMBER,  # read-only
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Velocity_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    # SRAM
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),  # read-only
    "Present_Velocity": (58, 2),  # read-only
    "Present_Load": (60, 2),  # read-only
    "Present_Voltage": (62, 1),  # read-only
    "Present_Temperature": (63, 1),  # read-only
    "Status": (65, 1),  # read-only
    "Moving": (66, 1),  # read-only
    "Present_Current": (69, 2),  # read-only
    "Goal_Position_2": (71, 2),  # read-only
    # Factory
    "Moving_Velocity": (80, 1),
    "Moving_Velocity_Threshold": (80, 1),
    "DTs": (81, 1),  # (ms)
    "Velocity_Unit_factor": (82, 1),
    "Hts": (83, 1),  # (ns) valid for firmware >= 2.54, other versions keep 0
    "Maximum_Velocity_Limit": (84, 1),
    "Maximum_Acceleration": (85, 1),
    "Acceleration_Multiplier ": (86, 1),  # Acceleration multiplier in effect when acceleration is 0
}

#: motors/feetech/tables.py:154-163
STS_SMS_SERIES_BAUDRATE_TABLE: dict[int, int] = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}

#: motors/feetech/tables.py:207-216 (sign-magnitude sign bit per register)
STS_SMS_SERIES_ENCODINGS_TABLE: dict[str, int] = {
    "Present_Load": 10,
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Goal_Velocity": 15,
    "Goal_Speed": 15,
    "Present_Position": 15,
    "Present_Velocity": 15,
    "Present_Speed": 15,
}

STS3215_MODEL = "sts3215"
STS3215_MODEL_NUMBER = 777  # tables.py:242-247 MODEL_NUMBER_TABLE["sts3215"]
STS3215_RESOLUTION = 4096  # tables.py:186-194 MODEL_RESOLUTION["sts3215"]
STS3215_PROTOCOL = 0  # tables.py:249-257 MODEL_PROTOCOL["sts3215"]
DEFAULT_BAUDRATE = 1_000_000  # motors/feetech/feetech.py:44
DEFAULT_TIMEOUT_MS = 1000  # motors/feetech/feetech.py:45
NORMALIZED_DATA = ("Goal_Position", "Present_Position")  # motors/feetech/feetech.py:47
#: MotorNormMode.DEGREES: (val - mid) * 360 / (resolution - 1), motors_bus.py:874-877.
DEGREES_PER_TICK = 360.0 / (STS3215_RESOLUTION - 1)

#: Registers the bring-up kit reads. It never writes any register.
DIAGNOSTIC_REGISTERS: tuple[str, ...] = (
    "Model_Number",
    "Firmware_Major_Version",
    "Firmware_Minor_Version",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Max_Temperature_Limit",
    "Max_Voltage_Limit",
    "Min_Voltage_Limit",
    "Homing_Offset",
    "Torque_Enable",
    "Present_Position",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
    "Status",
    "Present_Current",
)

#: Public symbols the Suite imports.
REQUIRED_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("lerobot.motors", "Motor"),
    ("lerobot.motors", "MotorCalibration"),
    ("lerobot.motors", "MotorNormMode"),
    ("lerobot.motors.feetech", "FeetechMotorsBus"),
    ("lerobot.motors.feetech", "OperatingMode"),
    ("lerobot.robots.so_follower", "SO101Follower"),
    ("lerobot.robots.so_follower", "SO101FollowerConfig"),
    ("lerobot.teleoperators.so_leader", "SO101Leader"),
    ("lerobot.teleoperators.so_leader", "SO101LeaderConfig"),
    ("lerobot.utils.constants", "HF_LEROBOT_CALIBRATION"),
)

#: Parameters the Suite passes to ``FeetechMotorsBus`` methods. ``disable_torque``
#: is FeetechMotorsBus's own (motors/feetech/feetech.py:291); the rest come from
#: ``SerialMotorsBus`` in motors/motors_bus.py: connect 514, disconnect 547,
#: set_timeout 693, set_baudrate 711, ping 948, read 995, sync_read 1128.
REQUIRED_PARAMETERS: dict[str, tuple[str, ...]] = {
    "connect": ("handshake",),
    "disconnect": ("disable_torque",),
    "disable_torque": ("motors", "num_retry"),
    "set_timeout": ("timeout_ms",),
    "set_baudrate": ("baudrate",),
    "ping": ("motor", "num_retry", "raise_on_error"),
    "read": ("data_name", "motor", "normalize", "num_retry"),
    "sync_read": ("data_name", "motors", "normalize", "num_retry"),
}


def compare_control_table(installed: Mapping[str, Any]) -> list[str]:
    """Differences between an installed control table and the pinned one."""
    problems: list[str] = []
    for name, expected in STS_SMS_SERIES_CONTROL_TABLE.items():
        if name not in installed:
            problems.append(f"register {name!r} is missing")
        elif tuple(installed[name]) != expected:
            problems.append(f"register {name!r} is {tuple(installed[name])}, pinned {expected}")
    for name in installed:
        if name not in STS_SMS_SERIES_CONTROL_TABLE:
            problems.append(f"register {name!r} is not in the pinned table")
    return problems


@dataclass
class LeRobotSurface:
    """What :func:`probe_lerobot_surface` found."""

    installed: bool = False
    version: str | None = None
    importable: bool = False
    sdk_importable: bool = False
    lerobot_home_set: bool = False
    import_error: str | None = None
    missing_symbols: list[str] = field(default_factory=list)
    signature_problems: list[str] = field(default_factory=list)
    table_problems: list[str] = field(default_factory=list)
    tables_checked: bool = False

    @property
    def version_ok(self) -> bool:
        return self.version == LEROBOT_PINNED_VERSION

    def problems(self) -> list[str]:
        """What is wrong, limited to what the probe actually checked."""
        found: list[str] = []
        if self.lerobot_home_set:
            found.append(
                "LEROBOT_HOME is set; LeRobot 0.6.1 raises ValueError on import "
                "(utils/constants.py:61-65). Use HF_LEROBOT_HOME instead."
            )
        if not self.installed:
            found.append("LeRobot is not installed")
            return found
        if self.version is None:
            found.append(self.import_error or "the installed LeRobot version could not be read")
            return found
        if not self.version_ok:
            found.append(
                f"LeRobot {self.version} is installed; the Suite pins {LEROBOT_PINNED_VERSION}"
            )
        if self.lerobot_home_set:
            return found  # the import was skipped, so nothing else was checked
        if not self.importable and self.import_error:
            found.append(f"LeRobot failed to import: {self.import_error}")
        if not self.sdk_importable:
            found.append("the Feetech SDK (scservo_sdk, feetech-servo-sdk) is not importable")
        found.extend(f"missing symbol {name}" for name in self.missing_symbols)
        found.extend(self.signature_problems)
        found.extend(self.table_problems)
        return found

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version_ok"] = self.version_ok
        payload["pinned_version"] = LEROBOT_PINNED_VERSION
        return payload


def probe_lerobot_surface(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    version_of: Callable[[str], str] = importlib.metadata.version,
    environ: Mapping[str, str] | None = None,
) -> LeRobotSurface:
    """Import LeRobot and compare the parts the Suite uses with the pin."""
    env = os.environ if environ is None else environ
    surface = LeRobotSurface(lerobot_home_set="LEROBOT_HOME" in env)
    try:
        surface.version = version_of("lerobot")
        surface.installed = True
    except importlib.metadata.PackageNotFoundError:
        return surface
    except Exception as exc:  # broken metadata is still "installed but unusable"
        surface.installed = True
        surface.import_error = f"could not read the LeRobot version: {exc}"
        return surface
    if surface.lerobot_home_set:
        surface.import_error = "skipped: LEROBOT_HOME would make the import raise"
        return surface

    try:
        importer("scservo_sdk")
        surface.sdk_importable = True
    except Exception:
        surface.sdk_importable = False

    modules: dict[str, Any] = {}
    for module_name in sorted({module for module, _ in REQUIRED_SYMBOLS}):
        try:
            modules[module_name] = importer(module_name)
        except Exception as exc:
            surface.import_error = surface.import_error or f"{module_name}: {exc}"
    surface.importable = len(modules) == len({module for module, _ in REQUIRED_SYMBOLS})
    for module_name, symbol in REQUIRED_SYMBOLS:
        module = modules.get(module_name)
        if module is None or not hasattr(module, symbol):
            surface.missing_symbols.append(f"{module_name}.{symbol}")

    bus_class = getattr(modules.get("lerobot.motors.feetech"), "FeetechMotorsBus", None)
    if bus_class is not None:
        for method_name, parameters in REQUIRED_PARAMETERS.items():
            method = getattr(bus_class, method_name, None)
            if method is None:
                surface.signature_problems.append(f"FeetechMotorsBus.{method_name} is missing")
                continue
            try:
                accepted = set(inspect.signature(method).parameters)
            except (TypeError, ValueError):
                continue
            missing = [name for name in parameters if name not in accepted]
            if missing:
                surface.signature_problems.append(
                    f"FeetechMotorsBus.{method_name} lacks parameters {missing}"
                )
        baudrate = getattr(bus_class, "default_baudrate", None)
        if baudrate is not None and baudrate != DEFAULT_BAUDRATE:
            surface.table_problems.append(
                f"FeetechMotorsBus.default_baudrate is {baudrate}, pinned {DEFAULT_BAUDRATE}"
            )

    try:
        tables = importer("lerobot.motors.feetech.tables")
    except Exception as exc:  # an unchecked table is a problem, not a pass
        surface.table_problems.append(f"lerobot.motors.feetech.tables did not import: {exc}")
        tables = None
    if tables is not None:
        surface.tables_checked = True
        table = getattr(tables, "STS_SMS_SERIES_CONTROL_TABLE", None)
        if isinstance(table, Mapping):
            surface.table_problems.extend(compare_control_table(table))
        else:
            surface.table_problems.append("STS_SMS_SERIES_CONTROL_TABLE is missing")
        expectations: tuple[tuple[str, Any, Any], ...] = (
            ("MODEL_NUMBER_TABLE", STS3215_MODEL, STS3215_MODEL_NUMBER),
            ("MODEL_RESOLUTION", STS3215_MODEL, STS3215_RESOLUTION),
            ("MODEL_PROTOCOL", STS3215_MODEL, STS3215_PROTOCOL),
        )
        for table_name, key, expected in expectations:
            actual = (getattr(tables, table_name, None) or {}).get(key)
            if actual != expected:
                surface.table_problems.append(
                    f"{table_name}[{key!r}] is {actual!r}, pinned {expected!r}"
                )
        baud_table = getattr(tables, "STS_SMS_SERIES_BAUDRATE_TABLE", None)
        if dict(baud_table or {}) != STS_SMS_SERIES_BAUDRATE_TABLE:
            surface.table_problems.append("STS_SMS_SERIES_BAUDRATE_TABLE differs from the pin")
        encodings = getattr(tables, "STS_SMS_SERIES_ENCODINGS_TABLE", None)
        if dict(encodings or {}) != STS_SMS_SERIES_ENCODINGS_TABLE:
            surface.table_problems.append("STS_SMS_SERIES_ENCODINGS_TABLE differs from the pin")
    return surface
