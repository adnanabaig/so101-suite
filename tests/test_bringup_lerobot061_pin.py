"""Pin of the LeRobot 0.6.1 surface the Suite relies on.

Provenance: the PyPI sdist ``lerobot-0.6.1.tar.gz``, sha256
``2111d67d72d9f6c2379e3b8d3ea41b58855066314d88f9c0893579b7707f04b5``. Paths are
relative to ``lerobot-0.6.1/src/lerobot`` in that archive, and line numbers refer
to it.

The tables below were copied by hand from that sdist, independently of
``so101.bringup.lerobot_surface``. Comparing the two catches an edit to the
Suite's copy that drifts from upstream.

Layers:

* Always: the Suite's pinned constants against the copies here, the registers the
  Suite reads, and the pyproject pins.
* ``SO101_TEST_LEROBOT_SDIST`` set to the path of ``lerobot-0.6.1.tar.gz``: the
  archive's sha256, the sha256 of each source file the Suite depends on, and an
  AST read of those files compared with the pin, cited line ranges included.
  Nothing from the archive is executed.
* LeRobot importable: ``probe_lerobot_surface()`` against the installed package.
  The Feetech SDK is a robot-host install question that preflight's ``lerobot``
  check reports, so this layer does not require it.
* Fake importers: each problem ``probe_lerobot_surface`` can report.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import tarfile
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.requirements import Requirement

from so101 import JOINT_NAMES
from so101.bringup import lerobot_surface as ls
from so101.hardware import health, lerobot_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
SDIST_ENV = "SO101_TEST_LEROBOT_SDIST"
SDIST_SHA256 = "2111d67d72d9f6c2379e3b8d3ea41b58855066314d88f9c0893579b7707f04b5"
SDIST_ROOT = "lerobot-0.6.1/"
LEROBOT_SRC = "src/lerobot/"

# motors/feetech/tables.py:41-101, copied from the sdist. The first three values are
# FIRMWARE_MAJOR_VERSION, FIRMWARE_MINOR_VERSION and MODEL_NUMBER (tables.py:15-17).
# Upstream's quirks are kept: the trailing space in "Acceleration_Multiplier " and
# the shared address 80 of "Moving_Velocity" and "Moving_Velocity_Threshold".
VERBATIM_STS_SMS_SERIES_CONTROL_TABLE: dict[str, tuple[int, int]] = {
    # EPROM
    "Firmware_Major_Version": (0, 1),  # read-only
    "Firmware_Minor_Version": (1, 1),  # read-only
    "Model_Number": (3, 2),  # read-only
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

# motors/feetech/tables.py:154-163
VERBATIM_STS_SMS_SERIES_BAUDRATE_TABLE: dict[int, int] = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}

# motors/feetech/tables.py:207-216 (sign-magnitude sign bit per register)
VERBATIM_STS_SMS_SERIES_ENCODINGS_TABLE: dict[str, int] = {
    "Present_Load": 10,
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Goal_Velocity": 15,
    "Goal_Speed": 15,
    "Present_Position": 15,
    "Present_Velocity": 15,
    "Present_Speed": 15,
}

# Where each pinned value sits in motors/feetech/tables.py. Code and docs cite these.
TABLES_LINES: dict[str, tuple[int, int]] = {
    "FIRMWARE_MAJOR_VERSION": (15, 15),
    "FIRMWARE_MINOR_VERSION": (16, 16),
    "MODEL_NUMBER": (17, 17),
    "STS_SMS_SERIES_CONTROL_TABLE": (41, 101),
    "STS_SMS_SERIES_BAUDRATE_TABLE": (154, 163),
    "MODEL_RESOLUTION": (186, 194),
    "STS_SMS_SERIES_ENCODINGS_TABLE": (207, 216),
    "MODEL_NUMBER_TABLE": (242, 247),
    "MODEL_PROTOCOL": (249, 257),
}

# Motor(id, "sts3215", ...) in robots/so_follower/so_follower.py:54-59 and
# teleoperators/so_leader/so_leader.py:46-51.
SO101_MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# The class and line that define each FeetechMotorsBus method the Suite calls.
BUS_METHOD_LINES: dict[str, tuple[str, int]] = {
    "connect": ("SerialMotorsBus", 514),
    "disconnect": ("SerialMotorsBus", 547),
    "disable_torque": ("FeetechMotorsBus", 291),
    "set_timeout": ("SerialMotorsBus", 693),
    "set_baudrate": ("SerialMotorsBus", 711),
    "ping": ("SerialMotorsBus", 948),
    "read": ("SerialMotorsBus", 995),
    "sync_read": ("SerialMotorsBus", 1128),
}


# --- the Suite's pinned copy -------------------------------------------------------------


def test_suite_control_table_is_the_verbatim_upstream_table() -> None:
    assert ls.STS_SMS_SERIES_CONTROL_TABLE == VERBATIM_STS_SMS_SERIES_CONTROL_TABLE
    assert list(ls.STS_SMS_SERIES_CONTROL_TABLE) == list(VERBATIM_STS_SMS_SERIES_CONTROL_TABLE)
    assert ls.compare_control_table(VERBATIM_STS_SMS_SERIES_CONTROL_TABLE) == []


def test_suite_keeps_upstream_quirks() -> None:
    table = ls.STS_SMS_SERIES_CONTROL_TABLE
    assert "Acceleration_Multiplier " in table
    assert "Acceleration_Multiplier" not in table
    assert table["Moving_Velocity"] == table["Moving_Velocity_Threshold"] == (80, 1)


def test_suite_baud_and_encoding_tables_match() -> None:
    assert ls.STS_SMS_SERIES_BAUDRATE_TABLE == VERBATIM_STS_SMS_SERIES_BAUDRATE_TABLE
    assert ls.STS_SMS_SERIES_ENCODINGS_TABLE == VERBATIM_STS_SMS_SERIES_ENCODINGS_TABLE


def test_suite_model_constants_match() -> None:
    assert ls.LEROBOT_PINNED_VERSION == "0.6.1"
    assert ls.LEROBOT_SDIST_SHA256 == SDIST_SHA256
    assert ls.FIRMWARE_MAJOR_VERSION == (0, 1)
    assert ls.FIRMWARE_MINOR_VERSION == (1, 1)
    assert ls.MODEL_NUMBER == (3, 2)
    assert ls.STS3215_MODEL == "sts3215"
    assert ls.STS3215_MODEL_NUMBER == 777  # tables.py:243
    assert ls.STS3215_RESOLUTION == 4096  # tables.py:190
    assert ls.STS3215_PROTOCOL == 0  # tables.py:253
    assert ls.DEFAULT_BAUDRATE == 1_000_000  # feetech.py:44
    assert ls.DEFAULT_TIMEOUT_MS == 1000  # feetech.py:45
    assert ls.NORMALIZED_DATA == ("Goal_Position", "Present_Position")  # feetech.py:47
    # MotorNormMode.DEGREES divides by (resolution - 1): motors_bus.py:874-877.
    assert ls.DEGREES_PER_TICK == pytest.approx(360 / 4095)


def test_registers_the_suite_reads_are_in_the_table() -> None:
    for group in (ls.DIAGNOSTIC_REGISTERS, health.HEALTH_REGISTERS, health.LIMIT_REGISTERS):
        assert len(set(group)) == len(group)
        assert set(group) <= set(VERBATIM_STS_SMS_SERIES_CONTROL_TABLE)


def test_health_register_addresses_come_from_the_table() -> None:
    assert set(health.UNIT_PROVENANCE) == {*health.HEALTH_REGISTERS, *health.LIMIT_REGISTERS}
    for name, provenance in health.UNIT_PROVENANCE.items():
        assert provenance.register_name == name
        expected = VERBATIM_STS_SMS_SERIES_CONTROL_TABLE[name]
        assert (provenance.address, provenance.size) == expected, name
        assert "tables.py" in provenance.source


def test_unit_scales_stay_marked_unverified() -> None:
    # LeRobot 0.6.1 states no physical units. A scale may become "verified" only with a
    # cited source, such as a Feetech STS3215 manual; update this test with that citation.
    assert {provenance.status for provenance in health.UNIT_PROVENANCE.values()} == {"unverified"}


def test_present_load_sign_bit_is_pinned() -> None:
    assert VERBATIM_STS_SMS_SERIES_ENCODINGS_TABLE["Present_Load"] == 10
    assert "Sign bit 10" in health.UNIT_PROVENANCE["Present_Load"].source


def test_so101_motor_ids_and_calibration_names() -> None:
    assert list(SO101_MOTOR_IDS) == JOINT_NAMES
    assert health.JOINT_IDS == SO101_MOTOR_IDS
    assert lerobot_paths.FOLLOWER_NAME == "so_follower"  # so_follower.py:44
    assert lerobot_paths.LEADER_NAME == "so_leader"  # so_leader.py:37
    assert lerobot_paths.LEROBOT_ROBOTS_DIR == "robots"  # utils/constants.py:43
    assert lerobot_paths.LEROBOT_TELEOPERATORS_DIR == "teleoperators"  # utils/constants.py:44


def _extras() -> dict[str, list[Requirement]]:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extras = tomllib.loads(text)["project"]["optional-dependencies"]
    return {name: [Requirement(item) for item in items] for name, items in extras.items()}


def _requirement(requirements: list[Requirement], name: str) -> Requirement:
    matches = [req for req in requirements if req.name == name]
    assert len(matches) == 1, f"expected one {name} requirement, found {matches}"
    return matches[0]


def test_pyproject_pins_lerobot_061() -> None:
    extras = _extras()
    for extra in ("robot", "train"):
        lerobot = _requirement(extras[extra], "lerobot")
        assert str(lerobot.specifier) == f"=={ls.LEROBOT_PINNED_VERSION}", extra
    robot = extras["robot"]
    assert "feetech" in _requirement(robot, "lerobot").extras
    assert _requirement(robot, "pyserial").specifier.contains("3.5")
    assert _requirement(robot, "feetech-servo-sdk").specifier.contains("1.0.0")


# --- fake importers: what probe_lerobot_surface reports ------------------------------------


class FakeFeetechMotorsBus:
    default_baudrate = 1_000_000

    def connect(self, handshake: bool = True) -> None:
        return None

    def disconnect(self, disable_torque: bool = True) -> None:
        return None

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        return None

    def set_timeout(self, timeout_ms: int | None = None) -> None:
        return None

    def set_baudrate(self, baudrate: int) -> None:
        return None

    def ping(self, motor: Any, num_retry: int = 0, raise_on_error: bool = False) -> None:
        return None

    def read(
        self, data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        return None

    def sync_read(
        self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        return None


TABLES_MODULE = "lerobot.motors.feetech.tables"
SDK_PROBLEM = "the Feetech SDK (scservo_sdk, feetech-servo-sdk) is not importable"


def _fake_modules() -> dict[str, Any]:
    return {
        "scservo_sdk": SimpleNamespace(),
        "lerobot.motors": SimpleNamespace(
            Motor=object, MotorCalibration=object, MotorNormMode=object
        ),
        "lerobot.motors.feetech": SimpleNamespace(
            FeetechMotorsBus=FakeFeetechMotorsBus, OperatingMode=object
        ),
        "lerobot.robots.so_follower": SimpleNamespace(
            SO101Follower=object, SO101FollowerConfig=object
        ),
        "lerobot.teleoperators.so_leader": SimpleNamespace(
            SO101Leader=object, SO101LeaderConfig=object
        ),
        "lerobot.utils.constants": SimpleNamespace(HF_LEROBOT_CALIBRATION=Path("calibration")),
        TABLES_MODULE: SimpleNamespace(
            STS_SMS_SERIES_CONTROL_TABLE=dict(VERBATIM_STS_SMS_SERIES_CONTROL_TABLE),
            STS_SMS_SERIES_BAUDRATE_TABLE=dict(VERBATIM_STS_SMS_SERIES_BAUDRATE_TABLE),
            STS_SMS_SERIES_ENCODINGS_TABLE=dict(VERBATIM_STS_SMS_SERIES_ENCODINGS_TABLE),
            MODEL_NUMBER_TABLE={"sts3215": 777},
            MODEL_RESOLUTION={"sts3215": 4096},
            MODEL_PROTOCOL={"sts3215": 0},
        ),
    }


def _probe(
    modules: dict[str, Any] | None = None,
    *,
    version: str | Exception = "0.6.1",
    environ: dict[str, str] | None = None,
) -> tuple[ls.LeRobotSurface, list[str]]:
    available = _fake_modules() if modules is None else modules
    imported: list[str] = []

    def importer(name: str) -> Any:
        imported.append(name)
        if name not in available:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return available[name]

    def version_of(name: str) -> str:
        assert name == "lerobot"
        if isinstance(version, Exception):
            raise version
        return version

    surface = ls.probe_lerobot_surface(
        importer=importer, version_of=version_of, environ=environ or {}
    )
    return surface, imported


def test_probe_passes_a_matching_install() -> None:
    surface, imported = _probe()
    assert surface.problems() == []
    assert surface.installed and surface.version_ok
    assert surface.importable and surface.sdk_importable and surface.tables_checked
    assert {"scservo_sdk", TABLES_MODULE} <= set(imported)
    payload = surface.to_dict()
    assert payload["version_ok"] is True and payload["pinned_version"] == "0.6.1"


def test_probe_reports_lerobot_missing_without_importing() -> None:
    surface, imported = _probe(version=importlib.metadata.PackageNotFoundError("lerobot"))
    assert surface.problems() == ["LeRobot is not installed"]
    assert not surface.installed and imported == []


def test_probe_reports_unreadable_metadata() -> None:
    surface, imported = _probe(version=ValueError("bad METADATA"))
    assert surface.installed and surface.version is None and imported == []
    assert surface.problems() == ["could not read the LeRobot version: bad METADATA"]


def test_probe_reports_a_wrong_version_and_still_checks_the_rest() -> None:
    surface, _ = _probe(version="0.5.0")
    assert surface.problems() == ["LeRobot 0.5.0 is installed; the Suite pins 0.6.1"]
    assert surface.tables_checked


def test_probe_skips_the_import_when_lerobot_home_is_set() -> None:
    surface, imported = _probe(environ={"LEROBOT_HOME": "elsewhere"})
    assert imported == []
    assert surface.import_error == "skipped: LEROBOT_HOME would make the import raise"
    problems = surface.problems()
    assert len(problems) == 1 and problems[0].startswith("LEROBOT_HOME is set")
    assert "HF_LEROBOT_HOME" in problems[0]


def test_probe_with_lerobot_home_and_a_wrong_version_reports_both() -> None:
    surface, _ = _probe(version="0.5.0", environ={"LEROBOT_HOME": "elsewhere"})
    problems = surface.problems()
    assert problems[0].startswith("LEROBOT_HOME is set")
    assert problems[1:] == ["LeRobot 0.5.0 is installed; the Suite pins 0.6.1"]


def test_probe_reports_a_missing_feetech_sdk() -> None:
    modules = _fake_modules()
    del modules["scservo_sdk"]
    surface, _ = _probe(modules)
    assert surface.problems() == [SDK_PROBLEM]


def test_probe_reports_a_module_that_fails_to_import() -> None:
    modules = _fake_modules()
    del modules["lerobot.robots.so_follower"]
    surface, _ = _probe(modules)
    assert not surface.importable
    error = "lerobot.robots.so_follower: No module named 'lerobot.robots.so_follower'"
    assert surface.import_error == error
    assert surface.problems() == [
        f"LeRobot failed to import: {error}",
        "missing symbol lerobot.robots.so_follower.SO101Follower",
        "missing symbol lerobot.robots.so_follower.SO101FollowerConfig",
    ]


def test_probe_reports_a_missing_symbol() -> None:
    modules = _fake_modules()
    del modules["lerobot.teleoperators.so_leader"].SO101LeaderConfig
    surface, _ = _probe(modules)
    assert surface.problems() == [
        "missing symbol lerobot.teleoperators.so_leader.SO101LeaderConfig"
    ]


def _bus_without(method: str) -> type:
    attributes = {
        key: value
        for key, value in vars(FakeFeetechMotorsBus).items()
        if not key.startswith("__") and key != method
    }
    return type("FeetechMotorsBus", (), attributes)


class OldConnectBus(FakeFeetechMotorsBus):
    def connect(self) -> None:
        return None


class SlowBaudBus(FakeFeetechMotorsBus):
    default_baudrate = 500_000


@pytest.mark.parametrize(
    ("bus", "problem"),
    [
        (_bus_without("set_baudrate"), "FeetechMotorsBus.set_baudrate is missing"),
        (OldConnectBus, "FeetechMotorsBus.connect lacks parameters ['handshake']"),
        (SlowBaudBus, "FeetechMotorsBus.default_baudrate is 500000, pinned 1000000"),
    ],
    ids=["missing-method", "missing-parameter", "default-baudrate"],
)
def test_probe_reports_bus_class_drift(bus: type, problem: str) -> None:
    modules = _fake_modules()
    modules["lerobot.motors.feetech"].FeetechMotorsBus = bus
    surface, _ = _probe(modules)
    assert surface.problems() == [problem]


def _rename_acceleration_multiplier(table: dict[str, Any]) -> None:
    table["Acceleration_Multiplier"] = table.pop("Acceleration_Multiplier ")


@pytest.mark.parametrize(
    ("edit", "problems"),
    [
        (
            lambda table: table.__setitem__("Present_Voltage", (63, 1)),
            ["register 'Present_Voltage' is (63, 1), pinned (62, 1)"],
        ),
        (lambda table: table.pop("Status"), ["register 'Status' is missing"]),
        (
            _rename_acceleration_multiplier,
            [
                "register 'Acceleration_Multiplier ' is missing",
                "register 'Acceleration_Multiplier' is not in the pinned table",
            ],
        ),
        (lambda table: table.__setitem__("ID", [5, 1]), []),
    ],
    ids=["moved", "missing", "renamed", "list-values-are-equal"],
)
def test_probe_reports_control_table_drift(
    edit: Callable[[dict[str, Any]], object], problems: list[str]
) -> None:
    modules = _fake_modules()
    edit(modules[TABLES_MODULE].STS_SMS_SERIES_CONTROL_TABLE)
    surface, _ = _probe(modules)
    assert surface.problems() == problems


@pytest.mark.parametrize(
    ("attribute", "value", "problem"),
    [
        (
            "MODEL_NUMBER_TABLE",
            {"sts3215": 778},
            "MODEL_NUMBER_TABLE['sts3215'] is 778, pinned 777",
        ),
        ("MODEL_RESOLUTION", {}, "MODEL_RESOLUTION['sts3215'] is None, pinned 4096"),
        ("MODEL_PROTOCOL", {"sts3215": 1}, "MODEL_PROTOCOL['sts3215'] is 1, pinned 0"),
        (
            "STS_SMS_SERIES_BAUDRATE_TABLE",
            {**VERBATIM_STS_SMS_SERIES_BAUDRATE_TABLE, 1_000_000: 1},
            "STS_SMS_SERIES_BAUDRATE_TABLE differs from the pin",
        ),
        (
            "STS_SMS_SERIES_ENCODINGS_TABLE",
            {**VERBATIM_STS_SMS_SERIES_ENCODINGS_TABLE, "Present_Load": 15},
            "STS_SMS_SERIES_ENCODINGS_TABLE differs from the pin",
        ),
        ("STS_SMS_SERIES_CONTROL_TABLE", None, "STS_SMS_SERIES_CONTROL_TABLE is missing"),
    ],
)
def test_probe_reports_model_table_drift(attribute: str, value: Any, problem: str) -> None:
    modules = _fake_modules()
    setattr(modules[TABLES_MODULE], attribute, value)
    surface, _ = _probe(modules)
    assert surface.problems() == [problem]


def test_probe_fails_closed_when_the_tables_do_not_import() -> None:
    modules = _fake_modules()
    del modules[TABLES_MODULE]
    surface, _ = _probe(modules)
    assert not surface.tables_checked
    assert surface.problems() == [
        f"{TABLES_MODULE} did not import: No module named {TABLES_MODULE!r}"
    ]


def test_compare_control_table_reports_every_difference() -> None:
    installed: dict[str, Any] = dict(VERBATIM_STS_SMS_SERIES_CONTROL_TABLE)
    installed["Lock"] = (56, 1)
    del installed["Moving"]
    installed["Goal_Speed"] = (46, 2)
    assert ls.compare_control_table(installed) == [
        "register 'Lock' is (56, 1), pinned (55, 1)",
        "register 'Moving' is missing",
        "register 'Goal_Speed' is not in the pinned table",
    ]


# --- the installed package -----------------------------------------------------------------

LEROBOT_INSTALLED = importlib.util.find_spec("lerobot") is not None


@pytest.fixture
def installed_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    if not LEROBOT_INSTALLED:
        pytest.skip("LeRobot is not installed here; robot hosts run this check")
    # LEROBOT_HOME makes LeRobot 0.6.1 raise on import (utils/constants.py:61-65).
    # Preflight reports that separately; this test pins the API surface.
    monkeypatch.delenv("LEROBOT_HOME", raising=False)


@pytest.mark.usefixtures("installed_lerobot")
def test_installed_lerobot_matches_the_pin() -> None:
    surface = ls.probe_lerobot_surface()
    assert surface.version == ls.LEROBOT_PINNED_VERSION
    assert surface.importable, surface.import_error
    assert surface.missing_symbols == []
    assert surface.signature_problems == []
    assert surface.table_problems == []
    assert surface.tables_checked


@pytest.mark.usefixtures("installed_lerobot")
def test_installed_lerobot_names_and_tables() -> None:
    follower = importlib.import_module("lerobot.robots.so_follower")
    leader = importlib.import_module("lerobot.teleoperators.so_leader")
    tables = importlib.import_module(TABLES_MODULE)
    assert follower.SO101Follower.name == lerobot_paths.FOLLOWER_NAME
    assert leader.SO101Leader.name == lerobot_paths.LEADER_NAME
    assert tables.MODEL_CONTROL_TABLE["sts3215"] is tables.STS_SMS_SERIES_CONTROL_TABLE


# --- the sdist, read without running it ------------------------------------------------------


@pytest.fixture(scope="module")
def sdist_path() -> Path:
    location = os.environ.get(SDIST_ENV)
    if not location:
        pytest.skip(f"set {SDIST_ENV} to the path of lerobot-0.6.1.tar.gz to check the source")
    path = Path(location)
    if not path.is_file():
        pytest.fail(f"{SDIST_ENV}={location} is not a file")
    return path


Sdist = Callable[[str], bytes]


@pytest.fixture(scope="module")
def sdist(sdist_path: Path) -> Iterator[Sdist]:
    """Bytes of a file in the pinned sdist, by path under ``lerobot-0.6.1/``."""
    if hashlib.sha256(sdist_path.read_bytes()).hexdigest() != SDIST_SHA256:
        pytest.fail(f"{sdist_path} is not the pinned lerobot-0.6.1 sdist")
    cache: dict[str, bytes] = {}
    with tarfile.open(sdist_path, "r:gz") as archive:

        def read(name: str) -> bytes:
            if name not in cache:
                member = archive.extractfile(SDIST_ROOT + name)  # KeyError when absent
                assert member is not None, name
                cache[name] = member.read()
            return cache[name]

        yield read


def _source(sdist: Sdist, name: str) -> str:
    """A LeRobot source file, by path under ``src/lerobot/``."""
    return sdist(LEROBOT_SRC + name).decode("utf-8")


def _literal(node: ast.expr, names: dict[str, Any]) -> Any:
    """A literal, where a bare name refers to an earlier top-level value."""
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise ValueError("dict unpacking is not a literal")
            result[_literal(key, names)] = _literal(value, names)
        return result
    if isinstance(node, ast.List):
        return [_literal(item, names) for item in node.elts]
    return ast.literal_eval(node)


def _top_level_values(source: str) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    """Literal top-level assignments of a module, and the lines each one spans."""
    values: dict[str, Any] = {}
    lines: dict[str, tuple[int, int]] = {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = _literal(node.value, values)
        except (ValueError, KeyError, TypeError):  # a call or another non-literal
            continue
        lines[target.id] = (node.lineno, node.end_lineno or node.lineno)
    return values, lines


def _class_node(source: str, name: str) -> ast.ClassDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _class_attribute(cls: ast.ClassDef, name: str) -> tuple[ast.expr, int]:
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value, node.lineno
    raise AssertionError(f"{cls.name}.{name} not found")


def _methods(cls: ast.ClassDef) -> dict[str, tuple[list[str], int]]:
    found: dict[str, tuple[list[str], int]] = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = node.args
            names = [a.arg for a in (*arguments.posonlyargs, *arguments.args)]
            names += [a.arg for a in arguments.kwonlyargs]
            found[node.name] = (names, node.lineno)
    return found


def _bound_names(source: str) -> set[str]:
    """Names a module binds at top level."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _motor_table(source: str) -> dict[str, tuple[int, str, int]]:
    """``{joint: (id, model, line)}`` from the ``motors={...: Motor(...)}`` literal."""
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Dict) and node.values):
            continue
        calls = [
            value
            for value in node.values
            if isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Motor"
        ]
        if len(calls) != len(node.values):
            continue
        table: dict[str, tuple[int, str, int]] = {}
        for key, call in zip(node.keys, calls, strict=True):
            assert isinstance(key, ast.Constant)
            motor_id = ast.literal_eval(call.args[0])
            model = ast.literal_eval(call.args[1])
            table[key.value] = (motor_id, model, key.lineno)
        return table
    raise AssertionError("no Motor table found")


def _alias_target(source: str, name: str) -> str | None:
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            return node.value.id
    return None


def test_sdist_is_the_pinned_archive(sdist_path: Path) -> None:
    assert hashlib.sha256(sdist_path.read_bytes()).hexdigest() == SDIST_SHA256


def test_sdist_source_files_match_their_pinned_hashes(sdist: Sdist) -> None:
    for name, expected in ls.LEROBOT_SOURCE_SHA256.items():
        digest = hashlib.sha256(sdist(LEROBOT_SRC + name)).hexdigest()
        assert digest == expected, name


def test_sdist_tables_match_the_pin(sdist: Sdist) -> None:
    values, lines = _top_level_values(_source(sdist, "motors/feetech/tables.py"))
    control = values["STS_SMS_SERIES_CONTROL_TABLE"]
    assert control == VERBATIM_STS_SMS_SERIES_CONTROL_TABLE
    assert list(control) == list(VERBATIM_STS_SMS_SERIES_CONTROL_TABLE)
    assert values["STS_SMS_SERIES_BAUDRATE_TABLE"] == VERBATIM_STS_SMS_SERIES_BAUDRATE_TABLE
    assert values["STS_SMS_SERIES_ENCODINGS_TABLE"] == VERBATIM_STS_SMS_SERIES_ENCODINGS_TABLE
    assert values["FIRMWARE_MAJOR_VERSION"] == ls.FIRMWARE_MAJOR_VERSION
    assert values["FIRMWARE_MINOR_VERSION"] == ls.FIRMWARE_MINOR_VERSION
    assert values["MODEL_NUMBER"] == ls.MODEL_NUMBER
    assert values["MODEL_NUMBER_TABLE"]["sts3215"] == ls.STS3215_MODEL_NUMBER
    assert values["MODEL_RESOLUTION"]["sts3215"] == ls.STS3215_RESOLUTION
    assert values["MODEL_PROTOCOL"]["sts3215"] == ls.STS3215_PROTOCOL
    # The STS3215 uses the STS/SMS tables (tables.py:176-226).
    assert values["MODEL_CONTROL_TABLE"]["sts3215"] is control
    assert values["MODEL_BAUDRATE_TABLE"]["sts3215"] is values["STS_SMS_SERIES_BAUDRATE_TABLE"]
    assert values["MODEL_ENCODING_TABLE"]["sts3215"] is values["STS_SMS_SERIES_ENCODINGS_TABLE"]
    for name, span in TABLES_LINES.items():
        assert lines[name] == span, name


def test_sdist_feetech_defaults(sdist: Sdist) -> None:
    source = _source(sdist, "motors/feetech/feetech.py")
    values, lines = _top_level_values(source)
    assert (values["DEFAULT_BAUDRATE"], lines["DEFAULT_BAUDRATE"]) == (
        ls.DEFAULT_BAUDRATE,
        (44, 44),
    )
    assert (values["DEFAULT_TIMEOUT_MS"], lines["DEFAULT_TIMEOUT_MS"]) == (
        ls.DEFAULT_TIMEOUT_MS,
        (45, 45),
    )
    assert tuple(values["NORMALIZED_DATA"]) == ls.NORMALIZED_DATA
    assert lines["NORMALIZED_DATA"] == (47, 47)
    value, _ = _class_attribute(_class_node(source, "FeetechMotorsBus"), "default_baudrate")
    assert isinstance(value, ast.Name) and value.id == "DEFAULT_BAUDRATE"


def test_sdist_bus_methods_accept_what_the_suite_passes(sdist: Sdist) -> None:
    feetech = _source(sdist, "motors/feetech/feetech.py")
    motors_bus = _source(sdist, "motors/motors_bus.py")
    chain = [
        _class_node(feetech, "FeetechMotorsBus"),
        _class_node(motors_bus, "SerialMotorsBus"),
        _class_node(motors_bus, "MotorsBusBase"),
    ]
    # The method resolution order this lookup assumes.
    assert [[ast.unparse(base) for base in cls.bases] for cls in chain] == [
        ["SerialMotorsBus"],
        ["MotorsBusBase"],
        ["abc.ABC"],
    ]
    assert set(BUS_METHOD_LINES) == set(ls.REQUIRED_PARAMETERS)
    for method, required in ls.REQUIRED_PARAMETERS.items():
        owner = next(cls for cls in chain if method in _methods(cls))
        accepted, line = _methods(owner)[method]
        assert set(required) <= set(accepted), (method, owner.name, accepted)
        assert (owner.name, line) == BUS_METHOD_LINES[method], method


def test_sdist_exports_the_symbols_the_suite_imports(sdist: Sdist) -> None:
    for module, symbol in ls.REQUIRED_SYMBOLS:
        path = module.removeprefix("lerobot.").replace(".", "/")
        try:
            source = _source(sdist, path + "/__init__.py")
        except KeyError:
            source = _source(sdist, path + ".py")
        assert symbol in _bound_names(source), f"{module}.{symbol}"


def test_sdist_so101_motor_ids_and_calibration_names(sdist: Sdist) -> None:
    follower = _source(sdist, "robots/so_follower/so_follower.py")
    leader = _source(sdist, "teleoperators/so_leader/so_leader.py")
    expected = {name: (motor_id, "sts3215") for name, motor_id in SO101_MOTOR_IDS.items()}
    for source, span in ((follower, (54, 59)), (leader, (46, 51))):
        motors = _motor_table(source)
        assert {name: (row[0], row[1]) for name, row in motors.items()} == expected
        rows = [row[2] for row in motors.values()]
        assert (min(rows), max(rows)) == span
    # Calibration lives in <calibration>/robots/<name>/ and <calibration>/teleoperators/<name>/.
    value, line = _class_attribute(_class_node(follower, "SOFollower"), "name")
    assert (ast.literal_eval(value), line) == (lerobot_paths.FOLLOWER_NAME, 44)
    value, line = _class_attribute(_class_node(leader, "SOLeader"), "name")
    assert (ast.literal_eval(value), line) == (lerobot_paths.LEADER_NAME, 37)
    assert _alias_target(follower, "SO101Follower") == "SOFollower"
    assert _alias_target(leader, "SO101Leader") == "SOLeader"


def test_sdist_constants(sdist: Sdist) -> None:
    source = _source(sdist, "utils/constants.py")
    values, lines = _top_level_values(source)
    assert (values["ROBOTS"], lines["ROBOTS"]) == (lerobot_paths.LEROBOT_ROBOTS_DIR, (43, 43))
    assert (values["TELEOPERATORS"], lines["TELEOPERATORS"]) == (
        lerobot_paths.LEROBOT_TELEOPERATORS_DIR,
        (44, 44),
    )
    guard = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.If) and "LEROBOT_HOME" in ast.unparse(node.test)
    )
    assert (guard.lineno, guard.end_lineno) == (61, 65)
    assert isinstance(guard.body[0], ast.Raise)
    assert "ValueError" in ast.unparse(guard.body[0])


def test_sdist_packaging_matches_the_robot_extra(sdist: Sdist) -> None:
    upstream = tomllib.loads(sdist("pyproject.toml").decode("utf-8"))["project"]
    assert upstream["version"] == ls.LEROBOT_PINNED_VERSION
    extras = upstream["optional-dependencies"]
    feetech = [Requirement(item) for item in extras["feetech"]]
    pyserial = [Requirement(item) for item in extras["pyserial-dep"]]
    robot = _extras()["robot"]
    sdk = _requirement(robot, "feetech-servo-sdk")
    assert sdk.specifier == _requirement(feetech, "feetech-servo-sdk").specifier
    assert _requirement(pyserial, "pyserial").specifier.contains("3.5")
    assert _requirement(robot, "pyserial").specifier.contains("3.5")
    # docs/bring-up.md and preflight remediations name these LeRobot commands.
    for script in ("lerobot-find-port", "lerobot-calibrate", "lerobot-setup-motors"):
        assert script in upstream["scripts"], script
