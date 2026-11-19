"""Central configuration: SO101_* settings from the environment and .env files.

Where a setting comes from, highest priority first:

1. an ``SO101_*`` variable in the process environment;
2. ``./.env`` in the folder the command runs in (a source checkout's own);
3. ``<data folder>/.env``: ``SO101_HOME`` if set, else the per-user folder
   (``~/.local/share/so101`` on Linux, ``$XDG_DATA_HOME/so101`` when that is
   set; see :func:`so101.platform_info.data_home`). The installer creates it
   with absolute ``SO101_DATA_DIR``, ``SO101_DATASETS_DIR`` and
   ``SO101_OUTPUTS_DIR``, so the installed suite finds the same settings and
   data from any folder;
4. the defaults below. The storage folders default to ``./data``,
   ``./datasets`` and ``./outputs`` in the folder the command runs in.

So on the G15 ``so101 doctor`` reads ``~/.local/share/so101/.env`` whatever
the working directory; in a checkout without an installed suite, ``.env`` and
``./data`` work as before. :func:`env_files` lists the files in effect, and
``so101 info`` and ``so101 doctor`` print them.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .atomic_io import durable_makedirs
from .platform_info import ARM_LINKS, current_data_home, current_env_files

if TYPE_CHECKING:
    from .hardware.camera_spec import CameraSpec

# ``model_config["env_file"]`` value meaning "the suite's rule" (module
# docstring): resolved on every ``Settings()``, so it follows the working
# directory and SO101_HOME at that moment. Tests set ``env_file`` to None.
AUTO_ENV_FILES = "so101:auto"

# The serial ports of the macOS adapters this suite was first set up with. They
# stay the defaults off Linux (they may be the user's real adapters).
MACOS_DEFAULT_PORTS = {
    "follower": "/dev/tty.usbmodem585A0077531",
    "leader": "/dev/tty.usbmodem585A0077541",
}


def default_arm_port(arm: Literal["follower", "leader"], platform: str | None = None) -> str:
    """The default serial port of an arm on this OS.

    Linux: ``/dev/so101-follower`` and ``/dev/so101-leader``, the udev names
    ``install/install.sh --follower-serial ... --leader-serial ...`` gives the
    adapters (:data:`so101.platform_info.ARM_LINKS`). Elsewhere the macOS
    adapters' device files, unchanged.
    """

    if (platform or sys.platform).startswith("linux"):
        return ARM_LINKS[0] if arm == "follower" else ARM_LINKS[1]
    return MACOS_DEFAULT_PORTS[arm]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SO101_",
        env_file=AUTO_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        # Validation errors must not echo values back: SO101_CAMERAS may hold
        # rtsp://user:password@ URLs, and tokens and keys live here too. Field
        # validators name the problem (and redact credentials) themselves.
        hide_input_in_errors=True,
    )

    # --- Hardware -------------------------------------------------------
    mock: bool = Field(
        default=False,
        description="Force mock hardware (no serial/cameras/LeRobot needed).",
    )
    # Linux: the udev names install.sh creates; macOS and others: see MACOS_DEFAULT_PORTS.
    follower_port: str = Field(default_factory=lambda: default_arm_port("follower"))
    leader_port: str = Field(default_factory=lambda: default_arm_port("leader"))
    follower_id: str = "so101-follower"
    leader_id: str = "so101-leader"
    teleop_max_velocity_deg_s: float = Field(default=60.0, gt=0, le=450, allow_inf_nan=False)
    baudrate: int = 1_000_000
    # Comma-separated camera entries: an OpenCV index or device path/URL, or an
    # Intel RealSense as realsense[:<serial>][:<W>x<H>[@<fps>]][+depth]. A
    # "<name>=" prefix names the recorded feature. See docs/realsense.md.
    cameras: str = "0"
    camera_capture_mode: Literal["direct", "shared"] = "direct"
    # OpenCV capture requests; None preserves the device's default. Actual
    # negotiated values are reported by CameraManager.describe().
    camera_width: int | None = Field(default=None, ge=16, le=4096)
    camera_height: int | None = Field(default=None, ge=16, le=4096)
    camera_fps: int | None = Field(default=None, ge=1, le=300)
    camera_fourcc: Literal["MJPG", "YUYV"] | None = None
    # What backs the robot: "mock" (MockBus), "sim" (MuJoCo physics, see
    # docs/simulation.md) or "real". Unset follows SO101_MOCK. SO101_MOCK=1
    # together with SO101_BACKEND=real or sim is a configuration error.
    backend: Literal["mock", "sim", "real"] | None = None

    # --- Simulation (docs/simulation.md) ----------------------------------
    # /api/simulation engine: "auto" serves MuJoCo physics when the [sim] extra
    # is installed and the kinematic preview otherwise.
    sim_engine: Literal["auto", "kinematic", "mujoco"] = "auto"
    sim_task: Literal["reach", "pick_lift", "pick_place"] = "pick_lift"
    sim_seed: int = 0
    sim_cameras: str = "front,wrist_cam"  # comma-separated simulated camera names
    sim_image_size: str = "320x240"  # WxH of simulated camera frames
    sim_realtime: bool = True  # pace the sim backend's physics to the wall clock

    # --- Storage --------------------------------------------------------
    # Relative to the folder a command runs in, unless set (the installer sets
    # absolute ones in <data folder>/.env).
    data_dir: Path = Path("./data")
    datasets_dir: Path = Path("./datasets")
    outputs_dir: Path = Path("./outputs")

    # --- Service --------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
    )
    # Extra Host header names the server answers to, comma-separated, on top of
    # localhost, 127.0.0.1, ::1 and a non-wildcard ``host``. "*.lan" matches
    # subdomains; a bare "*" is refused. See docs/control-plane-security.md.
    allowed_hosts: str = ""
    # Installed-artifact E2E identity nonce. Empty in every production setup.
    # GuardMiddleware accepts its header only on mock/sim backends and never
    # logs or returns the value (scripts/installed_e2e.py).
    e2e_token: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    # Built web UI directory (must contain index.html). Empty = packaged
    # ``so101/web``, then the source checkout's ``frontend/dist``.
    ui_dir: str = ""

    # --- Hugging Face ---------------------------------------------------
    hf_token: str = ""
    hf_user: str = ""

    # --- Training -------------------------------------------------------
    wandb_project: str = "so101"
    wandb_enabled: bool = False
    # Policy inference device: "" or "auto" picks cuda, then mps, then cpu;
    # or force cpu | cuda | cuda:<n> | mps (fails if unavailable).
    policy_device: str = ""

    # --- Optional proposal-only intelligent planner --------------------
    agent_provider: str = "deterministic"
    agent_model: str = ""
    agent_timeout_seconds: float = 20.0
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Bring-up: servo health (see docs/bring-up.md) ------------------
    # These are suite defaults, not Feetech specifications. "7.4v" is the
    # standard STS3215 variant (the SO-101 kit powers it from 5 V); "12v" is the
    # high-torque variant on a 12 V supply. Bands are volts:
    # (critical_low, warn_low, warn_high, critical_high), inclusive.
    health_follower_supply: Literal["7.4v", "12v"] = "7.4v"
    health_leader_supply: Literal["7.4v", "12v"] = "7.4v"
    health_supply_band_7v4: tuple[float, float, float, float] = (4.0, 4.5, 8.0, 8.5)
    health_supply_band_12v: tuple[float, float, float, float] = (9.5, 10.5, 13.0, 14.0)
    # Temperature thresholds sit this far below the servo's Max_Temperature_Limit
    # register when it reads back plausibly, else the fallbacks apply.
    health_temp_warn_margin_c: float = 15.0
    health_temp_critical_margin_c: float = 5.0
    health_temp_fallback_warn_c: float = 55.0
    health_temp_fallback_critical_c: float = 65.0
    health_load_warn_pct: float | None = 80.0
    health_load_critical_pct: float | None = None
    health_current_warn_ma: float | None = 2000.0
    health_current_critical_ma: float | None = None
    health_status_nonzero_level: Literal["ok", "warn", "critical"] = "warn"
    health_poll_hz: float = 1.0
    health_max_consecutive_failures: int = 3

    # --- Bring-up: rest pose (see so101.hardware.rest_pose) -------------
    # Six numbers in JOINT_NAMES order ("0,-95,92.8,-85,0,5", MODEL_REST_POSE)
    # or a JSON object keyed by joint name. Empty means unset.
    rest_pose: str = ""

    # --- Stops: Halt and Collapse (docs/control-plane-safety.md) ----------
    # A halt holds position for at most this long, then becomes a Collapse.
    halt_hold_seconds: float = 60.0
    # While no operator browser has a fresh supervision heartbeat, a halt
    # holds only this long before it becomes a Collapse.
    supervision_grace_seconds: float = 5.0
    # Collapse caps every joint at this speed on its way to the rest pose.
    collapse_speed_deg_s: float = 15.0
    # Bound on the Collapse descent. Unset: distance / speed + 2 s, at most
    # COLLAPSE_TIMEOUT_MAX_S. A set value may not exceed that cap either.
    collapse_timeout_seconds: float | None = None
    # Halt and Collapse grip rule, in 0..100 gripper units: the gripper goal
    # never opens past its present position and squeezes at most this far
    # below it, so a held object is not dropped and a stalled grasp does not
    # cook the gripper servo during a long hold.
    grip_hold_margin: float = 5.0

    def __init__(self, **values: Any) -> None:
        # The env files are worked out per instance (AUTO_ENV_FILES), so a
        # Settings() made in another folder or with another SO101_HOME reads
        # that folder's files. An explicit _env_file still wins.
        if "_env_file" not in values and type(self).model_config.get("env_file") == AUTO_ENV_FILES:
            values["_env_file"] = current_env_files()
        super().__init__(**values)

    @field_validator("data_dir", "datasets_dir", "outputs_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("health_supply_band_7v4", "health_supply_band_12v")
    @classmethod
    def _check_supply_band(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if any(not math.isfinite(x) for x in v) or not (v[0] < v[1] < v[2] < v[3]):
            raise ValueError(
                "supply band must be finite and strictly increasing: "
                "(critical_low, warn_low, warn_high, critical_high)"
            )
        return v

    @field_validator(
        "health_load_warn_pct",
        "health_load_critical_pct",
        "health_current_warn_ma",
        "health_current_critical_ma",
        mode="before",
    )
    @classmethod
    def _optional_threshold(cls, v: object) -> object:
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "off"}:
            return None
        return v

    @field_validator(
        "health_temp_warn_margin_c",
        "health_temp_critical_margin_c",
        "health_temp_fallback_warn_c",
        "health_temp_fallback_critical_c",
        "health_load_warn_pct",
        "health_load_critical_pct",
        "health_current_warn_ma",
        "health_current_critical_ma",
    )
    @classmethod
    def _finite_non_negative(cls, v: float | None) -> float | None:
        if v is not None and (not math.isfinite(v) or v < 0):
            raise ValueError("health thresholds must be finite and non-negative")
        return v

    @field_validator("health_poll_hz")
    @classmethod
    def _check_health_poll_hz(cls, v: float) -> float:
        if not math.isfinite(v) or not 1.0 <= v <= 2.0:
            raise ValueError("health_poll_hz must be between 1 and 2")
        return v

    @field_validator("health_max_consecutive_failures")
    @classmethod
    def _check_health_failures(cls, v: int) -> int:
        if v < 1:
            raise ValueError("health_max_consecutive_failures must be at least 1")
        return v

    @field_validator("rest_pose")
    @classmethod
    def _check_rest_pose(cls, v: str) -> str:
        if not v.strip():
            return ""
        from .hardware.rest_pose import parse_rest_pose_setting

        parse_rest_pose_setting(v)  # raises ValueError on a malformed pose
        return v.strip()

    @field_validator("cameras")
    @classmethod
    def _check_cameras(cls, v: str) -> str:
        from .hardware.camera_spec import parse_camera_setting

        parse_camera_setting(v)  # raises ValueError naming the bad entry
        return v  # unchanged, so camera_ids stays exactly what was configured

    @field_validator(
        "halt_hold_seconds",
        "supervision_grace_seconds",
        "collapse_speed_deg_s",
        "grip_hold_margin",
    )
    @classmethod
    def _finite_positive(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("must be finite and positive")
        return v

    @field_validator("collapse_timeout_seconds", mode="before")
    @classmethod
    def _blank_collapse_timeout_is_unset(cls, v: object) -> object:
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "auto"}:
            return None
        return v

    @field_validator("collapse_timeout_seconds")
    @classmethod
    def _check_collapse_timeout(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if not math.isfinite(v) or v <= 0:
            raise ValueError("collapse_timeout_seconds must be finite and positive")
        if v > COLLAPSE_TIMEOUT_MAX_S:
            raise ValueError(
                f"collapse_timeout_seconds may not exceed {COLLAPSE_TIMEOUT_MAX_S:g} s "
                "(design doc section 5 bounds the descent)"
            )
        return v

    @property
    def camera_ids(self) -> list[str]:
        return [c.strip() for c in self.cameras.split(",") if c.strip()]

    @property
    def camera_specs(self) -> list[CameraSpec]:
        """The parsed SO101_CAMERAS entries, in configured order."""
        from .hardware.camera_spec import parse_camera_setting

        return parse_camera_setting(self.cameras)

    @property
    def camera_streams(self) -> list[str]:
        """Stream keys: every entry (color), then ``<entry>#depth`` per +depth entry."""
        from .hardware.camera_spec import stream_keys

        return stream_keys(self.camera_specs)

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.datasets_dir, self.outputs_dir):
            durable_makedirs(d)

    @field_validator("backend", mode="before")
    @classmethod
    def _blank_backend_is_unset(cls, v: object) -> object:
        return (v.strip().lower() or None) if isinstance(v, str) else v

    @field_validator("sim_engine", mode="before")
    @classmethod
    def _blank_engine_is_auto(cls, v: object) -> object:
        return (v.strip().lower() or "auto") if isinstance(v, str) else v

    @field_validator("sim_cameras")
    @classmethod
    def _check_sim_cameras(cls, v: str) -> str:
        names = [name.strip() for name in v.split(",") if name.strip()]
        unknown = sorted(set(names) - set(SIM_CAMERAS))
        if unknown:
            raise ValueError(f"unknown sim cameras {unknown}; choose from {list(SIM_CAMERAS)}")
        return ",".join(dict.fromkeys(names))

    @field_validator("sim_image_size")
    @classmethod
    def _check_sim_image_size(cls, v: str) -> str:
        width, height = parse_image_size(v)
        return f"{width}x{height}"

    @field_validator("sim_seed")
    @classmethod
    def _check_sim_seed(cls, v: int) -> int:
        if v < 0:  # numpy's generators take only non-negative seeds
            raise ValueError("sim_seed must be at least 0")
        return v

    @model_validator(mode="after")
    def _resolve_backend(self) -> Settings:
        if self.mock and self.backend in ("real", "sim"):
            raise ValueError(
                f"SO101_MOCK=1 conflicts with SO101_BACKEND={self.backend}; set only one"
            )
        if self.backend is None:
            self.backend = "mock" if self.mock else "real"
        elif self.backend == "mock":
            self.mock = True
        if self.backend == "real" and self.e2e_token.get_secret_value():
            raise ValueError(
                "SO101_E2E_TOKEN is test-only and cannot be configured with SO101_BACKEND=real"
            )
        return self

    @property
    def sim_camera_names(self) -> list[str]:
        return [name for name in self.sim_cameras.split(",") if name]

    @property
    def sim_image_wh(self) -> tuple[int, int]:
        return parse_image_size(self.sim_image_size)


# Collapse descent bound: the derived default is distance / speed + margin, at
# most the cap (design doc section 5 "Collapse").
COLLAPSE_TIMEOUT_MARGIN_S = 2.0
COLLAPSE_TIMEOUT_MAX_S = 12.0

# Mirrors so101.simulation (world.CAMERAS, render.MIN_SIZE, world.OFFSCREEN_SIZE;
# pinned by tests) so that reading settings never imports MuJoCo.
SIM_CAMERAS = ("front", "wrist_cam")
SIM_IMAGE_MIN = 16
SIM_IMAGE_MAX = (1280, 960)


def parse_image_size(value: str) -> tuple[int, int]:
    """``"320x240"`` -> ``(320, 240)``, within the simulator's render limits."""

    width_text, sep, height_text = value.strip().lower().partition("x")
    try:
        if not sep:
            raise ValueError
        width, height = int(width_text), int(height_text)
    except ValueError:
        raise ValueError(f"image size must look like 320x240, got {value!r}") from None
    if not (
        SIM_IMAGE_MIN <= width <= SIM_IMAGE_MAX[0] and SIM_IMAGE_MIN <= height <= SIM_IMAGE_MAX[1]
    ):
        raise ValueError(
            f"image size {width}x{height} is outside {SIM_IMAGE_MIN}x{SIM_IMAGE_MIN}.."
            f"{SIM_IMAGE_MAX[0]}x{SIM_IMAGE_MAX[1]}"
        )
    return width, height


STORAGE_SETTINGS = ("data_dir", "datasets_dir", "outputs_dir")


def describe_sources(sources: dict[str, Any]) -> str:
    """One line for :func:`settings_sources`: the env files and storage folders."""

    files = [
        f"{entry['path']} ({'read' if entry['exists'] else 'not found'})"
        for entry in sources["env_files"]
    ]
    text = "Settings files, later ones win: " + ("; ".join(files) or "none (env files are off)")
    return (
        f"{text}. Data {sources['data_dir']}, datasets {sources['datasets_dir']}, "
        f"outputs {sources['outputs_dir']}."
    )


def env_files() -> tuple[Path, ...]:
    """The .env files ``Settings()`` reads now, lowest priority first (later ones win)."""

    configured = Settings.model_config.get("env_file")
    if configured == AUTO_ENV_FILES:
        return current_env_files()
    if not configured:
        return ()
    if isinstance(configured, str | os.PathLike):
        return (Path(configured),)
    return tuple(Path(item) for item in configured)


def settings_sources(settings: Settings) -> dict[str, Any]:
    """Where ``settings`` came from, for ``so101 info``, ``so101 doctor`` and diagnostics.

    ``storage_from_cwd`` names the storage folders no setting placed, which
    therefore follow the folder the command runs in. ``warning`` is set when
    that probably isn't what was meant: the data folder has a .env (the suite
    is installed), the command runs elsewhere, and that .env doesn't set them.
    """

    files = env_files()
    auto = Settings.model_config.get("env_file") == AUTO_ENV_FILES
    roles = ["data folder", "current folder"] if auto else []
    data_home = current_data_home()
    from_cwd = [name for name in STORAGE_SETTINGS if name not in settings.model_fields_set]
    try:
        cwd: Path | None = Path.cwd()
    except OSError:
        cwd = None
    home_env = data_home / ".env"
    warning = None
    if (
        from_cwd
        and home_env in files
        and home_env.is_file()
        and (cwd is None or os.path.realpath(cwd) != os.path.realpath(data_home))
    ):
        names = ", ".join(f"SO101_{name.upper()}" for name in from_cwd)
        warning = (
            f"{home_env} doesn't set {names}, so this command keeps that data in the "
            f"current folder ({cwd}), not in {data_home}. Add absolute paths for them to "
            f"{home_env} (re-running install/install.sh adds them), or run from {data_home}."
        )
    return {
        # Lowest priority first: a later file overrides an earlier one.
        "env_files": [
            {
                "path": str(path),
                "exists": path.is_file(),
                "role": roles[index] if index < len(roles) else "configured",
            }
            for index, path in enumerate(files)
        ],
        "data_home": str(data_home),
        **{name: str(getattr(settings, name)) for name in STORAGE_SETTINGS},
        "storage_from_cwd": from_cwd,
        "warning": warning,
    }


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
