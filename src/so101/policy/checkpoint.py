"""Policy checkpoint discovery, validation and backend gating.

Every entry point that is about to move an arm (``so101 run``, ``so101 eval``,
``POST /api/models/rollout/start``) calls :func:`preflight_policy` first. A
mistyped path, a directory that is not a checkpoint, or a checkpoint that must
not run on the active backend then fails *before* a control loop starts, in
mock mode as well as on hardware.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from ..config import get_settings

Backend = Literal["mock", "sim", "real"]
CheckpointKind = Literal["numpy_bc_mlp", "mock", "lerobot"]


class ModelNotFoundError(FileNotFoundError):
    """The model path does not exist and is not a registered model name."""


class InvalidCheckpointError(ValueError):
    """The path exists but is not a loadable policy checkpoint."""


class PolicyBackendError(RuntimeError):
    """The checkpoint is valid but must not run on the active backend (fail closed)."""


class MissingTaskError(InvalidCheckpointError):
    """A language-conditioned policy needs a task string and none is known (audit B6)."""


# LeRobot policies that read a language instruction (``observation["task"]``).
# A checkpoint is also treated as language-conditioned when its saved
# preprocessor runs LeRobot's tokenizer step.
LANGUAGE_POLICY_TYPES = frozenset({"smolvla", "pi0", "pi0_fast", "pi05", "xvla", "multi_task_dit"})
_TOKENIZER_STEP = "tokenizer_processor"


DEFAULT_CONTROL_FPS = 30
MIN_CONTROL_FPS = 1
MAX_CONTROL_FPS = 240


class _TrainedAt(Protocol):
    @property
    def fps(self) -> int | None: ...


def resolve_control_fps(requested: int | None, info: _TrainedAt | None = None) -> int:
    """The 1..240 Hz control rate: explicit, else training data's, else 30.

    ``info`` is a :class:`CheckpointInfo` or a ``CheckpointDefaults``. A policy
    trained on 15 fps data executes its action chunks at 2x speed when driven
    at 30 fps (audit B18).
    """

    raw = requested if requested is not None else (info.fps if info is not None else None)
    raw = DEFAULT_CONTROL_FPS if raw is None else raw
    if isinstance(raw, bool):
        raise ValueError(f"fps must be between {MIN_CONTROL_FPS} and {MAX_CONTROL_FPS}")
    try:
        fps = int(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"fps must be between {MIN_CONTROL_FPS} and {MAX_CONTROL_FPS}") from exc
    if fps != raw or not MIN_CONTROL_FPS <= fps <= MAX_CONTROL_FPS:
        raise ValueError(f"fps must be between {MIN_CONTROL_FPS} and {MAX_CONTROL_FPS}")
    return fps


def active_backend() -> Backend:
    """Return the robot backend this process drives: ``mock``, ``sim`` or ``real``.

    ``SO101_BACKEND=sim`` (added by the simulation stream) wins; otherwise
    ``SO101_MOCK=1`` means mock, and anything else is real hardware.
    """

    settings = get_settings()
    configured = str(getattr(settings, "backend", None) or "").strip().lower()
    if configured == "sim":
        return "sim"
    if settings.mock:
        return "mock"
    return "real"


def _active_sim_environment_state() -> tuple[str, tuple[str, ...]]:
    """Return the configured simulation task and its privileged-state layout.

    Importing the task registry requires the ``sim`` extra. A privileged
    checkpoint must never run when that layout cannot be established: treating
    an import or registry failure as compatible would let positional object
    features be interpreted under the wrong task.
    """

    task_name = str(get_settings().sim_task)
    try:
        from ..simulation.tasks import TASKS
    except ImportError as exc:
        raise PolicyBackendError(
            "cannot validate the checkpoint's environment state against the active "
            "simulation task; install the sim extra"
        ) from exc
    task = TASKS.get(task_name)
    if task is None:
        raise PolicyBackendError(
            f"cannot validate the checkpoint's environment state: unknown simulation "
            f"task {task_name!r}"
        )
    return task_name, tuple(task.env_state_names)


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    kind: CheckpointKind
    policy_type: str
    label: str
    is_mock: bool = False
    privileged_state: bool = False
    # What the checkpoint was trained on (see training.run_manifest), plus
    # whether it needs a language instruction. ``preflight_policy`` replaces
    # ``task`` with the task it resolved.
    task: str | None = None
    fps: int | None = None
    needs_task: bool = False
    dataset: str | None = None
    # The simulator object-state features a privileged-state baseline reads
    # (``observation.environment_state``); empty for every other checkpoint.
    env_state_names: tuple[str, ...] = ()


def missing_lerobot_runtime() -> list[str]:
    """The LeRobot inference dependencies that are not importable."""

    return [name for name in ("torch", "lerobot") if importlib.util.find_spec(name) is None]


def uses_language_task(path: str | Path, policy_type: str) -> bool:
    """True when the checkpoint conditions on ``observation["task"]`` (a prompt)."""

    if policy_type.strip().lower() in LANGUAGE_POLICY_TYPES:
        return True
    preprocessor = Path(path) / "policy_preprocessor.json"
    if not preprocessor.is_file():
        return False
    try:
        config = json.loads(preprocessor.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    steps = config.get("steps") if isinstance(config, dict) else None
    return any(
        isinstance(step, dict) and step.get("registry_name") == _TOKENIZER_STEP
        for step in steps or []
    )


def resolve_registered_model(model: str | Path) -> Path:
    """Resolve ``model`` only if it is a model the registry lists (audit B16).

    Accepts a registry name (``<run>@<label>``) or a path that is, or resolves
    to, a listed checkpoint under the outputs directory. API callers use this
    so a request cannot point the robot at an arbitrary directory or Hub id.
    """

    from ..models import find_registered_model

    raw = str(model).strip()
    if not raw:
        raise ModelNotFoundError("model path is empty")
    registered = find_registered_model(raw)
    if registered is None:
        raise ModelNotFoundError(
            f"model not found: {raw!r} is not a registered model; choose one from "
            "GET /api/models (checkpoints under the outputs directory)"
        )
    return Path(registered.path)


def resolve_model_path(model: str | Path) -> Path:
    """Resolve a checkpoint path or a registry name (``<run>@<label>``)."""

    raw = str(model).strip()
    if not raw:
        raise ModelNotFoundError("model path is empty")
    path = Path(raw).expanduser()
    if path.exists():
        return path
    from ..models import get_model

    registered = get_model(raw)
    if registered is not None and Path(registered.path).exists():
        return Path(registered.path)
    raise ModelNotFoundError(
        f"model not found: {raw!r} is neither an existing checkpoint directory nor a "
        "registered model name (see `so101 models ls`)"
    )


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    """Classify a checkpoint directory without loading any weights."""

    path = Path(path)
    if not path.is_dir():
        raise InvalidCheckpointError(f"{path} is not a checkpoint directory")
    from ..training.run_manifest import checkpoint_defaults
    from .baseline import BASELINE_TYPE, describe_baseline_checkpoint, is_baseline_checkpoint

    if is_baseline_checkpoint(path):
        # Reads and checks the manifest (a damaged one raises
        # InvalidCheckpointError); the weights are verified by preflight_policy.
        description = describe_baseline_checkpoint(path)
        defaults = checkpoint_defaults(path)
        return CheckpointInfo(
            path=path,
            kind="numpy_bc_mlp",
            policy_type=BASELINE_TYPE,
            label=description.label,
            privileged_state=description.privileged_state,
            task=description.task or defaults.task,
            fps=description.fps or defaults.fps,
            dataset=description.dataset or defaults.dataset,
            env_state_names=description.env_state_names,
        )
    config = path / "config.json"
    if not config.is_file():
        raise InvalidCheckpointError(
            f"{path} is not a policy checkpoint: expected config.json (LeRobot) "
            "or policy_manifest.json (numpy baseline)"
        )
    try:
        meta = json.loads(config.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise InvalidCheckpointError(f"invalid policy config JSON: {config}") from exc
    if not isinstance(meta, dict):
        raise InvalidCheckpointError(f"policy config is not a JSON object: {config}")
    # LeRobot writes the policy kind as "type"; so101's mock checkpoints also
    # carry "policy_type" (audit B21).
    policy_type = str(meta.get("type") or meta.get("policy_type") or "unknown")
    defaults = checkpoint_defaults(path)
    if meta.get("mock"):
        return CheckpointInfo(
            path=path,
            kind="mock",
            policy_type=policy_type,
            label=f"mock placeholder ({policy_type}); moves only the mock arm",
            is_mock=True,
            task=defaults.task,
            fps=defaults.fps,
            dataset=defaults.dataset,
        )
    return CheckpointInfo(
        path=path,
        kind="lerobot",
        policy_type=policy_type,
        label=f"LeRobot {policy_type}",
        task=defaults.task,
        fps=defaults.fps,
        needs_task=uses_language_task(path, policy_type),
        dataset=defaults.dataset,
    )


def preflight_policy(
    model: str | Path,
    *,
    backend: Backend | None = None,
    task: str | None = None,
    registered_only: bool = False,
) -> CheckpointInfo:
    """Validate that ``model`` exists and may run on ``backend`` before any motion.

    ``task`` is the language instruction for the rollout; when it is empty the
    task the checkpoint was trained on is used (audit B6). A language-conditioned
    policy with no task fails here instead of running on an empty prompt.
    ``registered_only`` restricts ``model`` to registry entries (API callers).

    Raises :class:`ModelNotFoundError`, :class:`InvalidCheckpointError` (and its
    :class:`MissingTaskError`) or :class:`PolicyBackendError`. LeRobot weights
    are not loaded here (the control loop loads them once); their runtime
    dependencies are checked. A numpy baseline is fully verified, and a
    privileged-state one (trained on simulator object state) is refused unless
    ``backend`` is ``"sim"``. The returned info carries the resolved task.
    """

    target = backend or active_backend()
    path = resolve_registered_model(model) if registered_only else resolve_model_path(model)
    info = inspect_checkpoint(path)
    if info.kind == "numpy_bc_mlp":
        from .baseline import load_baseline_model, privileged_refusal

        # A privileged-state baseline reads simulator object state that no real
        # (or mock) arm provides: it runs in simulation only, in every mode.
        if info.privileged_state and target != "sim":
            raise PolicyBackendError(privileged_refusal(info.path, target, info.env_state_names))
        if info.privileged_state:
            task_name, provided = _active_sim_environment_state()
            if info.env_state_names != provided:
                raise PolicyBackendError(
                    f"{info.label} expects environment state {list(info.env_state_names)}, but "
                    f"the active simulation task {task_name!r} provides {list(provided)}; "
                    "select the matching SO101_SIM_TASK or a compatible checkpoint"
                )
        load_baseline_model(info.path)  # verify weights hash, shapes and dtypes now
    if info.kind == "mock" and target != "mock":
        raise PolicyBackendError(
            f"refusing to load a moving mock checkpoint on the {target} backend while "
            "SO101_MOCK=0; mock checkpoints contain no trained weights"
        )
    if info.kind == "lerobot" and target != "mock":
        missing = missing_lerobot_runtime()
        if missing:
            raise PolicyBackendError(
                f"cannot run LeRobot {info.policy_type} checkpoint: {', '.join(missing)} "
                "not installed (install the robot/train extras)"
            )
    explicit = task.strip() if isinstance(task, str) else ""
    resolved = explicit or info.task
    # Mock mode runs MockPolicy for LeRobot checkpoints, so no prompt is read.
    if info.needs_task and not resolved and target != "mock":
        raise MissingTaskError(
            f"{info.label} is language-conditioned and needs a task instruction, but none "
            "was given and the checkpoint does not record a single training task; pass "
            '--task (CLI) or "task" (API) with the instruction used when recording'
        )
    return replace(info, task=resolved or None)
