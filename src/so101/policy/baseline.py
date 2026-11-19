"""Numpy behaviour-cloning baseline: a small MLP policy that needs no torch.

The baseline maps the arm's joint state (plus, for simulation datasets, the
privileged ``observation.environment_state``) to the next absolute joint
target. :mod:`so101.training.bc_baseline` trains it in-process
(``so101 train <repo> --policy bc_baseline``); :func:`load_baseline_policy`
serves it.

Policy protocol (the simulator's, LeRobot-style)::

    policy.reset()
    action = policy.select_action({"observation.state": state, ...})

``action`` is float32 with shape (6,): absolute targets in ``JOINT_NAMES``
order, degrees for the arm joints and 0 (closed) to 100 (open) for the
gripper, clipped to the SO-101 joint limits.
:class:`so101.policy.server.JointDictPolicy` adapts it to the
``{joint: value}`` form that the robot control loops use.

Privileged state: a checkpoint trained with ``observation.environment_state``
(true object positions that only the simulator knows) is a "privileged-state
sim baseline". It refuses every backend except ``sim``. If it is somehow run
anyway, it fails at its first step because the observation has no
environment state.

Checkpoint layout (``<run>/checkpoints/last/pretrained_model/``):

``policy_manifest.json``
    Type ``numpy_bc_mlp``, version, feature spec, normalization statistics,
    network shape, weights hash, dataset (repo id, episodes, fps, task),
    training configuration and metrics, and ``privileged_state``.
``weights.npz``
    Float32 layer weights ``W0..Wn`` and biases ``b0..bn``, written
    byte-deterministically. The manifest records its SHA-256.
``train_log.jsonl``
    One JSON object per held-out evaluation during training.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import JOINT_NAMES, __version__, atomic_io
from ..hardware.safety import DEFAULT_LIMITS
from .checkpoint import Backend, InvalidCheckpointError, PolicyBackendError

BASELINE_TYPE = "numpy_bc_mlp"
BASELINE_VERSION = 1
MANIFEST_NAME = "policy_manifest.json"
WEIGHTS_NAME = "weights.npz"
TRAIN_LOG_NAME = "train_log.jsonl"
PRIVILEGED_LABEL = "privileged-state sim baseline"
STATE_ONLY_LABEL = "state-only BC baseline"

STATE_KEY = "observation.state"
ENV_STATE_KEY = "observation.environment_state"
STATE_NAMES: tuple[str, ...] = tuple(f"{joint}.pos" for joint in JOINT_NAMES)
ACTIVATIONS = ("relu", "tanh")
TIME_FEATURES = ("none", "time")
TARGETS = ("delta", "absolute")
MAX_HISTORY = 16
# Normalized inputs are clipped to +-5 standard deviations, so a reading far
# outside the training data cannot drive the network arbitrarily far.
INPUT_CLIP = 5.0
# Target dimensions that vary less than this (suite units) are not worth
# fitting more precisely; the floor keeps their normalized noise bounded.
OUTPUT_STD_FLOOR = 1e-2
# Manifest keys derived from the model itself (everything else is metadata).
CORE_MANIFEST_KEYS = frozenset(
    {
        "type",
        "version",
        "label",
        "privileged_state",
        "so101_version",
        "features",
        "normalization",
        "network",
        "weights",
        "action_limits",
    }
)

_N_JOINTS = len(JOINT_NAMES)
_LOW = np.asarray([DEFAULT_LIMITS[joint][0] for joint in JOINT_NAMES], dtype=np.float32)
_HIGH = np.asarray([DEFAULT_LIMITS[joint][1] for joint in JOINT_NAMES], dtype=np.float32)
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def baseline_label(privileged_state: bool) -> str:
    return PRIVILEGED_LABEL if privileged_state else STATE_ONLY_LABEL


def privileged_refusal(path: str | Path, backend: str, env_state_names: Sequence[str] = ()) -> str:
    """The message for a privileged-state checkpoint asked to run off the simulator."""

    names = f" ({', '.join(env_state_names)})" if env_state_names else ""
    return (
        f"{path} is a {PRIVILEGED_LABEL}: it was trained on {ENV_STATE_KEY}{names}, true "
        f"object state that only the simulator provides, so it refuses the {backend} backend. "
        "Evaluate it in simulation, or train a state-only baseline for the arm: "
        "so101 train <repo> --policy bc_baseline --state-only"
    )


# --- features -------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """What the network reads and predicts. Training and inference share it.

    Inputs, in order: the joint state; ``history - 1`` frame-to-frame state
    differences (velocity), most recent first; the environment state (only
    for a privileged checkpoint); and ``time_s``, the time since the episode
    started (``time_feature="time"``). The first frame of an episode is
    repeated before it, so its differences are zero.

    ``target`` is what the network predicts: ``delta`` (the action minus the
    current state) or ``absolute`` (the action itself). The policy output is
    always the absolute target. ``fps`` is the dataset's frame rate: the
    policy counts steps, so ``time_s`` is ``step / fps``.
    """

    env_state_names: tuple[str, ...] = ()
    history: int = 1
    time_feature: str = "none"
    target: str = "delta"
    fps: int = 30

    def __post_init__(self) -> None:
        if isinstance(self.history, bool) or not isinstance(self.history, int):
            raise ValueError("history must be an integer")
        if not 1 <= self.history <= MAX_HISTORY:
            raise ValueError(f"history must be from 1 to {MAX_HISTORY} frames")
        if self.time_feature not in TIME_FEATURES:
            raise ValueError(f"time_feature must be one of {', '.join(TIME_FEATURES)}")
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {', '.join(TARGETS)}")
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps < 1:
            raise ValueError("fps must be a positive integer")
        names = tuple(self.env_state_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("environment state names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("environment state names must be unique")
        object.__setattr__(self, "env_state_names", names)

    @property
    def privileged_state(self) -> bool:
        return bool(self.env_state_names)

    @property
    def input_names(self) -> list[str]:
        names = list(STATE_NAMES)
        for lag in range(1, self.history):
            names += [f"{name}.diff{lag}" for name in STATE_NAMES]
        names += [f"env.{name}" for name in self.env_state_names]
        if self.time_feature == "time":
            names.append("time_s")
        return names

    @property
    def input_dim(self) -> int:
        return len(self.input_names)

    def episode_inputs(self, state: np.ndarray, env_state: np.ndarray | None = None) -> np.ndarray:
        """Inputs for every frame of one episode (vectorized; used for training)."""

        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != _N_JOINTS:
            raise ValueError(f"state must have shape (frames, {_N_JOINTS})")
        frames = state.shape[0]
        index = np.arange(frames)
        parts = [state]
        for lag in range(1, self.history):
            now = np.maximum(index - lag + 1, 0)
            before = np.maximum(index - lag, 0)
            parts.append(state[now] - state[before])
        if self.env_state_names:
            env = self._env_rows(env_state, frames)
            parts.append(env)
        if self.time_feature == "time":
            parts.append((index.astype(np.float64) / self.fps).astype(np.float32)[:, None])
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    def step_inputs(
        self, window: Sequence[np.ndarray], env_state: np.ndarray | None, step: int
    ) -> np.ndarray:
        """Inputs for one control step; ``window`` is the last ``history`` states, oldest first."""

        if len(window) != self.history:
            raise ValueError(f"expected {self.history} states, got {len(window)}")
        parts = [np.asarray(window[-1], dtype=np.float32)]
        for lag in range(1, self.history):
            parts.append(
                np.asarray(window[-lag], dtype=np.float32)
                - np.asarray(window[-lag - 1], dtype=np.float32)
            )
        if self.env_state_names:
            parts.append(self._env_rows(env_state, 1)[0])
        if self.time_feature == "time":
            parts.append(np.asarray([step / self.fps], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _env_rows(self, env_state: np.ndarray | None, frames: int) -> np.ndarray:
        if env_state is None:
            raise ValueError(
                f"{PRIVILEGED_LABEL} needs {ENV_STATE_KEY} "
                f"({len(self.env_state_names)} values: {', '.join(self.env_state_names)})"
            )
        env = np.asarray(env_state, dtype=np.float32).reshape(frames, -1)
        if env.shape[1] != len(self.env_state_names):
            raise ValueError(
                f"{ENV_STATE_KEY} must have {len(self.env_state_names)} values "
                f"({', '.join(self.env_state_names)}), got {env.shape[1]}"
            )
        return env

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": list(STATE_NAMES),
            "environment_state": list(self.env_state_names),
            "history": self.history,
            "time_feature": self.time_feature,
            "target": self.target,
            "fps": self.fps,
            "inputs": self.input_names,
            "action": list(STATE_NAMES),
            "units": "degrees; gripper 0 (closed) to 100 (open)",
            "input_clip_sigma": INPUT_CLIP,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FeatureSpec:
        if list(raw.get("state", [])) != list(STATE_NAMES):
            raise ValueError(f"state features must be {list(STATE_NAMES)}")
        if list(raw.get("action", [])) != list(STATE_NAMES):
            raise ValueError(f"action features must be {list(STATE_NAMES)}")
        env = raw.get("environment_state", [])
        if not isinstance(env, list):
            raise ValueError("environment_state must be a list of names")
        spec = cls(
            env_state_names=tuple(env),
            history=raw.get("history", 1),
            time_feature=raw.get("time_feature", "none"),
            target=raw.get("target", "delta"),
            fps=raw.get("fps", 30),
        )
        inputs = raw.get("inputs")
        if inputs is not None and list(inputs) != spec.input_names:
            raise ValueError("the recorded input names do not match the feature spec")
        return spec


@dataclass(frozen=True)
class Normalization:
    """Per-feature standardization fitted on the training split.

    An input with no variation in the training data has ``input_std`` 0 and
    is ignored (its normalized value is always 0).
    """

    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray

    def __post_init__(self) -> None:
        for name in ("input_mean", "input_std", "output_mean", "output_std"):
            array = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains a non-finite value")
            object.__setattr__(self, name, array)
        if self.input_mean.shape != self.input_std.shape:
            raise ValueError("input_mean and input_std differ in length")
        if self.output_mean.shape != (_N_JOINTS,) or self.output_std.shape != (_N_JOINTS,):
            raise ValueError(f"output statistics must have {_N_JOINTS} values")
        if (self.input_std < 0).any() or (self.output_std <= 0).any():
            raise ValueError("standard deviations must be positive (input_std may be 0)")

    @classmethod
    def fit(cls, inputs: np.ndarray, outputs: np.ndarray) -> Normalization:
        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(outputs, dtype=np.float64)
        input_mean = x.mean(axis=0)
        input_std = x.std(axis=0)
        constant = input_std <= 1e-6 * np.maximum(1.0, np.abs(input_mean))
        input_std = np.where(constant, 0.0, input_std)
        return cls(
            input_mean=input_mean,
            input_std=input_std,
            output_mean=y.mean(axis=0),
            output_std=np.maximum(y.std(axis=0), OUTPUT_STD_FLOOR),
        )

    @property
    def input_dim(self) -> int:
        return int(self.input_mean.shape[0])

    def input_scale(self) -> np.ndarray:
        safe = np.where(self.input_std > 0, self.input_std, np.float32(1.0))
        return np.where(self.input_std > 0, np.float32(1.0) / safe, np.float32(0.0)).astype(
            np.float32
        )

    def normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        z = (np.asarray(inputs, dtype=np.float32) - self.input_mean) * self.input_scale()
        return np.clip(z, -INPUT_CLIP, INPUT_CLIP).astype(np.float32, copy=False)

    def normalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        y = (np.asarray(outputs, dtype=np.float32) - self.output_mean) / self.output_std
        return y.astype(np.float32, copy=False)

    def denormalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        y = np.asarray(outputs, dtype=np.float32) * self.output_std + self.output_mean
        return y.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        # float32 -> Python float is exact, and JSON round-trips it exactly.
        return {
            name: [float(value) for value in getattr(self, name)]
            for name in ("input_mean", "input_std", "output_mean", "output_std")
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Normalization:
        values: dict[str, np.ndarray] = {}
        for name in ("input_mean", "input_std", "output_mean", "output_std"):
            entry = raw.get(name)
            if not isinstance(entry, list) or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in entry
            ):
                raise ValueError(f"normalization.{name} must be a list of numbers")
            values[name] = np.asarray(entry, dtype=np.float32)
        return cls(**values)


# --- network --------------------------------------------------------------------


def _activate(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(values, np.float32(0.0))
    return np.tanh(values)


@dataclass
class MLP:
    """A float32 multilayer perceptron: ``activation`` between layers, linear output."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {', '.join(ACTIVATIONS)}")
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("an MLP needs one bias per weight matrix")
        self.weights = [np.asarray(w, dtype=np.float32) for w in self.weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in self.biases]
        previous = None
        for index, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            if w.ndim != 2 or b.shape != (w.shape[1],):
                raise ValueError(f"layer {index}: W{index} {w.shape} and b{index} {b.shape} differ")
            if previous is not None and w.shape[0] != previous:
                raise ValueError(f"layer {index} takes {w.shape[0]} inputs, expected {previous}")
            if not (np.isfinite(w).all() and np.isfinite(b).all()):
                raise ValueError(f"layer {index} contains a non-finite weight")
            previous = w.shape[1]

    @classmethod
    def initialize(
        cls, sizes: Sequence[int], rng: np.random.Generator, activation: str = "relu"
    ) -> MLP:
        """He-uniform (ReLU) or LeCun-uniform (tanh) weights, zero biases.

        The output layer starts 10x smaller, so a fresh network predicts close
        to the mean target.
        """

        if len(sizes) < 2 or any(int(size) < 1 for size in sizes):
            raise ValueError("an MLP needs at least an input and an output size")
        gain = np.sqrt(2.0) if activation == "relu" else 1.0
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        layers = list(zip(sizes[:-1], sizes[1:], strict=True))
        for index, (fan_in, fan_out) in enumerate(layers):
            limit = gain * np.sqrt(3.0 / fan_in)
            if index == len(layers) - 1:
                limit *= 0.1
            weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32))
            biases.append(np.zeros(fan_out, dtype=np.float32))
        return cls(weights=weights, biases=biases, activation=activation)

    @property
    def sizes(self) -> list[int]:
        return [int(self.weights[0].shape[0])] + [int(w.shape[1]) for w in self.weights]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.asarray(inputs, dtype=np.float32)
        last = len(self.weights) - 1
        for index, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            hidden = hidden @ w + b
            if index < last:
                hidden = _activate(hidden, self.activation)
        return hidden

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for index, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            out[f"W{index}"] = w
            out[f"b{index}"] = b
        return out


@dataclass
class BaselineModel:
    """Feature spec + normalization + network: everything inference needs."""

    spec: FeatureSpec
    normalization: Normalization
    mlp: MLP

    def __post_init__(self) -> None:
        sizes = self.mlp.sizes
        if sizes[0] != self.spec.input_dim or self.normalization.input_dim != self.spec.input_dim:
            raise ValueError(
                f"the feature spec has {self.spec.input_dim} inputs but the network takes "
                f"{sizes[0]} and the normalization has {self.normalization.input_dim}"
            )
        if sizes[-1] != _N_JOINTS:
            raise ValueError(f"the network must output {_N_JOINTS} targets, not {sizes[-1]}")

    def predict(self, inputs: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Absolute joint targets for feature rows and the current state of each row."""

        z = self.normalization.normalize_inputs(inputs)
        y = self.normalization.denormalize_outputs(self.mlp.forward(z))
        if self.spec.target == "delta":
            y = np.asarray(states, dtype=np.float32) + y
        return np.clip(y, _LOW, _HIGH).astype(np.float32, copy=False)


# --- policy ---------------------------------------------------------------------


def _observation_vector(observation: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    value = observation.get(key)
    if value is None:
        raise ValueError(f"observation has no {key!r}")
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{key} is not numeric") from None
    if array.shape != (size,):
        raise ValueError(f"{key} must have {size} values, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains a non-finite value")
    return array


class BaselinePolicy:
    """The numpy BC baseline behind the ``reset``/``select_action`` protocol.

    ``select_action`` returns float32 (6,) absolute targets, clipped to the
    SO-101 joint limits. The policy keeps the last ``history`` states and
    counts steps since :meth:`reset` (for ``time_s``), so it assumes it is
    called at the dataset's frame rate (``fps``).
    """

    def __init__(
        self,
        model: BaselineModel,
        manifest: Mapping[str, Any] | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.manifest: dict[str, Any] = dict(manifest or {})
        self.path = Path(path) if path is not None else None
        self._window: deque[np.ndarray] = deque(maxlen=model.spec.history)
        self._step = 0

    @property
    def spec(self) -> FeatureSpec:
        return self.model.spec

    @property
    def privileged_state(self) -> bool:
        return self.model.spec.privileged_state

    @property
    def label(self) -> str:
        return baseline_label(self.privileged_state)

    @property
    def fps(self) -> int:
        return self.model.spec.fps

    def reset(self) -> None:
        self._window.clear()
        self._step = 0

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping of feature name to array")
        state = _observation_vector(observation, STATE_KEY, _N_JOINTS)
        env = None
        if self.privileged_state:
            names = self.spec.env_state_names
            if observation.get(ENV_STATE_KEY) is None:
                raise ValueError(
                    f"{PRIVILEGED_LABEL}: the observation has no {ENV_STATE_KEY} "
                    f"({len(names)} values: {', '.join(names)}); only the simulator provides it"
                )
            env = _observation_vector(observation, ENV_STATE_KEY, len(names))
        if not self._window:
            self._window.extend(state for _ in range(self.spec.history))
        else:
            self._window.append(state)
        inputs = self.spec.step_inputs(list(self._window), env, self._step)
        action = self.model.predict(inputs[None, :], state[None, :])[0]
        self._step += 1
        return action


# --- checkpoints ----------------------------------------------------------------


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """An ``.npz`` archive with fixed member timestamps: same arrays, same bytes."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(
                member, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def _write_file(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def build_manifest(
    model: BaselineModel, *, weights_sha256: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The manifest for ``model``; ``extra`` adds metadata (dataset, training, metrics)."""

    manifest: dict[str, Any] = {
        "type": BASELINE_TYPE,
        "version": BASELINE_VERSION,
        "label": baseline_label(model.spec.privileged_state),
        "privileged_state": model.spec.privileged_state,
        "so101_version": __version__,
        "features": model.spec.to_dict(),
        "normalization": model.normalization.to_dict(),
        "network": {"sizes": model.mlp.sizes, "activation": model.mlp.activation},
        "weights": {"file": WEIGHTS_NAME, "sha256": weights_sha256, "dtype": "float32"},
        "action_limits": {joint: list(DEFAULT_LIMITS[joint]) for joint in JOINT_NAMES},
    }
    for key, value in (extra or {}).items():
        if key in CORE_MANIFEST_KEYS:
            continue  # derived from the model; never overridden by metadata
        manifest[key] = value
    return manifest


def manifest_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The metadata part of a manifest (what :func:`build_manifest` takes as ``extra``)."""

    return {key: value for key, value in manifest.items() if key not in CORE_MANIFEST_KEYS}


def save_baseline_checkpoint(
    directory: str | Path,
    model: BaselineModel,
    *,
    extra: Mapping[str, Any] | None = None,
    train_log: Iterable[Mapping[str, Any]] = (),
) -> Path:
    """Write a checkpoint into ``directory`` (which must not exist), atomically.

    Files are staged in a sibling directory and renamed into place, so the
    model registry never sees a partial checkpoint. A storage error syncing
    the parent after that rename is reported even though the complete target
    necessarily remains visible; callers must inspect it before retrying.
    """

    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    atomic_io.durable_makedirs(target.parent)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=target.parent))
    try:
        weights = _npz_bytes(model.mlp.arrays())
        _write_file(stage / WEIGHTS_NAME, weights)
        log_lines = "".join(
            json.dumps(dict(entry), sort_keys=True, allow_nan=False) + "\n" for entry in train_log
        )
        _write_file(stage / TRAIN_LOG_NAME, log_lines.encode("utf-8"))
        manifest = build_manifest(
            model, weights_sha256=hashlib.sha256(weights).hexdigest(), extra=extra
        )
        text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        _write_file(stage / MANIFEST_NAME, text.encode("utf-8"))
        # File fsync alone does not make the staged directory entries durable.
        # Flush them before publication, then flush the parent so the rename
        # itself survives an immediate power loss.
        atomic_io.fsync_directory(stage)
        os.replace(stage, target)
        atomic_io.fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def is_baseline_checkpoint(path: str | Path) -> bool:
    """True when ``path`` holds a numpy baseline checkpoint (``policy_manifest.json``).

    A manifest that cannot be parsed still counts, so loading it reports what
    is wrong instead of "unknown checkpoint".
    """

    manifest = Path(path).expanduser() / MANIFEST_NAME
    try:
        if not manifest.is_file():
            return False
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return True
    return isinstance(raw, dict) and raw.get("type") == BASELINE_TYPE


def read_baseline_manifest(path: str | Path) -> dict[str, Any]:
    """Parse and check a baseline manifest. Raises :class:`InvalidCheckpointError`."""

    manifest_path = Path(path).expanduser() / MANIFEST_NAME
    if not manifest_path.is_file():
        raise InvalidCheckpointError(f"{Path(path)} has no {MANIFEST_NAME}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise InvalidCheckpointError(
            f"unreadable baseline manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidCheckpointError(f"{manifest_path} is not a JSON object")
    if raw.get("type") != BASELINE_TYPE:
        raise InvalidCheckpointError(
            f"{manifest_path} declares type {raw.get('type')!r}, not {BASELINE_TYPE!r}"
        )
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version:
        raise InvalidCheckpointError(f"{manifest_path}: invalid version {version!r}")
    if version > BASELINE_VERSION:
        raise InvalidCheckpointError(
            f"{manifest_path}: {BASELINE_TYPE} version {version} is newer than this so101 "
            f"supports ({BASELINE_VERSION}); upgrade so101"
        )
    return raw


def _manifest_spec(manifest: Mapping[str, Any], where: Path) -> FeatureSpec:
    features = manifest.get("features")
    if not isinstance(features, dict):
        raise InvalidCheckpointError(f"{where}: manifest has no feature spec")
    try:
        spec = FeatureSpec.from_dict(features)
    except (TypeError, ValueError) as exc:
        raise InvalidCheckpointError(f"{where}: invalid feature spec: {exc}") from exc
    if manifest.get("privileged_state") is not spec.privileged_state:
        # Fail closed: never trust a label that disagrees with the inputs.
        raise InvalidCheckpointError(
            f"{where}: privileged_state={manifest.get('privileged_state')!r} disagrees with "
            f"the feature spec (environment state: {list(spec.env_state_names)})"
        )
    return spec


@dataclass(frozen=True)
class BaselineDescription:
    """What a baseline checkpoint is, read from its manifest without the weights."""

    path: Path
    privileged_state: bool
    label: str
    env_state_names: tuple[str, ...]
    fps: int
    task: str | None
    dataset: str | None


def describe_baseline_checkpoint(path: str | Path) -> BaselineDescription:
    """Read and check a baseline manifest (weights are not loaded)."""

    directory = Path(path).expanduser()
    manifest = read_baseline_manifest(directory)
    spec = _manifest_spec(manifest, directory)
    weights = manifest.get("weights")
    name = weights.get("file") if isinstance(weights, dict) else None
    if name != WEIGHTS_NAME or not (directory / WEIGHTS_NAME).is_file():
        raise InvalidCheckpointError(f"{directory}: {WEIGHTS_NAME} is missing")
    dataset = manifest.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    task = manifest.get("task") or dataset.get("task")
    repo = dataset.get("repo_id")
    return BaselineDescription(
        path=directory,
        privileged_state=spec.privileged_state,
        label=baseline_label(spec.privileged_state),
        env_state_names=spec.env_state_names,
        fps=spec.fps,
        task=task if isinstance(task, str) and task.strip() else None,
        dataset=repo if isinstance(repo, str) and repo else None,
    )


def load_baseline_model(path: str | Path) -> tuple[BaselineModel, dict[str, Any]]:
    """Load and verify a checkpoint: manifest, weights hash, shapes and dtypes."""

    directory = Path(path).expanduser()
    manifest = read_baseline_manifest(directory)
    spec = _manifest_spec(manifest, directory)
    return _load_model(directory, manifest, spec), manifest


def _load_model(directory: Path, manifest: Mapping[str, Any], spec: FeatureSpec) -> BaselineModel:
    weights_meta = manifest.get("weights")
    network = manifest.get("network")
    normalization = manifest.get("normalization")
    if not isinstance(weights_meta, dict) or weights_meta.get("file") != WEIGHTS_NAME:
        raise InvalidCheckpointError(f"{directory}: manifest does not name {WEIGHTS_NAME}")
    if not isinstance(network, dict) or not isinstance(normalization, dict):
        raise InvalidCheckpointError(f"{directory}: manifest has no network or normalization")
    weights_path = directory / WEIGHTS_NAME
    try:
        data = weights_path.read_bytes()
    except OSError as exc:
        raise InvalidCheckpointError(f"{directory}: cannot read {WEIGHTS_NAME}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != weights_meta.get("sha256"):
        raise InvalidCheckpointError(
            f"{weights_path} does not match the manifest (sha256 {digest[:12]}..., expected "
            f"{str(weights_meta.get('sha256'))[:12]}...); the checkpoint is damaged or mixed"
        )
    sizes = network.get("sizes")
    if not isinstance(sizes, list) or len(sizes) < 2:
        raise InvalidCheckpointError(f"{directory}: manifest network sizes are invalid")
    layers = len(sizes) - 1
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            expected = {f"W{i}" for i in range(layers)} | {f"b{i}" for i in range(layers)}
            if set(archive.files) != expected:
                raise InvalidCheckpointError(
                    f"{weights_path} holds {sorted(archive.files)}, expected {sorted(expected)}"
                )
            arrays = {name: archive[name] for name in expected}
    except InvalidCheckpointError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise InvalidCheckpointError(
            f"{weights_path} is not a valid weights archive: {exc}"
        ) from exc
    if any(array.dtype != np.float32 for array in arrays.values()):
        raise InvalidCheckpointError(f"{weights_path}: weights must be float32")
    try:
        mlp = MLP(
            weights=[arrays[f"W{i}"] for i in range(layers)],
            biases=[arrays[f"b{i}"] for i in range(layers)],
            activation=str(network.get("activation", "relu")),
        )
        if mlp.sizes != [int(size) for size in sizes]:
            raise ValueError(f"layer sizes {mlp.sizes} differ from the manifest's {sizes}")
        model = BaselineModel(
            spec=spec, normalization=Normalization.from_dict(normalization), mlp=mlp
        )
    except (TypeError, ValueError) as exc:
        raise InvalidCheckpointError(f"{directory}: {exc}") from exc
    return model


def load_baseline_policy(path: str | Path, *, backend: Backend | None = None) -> BaselinePolicy:
    """Load a numpy baseline checkpoint as a :class:`BaselinePolicy`.

    ``backend`` is the robot backend the policy will drive. Entry points that
    move an arm pass it (``so101.policy.server.load_policy`` passes the active
    backend), and a privileged-state checkpoint is refused with
    :class:`PolicyBackendError` unless it is ``"sim"``. ``None`` means the
    caller is the in-process simulator (the simulator's evaluation hook) or
    only inspects the policy; a privileged policy then still fails at its
    first step without ``observation.environment_state``, which only the
    simulator provides.

    Raises :class:`InvalidCheckpointError` for a damaged or incompatible
    checkpoint (it is fully verified before it is returned).
    """

    directory = Path(path).expanduser()
    manifest = read_baseline_manifest(directory)
    spec = _manifest_spec(manifest, directory)
    if spec.privileged_state and backend is not None and backend != "sim":
        raise PolicyBackendError(privileged_refusal(directory, backend, spec.env_state_names))
    model = _load_model(directory, manifest, spec)
    return BaselinePolicy(model, manifest, directory)


__all__ = [
    "BASELINE_TYPE",
    "BASELINE_VERSION",
    "ENV_STATE_KEY",
    "MANIFEST_NAME",
    "PRIVILEGED_LABEL",
    "STATE_KEY",
    "STATE_ONLY_LABEL",
    "TRAIN_LOG_NAME",
    "WEIGHTS_NAME",
    "BaselineDescription",
    "BaselineModel",
    "BaselinePolicy",
    "FeatureSpec",
    "MLP",
    "Normalization",
    "baseline_label",
    "build_manifest",
    "describe_baseline_checkpoint",
    "is_baseline_checkpoint",
    "load_baseline_model",
    "load_baseline_policy",
    "manifest_metadata",
    "privileged_refusal",
    "read_baseline_manifest",
    "save_baseline_checkpoint",
]
