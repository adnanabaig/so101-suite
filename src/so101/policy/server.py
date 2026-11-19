"""Load a trained policy and run inference.

Wraps LeRobot's policy classes. A ``MockPolicy`` provides a smooth, safe
output so rollouts/eval run without a trained checkpoint or torch.

Language-conditioned policies (SmolVLA, Pi0, ...) are served with the task
they were trained on unless a task is given explicitly (audit B6). Inference
runs on CUDA, then Apple MPS, then CPU unless ``SO101_POLICY_DEVICE`` picks a
device (audit B15).

The numpy behaviour-cloning baseline (``numpy_bc_mlp``,
:mod:`so101.policy.baseline`) is served in every mode, mock included, through
:class:`JointDictPolicy`; a privileged-state one runs only in simulation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .. import JOINT_NAMES
from ..config import get_settings
from ..logging import get_logger

log = get_logger(__name__)


class Policy(Protocol):
    def select_action(self, observation: dict[str, np.ndarray]) -> dict[str, float]: ...
    def reset(self) -> None: ...


class MockPolicy:
    """Emits a slow, bounded sinusoidal trajectory — visibly 'doing something'
    while staying inside safety limits."""

    label = "deterministic MockPolicy stand-in (checkpoint weights were not loaded)"
    privileged_state = False

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def reset(self) -> None:
        self._t0 = time.monotonic()

    def select_action(self, observation: dict[str, np.ndarray]) -> dict[str, float]:
        t = time.monotonic() - self._t0
        return {
            "shoulder_pan": 20 * math.sin(t * 0.3),
            "shoulder_lift": 15 * math.sin(t * 0.25 + 1),
            "elbow_flex": 20 * math.sin(t * 0.35 + 2),
            "wrist_flex": 10 * math.sin(t * 0.4),
            "wrist_roll": 30 * math.sin(t * 0.2),
            "gripper": 50 + 40 * math.sin(t * 0.5),
        }


class JointDictPolicy:
    """Serve an array policy as the ``{joint: degrees}`` dict the control loops use.

    ``inner`` follows the LeRobot-style protocol (``reset()`` and
    ``select_action(observation) -> float32 (6,)``: absolute targets, degrees
    with the gripper 0..100, in ``JOINT_NAMES`` order), like the numpy
    baseline and the simulator's policies. Every action is checked for shape
    and finiteness before it reaches a control loop; the follower's safety
    layer still clamps and rate-limits it.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    @property
    def label(self) -> str:
        return str(getattr(self.inner, "label", type(self.inner).__name__))

    @property
    def privileged_state(self) -> bool:
        return bool(getattr(self.inner, "privileged_state", False))

    def reset(self) -> None:
        self.inner.reset()

    def select_action(self, observation: dict[str, np.ndarray]) -> dict[str, float]:
        vector = np.asarray(self.inner.select_action(observation), dtype=np.float64).reshape(-1)
        if vector.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"SO-101 policy must return {len(JOINT_NAMES)} actions, got {vector.size}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("policy returned a non-finite action")
        return {joint: float(vector[index]) for index, joint in enumerate(JOINT_NAMES)}


_DEVICE_NAMES = ("auto", "cpu", "cuda", "mps")


def select_torch_device(torch: Any, preferred: str = "") -> Any:
    """Return the inference device: an explicit choice, else cuda, mps, cpu.

    ``preferred`` is ``SO101_POLICY_DEVICE``: empty or ``auto`` selects
    automatically; ``cpu``, ``cuda``, ``cuda:<n>`` or ``mps`` is required to
    be available (a misconfigured device fails instead of silently moving the
    policy to another one).
    """

    choice = (preferred or "").strip().lower() or "auto"
    base = choice.split(":", 1)[0]
    if base not in _DEVICE_NAMES or (":" in choice and base != "cuda"):
        raise ValueError(
            f"SO101_POLICY_DEVICE={preferred!r} is not a device; use auto, cpu, cuda, "
            "cuda:<index> or mps"
        )
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)

    def available(name: str) -> bool:
        if name == "cpu":
            return True
        if name == "cuda":
            return bool(torch.cuda.is_available())
        if name == "mps":
            return bool(mps is not None and mps.is_available())
        return False

    if choice != "auto":
        if not available(base):
            raise RuntimeError(
                f"SO101_POLICY_DEVICE={choice} is not available on this machine; "
                "unset it to select a device automatically"
            )
        return torch.device(choice)
    for name in ("cuda", "mps"):
        if available(name):
            return torch.device(name)
    return torch.device("cpu")


class LeRobotPolicy:
    """Adapter over a LeRobot policy and its saved processor pipelines.

    ``task`` is the language instruction passed to the policy's preprocessor
    on every step; policies without a language input ignore it.
    """

    def __init__(self, model_path: str | Path, task: str | None = None) -> None:
        self.model_path = str(model_path)
        self.task = task or None
        self._policy: Any | None = None
        self._preprocessor: Any | None = None
        self._postprocessor: Any | None = None
        self._prepare_observation = None
        self._device: Any = "cpu"
        self._action_names = [f"{joint}.pos" for joint in JOINT_NAMES]
        self._modern = False

    @staticmethod
    def _normalize_action_names(names: Any) -> list[str] | None:
        if names is None:
            return None
        if isinstance(names, dict):
            names = [name for group in names.values() for name in group]
        normalized = [name if str(name).endswith(".pos") else f"{name}.pos" for name in names]
        canonical = [f"{joint}.pos" for joint in JOINT_NAMES]
        if set(normalized) != set(canonical):
            raise ValueError(
                "checkpoint action joints do not match the canonical SO-101 layout: "
                f"{normalized} != {canonical}"
            )
        return normalized

    def _ensure_modern(self, torch: Any) -> bool:
        """Load current LeRobot APIs; return False only when that API is absent."""
        try:
            from lerobot.configs import PreTrainedConfig  # type: ignore[import-not-found]
            from lerobot.policies import (  # type: ignore[import-not-found]
                get_policy_class,
                make_pre_post_processors,
                prepare_observation_for_inference,
            )
        except (ImportError, ModuleNotFoundError, AttributeError):
            return False

        policy_cfg = PreTrainedConfig.from_pretrained(self.model_path)
        self._device = select_torch_device(torch, get_settings().policy_device)
        policy_cfg.device = self._device.type
        policy_cls = get_policy_class(policy_cfg.type)
        self._policy = policy_cls.from_pretrained(
            self.model_path,
            config=policy_cfg,
            revision=getattr(policy_cfg, "pretrained_revision", None),
        ).to(self._device)
        self._policy.eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=self.model_path,
            pretrained_revision=getattr(policy_cfg, "pretrained_revision", None),
            preprocessor_overrides={"device_processor": {"device": self._device.type}},
        )
        self._prepare_observation = prepare_observation_for_inference
        action_names = self._normalize_action_names(
            getattr(policy_cfg, "action_feature_names", None)
        )
        if action_names is not None:
            self._action_names = action_names
        self._modern = True
        return True

    def _ensure_legacy(self, torch: Any) -> None:
        """Compatibility path for pre-processor-pipeline LeRobot releases."""
        from lerobot.common.policies.factory import (  # type: ignore[import-not-found]
            get_policy_class,
        )

        model_dir = Path(self.model_path)
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"legacy LeRobot checkpoint is missing {config_path}; cannot load policy"
            )
        cfg = json.loads(config_path.read_text())
        policy_type = cfg.get("policy_type") or cfg.get("type")
        if not policy_type:
            raise ValueError(f"checkpoint config {config_path} has no policy type")
        policy_cls = get_policy_class(policy_type)
        self._device = select_torch_device(torch, get_settings().policy_device)
        self._policy = policy_cls.from_pretrained(model_dir).to(self._device)
        self._policy.eval()

    def _ensure(self):
        if self._policy is not None:
            return
        import torch  # type: ignore

        if not self._ensure_modern(torch):
            self._ensure_legacy(torch)

    def reset(self) -> None:
        self._ensure()
        assert self._policy is not None
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        if self._preprocessor is not None and hasattr(self._preprocessor, "reset"):
            self._preprocessor.reset()
        if self._postprocessor is not None and hasattr(self._postprocessor, "reset"):
            self._postprocessor.reset()

    @property
    def input_feature_keys(self) -> frozenset[str] | None:
        """The observation keys the checkpoint declares (``config.input_features``).

        The runner passes only these, so a policy trained without a depth
        stream is never handed one.
        """
        self._ensure()
        features = getattr(getattr(self._policy, "config", None), "input_features", None)
        return frozenset(features) if features else None

    def select_action(self, observation: dict[str, np.ndarray]) -> dict[str, float]:
        self._ensure()
        assert self._policy is not None
        import torch  # type: ignore

        obs = {key: np.asarray(value) for key, value in observation.items()}
        with torch.inference_mode():
            if self._modern:
                assert self._prepare_observation is not None
                batch = self._prepare_observation(
                    obs,
                    self._device,
                    task=self.task,
                    robot_type="so_follower",
                )
                batch = self._preprocessor(batch)
                action = self._policy.select_action(batch)
                action = self._postprocessor(action)
            else:
                batch = {
                    key: torch.as_tensor(value).unsqueeze(0).to(self._device)
                    for key, value in obs.items()
                }
                action = self._policy.select_action(batch)

        if isinstance(action, dict):
            if "action" not in action:
                raise ValueError(
                    f"policy returned an action mapping without 'action': {action.keys()}"
                )
            action = action["action"]
        vector = action.squeeze(0).detach().cpu().numpy().reshape(-1)
        if len(vector) != len(JOINT_NAMES):
            raise ValueError(
                f"SO-101 policy must return {len(JOINT_NAMES)} actions, got {len(vector)}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("policy returned a non-finite action")

        by_feature = {
            feature_name: float(vector[index])
            for index, feature_name in enumerate(self._action_names)
        }
        return {joint: by_feature[f"{joint}.pos"] for joint in JOINT_NAMES}


def load_policy(model_path: str | Path, *, task: str | None = None) -> Policy:
    """Load a policy, using ``MockPolicy`` only in explicit mock mode.

    ``task`` defaults to the single task the checkpoint was trained on (from
    its run manifest or ``train_config.json`` dataset). A language-conditioned
    LeRobot policy with no known task is refused rather than served an empty
    prompt.

    A numpy baseline checkpoint is loaded for real in every mode (it needs no
    torch), verified, and refused with
    :class:`~so101.policy.checkpoint.PolicyBackendError` when it is a
    privileged-state baseline and the active backend is not the simulator.
    """
    settings = get_settings()
    path = Path(model_path)
    from .baseline import is_baseline_checkpoint, load_baseline_policy
    from .checkpoint import active_backend

    if is_baseline_checkpoint(path):
        baseline = load_baseline_policy(path, backend=active_backend())
        log.info("loaded %s from %s", baseline.label, path)
        return JointDictPolicy(baseline)
    if settings.mock:
        log.info("mock mode: loading MockPolicy")
        return MockPolicy()
    cfg = path / "config.json"
    meta: dict[str, Any] = {}
    if cfg.exists():
        try:
            loaded = json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid policy config JSON: {cfg}") from exc
        meta = loaded if isinstance(loaded, dict) else {}
        if meta.get("mock"):
            raise RuntimeError("refusing to load a moving mock checkpoint while SO101_MOCK=0")

    from ..training.run_manifest import checkpoint_defaults
    from .checkpoint import MissingTaskError, uses_language_task

    defaults = checkpoint_defaults(path)
    policy_type = str(meta.get("type") or meta.get("policy_type") or defaults.policy_type or "")
    resolved_task = (task or "").strip() or defaults.task
    if not resolved_task and uses_language_task(path, policy_type):
        raise MissingTaskError(
            f"{path} is a language-conditioned {policy_type} policy and no task is known; "
            "pass the instruction used when recording (so101 run --task ...)"
        )
    # Fail before any motor is touched. LeRobotPolicy._ensure deliberately
    # propagates dependency, checkpoint and processor errors in real mode.
    policy = LeRobotPolicy(model_path, task=resolved_task)
    policy._ensure()
    return policy
