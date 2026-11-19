"""Train the numpy behaviour-cloning baseline in-process (``--policy bc_baseline``).

No torch and no subprocess: the trainer reads the dataset into memory with
:mod:`so101.data.loader`, fits a small MLP with AdamW on CPU and writes a
:mod:`so101.policy.baseline` checkpoint. It works with the core install, in
mock mode and on a real-robot setup without the ``[train]`` extra.

Data and split
    The selected episodes (``kept`` by default: success plus unlabeled; see
    the loader) are split by episode, never by frame, into a training set and
    a held-out set (``val_fraction``, at least one episode when there are two
    or more). Normalization statistics come from the training episodes only.
Objective
    Mean squared error between the network output and the normalized target
    (the next action minus the current state for ``target=delta``).
Optimizer
    AdamW (decoupled weight decay on weight matrices only), cosine learning
    rate decay to 10% of ``lr``, minibatches drawn without replacement from a
    fresh permutation every epoch.
Early stopping
    Every ``eval_every`` steps the held-out loss is measured; the best weights
    are kept and training stops after ``patience`` evaluations without
    improvement. Without a held-out episode the final weights are kept.
Determinism
    Every random draw comes from ``numpy.random.SeedSequence(seed)``, so the
    same data, configuration and seed produce byte-identical weights on the
    same machine and numpy build.

Trainer overrides (``--arg KEY=VALUE`` or the API's ``extra_args``) are
listed in :data:`BC_ARGS`; steps and batch size come from the job.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from .. import JOINT_NAMES, __version__
from ..data.loader import TrainingData, load_training_data
from ..policy.baseline import (
    MAX_HISTORY,
    MLP,
    TARGETS,
    TIME_FEATURES,
    BaselineModel,
    FeatureSpec,
    Normalization,
    baseline_label,
    save_baseline_checkpoint,
)
from .jobs import TrainingJob
from .preflight import InvalidTrainingArgsError, job_episode_selection

CHECKPOINT_SUBDIR = Path("checkpoints") / "last" / "pretrained_model"
_ADAM_BETAS = (0.9, 0.999)
_ADAM_EPS = 1e-8
_MIN_LR_FRACTION = 0.1
# A held-out loss must beat the best by this relative margin to reset patience.
_IMPROVEMENT = 1e-4
_MAX_HIDDEN_LAYERS = 4
_MAX_HIDDEN_UNITS = 1024


class TrainingCancelled(Exception):
    """The cancel event was set; no checkpoint was written."""


@dataclass(frozen=True)
class BCConfig:
    """Hyperparameters of one baseline training run (all recorded in the manifest)."""

    steps: int = 3000
    batch_size: int = 256
    seed: int = 0
    episodes: str = "kept"
    state_only: bool = False
    hidden: tuple[int, ...] = (128, 128)
    activation: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    patience: int = 10
    eval_every: int = 100
    history: int = 1
    time_feature: str = "none"
    target: str = "delta"

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        object.__setattr__(self, "hidden", tuple(int(size) for size in self.hidden))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hidden"] = list(self.hidden)
        return out


def _parse_int(key: str, value: str, low: int, high: int) -> int:
    try:
        number = int(value.strip())
    except ValueError:
        raise InvalidTrainingArgsError(f"{key} must be an integer, not {value!r}") from None
    if not low <= number <= high:
        raise InvalidTrainingArgsError(f"{key} must be from {low} to {high}, got {number}")
    return number


def _parse_float(key: str, value: str, low: float, high: float) -> float:
    try:
        number = float(value.strip())
    except ValueError:
        raise InvalidTrainingArgsError(f"{key} must be a number, not {value!r}") from None
    if not (math.isfinite(number) and low <= number <= high):
        raise InvalidTrainingArgsError(f"{key} must be from {low:g} to {high:g}, got {value!r}")
    return number


def _parse_choice(key: str, value: str, choices: tuple[str, ...]) -> str:
    cleaned = value.strip().lower()
    if cleaned not in choices:
        raise InvalidTrainingArgsError(f"{key} must be one of {', '.join(choices)}, not {value!r}")
    return cleaned


def _parse_bool(key: str, value: str) -> bool:
    cleaned = value.strip().lower()
    if cleaned in ("1", "true", "yes", "on"):
        return True
    if cleaned in ("0", "false", "no", "off"):
        return False
    raise InvalidTrainingArgsError(f"{key} must be true or false, not {value!r}")


def _parse_hidden(key: str, value: str) -> tuple[int, ...]:
    parts = [part for part in value.replace("x", ",").split(",") if part.strip()]
    if not 1 <= len(parts) <= _MAX_HIDDEN_LAYERS:
        raise InvalidTrainingArgsError(
            f"{key} must list 1 to {_MAX_HIDDEN_LAYERS} layer sizes, e.g. 128,128"
        )
    return tuple(_parse_int(key, part, 1, _MAX_HIDDEN_UNITS) for part in parts)


# key -> (parser, help). Steps and batch size are job fields, not overrides.
BC_ARGS: dict[str, tuple[Callable[[str, str], Any], str]] = {
    "seed": (lambda k, v: _parse_int(k, v, 0, 2**32 - 1), "random seed (default 0)"),
    "episodes": (
        lambda k, v: _parse_choice(k, v, ("kept", "success", "all")),
        "episode selection: kept (success + unlabeled, default), success or all",
    ),
    "state_only": (
        _parse_bool,
        "ignore observation.environment_state so the policy can drive a real arm",
    ),
    "hidden": (_parse_hidden, "hidden layer sizes (default 128,128)"),
    "activation": (lambda k, v: _parse_choice(k, v, ("relu", "tanh")), "relu (default) or tanh"),
    "lr": (lambda k, v: _parse_float(k, v, 1e-6, 1.0), "Adam learning rate (default 1e-3)"),
    "weight_decay": (
        lambda k, v: _parse_float(k, v, 0.0, 1.0),
        "decoupled weight decay (default 1e-4)",
    ),
    "val_fraction": (
        lambda k, v: _parse_float(k, v, 0.0, 0.5),
        "share of episodes held out for early stopping (default 0.2)",
    ),
    "patience": (
        lambda k, v: _parse_int(k, v, 1, 1000),
        "held-out evaluations without improvement before stopping (default 10)",
    ),
    "eval_every": (
        lambda k, v: _parse_int(k, v, 1, 100_000),
        "steps between held-out evaluations (default 100)",
    ),
    "history": (
        lambda k, v: _parse_int(k, v, 1, MAX_HISTORY),
        "states the policy sees: 1 = current only, 2+ adds velocities (default 1)",
    ),
    "time_feature": (
        lambda k, v: _parse_choice(k, v, TIME_FEATURES),
        "none (default) or time: add seconds since the episode started",
    ),
    "target": (
        lambda k, v: _parse_choice(k, v, TARGETS),
        "delta (default: action minus state) or absolute",
    ),
}


def validate_bc_args(args: Mapping[str, str]) -> dict[str, Any]:
    """Parse ``bc_baseline`` trainer overrides; raises :class:`InvalidTrainingArgsError`."""

    parsed: dict[str, Any] = {}
    for key, value in args.items():
        entry = BC_ARGS.get(key)
        if entry is None:
            raise InvalidTrainingArgsError(
                f"extra_args[{key!r}] is not a bc_baseline option; allowed: "
                f"{', '.join(sorted(BC_ARGS))} (steps and batch size are job settings)"
            )
        parser, _help = entry
        parsed[key] = parser(key, value)
    return parsed


def bc_config_for(job: TrainingJob) -> BCConfig:
    """The :class:`BCConfig` for a job: its steps, batch size and overrides.

    The episode selection is the job's ``episodes``, or ``extra_args["episodes"]``
    when only that is set (both work; they must agree when both are set).
    """

    args = validate_bc_args(job.extra_args)
    args["episodes"] = job_episode_selection(job)
    return BCConfig(steps=job.steps, batch_size=job.batch_size, **args)


# --- training -------------------------------------------------------------------


@dataclass
class _Arrays:
    episodes: list[int]
    inputs: np.ndarray
    states: np.ndarray
    actions: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.inputs.shape[0])


def _stack(data: TrainingData, indices: list[int], spec: FeatureSpec) -> _Arrays:
    episodes = [data.episodes[i] for i in indices]
    k = spec.input_dim
    if not episodes:
        empty = np.zeros((0, len(JOINT_NAMES)), dtype=np.float32)
        return _Arrays([], np.zeros((0, k), dtype=np.float32), empty, empty.copy())
    inputs = np.concatenate([spec.episode_inputs(ep.state, ep.env_state) for ep in episodes])
    states = np.concatenate([ep.state for ep in episodes]).astype(np.float32, copy=False)
    actions = np.concatenate([ep.action for ep in episodes]).astype(np.float32, copy=False)
    return _Arrays([ep.episode for ep in episodes], inputs, states, actions)


def split_episodes(
    count: int, val_fraction: float, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """Episode indices for training and held-out evaluation (sorted)."""

    if count < 2 or val_fraction <= 0:
        return list(range(count)), []
    held_out = min(count - 1, max(1, int(round(count * val_fraction))))
    order = rng.permutation(count)
    return sorted(int(i) for i in order[held_out:]), sorted(int(i) for i in order[:held_out])


def _targets(arrays: _Arrays, spec: FeatureSpec) -> np.ndarray:
    if spec.target == "delta":
        return arrays.actions - arrays.states
    return arrays.actions


def _loss_and_grads(
    mlp: MLP, x: np.ndarray, y: np.ndarray
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    """Mean squared error and its gradients for one minibatch (float32)."""

    layers = len(mlp.weights)
    outputs = [x]
    hidden = x
    for index, (w, b) in enumerate(zip(mlp.weights, mlp.biases, strict=True)):
        hidden = hidden @ w + b
        if index < layers - 1:
            hidden = np.maximum(hidden, 0.0) if mlp.activation == "relu" else np.tanh(hidden)
        outputs.append(hidden)
    diff = hidden - y
    loss = float(np.mean(np.square(diff, dtype=np.float64)))
    grad = (diff * np.float32(2.0 / diff.size)).astype(np.float32, copy=False)
    grad_w: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * layers
    grad_b: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * layers
    for index in range(layers - 1, -1, -1):
        grad_w[index] = outputs[index].T @ grad
        grad_b[index] = grad.sum(axis=0)
        if index > 0:
            grad = grad @ mlp.weights[index].T
            activated = outputs[index]
            if mlp.activation == "relu":
                grad = grad * (activated > 0)
            else:
                grad = grad * (np.float32(1.0) - activated * activated)
    return loss, grad_w, grad_b


def _mse(mlp: MLP, x: np.ndarray, y: np.ndarray, chunk: int = 8192) -> float:
    if x.shape[0] == 0:
        return math.nan
    total = 0.0
    for start in range(0, x.shape[0], chunk):
        diff = mlp.forward(x[start : start + chunk]) - y[start : start + chunk]
        total += float(np.sum(np.square(diff, dtype=np.float64)))
    return total / (x.shape[0] * y.shape[1])


def _per_joint_mae(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = np.mean(np.abs(predicted.astype(np.float64) - actual), axis=0)
    return {joint: round(float(value), 4) for joint, value in zip(JOINT_NAMES, error, strict=True)}


@dataclass
class BCResult:
    """A trained baseline plus everything the manifest records about the run."""

    model: BaselineModel
    config: BCConfig
    metrics: dict[str, Any]
    train_log: list[dict[str, Any]]
    train_episodes: list[int]
    val_episodes: list[int]
    data: TrainingData
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return baseline_label(self.model.spec.privileged_state)


class BCTrainer:
    """Iterate :meth:`run` to train; progress is yielded as ``(step, loss, line)``.

    The trained model is in :attr:`result` once the iterator is exhausted.
    Setting ``cancel_event`` stops training and raises
    :class:`TrainingCancelled`.
    """

    def __init__(
        self,
        data: TrainingData,
        config: BCConfig,
        *,
        cancel_event: threading.Event | None = None,
        tasks: Sequence[str] = (),
    ) -> None:
        self.data = data
        self.config = config
        self.cancel_event = cancel_event
        self.tasks = list(tasks)
        self.result: BCResult | None = None

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def run(self) -> Iterator[tuple[int, float, str]]:
        cfg = self.config
        data = self.data
        started = time.monotonic()
        env_names = () if cfg.state_only else tuple(data.env_state_names)
        if not cfg.state_only and data.env_state_dim and not env_names:
            raise ValueError("environment state is present but has no names")
        spec = FeatureSpec(
            env_state_names=env_names,
            history=cfg.history,
            time_feature=cfg.time_feature,
            target=cfg.target,
            fps=max(1, int(data.fps)),
        )
        # A state-only spec has no environment-state inputs, so episode_inputs
        # ignores any env-state the episodes carry (the data is not modified).
        split_seed, init_seed, batch_seed = np.random.SeedSequence(cfg.seed).spawn(3)
        train_idx, val_idx = split_episodes(
            len(data.episodes), cfg.val_fraction, np.random.default_rng(split_seed)
        )
        train = _stack(data, train_idx, spec)
        val = _stack(data, val_idx, spec)
        if train.frames < 2:
            raise ValueError(
                f"{data.repo_id!r} has {train.frames} training frame(s) after the held-out "
                "split; record more episodes"
            )
        norm = Normalization.fit(train.inputs, _targets(train, spec))
        x_train = norm.normalize_inputs(train.inputs)
        y_train = norm.normalize_outputs(_targets(train, spec))
        x_val = norm.normalize_inputs(val.inputs)
        y_val = norm.normalize_outputs(_targets(val, spec))
        sizes = [spec.input_dim, *cfg.hidden, len(JOINT_NAMES)]
        mlp = MLP.initialize(sizes, np.random.default_rng(init_seed), cfg.activation)
        yield (
            0,
            math.nan,
            f"[bc] data: {data.describe()}; {train.frames} training frames from "
            f"{len(train.episodes)} episode(s), {val.frames} held-out frames from "
            f"{len(val.episodes)} episode(s)",
        )
        yield (
            0,
            math.nan,
            f"[bc] model: {baseline_label(spec.privileged_state)}; {spec.input_dim} inputs "
            f"({', '.join(spec.input_names[:8])}{', ...' if spec.input_dim > 8 else ''}), "
            f"hidden {'x'.join(str(s) for s in cfg.hidden)} {cfg.activation}, "
            f"target {spec.target}",
        )

        params = [*mlp.weights, *mlp.biases]
        decays = [True] * len(mlp.weights) + [False] * len(mlp.biases)
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        beta1, beta2 = _ADAM_BETAS
        batch = min(cfg.batch_size, train.frames)
        rng = np.random.default_rng(batch_seed)
        order = rng.permutation(train.frames)
        cursor = 0
        best_val = math.inf
        best_step = 0
        best_params = [p.copy() for p in params]
        stale = 0
        stopped_early = False
        window: list[float] = []
        log: list[dict[str, Any]] = []
        progress_every = max(1, cfg.steps // 100)
        step = 0
        lr = cfg.lr
        for step in range(1, cfg.steps + 1):
            if self._cancelled():
                raise TrainingCancelled("training cancelled")
            if cursor + batch > train.frames:
                order = rng.permutation(train.frames)
                cursor = 0
            index = order[cursor : cursor + batch]
            cursor += batch
            loss, grad_w, grad_b = _loss_and_grads(mlp, x_train[index], y_train[index])
            if not math.isfinite(loss):
                raise RuntimeError(
                    f"training diverged at step {step} (loss {loss}); lower lr (now {cfg.lr:g})"
                )
            window.append(loss)
            progress = (step - 1) / max(1, cfg.steps - 1)
            lr = cfg.lr * (
                _MIN_LR_FRACTION + (1 - _MIN_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress))
            )
            correction1 = 1 - beta1**step
            correction2 = 1 - beta2**step
            for p, g, m, v, decay in zip(
                params, [*grad_w, *grad_b], first, second, decays, strict=True
            ):
                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * (g * g)
                update = (m / correction1) / (np.sqrt(v / correction2) + _ADAM_EPS)
                if decay and cfg.weight_decay:
                    update = update + cfg.weight_decay * p
                p -= (lr * update).astype(np.float32, copy=False)
            evaluate = step % cfg.eval_every == 0 or step == cfg.steps
            if evaluate:
                train_loss = float(np.mean(window))
                window = []
                val_loss = _mse(mlp, x_val, y_val)
                entry: dict[str, Any] = {
                    "step": step,
                    "lr": round(lr, 8),
                    "train_loss": round(train_loss, 6),
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
                line = f"[bc] step {step}/{cfg.steps} train_loss {train_loss:.5f}"
                if val.frames:
                    entry["val_loss"] = round(val_loss, 6)
                    line += f" val_loss {val_loss:.5f}"
                    if val_loss < best_val * (1 - _IMPROVEMENT):
                        best_val = val_loss
                        best_step = step
                        best_params = [p.copy() for p in params]
                        stale = 0
                    else:
                        stale += 1
                log.append(entry)
                yield step, train_loss, line
                if val.frames and stale >= cfg.patience:
                    stopped_early = True
                    yield (
                        step,
                        train_loss,
                        f"[bc] early stop at step {step}: no held-out improvement in "
                        f"{cfg.patience} evaluations; keeping step {best_step}",
                    )
                    break
            elif step % progress_every == 0:
                percent = int(100 * step / cfg.steps)
                yield step, loss, f"Training: {percent:3d}%| {step}/{cfg.steps} loss={loss:.5f}"
        if val.frames:
            for p, best in zip(params, best_params, strict=True):
                p[...] = best
        else:
            best_step = step
        model = BaselineModel(spec=spec, normalization=norm, mlp=mlp)
        metrics = self._metrics(model, train, val, norm, spec)
        metrics.update(
            best_step=best_step,
            steps_run=step,
            stopped_early=stopped_early,
            wall_seconds=round(time.monotonic() - started, 3),
        )
        self.result = BCResult(
            model=model,
            config=cfg,
            metrics=metrics,
            train_log=log,
            train_episodes=train.episodes,
            val_episodes=val.episodes,
            data=data,
        )
        self.result.extra = manifest_extra(self.result, tasks=self.tasks)

    def _metrics(
        self,
        model: BaselineModel,
        train: _Arrays,
        val: _Arrays,
        norm: Normalization,
        spec: FeatureSpec,
    ) -> dict[str, Any]:
        target = val if val.frames else train
        x = norm.normalize_inputs(target.inputs)
        y = norm.normalize_outputs(_targets(target, spec))
        predicted = model.predict(target.inputs, target.states)
        mean_action = train.actions.astype(np.float64).mean(axis=0)
        mae = _per_joint_mae(predicted, target.actions)
        mean_mae = _per_joint_mae(
            np.broadcast_to(mean_action, target.actions.shape), target.actions
        )
        hold_mae = _per_joint_mae(target.states, target.actions)

        def overall(per_joint: dict[str, float]) -> float:
            return round(float(np.mean(list(per_joint.values()))), 4)

        return {
            "evaluated_on": "held_out" if val.frames else "train",
            "train_loss": round(_mse(model.mlp, *self._normalized(train, norm, spec)), 6),
            "val_loss": round(_mse(model.mlp, x, y), 6) if val.frames else None,
            "mae": overall(mae),
            "mae_per_joint": mae,
            "mean_predictor_mae": overall(mean_mae),
            "hold_predictor_mae": overall(hold_mae),
            "units": "degrees; gripper 0-100",
            "train_frames": train.frames,
            "val_frames": val.frames,
        }

    @staticmethod
    def _normalized(
        arrays: _Arrays, norm: Normalization, spec: FeatureSpec
    ) -> tuple[np.ndarray, np.ndarray]:
        return norm.normalize_inputs(arrays.inputs), norm.normalize_outputs(_targets(arrays, spec))


def fit_bc(
    data: TrainingData,
    config: BCConfig | None = None,
    *,
    cancel_event: threading.Event | None = None,
    on_line: Callable[[str], None] | None = None,
    tasks: Sequence[str] = (),
) -> BCResult:
    """Train to completion and return the result (no checkpoint is written)."""

    trainer = BCTrainer(data, config or BCConfig(), cancel_event=cancel_event, tasks=tasks)
    for _step, _loss, line in trainer.run():
        if on_line is not None:
            on_line(line)
    assert trainer.result is not None
    return trainer.result


def manifest_extra(result: BCResult, *, tasks: Sequence[str] = ()) -> dict[str, Any]:
    """Dataset, training and metric metadata for the checkpoint manifest.

    ``tasks`` are the dataset's task strings; a single task becomes the
    checkpoint's default task.
    """

    data = result.data
    tasks = list(dict.fromkeys(task for task in tasks if task))
    task = tasks[0] if len(tasks) == 1 else None
    used = sorted(episode.episode for episode in data.episodes)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": task,
        "dataset": {
            "repo_id": data.repo_id,
            "format": data.format,
            "fps": data.fps,
            "task": task,
            "tasks": tasks,
            "selection": data.selection,
            "episodes": used,
            "train_episodes": result.train_episodes,
            "val_episodes": result.val_episodes,
            "skipped": [{"episode": s.episode, "reason": s.reason} for s in data.skipped],
            "frames": data.total_frames,
            "environment_state": list(result.model.spec.env_state_names),
        },
        "training": {
            "trainer": "numpy_in_process",
            "config": result.config.to_dict(),
            "numpy_version": np.__version__,
            "so101_version": __version__,
        },
        "metrics": result.metrics,
    }


def train_bc_baseline(
    job: TrainingJob, cancel_event: threading.Event | None = None
) -> Iterator[tuple[int, float, str]]:
    """Train ``job`` and write its checkpoint; yields ``(step, loss, line)``.

    The checkpoint goes to ``<output_dir>/checkpoints/last/pretrained_model``.
    A cancelled job writes nothing.
    """

    from ..data.dataset import dataset_read_hold, get_dataset
    from .run_manifest import dataset_tasks

    config = bc_config_for(job)
    with dataset_read_hold(job.repo_id):
        data = load_training_data(
            job.repo_id,
            selection=config.episodes,
            env_state="ignore" if config.state_only else "auto",
        )
        info = get_dataset(job.repo_id)
        tasks = dataset_tasks(info) if info is not None else []
    trainer = BCTrainer(data, config, cancel_event=cancel_event, tasks=tasks)
    last = (0, math.nan)
    try:
        for step, loss, line in trainer.run():
            last = (step, loss)
            yield step, loss, line
    except TrainingCancelled:
        return
    result = trainer.result
    assert result is not None
    target = Path(job.output_dir) / CHECKPOINT_SUBDIR
    save_baseline_checkpoint(target, result.model, extra=result.extra, train_log=result.train_log)
    m = result.metrics
    where = "held-out" if m["evaluated_on"] == "held_out" else "training"
    yield (
        last[0],
        last[1],
        f"[bc] {where} action MAE {m['mae']:.3f} (mean predictor {m['mean_predictor_mae']:.3f}, "
        f"hold-position {m['hold_predictor_mae']:.3f}); best step {m['best_step']}",
    )
    yield last[0], last[1], f"[bc] saved {result.label} -> {target}"


def with_overrides(config: BCConfig, **overrides: Any) -> BCConfig:
    """``config`` with some fields replaced (unknown names raise ``TypeError``)."""

    known = {f.name for f in fields(BCConfig)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise TypeError(f"unknown BCConfig field(s): {', '.join(unknown)}")
    return replace(config, **overrides)


__all__ = [
    "BC_ARGS",
    "CHECKPOINT_SUBDIR",
    "BCConfig",
    "BCResult",
    "BCTrainer",
    "TrainingCancelled",
    "bc_config_for",
    "fit_bc",
    "manifest_extra",
    "split_episodes",
    "train_bc_baseline",
    "validate_bc_args",
    "with_overrides",
]
