"""Model registry: discover trained checkpoints under the outputs dir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import get_settings


@dataclass
class ModelInfo:
    name: str
    path: str
    policy_type: str
    is_mock: bool
    checkpoint: str  # "last", "best", or step label
    # Additive (v0.2): what kind of checkpoint this is and what it was trained
    # on. ``label`` is the operator-facing description, e.g. "privileged-state
    # sim baseline" or "LeRobot act".
    kind: str = ""
    label: str = ""
    privileged_state: bool = False
    needs_task: bool = False
    task: str | None = None
    fps: int | None = None
    dataset: str | None = None


def _describe(pm: Path, *, name: str, checkpoint: str) -> ModelInfo:
    from .policy.checkpoint import InvalidCheckpointError, inspect_checkpoint

    try:
        info = inspect_checkpoint(pm)
    except InvalidCheckpointError as exc:
        return ModelInfo(
            name=name,
            path=str(pm),
            policy_type="unknown",
            is_mock=False,
            checkpoint=checkpoint,
            label=f"unreadable checkpoint: {exc}",
        )
    return ModelInfo(
        name=name,
        path=str(pm),
        policy_type=info.policy_type,
        is_mock=info.is_mock,
        checkpoint=checkpoint,
        kind=info.kind,
        label=info.label,
        privileged_state=info.privileged_state,
        needs_task=info.needs_task,
        task=info.task,
        fps=info.fps,
        dataset=info.dataset,
    )


def list_models() -> list[ModelInfo]:
    """Find ``.../checkpoints/<label>/pretrained_model`` dirs in outputs."""
    root = get_settings().outputs_dir
    out: list[ModelInfo] = []
    if not root.exists():
        return out
    for pm in sorted(root.glob("**/pretrained_model")):
        run = pm.parent.parent.parent  # outputs/<run>/checkpoints/<label>/pretrained_model
        label = pm.parent.name
        out.append(_describe(pm, name=f"{run.name}@{label}", checkpoint=label))
    return out


def get_model(name: str) -> ModelInfo | None:
    for m in list_models():
        if m.name == name or m.path == name:
            return m
    return None


def find_registered_model(model: str) -> ModelInfo | None:
    """Return the registry entry for a name or a path that resolves to one.

    Paths are compared after resolving symlinks, so ``checkpoints/last/...``
    matches the step directory it points to.
    """

    raw = str(model).strip()
    if not raw:
        return None
    models = list_models()
    for m in models:
        if m.name == raw or m.path == raw:
            return m
    try:
        target = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    for m in models:
        try:
            if Path(m.path).resolve(strict=True) == target:
                return m
        except (OSError, RuntimeError):
            continue
    return None
