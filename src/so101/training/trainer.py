"""Build + run LeRobot training as a subprocess, parsing progress.

We run LeRobot's ``lerobot-train`` in a subprocess (rather than import its
trainer) so jobs are isolatable, cancellable, and crash-resistant. It starts
as ``<this python> -m lerobot.scripts.lerobot_train``, not from PATH: the
desktop launcher and ``so101-suite`` run the venv's Python without activating
the venv, so its console scripts may not be on PATH. A mock trainer simulates a
realistic loss curve when LeRobot isn't installed, so the whole train→serve UI
works end-to-end without a GPU. The ``bc_baseline`` preset is the exception:
it trains a real numpy policy in-process in every mode
(:mod:`so101.training.bc_baseline`).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

from ..config import get_settings
from ..data.dataset import dataset_operation_lock, get_dataset
from ..logging import get_logger
from .configs import PolicyPreset
from .jobs import TrainingJob
from .preflight import (
    EpisodePlan,
    job_episode_selection,
    plan_episodes,
    resolve_preset,
    validate_extra_args,
    validate_training_dataset,
)
from .processes import own_process_group, process_start_marker
from .run_manifest import write_run_manifest

log = get_logger(__name__)

# LeRobot 0.6.1's ``lerobot-train`` console script is
# ``lerobot.scripts.lerobot_train:main`` (its entry_points.txt), and that module
# runs ``main()`` under ``if __name__ == "__main__"`` (lerobot_train.py:809-810),
# so ``python -m`` starts the same trainer with the same ``sys.argv[1:]``.
TRAIN_MODULE = "lerobot.scripts.lerobot_train"
# lerobot_train.train() refuses to start without accelerate
# (``require_package("accelerate", extra="training")``, lerobot_train.py:231);
# the suite's [train] extra brings it through lerobot[training]. The robot
# extra installs lerobot without it, so the module alone is not enough.
TRAIN_REQUIRES: tuple[str, ...] = ("accelerate",)

# LeRobot logs "step:1K smpl:8K ep:2 epch:0.50 loss:0.123 ..." with the step
# rounded to a suffix (format_big_number), so a bare integer match reads
# "step:1K" as step 1. The tqdm "Training" bar carries the exact count.
_STEP_RE = re.compile(r"\bstep[:=]\s*(\d+(?:\.\d+)?)([KMBTQ]?)(?![\w.])")
_LOSS_RE = re.compile(r"\bloss[:=]\s*([0-9.eE+-]+)")
_PROGRESS_RE = re.compile(r"^\s*Training:.*?\|\s*(\d+)/(\d+)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_STEP_SUFFIX = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12, "Q": 10**15}


def is_progress_line(line: str) -> bool:
    """True for a tqdm "Training:" bar refresh (one per carriage return)."""

    return _ANSI_RE.sub("", line).lstrip().startswith("Training:")


def parse_step(line: str, current: int = 0) -> int:
    """Return the training step a log line reports (never moving backwards).

    The exact ``N/TOTAL`` of the tqdm bar wins; otherwise the rounded
    ``step:1K`` form is scaled by its suffix.
    """

    clean = _ANSI_RE.sub("", line)
    if match := _PROGRESS_RE.search(clean):
        return max(current, int(match.group(1)))
    if match := _STEP_RE.search(clean):
        value = float(match.group(1)) * _STEP_SUFFIX[match.group(2)]
        return max(current, int(round(value)))
    return current


def policy_inputs_without_depth(dataset_root: str | Path) -> dict[str, dict] | None:
    """``--policy.input_features`` for a LeRobot dataset with depth maps, minus the depth.

    LeRobot 0.6.1 makes every image feature a VISUAL policy input; a depth map
    becomes (1, H, W), which the 3-channel image backbones of the LeRobot
    presets cannot take. This mirrors ``dataset_to_policy_features`` for the
    remaining features. ``None`` when the dataset has no depth map (or its
    metadata cannot be read), so such commands stay unchanged.
    """
    try:
        info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    features = info.get("features") if isinstance(info, dict) else None
    if not isinstance(features, dict):
        return None
    inputs: dict[str, dict] = {}
    dropped = False
    for key, feature in features.items():
        if not isinstance(feature, dict):
            continue
        shape = [int(value) for value in feature.get("shape") or []]
        if feature.get("dtype") in ("image", "video"):
            extra = feature.get("info") or {}
            names = list(feature.get("names") or [])
            if len(shape) == 3 and len(names) == 3 and names[2] in ("channel", "channels"):
                shape = [shape[2], shape[0], shape[1]]  # (h, w, c) -> (c, h, w)
            if extra.get("is_depth_map") or extra.get("video.is_depth_map") or shape[:1] == [1]:
                dropped = True
                continue
            inputs[key] = {"type": "VISUAL", "shape": shape}
        elif key == "observation.environment_state":
            inputs[key] = {"type": "ENV", "shape": shape}
        elif key.startswith("observation"):
            inputs[key] = {"type": "STATE", "shape": shape}
    return inputs if dropped else None


def _module_found(name: str) -> bool:
    """True if ``name`` can be imported here. A dotted name imports its parent
    packages (``lerobot/__init__.py`` imports only its version), never the module."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # a missing parent package raises
        return False


def lerobot_train_available() -> bool:
    """True if this interpreter can run LeRobot training (``TRAIN_MODULE``).

    The trainer and ``/api/capabilities`` both ask this, so the UI shows
    training as available exactly when a job could start, whatever PATH is.
    """
    return all(_module_found(name) for name in (TRAIN_MODULE, *TRAIN_REQUIRES))


def train_launcher() -> list[str]:
    """The argv that starts ``lerobot-train`` with this interpreter."""
    return [sys.executable, "-m", TRAIN_MODULE]


def build_train_command(job: TrainingJob) -> list[str]:
    """Construct the ``lerobot-train`` argv for a job (:func:`train_launcher` first).

    Preset and job ``extra_args`` are validated (see
    :func:`~so101.training.preflight.validate_extra_args`) and placed before the
    flags so101 owns. Draccus keeps the last duplicate, so the output dir,
    dataset, episode selection, step budget and ``--policy.push_to_hub=false``
    cannot be overridden even by an argument that slipped past validation.

    The job's episode selection (``kept`` by default) is applied by outcome
    label (:func:`~so101.training.preflight.plan_episodes`):
    ``--dataset.episodes=[...]`` lists the episodes kept, and is left out when
    every episode is. Unreadable labels, an empty selection, or a Hub dataset
    with any selection but ``all`` raise
    :class:`~so101.data.loader.TrainingDataError`.
    """
    return _train_command(job)[0]


def _train_command(job: TrainingJob) -> tuple[list[str], EpisodePlan]:
    """:func:`build_train_command` plus the episode plan its command applies."""
    preset = resolve_preset(job.policy)
    if preset.trainer != "lerobot":
        raise ValueError(f"the {preset.name} preset does not train with lerobot-train")
    job_args = validate_extra_args(preset, job.extra_args)
    local_dataset = get_dataset(job.repo_id)
    if local_dataset is not None and local_dataset.format != "lerobot":
        raise RuntimeError(
            f"dataset {job.repo_id!r} uses {local_dataset.format}; convert and "
            "validate it as a LeRobot dataset before real training"
        )
    plan = plan_episodes(job.repo_id, job_episode_selection(job), local_dataset)
    cmd = [
        *train_launcher(),
        f"--policy.type={preset.policy_type}",
        # Overridable default; LeRobot falls back to an available device.
        "--policy.device=cuda",
    ]
    for k, v in {**preset.extra_args, **job_args}.items():
        cmd.append(f"--{k}={v}")
    cmd += [
        f"--dataset.repo_id={job.repo_id}",
        f"--output_dir={job.output_dir}",
        f"--steps={job.steps}",
        f"--batch_size={job.batch_size}",
        # LeRobot 0.6 defaults push_to_hub=True and then refuses to start without
        # a policy.repo_id. so101 never uploads a model as a side effect.
        "--policy.push_to_hub=false",
    ]
    if local_dataset is not None:
        cmd.append(f"--dataset.root={local_dataset.path}")
        if not job.include_depth and "policy.input_features" not in job_args:
            inputs = policy_inputs_without_depth(local_dataset.path)
            if inputs is not None:
                cmd.append(f"--policy.input_features={json.dumps(inputs, separators=(',', ':'))}")
    if (episodes_flag := plan.lerobot_flag()) is not None:
        cmd.append(episodes_flag)
    settings = get_settings()
    cmd.append(f"--wandb.enable={'true' if settings.wandb_enabled else 'false'}")
    if settings.wandb_enabled:
        cmd.append(f"--wandb.project={settings.wandb_project}")
    return cmd, plan


def _real_available() -> bool:
    return lerobot_train_available() and not get_settings().mock


def run_training(
    job: TrainingJob,
    cancel_event: threading.Event | None = None,
    on_start: Callable[[TrainingJob], None] | None = None,
) -> Generator[tuple[int, float, str], None, None]:
    """Run a job, yielding ``(step, loss, raw_line)`` as it progresses.

    Uses the real trainer when available; otherwise a deterministic mock that
    emits a decaying loss curve. Either way the dataset must exist locally
    (or be an explicit Hub dataset for real LeRobot training) before anything
    is written. ``on_start`` is called once the ``lerobot-train`` subprocess
    has started and ``job.pid``/``pgid``/``pid_start`` are set (the job runner
    records them at once, so a restarted server can find the trainer).
    """
    preset = resolve_preset(job.policy)
    validate_extra_args(preset, job.extra_args)
    if job.steps < 1 or job.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if preset.trainer == "numpy_in_process":
        # The numpy baseline trains for real in every mode (mock included):
        # it needs neither lerobot-train nor torch.
        yield from _run_numpy(job, preset, cancel_event=cancel_event)
        return
    if _real_available():
        validate_training_dataset(job.repo_id, preset, remote_dataset=job.remote_dataset)
        # The episode selection is read here, before anything is written or
        # started: unreadable labels or an empty selection fail the job.
        cmd, plan = _train_command(job)
        # lerobot-train refuses an existing output_dir unless resuming, so
        # create only the parent; the trainer creates the run directory.
        output = Path(job.output_dir)
        if output.exists():
            raise FileExistsError(f"training output directory already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        yield from _run_real(job, cmd, plan, cancel_event=cancel_event, on_start=on_start)
    elif get_settings().mock:
        info = validate_training_dataset(job.repo_id, preset, remote_dataset=job.remote_dataset)
        # The same selection as a real run, so the mock refuses what it would.
        plan = plan_episodes(job.repo_id, job_episode_selection(job), info)
        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
        write_run_manifest(job, preset, trainer="mock", episodes=plan)
        log.info("explicit mock mode: running deterministic trainer for %s", job.id)
        yield from _run_mock(job, cancel_event=cancel_event)
    else:
        raise RuntimeError(
            "lerobot-train is unavailable while SO101_MOCK=0; install the training "
            "stack or explicitly enable mock mode. No synthetic checkpoint was created."
        )


def _run_numpy(
    job: TrainingJob,
    preset: PolicyPreset,
    cancel_event: threading.Event | None = None,
) -> Iterator[tuple[int, float, str]]:
    from .bc_baseline import CHECKPOINT_SUBDIR, bc_config_for, train_bc_baseline

    validate_training_dataset(job.repo_id, preset, remote_dataset=job.remote_dataset)
    # The baseline applies the selection (and its quality checks) as it loads
    # the data; its checkpoint manifest lists the episodes used and skipped.
    plan = EpisodePlan(bc_config_for(job).episodes, episodes=None, skipped=None)
    output = Path(job.output_dir)
    checkpoint = output / CHECKPOINT_SUBDIR
    if checkpoint.exists():
        raise FileExistsError(f"a checkpoint already exists at {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(job, preset, trainer="numpy_in_process", episodes=plan)
    log.info("training %s in-process for job %s", preset.name, job.id)
    yield from train_bc_baseline(job, cancel_event=cancel_event)


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    """Terminate a trainer and its children without blocking the API thread."""

    if proc.poll() is not None:
        return
    # TODO(windows): this stops the trainer's whole process group, which
    # start_new_session=True creates on POSIX. Windows needs its own branch
    # (CREATE_NEW_PROCESS_GROUP plus a tree kill such as a job object or
    # `taskkill /T`); until then terminate() there stops only the direct child
    # and leaves DataLoader workers running. Out of scope for the G15 (Linux) pass.
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows fallback
            proc.terminate()
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows fallback
            proc.kill()
    except ProcessLookupError:
        pass


def _inherited_dataset_read_fd(repo_id: str) -> int | None:
    """A shared dataset-lock fd for the trainer subprocess to inherit.

    The API/CLI process holds its normal :func:`dataset_read_hold` while it
    validates and supervises training.  This is a second open file description:
    after ``Popen`` the parent closes only its copy, while ``lerobot-train``
    retains the shared ``flock`` until it exits.  Thus SIGKILL of the parent
    cannot make writers believe the trainer's lazily-read dataset is free.

    Process recovery is POSIX-only already (see :mod:`so101.training.processes`),
    and ``pass_fds`` is POSIX-only too.  The parent hold remains the Windows
    behavior until process-tree recovery is implemented there.
    """

    if os.name != "posix" or get_dataset(repo_id) is None:
        return None
    import fcntl

    path = dataset_operation_lock(repo_id).path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _run_real(
    job: TrainingJob,
    cmd: list[str],
    plan: EpisodePlan,
    cancel_event: threading.Event | None = None,
    on_start: Callable[[TrainingJob], None] | None = None,
) -> Iterator[tuple[int, float, str]]:
    preset = resolve_preset(job.policy)
    log.info("launching: %s", " ".join(cmd))
    dataset_lock_fd = _inherited_dataset_read_fd(job.repo_id)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # LeRobot's log lines and tqdm bar are UTF-8 whatever the locale; a
            # stray byte must not end the job with a UnicodeDecodeError.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            # POSIX: its own session and process group, so a stop reaches the
            # DataLoader workers too (see _terminate_process_tree).
            start_new_session=True,
            pass_fds=(dataset_lock_fd,) if dataset_lock_fd is not None else (),
        )
    finally:
        if dataset_lock_fd is not None:
            # Do not LOCK_UN: parent and child share this open file description.
            # Closing only our duplicate leaves the child's lock intact.
            os.close(dataset_lock_fd)
    job.pid = proc.pid
    job.pgid = own_process_group(proc.pid)
    job.pid_start = process_start_marker(proc.pid)

    output = Path(job.output_dir)
    manifest_written = False
    step, loss = 0, float("nan")
    assert proc.stdout is not None
    try:
        if on_start is not None:
            on_start(job)
        # Reading a subprocess pipe may block indefinitely when a trainer hangs.
        # A watcher makes cancellation effective even when no more log lines arrive.
        if cancel_event is not None:

            def terminate_when_cancelled() -> None:
                while proc.poll() is None:
                    if cancel_event.wait(timeout=0.25):
                        _terminate_process_tree(proc)
                        return

            threading.Thread(target=terminate_when_cancelled, daemon=True).start()

        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                break
            line = line.rstrip()
            # lerobot-train creates the run directory itself (B1); record the
            # run manifest beside its checkpoints as soon as it exists.
            if not manifest_written and output.is_dir():
                write_run_manifest(job, preset, trainer="lerobot", episodes=plan)
                manifest_written = True
            step = parse_step(line, step)
            if not is_progress_line(line) and (m := _LOSS_RE.search(line)):
                loss = float(m.group(1))
            yield step, loss, line
        proc.wait()
    finally:
        # Ctrl-C, a closed generator or an exception in the consumer must not
        # orphan the trainer: it runs in its own session, so the terminal's
        # SIGINT never reaches it.
        if proc.poll() is None:
            _terminate_process_tree(proc)
        if not manifest_written and output.is_dir():
            write_run_manifest(job, preset, trainer="lerobot", episodes=plan)
    if cancel_event is not None and cancel_event.is_set():
        return
    if proc.returncode != 0:
        raise RuntimeError(f"lerobot-train exited {proc.returncode}")


def _run_mock(
    job: TrainingJob,
    fast: bool = True,
    cancel_event: threading.Event | None = None,
) -> Iterator[tuple[int, float, str]]:
    import math

    n = min(job.steps, 2000)
    every = max(1, n // 100)
    for step in range(0, n + 1, every):
        if cancel_event is not None and cancel_event.is_set():
            return
        # Exponential decay + gentle ripple, deterministic in step.
        loss = 0.05 + 1.5 * math.exp(-step / (n / 4 + 1)) + 0.01 * math.sin(step / 50)
        yield step, loss, f"[mock] step:{step} loss:{loss:.4f}"
        if not fast:
            time.sleep(0.01)
    if cancel_event is not None and cancel_event.is_set():
        return
    # Drop a placeholder checkpoint so the model registry sees it.
    ckpt = Path(job.output_dir) / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy_type = resolve_preset(job.policy).policy_type
    (ckpt / "config.json").write_text(
        json.dumps({"type": policy_type, "policy_type": policy_type, "mock": True}),
        encoding="utf-8",
    )
    yield n, loss, "[mock] training complete"


if __name__ == "__main__":  # tiny manual smoke test
    j = TrainingJob(
        repo_id="user/demo", policy="act", steps=500, batch_size=8, output_dir="outputs/demo"
    )
    for _step, _loss, line in run_training(j):
        print(line, file=sys.stderr)
