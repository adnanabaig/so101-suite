"""``so101 sim``: MuJoCo physics demos, evaluation and renders from the command line.

so101 sim info
so101 sim render --camera front --out front.png
so101 sim demos user/sim_pick_lift --task pick_lift --episodes 50 --seed 0
so101 sim eval --policy expert --task pick_lift --episodes 100 --seed 1000

Runs never wait for the wall clock, so they are faster than real time, and
they are deterministic: the same task, seed and policy give the same joint
states, actions and environment states. Demos go through the same
``SafetyGuard`` and ``EpisodeRecorder`` as teleop recording, and the task's
success predicate labels every episode ``success`` or ``failure``.

Needs the ``[sim]`` extra (``pip install 'so101[sim]'``). MuJoCo is imported
only inside the commands, so the rest of the CLI works without it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import JOINT_NAMES, __version__
from .atomic_io import atomic_write_text
from .config import get_settings, parse_image_size
from .data.inspection import (
    SIM_OUTCOME_SOURCE as SIM_OUTCOME_SOURCE,
)
from .data.inspection import (
    LabelNotStoredError,
    persist_sim_outcome,
)

sim_app = typer.Typer(
    help="MuJoCo physics simulation: demos, evaluation and renders (needs the sim extra).",
    no_args_is_help=True,
)
console = Console()
err = Console(stderr=True)

# Demos keep recording this long after the first success, so they end settled.
DEMO_TAIL_S = 0.5
# Provenance of every `sim demos` run, kept in the dataset directory.
DEMOS_MANIFEST = "sim_demos.json"


# -- shared helpers -----------------------------------------------------------------------
# Messages carry paths, repo ids and exception text, so they are printed without
# Rich markup ("so101[sim]" must not turn into a style tag).
def _say(message: str, style: str | None = None) -> None:
    console.print(message, style=style, markup=False, highlight=False)


def _fail(message: str, style: str | None = "red") -> None:
    err.print(message, style=style, markup=False, highlight=False)


def _require_mujoco() -> str:
    try:
        import mujoco
    except ImportError as exc:
        _fail(f"the simulator needs MuJoCo ({exc}): pip install 'so101[sim]'")
        raise typer.Exit(1) from None
    return str(mujoco.__version__)


def _task(name: str | None) -> Any:
    from .simulation.tasks import make_task

    try:
        return make_task(name or get_settings().sim_task)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--task") from None


def _cameras(value: str) -> list[str]:
    from .simulation.world import CAMERAS

    names = list(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    unknown = [name for name in names if name not in CAMERAS]
    if unknown:
        raise typer.BadParameter(
            f"unknown cameras {unknown}; choose from {list(CAMERAS)}", param_hint="--cameras"
        )
    return names


def _image_size(value: str | None) -> tuple[int, int]:
    try:
        return parse_image_size(value or get_settings().sim_image_size)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--image-size") from None


def _renderer() -> Any:
    """The shared render thread, or exit when this host cannot render."""

    from .simulation.render import shared_renderer

    renderer = shared_renderer()
    if not renderer.start():
        _fail(
            f"rendering is unavailable on this host: {renderer.error}. "
            "Drop --cameras for state-only runs."
        )
        raise typer.Exit(1)
    return renderer


def _expert_config(noise_deg: float, waypoint_noise_mm: float) -> Any:
    from .simulation.expert import ExpertConfig

    return ExpertConfig(noise_deg=noise_deg, waypoint_noise_m=waypoint_noise_mm / 1000.0)


def _provenance(mujoco_version: str) -> dict[str, Any]:
    from .simulation import mjcf
    from .simulation.tasks import CONTROL_HZ
    from .simulation.world import DEFAULT_TIMESTEP_S

    return {
        "so101_version": __version__,
        "mujoco_version": mujoco_version,
        "model": {
            "repo": mjcf.MODEL_REPO,
            "path": mjcf.MODEL_REPO_PATH,
            "commit": mjcf.MODEL_COMMIT,
            "license": mjcf.MODEL_LICENSE,
        },
        "timestep_s": DEFAULT_TIMESTEP_S,
        "control_hz": CONTROL_HZ,
        "servo_mode": "sts3215",
    }


def _rate(successes: int, n: int) -> str:
    from .simulation.policy import wilson_interval

    low, high = wilson_interval(successes, n)
    return f"{successes}/{n} = {successes / n:.1%} (Wilson 95% {low:.1%}-{high:.1%})"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` atomically and durably: temp file, fsync, rename, directory fsync.

    Strict JSON: a NaN or infinity raises ``ValueError`` before anything is written.
    """

    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


class _OnceFilter(logging.Filter):
    """Drop repeats of a log message (the recorder warns once per episode)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message in self.seen:
            return False
        self.seen.add(message)
        return True


@contextmanager
def _quiet_repeats(logger_name: str) -> Iterator[None]:
    logger = logging.getLogger(logger_name)
    once = _OnceFilter()
    logger.addFilter(once)
    try:
        yield
    finally:
        logger.removeFilter(once)


@contextmanager
def _quiet_logs(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


@contextmanager
def _recording_backend() -> Iterator[str]:
    """Record as the ``sim`` backend for the duration of a command.

    Mock sessions already record the fallback format and stay as they are.
    """

    # The recorder fails closed without LeRobot unless the session is mock or
    # ``settings.backend == "sim"``; simulated demos present as the sim backend
    # so they record either way: a LeRobot dataset when LeRobot is installed
    # (the default install), else the fallback format.
    settings = get_settings()
    if settings.mock or settings.backend == "sim":
        yield str(settings.backend)
        return
    previous = os.environ.get("SO101_BACKEND")
    os.environ["SO101_BACKEND"] = "sim"
    get_settings.cache_clear()
    try:
        yield "sim"
    finally:
        if previous is None:
            os.environ.pop("SO101_BACKEND", None)
        else:
            os.environ["SO101_BACKEND"] = previous
        get_settings.cache_clear()


def label_episode(
    repo_id: str,
    episode: int,
    success: bool,
    detail: dict[str, Any],
    *,
    operation_lock_held: bool = False,
) -> str:
    """Store the predicate's verdict as the episode outcome; returns how it was stored.

    Both dataset formats store it (fallback: the episode's ``episode.json``;
    LeRobot: the dataset's ``so101_outcomes.json``), with the source
    :data:`SIM_OUTCOME_SOURCE`, the detail and the label history; a human
    label is never overwritten. Raises :class:`LabelNotStoredError` when the
    label cannot be stored: an unlabeled failure would train as a kept demo.
    """

    _outcome, stored = persist_sim_outcome(
        repo_id,
        episode,
        success,
        detail,
        operation_lock_held=operation_lock_held,
    )
    return stored


def _joints(values: Any) -> dict[str, float]:
    return {f"{joint}.pos": float(value) for joint, value in zip(JOINT_NAMES, values, strict=True)}


def _check_dataset_layout(
    repo_id: str,
    env_names: list[str],
    cameras: list[str],
    frame_shape: tuple[int, int, int] | None = None,
) -> None:
    """Refuse to append demos whose frame layout differs from the dataset's.

    The layout is the environment-state names, the cameras and, when
    ``frame_shape`` is given, each camera's frame shape (H, W, C).
    """

    from .data.dataset import resolved_dataset_path

    try:
        meta_path = resolved_dataset_path(repo_id) / "meta.json"
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="REPO_ID") from None
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"cannot read {meta_path}: {exc}")
        raise typer.Exit(1) from None
    if not isinstance(meta, dict) or not meta.get("episodes"):
        return
    have_env = list(meta.get("environment_state_names") or [])
    have_cameras = list(meta.get("camera_keys") or [])
    if have_env != env_names or have_cameras != cameras:
        _fail(
            f"{repo_id} already holds episodes with environment state {have_env} and "
            f"cameras {have_cameras}; this run records {env_names} and {cameras}. "
            "Use a new repo id."
        )
        raise typer.Exit(1)
    if frame_shape is None:
        return
    for camera, have in _recorded_shapes(meta_path.parent, meta, cameras).items():
        if have != frame_shape:
            _fail(
                f"{repo_id} already holds {camera} frames of shape {list(have)} (H, W, C); "
                f"this run records {list(frame_shape)}. Use --image-size {have[1]}x{have[0]} "
                "or a new repo id."
            )
            raise typer.Exit(1)


def _recorded_shapes(
    root: Path, meta: dict[str, Any], cameras: list[str]
) -> dict[str, tuple[int, ...]]:
    """Each camera's frame shape in a fallback dataset, where one is known.

    ``meta.json`` holds ``camera_shapes`` once ``so101 sim demos`` has written
    to the dataset. Older datasets lack it, so the shape comes from the first
    image chunk stored for that camera.
    """

    import numpy as np

    stored = meta.get("camera_shapes")
    stored = stored if isinstance(stored, dict) else {}
    storage = meta.get("camera_storage")
    storage = storage if isinstance(storage, dict) else {}
    shapes: dict[str, tuple[int, ...]] = {}
    for camera in cameras:
        shape = stored.get(camera)
        if isinstance(shape, list) and shape and all(isinstance(v, int) for v in shape):
            shapes[camera] = tuple(shape)
            continue
        directory = str(storage.get(camera, camera))
        for chunk in sorted(root.glob(f"episode_*/images/{directory}/chunk_*.npy")):
            try:
                array = np.load(chunk, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError):
                continue  # the dataset QC reports unreadable chunks
            if array.ndim == 4:
                shapes[camera] = tuple(int(value) for value in array.shape[1:])
                break
    return shapes


# -- commands -------------------------------------------------------------------------
@sim_app.command("info")
def info(as_json: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """MuJoCo, the renderer, model provenance, timestep and the task catalogue."""

    version = _require_mujoco()
    from .simulation.render import jpeg_supported
    from .simulation.render import shared_renderer as renderer_factory
    from .simulation.tasks import CONTROL_HZ, TASKS
    from .simulation.world import CAMERAS, DEFAULT_TIMESTEP_S

    settings = get_settings()
    renderer = renderer_factory()
    with _quiet_logs(as_json):  # log lines go to stdout and would corrupt the JSON
        render_ok = renderer.start()
    data = {
        **_provenance(version),
        "substeps_per_control_step": round(1.0 / (CONTROL_HZ * DEFAULT_TIMESTEP_S)),
        "engine_setting": settings.sim_engine,
        "backend": settings.backend,
        "render": {"available": render_ok, "error": renderer.error, "jpeg": jpeg_supported()},
        "cameras": list(CAMERAS),
        "defaults": {
            "task": settings.sim_task,
            "seed": settings.sim_seed,
            "cameras": settings.sim_camera_names,
            "image_size": settings.sim_image_size,
        },
        "tasks": [
            {
                "name": name,
                "episode_s": cls.episode_s,
                "instruction": cls.instruction,
                "env_state_names": list(cls.env_state_names),
            }
            for name, cls in TASKS.items()
        ],
    }
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    table = Table(title="SO-101 physics simulator")
    table.add_column("setting")
    table.add_column("value")
    model = data["model"]
    table.add_row("MuJoCo", version)
    table.add_row("model", escape(f"{model['repo']} {model['path']} @ {model['commit'][:12]}"))
    table.add_row(
        "timestep",
        f"{DEFAULT_TIMESTEP_S * 1000:.3f} ms; {data['substeps_per_control_step']} substeps "
        f"per {CONTROL_HZ:g} Hz control step",
    )
    table.add_row("servo model", "sts3215 (position servo, torque-limited)")
    status = "available" if render_ok else f"unavailable: {renderer.error}"
    jpeg = "yes" if jpeg_supported() else "no (PNG only)"
    table.add_row("renderer", escape(f"{status}; JPEG {jpeg}"))
    table.add_row("cameras", ", ".join(CAMERAS))
    table.add_row("backend / engine", f"{settings.backend} / {settings.sim_engine}")
    console.print(table)
    tasks = Table(title="tasks")
    for column in ("task", "episode", "instruction", "observation.environment_state"):
        tasks.add_column(column)
    for entry in data["tasks"]:
        tasks.add_row(
            entry["name"],
            f"{entry['episode_s']:g} s",
            escape(entry["instruction"]),
            ", ".join(entry["env_state_names"]),
        )
    console.print(tasks)


@sim_app.command("render")
def render(
    out: Path = typer.Option(
        ..., "--out", "-o", help="Image file: .png, or .jpg/.jpeg (needs Pillow or OpenCV)."
    ),
    camera: str = typer.Option("front", help="front | wrist_cam"),
    task: str | None = typer.Option(None, help="reach | pick_lift | pick_place"),
    seed: int | None = typer.Option(None, min=0, help="Layout seed (default: SO101_SIM_SEED)."),
    width: int = typer.Option(640, help="Pixels."),
    height: int = typer.Option(480, help="Pixels."),
) -> None:
    """Render one camera at a task's seeded start state."""

    from .config import SIM_CAMERAS

    suffix = out.suffix.lower()
    fmt = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}.get(suffix)
    if fmt is None:
        raise typer.BadParameter("use a .png, .jpg or .jpeg file name", param_hint="--out")
    if camera not in SIM_CAMERAS:
        raise typer.BadParameter(f"choose from {list(SIM_CAMERAS)}", param_hint="--camera")
    _require_mujoco()
    from .simulation.render import check_size, encode_frame, jpeg_supported

    try:
        check_size(width, height)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--width/--height") from None
    if fmt == "jpeg" and not jpeg_supported():
        _fail("JPEG needs Pillow or OpenCV; write a .png instead")
        raise typer.Exit(1)
    task_obj = _task(task)
    world = task_obj.make_world()
    layout_seed = get_settings().sim_seed if seed is None else seed
    task_obj.reset(world, layout_seed)
    frame = _renderer().render(world.snapshot(), camera, width, height)
    data, media_type = encode_frame(frame, fmt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    _say(
        f"{out} · {task_obj.name} seed {layout_seed} · {camera} {width}x{height} "
        f"· {media_type} · {len(data)} bytes",
        "green",
    )


@sim_app.command("demos")
def demos(
    repo_id: str = typer.Argument(..., help="Dataset to create or extend, e.g. user/sim_pick."),
    task: str | None = typer.Option(None, help="reach | pick_lift | pick_place"),
    episodes: int = typer.Option(10, min=1, help="Episodes to record."),
    seed: int | None = typer.Option(
        None, min=0, help="Episode i uses seed + i (default: SO101_SIM_SEED)."
    ),
    cameras: str = typer.Option(
        "", help="Cameras to record, comma-separated: front,wrist_cam (default: none)."
    ),
    image_size: str | None = typer.Option(
        None, "--image-size", help="WxH of recorded frames (default: SO101_SIM_IMAGE_SIZE)."
    ),
    noise_deg: float = typer.Option(
        0.0, min=0.0, help="Expert joint-target noise, degrees (correlated), for diverse demos."
    ),
    waypoint_noise_mm: float = typer.Option(
        0.0, min=0.0, help="Expert grasp/place point noise, millimetres."
    ),
) -> None:
    """Record scripted-expert demonstrations into a dataset, labeled by the task predicate."""

    version = _require_mujoco()
    camera_names = _cameras(cameras)
    size = _image_size(image_size)
    from .data.recorder import EpisodeRecorder, RecordingConfig
    from .simulation.expert import ScriptedExpert
    from .simulation.policy import episode_seeds, wilson_interval
    from .simulation.render import frame_source
    from .simulation.rollout import ENV_STATE_KEY, IMAGE_KEY_PREFIX, STATE_KEY, EpisodeRunner
    from .teleop.controller import TeleopStep

    task_obj = _task(task)
    env_names = list(task_obj.env_state_names)
    width, height = size
    frame_shape = (height, width, 3)  # frames are H x W x RGB
    _check_dataset_layout(repo_id, env_names, camera_names, frame_shape if camera_names else None)
    frames = frame_source(_renderer()) if camera_names else None
    runner = EpisodeRunner(task_obj, cameras=camera_names, image_size=size, frames=frames)
    expert_cfg = _expert_config(noise_deg, waypoint_noise_mm)
    expert = ScriptedExpert(task_obj, expert_cfg)
    base_seed = get_settings().sim_seed if seed is None else seed
    seeds = episode_seeds(base_seed, episodes)
    cfg = RecordingConfig(
        repo_id=repo_id,
        task=task_obj.instruction,
        fps=round(runner.control_hz),
        camera_keys=camera_names,
        environment_state_names=env_names,
        camera_feature_names={camera: camera for camera in camera_names},
    )
    features = ["observation.state", "action", "desired_action", ENV_STATE_KEY]
    features += [IMAGE_KEY_PREFIX + camera for camera in camera_names]
    run: dict[str, Any] = {
        "created_at": time.time(),
        "task": task_obj.name,
        "instruction": task_obj.instruction,
        "policy": "expert",
        "expert": asdict(expert_cfg),
        "seed": base_seed,
        "seeds": seeds,
        "tail_s": DEMO_TAIL_S,
        "fps": cfg.fps,
        "features": features,
        "env_state_names": env_names,
        "cameras": camera_names,
        "image_size": list(size) if camera_names else None,
        "provenance": _provenance(version),
        "episodes": [],
    }
    started = time.perf_counter()
    sim_time = 0.0
    recorder: EpisodeRecorder | None = None

    def record_step(step: Any) -> None:
        assert recorder is not None
        observation = step.observation
        recorder.add(
            TeleopStep(
                t=step.time_s,
                observation=_joints(observation[STATE_KEY]),
                action=_joints(step.command),
                desired_action=_joints(step.action),
            ),
            {camera: observation[IMAGE_KEY_PREFIX + camera] for camera in camera_names},
            environment_state=observation[ENV_STATE_KEY],
        )

    def discard(reason: str) -> None:
        nonlocal recorder
        if recorder is None:
            return
        take, recorder = recorder, None
        try:
            kept = take.abort(reason)
        except Exception as exc:  # never mask what stopped the run
            _fail(f"could not clean up the unsaved episode: {exc}")
            return
        if kept is not None:
            _fail(f"the unsaved episode's frames are kept apart in {kept}", "yellow")

    def summary(note: str) -> str:
        done = run["episodes"]
        wins = sum(1 for entry in done if entry["success"])
        if not done:
            return f"{note}: no episode saved"
        return f"{note}: {len(done)} episode(s) -> {repo_id}; expert {_rate(wins, len(done))}"

    _say(
        f"recording {episodes} {task_obj.name} demos -> {repo_id} "
        f"(seeds {seeds[0]}..{seeds[-1]}, cameras {', '.join(camera_names) or 'none'})",
        "cyan",
    )
    try:
        with _recording_backend() as backend, _quiet_repeats("so101.data.recorder"):
            run["recorded_as_backend"] = backend
            for episode_seed in seeds:
                recorder = EpisodeRecorder(cfg)
                result = runner.run(
                    expert,
                    episode_seed,
                    on_step=record_step,
                    tail_s=DEMO_TAIL_S,
                    policy_reset=partial(expert.reset, episode_seed),
                )
                frame_count = recorder.num_frames
                saved = recorder
                index = saved.save_episode(retain_operation_lock=True)
                recorder = None
                unlabeled: LabelNotStoredError | None = None
                try:
                    sim_time += result.sim_time_s
                    detail = {
                        "task": result.task,
                        "seed": episode_seed,
                        "success": result.success,
                        "first_success_s": result.first_success_s,
                        "sim_time_s": result.sim_time_s,
                        "layout": result.layout,
                        "summary": result.summary,
                    }
                    try:
                        stored = label_episode(
                            repo_id,
                            index,
                            result.success,
                            detail,
                            operation_lock_held=True,
                        )
                    except LabelNotStoredError as exc:
                        unlabeled, stored = exc, f"NOT LABELED ({exc.cause})"
                finally:
                    saved.release_operation_lock()
                outcome = "success" if result.success else "failure"
                run["episodes"].append(
                    {
                        "episode": index,
                        "seed": episode_seed,
                        "success": result.success,
                        "outcome": outcome,
                        "label": stored,
                        "frames": frame_count,
                        "first_success_s": result.first_success_s,
                        "layout": result.layout,
                        "summary": result.summary,
                    }
                )
                _save_manifest(repo_id, run)
                when = f" at {result.first_success_s:.2f} s" if result.success else ""
                _say(
                    f"  episode {index} · seed {episode_seed} · {outcome}{when} · "
                    f"{frame_count} frames · {stored}"
                )
                if unlabeled is not None:
                    raise unlabeled  # stop: more demos would be unlabeled too
    except KeyboardInterrupt:
        discard("interrupted")
        _fail(summary("interrupted"), "yellow")
        raise typer.Exit(130) from None
    except LabelNotStoredError as exc:
        _fail(f"demo recording stopped: {exc}")
        _fail(summary("stopped"), None)
        raise typer.Exit(1) from None
    except (OSError, ValueError, RuntimeError) as exc:
        discard(f"error: {exc}")
        _fail(f"demo recording failed: {exc}")
        _fail(summary("stopped"), None)
        raise typer.Exit(1) from None
    wall = time.perf_counter() - started
    wins = sum(1 for entry in run["episodes"] if entry["success"])
    run["successes"] = wins
    run["wilson95"] = [round(value, 4) for value in wilson_interval(wins, episodes)]
    run["sim_time_s"] = round(sim_time, 3)
    run["wall_time_s"] = round(wall, 3)
    run["real_time_factor"] = round(sim_time / wall, 1) if wall > 0 else None
    manifest = _save_manifest(repo_id, run)
    _say(summary("recorded"), "green")
    _say(
        f"{sim_time:.1f} s simulated in {wall:.1f} s ({sim_time / max(wall, 1e-9):.0f}x real "
        f"time) · provenance -> {manifest}"
    )


def _read_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """``sim_demos.json``'s content, or why it cannot be used.

    An ``OSError`` (say, no permission) propagates: the file may be fine.
    """

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("runs"), list):
        return None, "not an object with a 'runs' list"
    if not all(isinstance(entry, dict) for entry in loaded["runs"]):
        return None, "a 'runs' entry is not an object"
    return loaded, None


def _keep_corrupt(path: Path) -> Path:
    """Move an unreadable file aside as ``<name>.corrupt-<time>`` and return the new path."""

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.corrupt-{stamp}")
    counter = 1
    while backup.exists():
        counter += 1
        backup = path.with_name(f"{path.name}.corrupt-{stamp}-{counter}")
    os.replace(path, backup)
    return backup


def _save_manifest(repo_id: str, run: dict[str, Any]) -> Path:
    """Add or update this run in the dataset's ``sim_demos.json``.

    An unreadable manifest is never wiped: it is kept as
    ``sim_demos.json.corrupt-<time>`` (and said so) before a new one starts.
    """

    from .data.dataset import dataset_operation_target, require_dataset

    root, operation_lock = dataset_operation_target(repo_id)
    with operation_lock:
        dataset = require_dataset(repo_id)
        locked_root = Path(dataset.path)
        if locked_root.resolve() != root.resolve():
            raise RuntimeError(
                f"dataset {repo_id!r} changed folders while its manifest lock was acquired"
            )
        path = locked_root / DEMOS_MANIFEST
        manifest: dict[str, Any] = {"schema_version": 1, "runs": []}
        if path.is_file():
            loaded, problem = _read_manifest(path)
            if loaded is not None:
                manifest = loaded
            else:
                backup = _keep_corrupt(path)
                _fail(
                    f"{path} could not be read ({problem}); kept it as {backup.name} and "
                    f"started a new {DEMOS_MANIFEST}. The runs it held are in that file.",
                    "yellow",
                )
        runs = [entry for entry in manifest["runs"] if entry.get("created_at") != run["created_at"]]
        manifest["runs"] = [*runs, run]
        _write_json(path, manifest)
        return path


@sim_app.command("eval")
def eval_policy(
    policy: str = typer.Option("expert", help="expert | hold | path to a trained model."),
    task: str | None = typer.Option(None, help="reach | pick_lift | pick_place"),
    episodes: int = typer.Option(20, min=1, help="Episodes (seeded seed + i)."),
    seed: int | None = typer.Option(
        None, min=0, help="First episode seed (default: SO101_SIM_SEED)."
    ),
    cameras: str = typer.Option(
        "", help="Cameras the policy observes, comma-separated (default: none)."
    ),
    image_size: str | None = typer.Option(
        None, "--image-size", help="WxH of observed frames (default: SO101_SIM_IMAGE_SIZE)."
    ),
    noise_deg: float = typer.Option(0.0, min=0.0, help="Expert only: joint-target noise."),
    waypoint_noise_mm: float = typer.Option(
        0.0, min=0.0, help="Expert only: grasp/place point noise."
    ),
    out: Path | None = typer.Option(
        None, help="Report file (default: <outputs>/sim_eval/<task>-<policy>-<time>.json)."
    ),
) -> None:
    """Run a policy closed-loop and report its success rate with a Wilson 95 % interval."""

    version = _require_mujoco()
    camera_names = _cameras(cameras)
    size = _image_size(image_size)
    from .simulation.policy import UnsupportedPolicyError, evaluate
    from .simulation.render import frame_source

    task_obj = _task(task)
    frames = frame_source(_renderer()) if camera_names else None
    first_seed = get_settings().sim_seed if seed is None else seed

    def progress(result: Any) -> None:
        when = f" at {result.first_success_s:.2f} s" if result.success else ""
        outcome = "success" if result.success else "failure"
        _say(f"  seed {result.seed} · {outcome}{when} · {result.steps} steps")

    _say(f"evaluating {policy} on {task_obj.name} ({episodes} episodes)", "cyan")
    try:
        report = evaluate(
            task_obj,
            policy,
            episodes=episodes,
            seed=first_seed,
            expert=_expert_config(noise_deg, waypoint_noise_mm),
            on_episode=progress,
            cameras=camera_names,
            image_size=size,
            frames=frames,
        )
    except UnsupportedPolicyError as exc:
        raise typer.BadParameter(str(exc), param_hint="--policy") from None
    except KeyboardInterrupt:
        _fail("evaluation interrupted; no report was written", "yellow")
        raise typer.Exit(130) from None
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        _fail(f"evaluation failed: {exc}")
        raise typer.Exit(1) from None
    data = report.as_dict() | {
        "created_at": time.time(),
        "first_seed": first_seed,
        "provenance": _provenance(version),
    }
    path = out or _default_report_path(task_obj.name, policy)
    try:
        _write_json(path, data)
    except (OSError, ValueError) as exc:
        _fail(f"could not write the report to {path}: {exc}")
        _fail(json.dumps(data, default=str), None)  # the result is not lost
        raise typer.Exit(1) from None
    _say(f"{task_obj.name} · {policy}: {_rate(report.successes, report.episodes)}", "green")
    _say(
        f"{report.sim_time_s:.1f} s simulated in {report.wall_time_s:.1f} s "
        f"({report.real_time_factor:.0f}x real time) · report -> {path}"
    )


def _default_report_path(task: str, policy: str) -> Path:
    label = policy if policy in ("expert", "hold") else Path(policy).expanduser().name
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "policy"
    directory = get_settings().outputs_dir / "sim_eval"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"{task}-{label}-{stamp}.json"
    counter = 1
    while path.exists():
        counter += 1
        path = directory / f"{task}-{label}-{stamp}-{counter}.json"
    return path


__all__ = [
    "DEMOS_MANIFEST",
    "SIM_OUTCOME_SOURCE",
    "LabelNotStoredError",
    "label_episode",
    "sim_app",
]
