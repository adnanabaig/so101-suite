"""``so101`` command-line interface — the full lifecycle without the UI.

so101 info
so101 calibrate follower
so101 teleop --duration 30
so101 record user/pick_place --episodes 5 --episode-seconds 20
so101 datasets ls
so101 replay user/pick_place 0
so101 train user/pick_place --policy act --steps 100000
so101 models ls
so101 run outputs/act/checkpoints/last/pretrained_model --duration 30
so101 eval  outputs/act/checkpoints/last/pretrained_model --episodes 10
so101 serve --port 8000
so101 estop | halt | collapse [--url URL]
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from . import JOINT_NAMES, __version__
from .cli_bringup import app as bringup_app
from .config import get_settings
from .logging import setup_logging

if TYPE_CHECKING:
    import httpx

    from .bringup.standalone import StandaloneAccess, StandaloneRefused
    from .config import Settings

app = typer.Typer(add_completion=False, help="SO-101 robot platform CLI")
console = Console()

#: Exit code when a command may not open the arm (a latch, a running server,
#: motor-device lock held, an unreadable latch store); the bring-up kit uses 3 as well.
EXIT_REFUSED = 3


def _refuse(exc: StandaloneRefused) -> NoReturn:
    """Print a standalone-gate refusal to stderr and exit EXIT_REFUSED."""
    from rich.markup import escape

    # Unwrapped, so a path or a JSON body copies and pastes as is.
    err = Console(stderr=True, soft_wrap=True)
    err.print(f"[red]refused:[/] {escape(exc.detail)}")
    err.print(escape(exc.remediation))
    raise typer.Exit(EXIT_REFUSED)


def _standalone_gate(purpose: str, *, motion: bool = False) -> StandaloneAccess:
    """Pass the standalone gate (so101.bringup.standalone) or exit EXIT_REFUSED.

    Every command that opens an arm without the server calls this before it
    opens anything: on any non-mock backend it refuses while the persisted
    E-stop, fault or unconfirmed torque-off is latched (or the latch store is
    unreadable), and on the real backend it also refuses while a server runs
    and holds the canonical motor-device locks until the caller releases the returned access.
    """
    from .bringup.standalone import StandaloneRefused, standalone_access

    try:
        return standalone_access(purpose, motion=motion)
    except StandaloneRefused as exc:
        _refuse(exc)


def _standalone_halt_notice() -> None:
    """Make a real standalone command's energized end state explicit."""

    from .hardware.backend import current_backend

    if current_backend(get_settings()) == "real":
        console.print(
            "[yellow]Halt:[/] the follower is holding with torque on. Keep supervising it; "
            "the next Suite/CLI process will refuse motion until it confirms torque off."
        )


@app.callback()
def _root() -> None:
    setup_logging(get_settings().log_level)


def _sim_settings(s: Settings) -> dict[str, Any]:
    """The SO101_SIM_* settings the sim backend runs with (doctor, info)."""
    return {
        "task": s.sim_task,
        "seed": s.sim_seed,
        "cameras": s.sim_camera_names,
        "image_size": s.sim_image_size,
        "realtime": s.sim_realtime,
    }


@app.command()
def info() -> None:
    """Show the backend and its devices, and which settings files and data folders apply.

    Under SO101_BACKEND=sim it lists the SO101_SIM_* settings instead of the
    serial ports and cameras, which the sim backend never opens.
    """
    from rich.markup import escape

    from .config import settings_sources
    from .hardware.backend import current_backend

    s = get_settings()
    sources = settings_sources(s)
    backend = current_backend(s)
    t = Table(title=f"SO-101 platform v{__version__}")
    t.add_column("setting")
    t.add_column("value")
    t.add_row("backend", backend)
    if backend == "sim":
        sim = _sim_settings(s)
        t.add_row("sim task", sim["task"])
        t.add_row("sim seed", str(sim["seed"]))
        t.add_row("sim cameras", ", ".join(sim["cameras"]) or "(none)")
        t.add_row("sim image size", sim["image_size"])
        t.add_row("sim realtime", str(sim["realtime"]).lower())
    else:
        t.add_row("follower port", s.follower_port)
        t.add_row("follower id", s.follower_id)
        t.add_row("leader port", s.leader_port)
        t.add_row("leader id", s.leader_id)
        from .hardware.cameras import configured_streams

        # Stream keys: credentials stripped, "<entry>#depth" for a RealSense depth stream.
        t.add_row("camera streams", ", ".join(configured_streams(s)) or "(none)")
    # Lowest priority first; the environment overrides every file.
    for entry in sources["env_files"]:
        state = "read" if entry["exists"] else "not found"
        t.add_row(f"settings ({entry['role']})", escape(f"{entry['path']} ({state})"))
    if not sources["env_files"]:
        t.add_row("settings", "no .env files (env files are off)")
    t.add_row("data folder", escape(sources["data_home"]))
    t.add_row("data dir", escape(str(s.data_dir)))
    t.add_row("datasets dir", escape(str(s.datasets_dir)))
    t.add_row("outputs dir", escape(str(s.outputs_dir)))
    console.print(t)
    if sources["warning"]:
        console.print(f"[yellow]warning:[/] {escape(sources['warning'])}")


@app.command("find-cameras")
def find_cameras(
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """List physical cameras for SO101_CAMERAS (real backend only)."""
    import json

    from .hardware.camera_discovery import (
        camera_discovery_refusal,
        discover_cameras,
        render_report,
    )

    refusal = camera_discovery_refusal(get_settings())
    if refusal is not None:
        typer.echo(f"refused: {refusal}", err=True)
        raise typer.Exit(EXIT_REFUSED)

    report = discover_cameras()
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(render_report(report))


@app.command()
def calibrate(
    arm: str = typer.Argument(..., help="follower|leader"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm non-interactively that the external catch/cutoff setup is ready.",
    ),
) -> None:
    """Calibrate an arm's range of motion."""
    from .hardware.calibration import calibrate as run_cal
    from .hardware.motors import make_bus

    if arm not in {"follower", "leader"}:
        raise typer.BadParameter("arm must be 'follower' or 'leader'")
    s = get_settings()
    from .hardware.backend import current_backend
    from .hardware.calibration import SIM_CALIBRATION_REFUSED

    backend = current_backend(s)
    if backend == "sim":
        typer.echo(SIM_CALIBRATION_REFUSED, err=True)
        raise typer.Exit(1)
    if backend == "real" and not yes:
        typer.echo(
            "Calibration is a hands-on joint sweep and will turn torque off. Before any "
            f"{arm} bus is opened, place the arm low over a padded mechanical catch that "
            "does not obstruct its sweep, keep your body outside the sweep envelope, keep "
            "fingers away from pinch points, and assign an operator to the physical cutoff. "
            "Do not touch the arm until the live status changes from preparation to "
            "'midpoint captured; now sweep'; then move it only by its links.",
            err=True,
        )
        typer.confirm(
            "Mechanical catch, clear sweep and physical cutoff confirmed; continue?",
            abort=True,
            err=True,
        )
    # The server refuses calibration while a latch is set; so does this.
    with _standalone_gate(f"so101 calibrate {arm}") as access:
        port = s.follower_port if arm == "follower" else s.leader_port
        bus = make_bus(port, name=arm)
        try:
            bus.connect()
            with console.status(f"preparing {arm} calibration… do not touch the arm") as status:
                res = run_cal(
                    bus,
                    arm,
                    progress=lambda fraction, message: status.update(
                        f"{arm} calibration {fraction:.0%} · {message}"
                    ),
                )
            path = res.save()
        finally:
            access.disconnect_device(bus, device_name=arm)
    console.print(f"[green]saved[/] {path}")


@app.command()
def teleop(
    duration: float = typer.Option(30.0),
    fps: int = typer.Option(30, min=1, max=240),
) -> None:
    """Teleoperate in mock/sim; real teleop runs through the Suite server."""
    from .hardware.follower import SO101Follower
    from .hardware.leader import SO101Leader
    from .teleop.controller import TeleopController

    if not math.isfinite(duration) or duration < 0 or isinstance(fps, bool) or not 1 <= fps <= 240:
        typer.echo(
            "cannot teleoperate: --duration must be finite and non-negative and --fps must "
            "be an integer from 1 to 240",
            err=True,
        )
        raise typer.Exit(1)
    with _standalone_gate("so101 teleop", motion=True) as access:
        with TeleopController(
            SO101Leader(), SO101Follower(), fps=fps, standalone_access=access
        ) as tc:
            with console.status(f"teleop for {duration}s (Ctrl-C to stop)"):
                try:
                    for _ in tc.run(duration):
                        pass
                except KeyboardInterrupt:
                    pass
    console.print("[green]teleop done[/]")
    _standalone_halt_notice()


@app.command()
def record(
    repo_id: str,
    episodes: int = typer.Option(5),
    episode_seconds: float = typer.Option(20.0),
    fps: int = typer.Option(30, min=1, max=240),
    task: str = typer.Option("Perform the task."),
    reset_seconds: float = typer.Option(
        5.0,
        help="Before each episode after the first: unrecorded teleop time for the arm to "
        "follow the leader back to its start pose.",
    ),
    prompt: bool = typer.Option(
        True,
        "--prompt/--no-prompt",
        help="Wait for you to reset the scene and confirm each episode after the first.",
    ),
) -> None:
    """Teleoperate + record in mock/sim; use the Suite server on real hardware.

    Nothing is recorded between episodes: you reset the scene and confirm the
    next take (--no-prompt skips the question), then the arm follows the leader
    for --reset-seconds before recording starts. Ctrl-C or an error discards
    only the take in progress; saved episodes are kept.
    """
    from .data.recorder import EpisodeRecorder, RecordingConfig
    from .hardware.cameras import CameraManager
    from .hardware.follower import SO101Follower
    from .hardware.leader import SO101Leader
    from .teleop.controller import TeleopController

    err = Console(stderr=True)
    if (
        episodes < 1
        or not math.isfinite(episode_seconds)
        or episode_seconds <= 0
        or isinstance(fps, bool)
        or not 1 <= fps <= 240
        or not math.isfinite(reset_seconds)
        or reset_seconds < 0
    ):
        err.print(
            "[red]cannot record:[/] --episodes, --episode-seconds and --fps must be positive "
            "and --reset-seconds must not be negative"
        )
        raise typer.Exit(1)
    cams = CameraManager()
    cfg = RecordingConfig(
        repo_id=repo_id,
        task=task,
        fps=fps,
        camera_keys=cams.streams,
        # Simulated cameras record under their own names, like `so101 sim demos`.
        camera_feature_names=cams.feature_names if cams.backend == "sim" else {},
    )
    saved = 0
    take: EpisodeRecorder | None = None
    stopped_early = False
    normal_halt = False
    access = _standalone_gate("so101 record", motion=True)

    def discard_take(reason: str) -> None:
        nonlocal take
        if take is None:
            return
        recorder, take = take, None
        try:
            kept = recorder.abort(reason)
        except Exception as exc:  # never mask what stopped the recording
            err.print(f"[red]could not clean up the unsaved take:[/] {exc}")
            return
        where = f"; its frames are kept apart in {kept}" if kept else ""
        err.print(f"[yellow]discarded the unsaved take{where}[/]")

    def stopped(note: str) -> None:
        err.print(f"[yellow]{note}:[/] {saved} of {episodes} episodes saved -> {repo_id}")

    try:
        cams.open()
        with TeleopController(
            SO101Leader(), SO101Follower(), fps=fps, standalone_access=access
        ) as tc:
            for ep in range(episodes):
                if ep:
                    if prompt and not typer.confirm(
                        f"Reset the scene, then record episode {ep + 1}/{episodes}?",
                        default=True,
                    ):
                        stopped("stopped")
                        stopped_early = True
                        break
                    if reset_seconds:
                        console.print(
                            f"[cyan]get ready: the arm follows the leader for "
                            f"{reset_seconds:g} s (not recorded)[/]"
                        )
                        for _ in tc.run(reset_seconds):
                            pass
                take = EpisodeRecorder(cfg)
                console.print(f"[cyan]recording episode {ep + 1}/{episodes}[/]")
                for step in tc.run(episode_seconds):
                    take.add(step, cams.read())
                frame_count = take.num_frames
                try:
                    idx = take.save_episode()
                except BaseException:
                    discard_take("save failed")
                    raise
                take = None
                saved += 1
                console.print(f"  saved episode {idx} ({frame_count} frames)")
        normal_halt = True
    except KeyboardInterrupt:
        discard_take("interrupted")
        stopped("interrupted")
        raise typer.Exit(130) from None
    except typer.Abort:  # Ctrl-C or end of input at the prompt
        stopped("stopped")
        raise typer.Exit(1) from None
    except (OSError, ValueError, RuntimeError) as exc:
        discard_take(f"error: {exc}")
        err.print(f"[red]recording failed:[/] {exc}")
        stopped("stopped")
        raise typer.Exit(1) from None
    finally:
        try:
            cams.close()
        finally:
            access.release()
    if stopped_early:
        if normal_halt:
            _standalone_halt_notice()
        return
    console.print(f"[green]recorded {episodes} episodes -> {repo_id}[/]")
    if normal_halt:
        _standalone_halt_notice()


datasets_app = typer.Typer(help="dataset management")
app.add_typer(datasets_app, name="datasets")


@datasets_app.command("ls")
def datasets_ls() -> None:
    from .data.dataset import list_datasets

    t = Table(title="datasets")
    for c in ("repo_id", "format", "episodes", "frames", "fps"):
        t.add_column(c)
    for d in list_datasets():
        t.add_row(d.repo_id, d.format, str(d.num_episodes), str(d.num_frames), str(d.fps))
    console.print(t)


@datasets_app.command("rm")
def datasets_rm(repo_id: str) -> None:
    from .data.dataset import DatasetBusyError, delete_dataset

    try:
        ok = delete_dataset(repo_id)
    except (DatasetBusyError, ValueError) as exc:
        Console(stderr=True).print(f"[red]cannot remove {repo_id}:[/] {exc}")
        raise typer.Exit(1) from None
    if not ok:
        Console(stderr=True).print(f"[red]dataset {repo_id!r} not found[/]")
        raise typer.Exit(1)
    console.print("[green]moved to recoverable trash[/]")


@app.command()
def replay(
    repo_id: str,
    episode: int = 0,
    fps: int | None = typer.Option(
        None, min=1, max=240, help="Playback rate (default: the dataset's recording fps)."
    ),
) -> None:
    """Replay in mock/sim; real replay runs through the Suite server."""
    from .bringup.standalone import StandaloneRefused
    from .data.replay import replay_episode

    # Outside the try below: typer.Exit is a RuntimeError.
    with _standalone_gate("so101 replay", motion=True):
        try:
            n = replay_episode(repo_id, episode, fps=fps)
        except StandaloneRefused as exc:
            _refuse(exc)
        except (OSError, ValueError, RuntimeError) as exc:
            Console(stderr=True).print(f"[red]replay failed:[/] {exc}")
            raise typer.Exit(1) from None
    console.print(f"[green]replayed {n} frames[/]")
    _standalone_halt_notice()


@app.command()
def train(
    repo_id: str,
    policy: str = typer.Option(
        "act", help="Preset: act, diffusion, smolvla, pi0, or bc_baseline (numpy, CPU)."
    ),
    steps: int | None = typer.Option(None, min=1, help="Training steps (default: the preset's)."),
    batch_size: int | None = typer.Option(None, min=1, help="Batch size (default: the preset's)."),
    output_dir: str = typer.Option(""),
    hub_dataset: bool = typer.Option(
        False,
        help="Real LeRobot training only: train on a Hugging Face Hub dataset. It has no "
        "outcome labels here, so it needs --episodes all.",
    ),
    seed: int | None = typer.Option(None, min=0, help="Random seed for the trainer."),
    episodes: str | None = typer.Option(
        None,
        help="Every preset: which episodes to learn from, by outcome label: kept (default: "
        "labelled success or unlabelled), success, or all (failures too).",
    ),
    state_only: bool = typer.Option(
        False,
        "--state-only",
        help="bc_baseline only: ignore simulator object state, so the policy can drive the arm.",
    ),
    include_depth: bool = typer.Option(
        False,
        "--include-depth",
        help="LeRobot presets: also feed RealSense depth maps to the policy (left out by "
        "default; the presets' image backbones take 3-channel color).",
    ),
    arg: list[str] = typer.Option(
        [],
        "--arg",
        help="Trainer override KEY=VALUE (repeatable), e.g. --arg policy.n_obs_steps=2.",
    ),
) -> None:
    """Train a policy on a dataset."""
    import math
    from contextlib import closing, nullcontext

    from .data.dataset import DatasetBusyError, DatasetNotFoundError, dataset_read_hold
    from .training.jobs import TrainingJob
    from .training.preflight import (
        UnknownPolicyError,
        plan_episodes,
        resolve_episode_selection,
        resolve_preset,
        validate_extra_args,
        validate_training_dataset,
    )
    from .training.trainer import run_training

    err = Console(stderr=True)
    try:
        preset = resolve_preset(policy)
        extra_args = _parse_trainer_args(arg)
        options: dict[str, str] = {}
        if seed is not None:
            options["seed"] = str(seed)
        if include_depth and preset.trainer != "lerobot":
            raise ValueError(f"--include-depth applies to LeRobot presets, not {preset.name}")
        if preset.trainer != "numpy_in_process" and state_only:
            raise ValueError(f"--state-only applies to the bc_baseline preset, not {preset.name}")
        if state_only:
            options["state_only"] = "true"
        for key, value in options.items():
            if extra_args.get(key, value) != value:
                raise ValueError(f"--{key.replace('_', '-')} conflicts with --arg {key}=...")
        extra_args.update(options)
        validate_extra_args(preset, extra_args)
        # --episodes, or a bc_baseline --arg episodes=... (they must agree).
        selection = resolve_episode_selection(episodes, extra_args)
        info = validate_training_dataset(repo_id, preset, remote_dataset=hub_dataset)
        plan_episodes(repo_id, selection, info)  # unreadable labels, nothing selected, Hub
    except (UnknownPolicyError, DatasetBusyError, DatasetNotFoundError, ValueError) as exc:
        err.print(f"[red]cannot train:[/] {exc}")
        raise typer.Exit(1) from None
    job = TrainingJob(
        repo_id=repo_id,
        policy=preset.name,
        steps=steps if steps is not None else preset.default_steps,
        batch_size=batch_size if batch_size is not None else preset.default_batch_size,
        output_dir="",
        extra_args=extra_args,
        remote_dataset=hub_dataset,
        include_depth=include_depth,
        episodes=selection,
    )
    # Nanoseconds plus the job id: two runs started in the same second must
    # not share a directory (lerobot-train refuses an existing one).
    job.output_dir = output_dir or str(
        get_settings().outputs_dir / f"{preset.name}_{time.time_ns()}_{job.id}"
    )
    console.print(f"[cyan]training {preset.name} on {repo_id} -> {job.output_dir}[/]")
    last_step, last_loss = 0, math.nan
    on_progress_line = False
    try:
        # LeRobot reads parquet/video lazily for the entire subprocess run.
        # The API JobRunner takes the same shared read hold; keep this CLI hold
        # outside the generator's close so interruption cannot release it while
        # trainer teardown is still using the dataset. The numpy baseline owns
        # its shorter load-only hold, and Hub datasets have no local root.
        hold = (
            dataset_read_hold(repo_id)
            if preset.trainer == "lerobot" and info is not None
            else nullcontext()
        )
        with hold:
            stream = run_training(job)
            # closing(): Ctrl-C stops the trainer subprocess (it runs in its own
            # session, so the terminal's SIGINT never reaches it).
            with closing(stream):
                for step, loss, line in stream:
                    last_step = step
                    if math.isfinite(loss):
                        last_loss = loss
                    if line.startswith("[bc]"):
                        # The numpy baseline's summary lines (data, model,
                        # evaluations, saved checkpoint) stay on screen.
                        if on_progress_line:
                            console.print()
                            on_progress_line = False
                        console.print(line, markup=False, highlight=False)
                    elif math.isfinite(loss):
                        console.print(f"  step {step:>7} loss {loss:.4f}", end="\r")
                        on_progress_line = True
    except KeyboardInterrupt:
        err.print("\n[yellow]training interrupted; the trainer was stopped[/]")
        raise typer.Exit(130) from None
    except (OSError, ValueError, RuntimeError) as exc:
        err.print(f"\n[red]training failed:[/] {exc}")
        raise typer.Exit(1) from None
    loss_text = f" loss {last_loss:.4f}" if math.isfinite(last_loss) else ""
    console.print(f"\n[green]done[/] final step {last_step}{loss_text}")


def _parse_trainer_args(items: list[str]) -> dict[str, str]:
    """Parse repeated ``--arg KEY=VALUE`` options (validation happens per preset)."""

    parsed: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--arg expects KEY=VALUE, got {item!r}")
        parsed[key.strip()] = value
    return parsed


models_app = typer.Typer(help="model registry")
app.add_typer(models_app, name="models")


@models_app.command("ls")
def models_ls() -> None:
    from .models import list_models

    t = Table(title="models")
    for c in ("name", "policy", "mock", "path", "label"):
        t.add_column(c)
    for m in list_models():
        t.add_row(m.name, m.policy_type, str(m.is_mock), m.path, m.label)
    console.print(t)


@app.command()
def run(
    model_path: str,
    duration: float = typer.Option(30.0, min=0.0),
    fps: int | None = typer.Option(
        None, min=1, max=240, help="Control rate (default: the training data's fps, else 30)."
    ),
    task: str | None = typer.Option(
        None,
        help="Language instruction for VLA policies (default: the checkpoint's training task).",
    ),
) -> None:
    """Run a policy in mock/sim; real rollout runs through the Suite server."""
    from .bringup.standalone import StandaloneRefused
    from .policy.checkpoint import preflight_policy, resolve_control_fps
    from .policy.runner import PolicyRunner
    from .policy.server import load_policy

    if not math.isfinite(duration) or duration < 0:
        typer.echo("cannot run: --duration must be finite and non-negative", err=True)
        raise typer.Exit(1)
    # Before the policy loads: a latch or a running server refuses at once.
    with _standalone_gate("so101 run", motion=True):
        try:
            info = preflight_policy(model_path, task=task)
            rate = resolve_control_fps(fps, info)
            policy = load_policy(info.path, task=info.task)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            Console(stderr=True).print(f"[red]cannot run {model_path}:[/] {exc}")
            raise typer.Exit(1) from None
        detail = f" · task {info.task!r}" if info.task else ""
        console.print(f"[cyan]{info.label}[/] · {info.path} · {rate} fps{detail}")
        runner = PolicyRunner(policy, fps=rate)
        try:
            with console.status(f"rollout for {duration}s"):
                res = runner.rollout(duration)
        except StandaloneRefused as exc:
            _refuse(exc)
    console.print(f"[green]rollout done[/]: {res.steps} steps in {res.duration:.1f}s")
    _standalone_halt_notice()


@app.command()
def eval(
    model_path: str,
    episodes: int = typer.Option(10),
    max_seconds: float = typer.Option(30.0),
    fps: int | None = typer.Option(
        None, min=1, max=240, help="Control rate (default: the training data's fps, else 30)."
    ),
    task: str | None = typer.Option(
        None,
        help="Language instruction for VLA policies (default: the checkpoint's training task).",
    ),
) -> None:
    """Evaluate in mock/sim; real evaluation runs through the Suite server.

    Before each rollout after the first you confirm the workspace is reset.
    Every scored trial is journaled as it happens, and Ctrl-C or a declined
    prompt still saves a partial report.
    """
    from .bringup.standalone import StandaloneRefused
    from .eval.evaluator import EvalAborted, EvalConfig, EvalRun
    from .eval.stats import success_summary
    from .policy.checkpoint import preflight_policy, resolve_control_fps

    err = Console(stderr=True)
    if episodes < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        err.print("[red]cannot evaluate:[/] --episodes and --max-seconds must be positive")
        raise typer.Exit(1)
    # Before the policy loads; released when the command exits.
    access = _standalone_gate("so101 eval", motion=True)
    halt_notice_run: Any = None
    try:
        try:
            info = preflight_policy(model_path, task=task)
            rate = resolve_control_fps(fps, info)
        except (OSError, ValueError, RuntimeError) as exc:
            err.print(f"[red]cannot evaluate {model_path}:[/] {exc}")
            raise typer.Exit(1) from None

        def ask(ep: int) -> bool:
            try:
                return typer.confirm(f"rollout {ep + 1}/{episodes}: success?", default=True)
            except typer.Abort:  # Ctrl-C or end of input at the prompt
                raise EvalAborted(f"stopped at the score prompt of rollout {ep + 1}") from None

        def reset_gate(ep: int) -> None:
            # The operator set the scene before starting; gate every later rollout
            # so the arm never moves into a scene that has not been reset (B12).
            if ep == 0:
                return
            try:
                go = typer.confirm(
                    f"Reset the workspace, then start rollout {ep + 1}/{episodes}?", default=True
                )
            except typer.Abort:
                go = False
            if not go:
                raise EvalAborted(f"stopped before rollout {ep + 1}")

        def source_note(source: str, label: str = "") -> str:
            where = {
                "real": "on the real arm",
                "sim": "in simulation (not a real-robot result)",
                "mock": "on the mock arm (not a real-robot result)",
            }.get(source, f"on {source or 'an unknown backend'}")
            return f"{where}; {label}" if label else where

        def print_partial() -> None:
            report = eval_run.report
            if report is None or not report.n_episodes:
                err.print("no rollout started; nothing was saved")
                return
            summary = success_summary(report.successes, report.n or 0)
            err.print(
                f"partial result after {report.n_episodes} of {report.requested_episodes} "
                f"rollouts ({report.interrupted} unscored): {summary} {source_note(report.source)}"
            )
            if eval_run.report_path is not None:
                err.print(f"partial report -> {eval_run.report_path}")

        eval_run = EvalRun(
            EvalConfig(
                model_path=str(info.path),
                n_episodes=episodes,
                max_seconds=max_seconds,
                fps=rate,
                task=info.task or "",
            ),
            score_fn=ask,
            on_reset=reset_gate,
            save=True,
        )
        halt_notice_run = eval_run
        try:
            report = eval_run.run()
        except KeyboardInterrupt:
            err.print("[yellow]evaluation interrupted[/]")
            print_partial()
            raise typer.Exit(130) from None
        except EvalAborted as exc:
            err.print(f"[yellow]evaluation {exc}[/]")
            print_partial()
            raise typer.Exit(1) from None
        except StandaloneRefused as exc:  # a latch set since the gate above
            _refuse(exc)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            err.print(f"[red]evaluation failed:[/] {exc}")
            print_partial()
            raise typer.Exit(1) from None
        summary = success_summary(report.successes, report.n or 0)
        console.print(f"[green]{summary}[/] {source_note(report.source, report.label)}")
        if eval_run.report_path is not None:
            console.print(f"report -> {eval_run.report_path}")
        else:
            err.print("[red]the report could not be saved[/] (see the log)")
            raise typer.Exit(1)
    finally:
        try:
            access.release()
        finally:
            if halt_notice_run is not None and halt_notice_run.ended_holding:
                _standalone_halt_notice()


@app.command()
def serve(host: str = "", port: int = 0) -> None:
    """Start the FastAPI backend (the web UI talks to this).

    Ctrl+C, SIGTERM and closing the terminal (SIGHUP) all stop it gracefully:
    the lifespan shutdown cancels training jobs and stops the arm, as with the
    launcher (so101.launcher.run_server).
    """
    from .launcher import run_server

    s = get_settings()
    run_server(host or s.host, port or s.port)


@app.command()
def doctor(
    json_out: bool = typer.Option(
        False, "--json", help="Print every check and the platform report as JSON."
    ),
) -> None:
    """Run non-invasive readiness diagnostics and print remediation.

    Exits 1 when any check fails, so ``so101 doctor && so101 teleop`` stops
    at a FAIL (devex-B13, docs/audit/v0.1/devex.md). The platform section
    (so101.platform_info: Python, GPU, sim rendering, serial and udev setup,
    RealSense, disk space) comes after the suite's own checks.
    """
    import json
    from dataclasses import asdict

    from rich.markup import escape

    from . import platform_info
    from .config import settings_sources
    from .hardware.backend import current_backend
    from .services.routers.operations import diagnostics

    settings = get_settings()
    backend = current_backend(settings)
    result = diagnostics()
    report = platform_info.collect()
    platform_checks = platform_info.checks(report)
    rank = {"pass": 0, "warn": 1, "fail": 2}
    statuses = [result.overall, *(check.status for check in platform_checks)]
    overall = max(statuses, key=lambda status: rank.get(status, 0))
    if json_out:
        payload = {
            "overall": overall,
            # Which backend made this report: only "real" says anything about the arm.
            "backend": backend,
            "sim": _sim_settings(settings) if backend == "sim" else None,
            "checks": [
                *(check.model_dump(mode="json") for check in result.checks),
                *(asdict(check) for check in platform_checks),
            ],
            "diagnostics": result.model_dump(mode="json"),
            "platform": report.to_dict(),
            # Which .env files and data folders this run used (so101.config).
            "settings": settings_sources(settings),
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        if overall == "fail":
            raise typer.Exit(1)
        return
    table = Table(
        title=f"SO-101 diagnostics · {escape(overall.upper())}",
        caption=f"backend: {backend}"
        + (" (a rehearsal: not evidence about a real arm)" if backend != "real" else ""),
    )
    table.add_column("status")
    table.add_column("check")
    table.add_column("detail")
    table.add_column("remediation")
    tones = {"pass": "green", "warn": "yellow", "fail": "red"}
    rows = [(c.status, c.label, c.detail, c.remediation) for c in result.checks]
    rows += [(c.status, c.label, c.detail, c.remediation) for c in platform_checks]
    for status, label, detail, remediation in rows:
        color = tones.get(status, "white")
        # Text from checks is data, not Rich markup: `.[robot]` must survive,
        # and a port path like [/dev/ttyACM0] must not raise MarkupError.
        table.add_row(
            f"[{color}]{escape(status.upper())}[/]",
            escape(label),
            escape(detail),
            escape(remediation or "—"),
        )
    console.print(table)
    if overall == "fail":
        raise typer.Exit(1)


def _default_backend_url() -> str:
    """The backend URL from settings; a wildcard bind maps to loopback."""
    s = get_settings()
    host = s.host.strip()
    if host in {"", "0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"  # IPv6 literal
    return f"http://{host}:{s.port}"


def _response_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:
        return response.text[:300]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)[:300]


@app.command("estop")
def estop(
    url: str = typer.Option(
        "",
        "--url",
        help="Backend base URL, e.g. http://127.0.0.1:8000. Defaults to SO101_HOST/SO101_PORT.",
    ),
) -> None:
    """Trigger the software E-stop on a running backend.

    Exits non-zero with a loud stderr message unless the backend confirmed the
    stop (C11 item 6, devex-B7, docs/audit/v0.1/devex.md).
    """
    import httpx

    base = (url.strip() or _default_backend_url()).rstrip("/")
    try:
        response = httpx.post(f"{base}/api/robot/estop", timeout=5)
    except Exception as exc:  # any failure to deliver must be loud
        failure = f"{type(exc).__name__}: {exc}"
    else:
        if response.is_success:
            typer.echo(f"E-STOP engaged at {base}: {_response_detail(response)}")
            return
        if response.status_code == 503:
            # The route latches before it answers: 503 means latched, but the
            # torque-off could not be confirmed (docs/operator-ui.md).
            typer.echo(
                f"E-STOP LATCHED at {base}, but torque-off is not confirmed — cut physical power",
                err=True,
            )
            typer.echo(_response_detail(response), err=True)
            raise typer.Exit(1)
        failure = f"HTTP {response.status_code}: {_response_detail(response)}"
    typer.echo(f"E-STOP NOT DELIVERED to {base} — use the power switch", err=True)
    typer.echo(failure, err=True)
    raise typer.Exit(1)


def _post_stop(url: str, path: str, body: dict | None = None) -> tuple[str, str | None]:
    """POST a stop to the backend. Returns (base URL, failure or None). Never raises."""
    import httpx

    base = (url.strip() or _default_backend_url()).rstrip("/")
    try:
        response = httpx.post(f"{base}{path}", json=body, timeout=5)
    except Exception as exc:  # any failure to deliver must be loud
        return base, f"{type(exc).__name__}: {exc}"
    if response.is_success:
        return base, None
    return base, f"HTTP {response.status_code}: {_response_detail(response)}"


_URL_HELP = "Backend base URL, e.g. http://127.0.0.1:8000. Defaults to SO101_HOST/SO101_PORT."


@app.command("halt")
def halt(url: str = typer.Option("", "--url", help=_URL_HELP)) -> None:
    """Halt the arm on a running backend: freeze in place, motors on.

    The hold lasts SO101_HALT_HOLD_SECONDS (5 s while no operator browser is
    supervising), then the arm collapses to its rest pose. Exits non-zero with
    a loud stderr message unless the backend confirmed the halt.
    """
    base, failure = _post_stop(url, "/api/robot/halt", {"reason": "so101 halt (CLI)"})
    if failure is None:
        typer.echo(f"HALT engaged at {base}: the arm holds position")
        return
    typer.echo(f"HALT NOT CONFIRMED by {base} — use `so101 estop` or the power switch", err=True)
    typer.echo(failure, err=True)
    raise typer.Exit(1)


@app.command("collapse")
def collapse(url: str = typer.Option("", "--url", help=_URL_HELP)) -> None:
    """Collapse the arm on a running backend: lower slowly to rest, then motors off.

    Exits non-zero with a loud stderr message unless the backend confirmed the
    descent started (or the motors were already off).
    """
    base, failure = _post_stop(url, "/api/robot/collapse", {"reason": "so101 collapse (CLI)"})
    if failure is None:
        typer.echo(f"COLLAPSE started at {base}: lowering to the rest pose, then motors off")
        return
    typer.echo(
        f"COLLAPSE NOT CONFIRMED by {base} — use `so101 estop` or the power switch", err=True
    )
    typer.echo(failure, err=True)
    raise typer.Exit(1)


@app.command()
def joints() -> None:
    """Print the SO-101 joint names."""
    console.print(JOINT_NAMES)


app.add_typer(__import__("so101.cli_sim", fromlist=["sim_app"]).sim_app, name="sim")
app.add_typer(bringup_app)  # so101 preflight | rest-pose | hil (bring-up kit)

if __name__ == "__main__":
    app()
