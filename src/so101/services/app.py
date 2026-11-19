"""FastAPI application factory: mounts routers, middleware, lifespan, static UI."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..bringup.ownership import (
    GlobalEnergizedIntentError,
    ServerBusLockError,
    read_global_energized_intent,
    take_server_bus_lock,
)
from ..config import Settings, get_settings
from ..hardware.backend import current_backend
from ..logging import setup_logging
from .agent_executor import recover_interrupted_proposals, shutdown_executor
from .guard import GuardMiddleware, GuardPolicy
from .jobs import get_runner
from .robot_manager import get_manager
from .routers import (
    agent,
    calibration,
    capabilities,
    control,
    datasets,
    eval,
    hil,
    mcap,
    models,
    operations,
    preflight,
    recording,
    robot,
    simulation,
    telemetry,
    teleop,
    tracking,
    training,
)
from .strict_json import StrictJSONMiddleware, install_numeric_error_handlers

logger = logging.getLogger(__name__)

#: What uvicorn runs, with ``--factory`` (``uvicorn.run(APP_FACTORY, factory=True)``).
APP_FACTORY = "so101.services.app:create_app"

# Paths the SPA fallback must never answer with index.html (F25, devex-B12).
_BACKEND_PREFIXES = ("api", "ws")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    # On the real backend the server owns the serial buses for its lifetime:
    # it holds each canonical device's host-global lock, so a standalone CLI command (teleop,
    # record, replay, run, eval, calibrate, preflight, rest-pose record) and
    # the server never drive the arm at once. A CLI command that holds the
    # lock stops the startup here, naming that command.
    try:
        bus_lock = take_server_bus_lock(settings)
    except ServerBusLockError as exc:
        logger.error("%s", exc)
        raise
    shutdown_errors: list[BaseException] = []
    mcap_supervisor = None
    try:
        manager = get_manager()  # eager-init the single hardware owner
        if bus_lock is not None and current_backend(settings) == "real":
            try:
                global_intent = read_global_energized_intent(settings)
            except GlobalEnergizedIntentError as exc:
                manager.adopt_global_energized_intent(None, error=exc)
            else:
                manager.adopt_global_energized_intent(global_intent)
                manager.publish_restored_recovery_intent()
        try:
            try:
                # A proposal still "executing" was cut off by a crash or restart
                # (sim-B9, docs/audit/v0.1/sim.md).
                recover_interrupted_proposals()
            except Exception:
                logger.exception("could not recover interrupted agent proposals")
            try:
                # Restore job records now, so a trainer a crashed server left running
                # is stopped at startup, not at the first training request.
                get_runner()
            except Exception:
                logger.exception("could not restore training jobs")
            mcap_supervisor = mcap.start_supervisor(manager)
            yield
        except BaseException as exc:
            # Startup and lifespan failures must not skip the hardware shutdown
            # below. Keep the original error and re-raise it after every teardown.
            shutdown_errors.append(exc)
        finally:
            if mcap_supervisor is not None:
                mcap_supervisor[0].set()
                mcap_supervisor[1].join(timeout=2)
            # Stop agent motion first, then end the eval: it saves a partial
            # report while the manager can still stop its rollout (data-B7,
            # docs/audit/v0.1/data.md). Every step is isolated so a persistence
            # or worker-cleanup error cannot skip the manager's safety Collapse.
            shutdown_steps: tuple[tuple[str, Callable[[], None]], ...] = (
                ("MCAP episode metadata", mcap.shutdown),
                ("agent executor", shutdown_executor),
                ("evaluation session", eval.shutdown_session),
                ("training runner", lambda: get_runner().shutdown()),
                ("robot manager safety", lambda: get_manager().shutdown()),
            )
            for name, shutdown in shutdown_steps:
                try:
                    shutdown()
                except BaseException as exc:
                    logger.exception("%s shutdown failed", name)
                    shutdown_errors.append(exc)
    finally:
        # Only after the manager's shutdown stop has run (or reported its failure).
        if bus_lock is not None:
            try:
                bus_lock.release()
            except BaseException as exc:
                logger.exception("could not release the server motor-bus lock")
                shutdown_errors.append(exc)

    if len(shutdown_errors) == 1:
        raise shutdown_errors[0]
    if shutdown_errors:
        raise BaseExceptionGroup("SO-101 application lifespan shutdown failed", shutdown_errors)


# --- built web UI ------------------------------------------------------------


def _has_index(directory: Path) -> bool:
    try:
        return (directory / "index.html").is_file()
    except (OSError, ValueError):
        return False


def _packaged_ui_dir(package: str = "so101") -> Path | None:
    """``<package>/web`` inside an installed wheel, when it is on the filesystem."""
    try:
        candidate = resources.files(package) / "web"
    except (ModuleNotFoundError, TypeError, ValueError):
        return None
    return candidate if isinstance(candidate, Path) else None


def _checkout_ui_dir() -> Path:
    """``frontend/dist`` of a source checkout (src/so101/services -> repo root)."""
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def resolve_ui_dir(settings: Settings) -> Path | None:
    """First directory with an index.html: SO101_UI_DIR, packaged, checkout.

    None means no UI is built; ``/`` then serves a JSON placeholder (devex-B3,
    docs/audit/v0.1/devex.md).
    """
    if settings.ui_dir:
        configured = Path(settings.ui_dir).expanduser()
        if _has_index(configured):
            return configured.resolve()
        logger.warning("SO101_UI_DIR=%s has no index.html; ignoring it", settings.ui_dir)
    for candidate in (_packaged_ui_dir(), _checkout_ui_dir()):
        if candidate is not None and _has_index(candidate):
            return candidate.resolve()
    return None


def _is_backend_path(path: str) -> bool:
    return path.lstrip("/").split("/", 1)[0] in _BACKEND_PREFIXES


# A UI path segment: no leading dot, so no "..", "." or dotfiles.
_UI_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_WINDOWS_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"] + [f"{port}{n}" for port in ("COM", "LPT") for n in range(10)]
)


def _is_safe_ui_path(path: str) -> bool:
    """True if ``path`` is a plain relative path of ordinary file names.

    This is a string check, made before any filesystem call. On Windows
    ``root / path`` discards ``root`` for ``\\\\host\\share``, ``//host/share``
    and ``D:\\x``, and resolving the result opens it: a UNC path sends the
    operator's NTLM hash to that host. A device name such as ``COM3`` or
    ``nul.txt`` opens the device in any directory. A cross-site ``<img>`` can
    request all of these, and GETs skip the Origin check.
    """
    if not path or path.startswith("/") or "\\" in path or ":" in path:
        return False
    for segment in path.split("/"):
        if not _UI_SEGMENT.fullmatch(segment):
            return False
        if segment.split(".", 1)[0].upper() in _WINDOWS_DEVICE_NAMES:
            return False
    return True


def _contained_file(root: Path, path: str) -> Path | None:
    """The file ``path`` names inside ``root``, or None if it escapes or is missing.

    ``root / path`` alone follows ``..`` segments, absolute paths and symlinks
    anywhere on disk (safety-B4, docs/audit/v0.1/safety.md). The string check
    comes first, so a hostile path never reaches the filesystem.
    """
    if not _is_safe_ui_path(path):
        return None
    try:
        target = (root / path).resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    except (OSError, ValueError, RuntimeError):
        pass
    return None


class _UIStaticFiles(StaticFiles):
    """StaticFiles that applies the UI path check before touching the disk.

    Starlette rejects a leading ``/`` or ``\\`` itself, but on Windows it still
    resolves ``D:\\x`` and opens device names such as ``COM3``.
    """

    def lookup_path(self, path: str) -> tuple[str, os.stat_result | None]:
        if not _is_safe_ui_path(path.replace(os.sep, "/")):
            return "", None
        return super().lookup_path(path)


def _mount_ui(app: FastAPI, root: Path) -> None:
    index = root / "index.html"
    if (root / "assets").is_dir():
        app.mount("/assets", _UIStaticFiles(directory=root / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def spa_index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:  # client-side routing fallback
        if _is_backend_path(path):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(_contained_file(root, path) or index)


def _mount_placeholder(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    def placeholder() -> JSONResponse:
        return JSONResponse(
            {
                "service": "so101",
                "version": __version__,
                "ui": "frontend not built — run `make ui` (dev) or `cd frontend && npm run build`",
                "docs": "/docs",
            }
        )


# --- application -------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="SO-101 Platform", version=__version__, lifespan=lifespan)
    settings = get_settings()
    # Fail at startup, not on the first request, if SO101_ALLOWED_HOSTS is unsafe.
    GuardPolicy.from_settings(settings)

    # add_middleware prepends: the guard, added last, runs first and rejects
    # before any body is read; strict JSON then runs before routing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(StrictJSONMiddleware)
    app.add_middleware(GuardMiddleware)
    install_numeric_error_handlers(app)

    for r in (
        capabilities,
        control,
        operations,
        simulation,
        tracking,
        agent,
        robot,
        teleop,
        calibration,
        recording,
        mcap,
        datasets,
        training,
        models,
        eval,
        telemetry,
        preflight,  # bring-up kit
        hil,  # bring-up kit
    ):
        app.include_router(r.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__, **get_manager().status()}

    @app.get("/api/e2e/identity", include_in_schema=False, response_model=None)
    def installed_e2e_identity() -> dict | JSONResponse:
        """Identify the ephemeral installed-wheel mock server without side effects."""

        current = get_settings()
        if not current.e2e_token.get_secret_value():
            # Token-unset production servers retain the pre-E2E 404 behavior.
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return {
            "ok": True,
            "kind": "so101.installed_e2e",
            "schema_version": 1,
            "backend": current_backend(current),
            "mock": current.mock,
            "version": __version__,
        }

    # Serve the built frontend if present (so the whole app is one process).
    ui_dir = resolve_ui_dir(settings)
    if ui_dir is not None:
        _mount_ui(app, ui_dir)
    else:
        _mount_placeholder(app)
    return app


def __getattr__(name: str) -> Any:
    """``so101.services.app:app``, built on first access, for older launch commands.

    Importing this module has no side effects: building the app reads the
    settings and creates the data folders, so it happens only when a server
    starts. Launchers run ``uvicorn --factory so101.services.app:create_app``
    (see :data:`APP_FACTORY`); ``uvicorn so101.services.app:app`` still works.
    """

    if name == "app":
        built = create_app()
        globals()["app"] = built  # later lookups find it without coming back here
        return built
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
