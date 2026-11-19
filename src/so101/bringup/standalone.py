"""May a command that runs without the Suite server energize or command an arm?

The server (RobotManager) restores the persisted safety latches before it
serves anything and refuses motion while one is set (docs/control-plane-safety.md,
"Safety latches"). Commands that build their own ``SO101Follower`` or
``SO101Leader`` pass this gate first. Real owned motion (teleop, record,
replay, run and eval, including ``replay_episode`` and ``PolicyRunner``) is
refused: only ``RobotManager`` can keep the watchdog, stop listener and
Halt-to-Collapse timer alive. Calibration and bounded bring-up operations may
still open a real bus through this gate:

- **mock:** nothing to check; there is no arm.
- **real:** :func:`so101.bringup.ownership.acquire_bus_access`: refuse when a
  server answers at ``SO101_HOST:SO101_PORT``, then hold the host-global locks
  for the canonical follower and leader serial devices until
  :meth:`StandaloneAccess.release`. The server takes those same locks for its
  lifetime (``so101.services.app``), including across data folders/checkouts.
  A request marked as motion is then refused and releases the locks.
- **real and sim:** merge the latches from
  ``<data_dir>/control_plane.sqlite3`` with a newer
  ``control_plane.latch-unsaved`` fallback marker, exactly as server startup
  does, and refuse while the E-stop, a fault, an unconfirmed torque-off or
  energized intent is latched. An unreadable store or active marker refuses
  too. No matching store row or marker means no latch was ever saved.

The gate only refuses; it never clears a latch. Clearing stays a deliberate
reset through the server. Within one owning thread the gate is re-entrant: a
library call made inside a CLI command that already passed it re-reads the
latches but reuses the command's device-lock set. An unrelated thread is another
owner and is refused. Standalone-owned motor devices also tear down through
the access: a closed host port is not accepted while its torque truth remains
``on`` or ``unconfirmed``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from ..config import get_settings
from ..hardware.backend import Backend, current_backend
from ..logging import get_logger
from ..services.store import (
    clear_standalone_energized_intent as _clear_energized_intent,
)
from ..services.store import (
    merge_standalone_latches as _merge_latches,
)
from ..services.store import (
    newest_identity_latch_marker,
)
from ..services.store import (
    persist_standalone_energized_intent as _persist_energized_intent,
)
from ..services.store import (
    persist_standalone_fault as _persist_standalone_fault,
)
from ..services.store import (
    persist_standalone_torque_off_failure as _persist_torque_off_failure,
)
from . import ownership

log = get_logger(__name__)

RefusalReason = Literal["latched", "unreadable", "server", "lock", "unsupported"]

#: Test seam, looked up at call time: probes for a running Suite server.
server_probe: ownership.ServerProber = ownership.probe_local_server

_RESET_BODY = json.dumps({"operator_id": "<your name>", "confirm": True})


class StandaloneRefused(RuntimeError):
    """The command may not open the arm; nothing was opened or energized."""

    def __init__(self, reason: RefusalReason, detail: str, remediation: str) -> None:
        super().__init__(f"{detail}. {remediation}")
        self.reason: RefusalReason = reason
        self.detail = detail
        self.remediation = remediation


def refuse_real_motion(purpose: str, *, settings: Any = None) -> None:
    """Reject a motion path that is not owned by RobotManager on real hardware."""

    settings = settings if settings is not None else get_settings()
    if current_backend(settings) != "real":
        return
    raise StandaloneRefused(
        "unsupported",
        f"real standalone motion is disabled for {purpose} because it cannot enforce the "
        "watchdog, software stops, supervision grace and timed Collapse",
        "Start `so101 serve`, then run teleop, recording, replay, rollout or evaluation "
        "through the Suite UI/API. Mock and simulation CLI motion remain available.",
    )


class _HeldLock:
    def __init__(self, lock: ownership.BusLock | ownership.DeviceBusLock) -> None:
        self.lock = lock
        self.count = 1
        self.owner_thread = threading.current_thread()
        self.release_blocked: str | None = None


#: Canonical device-lock sets this process holds, for re-entrant use within one command.
_held: dict[tuple[str, ...], _HeldLock] = {}
_held_guard = threading.Lock()


class StandaloneAccess:
    """Permission to open the arm, from :func:`standalone_access` until released."""

    def __init__(
        self,
        backend: Backend,
        purpose: str,
        *,
        lock_path: Path | None = None,
        lock_key: tuple[str, ...] | None = None,
        latches: dict[str, Any] | None = None,
        settings: Any = None,
    ) -> None:
        self.backend = backend
        self.purpose = purpose
        self.lock_path = lock_path
        self.lock_key = lock_key
        self.latches = latches
        self.settings = settings if settings is not None else get_settings()
        self._owner_thread = threading.current_thread() if lock_path is not None else None
        self._torque_failure_recorded = False
        self._energized_intent_recorded = False
        self._energized_device_name = "follower"
        self._released = False
        self._capture_context: Any = None
        # Real motion access is never issued: public entry points are
        # server-only. This also prevents a caller from omitting motion=True
        # and then using a bring-up access object to energize the follower.
        self._real_motion_allowed = backend != "real"

    @property
    def holds_bus_lock(self) -> bool:
        return self.lock_key is not None and not self._released

    def _require_owner_thread(self) -> None:
        if self._owner_thread is not None and threading.current_thread() is not self._owner_thread:
            raise RuntimeError("motor-bus device-lock access may only be used by its owning thread")

    def _require_motion_allowed(self) -> None:
        if not self._real_motion_allowed:
            refuse_real_motion(self.purpose, settings=self.settings)

    def disconnect_device(self, device: Any, *, device_name: str) -> None:
        """Disconnect an owned motor device and fail closed on uncertain torque.

        The latch is made durable while the motor-device locks are still held. Callers
        use this instead of a bare device ``disconnect()`` so another
        standalone command cannot slip in between the failed torque-off and
        its persisted fault.
        """

        self._require_owner_thread()
        if self._released:
            raise RuntimeError("standalone access was released before device disconnect")
        from ..hardware.motors import TorqueOffUnconfirmed

        try:
            device.disconnect()
        except TorqueOffUnconfirmed as exc:
            self.record_torque_off_failure(exc, device_name=device_name)
            raise
        except BaseException as disconnect_error:
            self._fail_if_torque_may_be_on(device, device_name=device_name, cause=disconnect_error)
            if device_name == "follower":
                self.clear_energized_intent()
            raise
        self._fail_if_torque_may_be_on(device, device_name=device_name)
        if device_name == "follower":
            self.clear_energized_intent()

    def disconnect_follower(self, follower: Any) -> None:
        """Compatibility wrapper for the standalone follower cleanup path."""

        self.disconnect_device(follower, device_name="follower")

    def connect_follower(self, follower: Any) -> None:
        """Write energized intent, then allow an owned follower to connect."""

        self._require_motion_allowed()
        self.prepare_to_energize()
        try:
            follower.connect()
        except BaseException:
            self.disconnect_follower(follower)
            raise

    def _fail_if_torque_may_be_on(
        self,
        device: Any,
        *,
        device_name: str,
        cause: BaseException | None = None,
    ) -> None:
        state, state_error = _device_torque_state(device)
        if state_error is None:
            try:
                if bool(state == "off"):
                    return
            except Exception as exc:
                state_error = exc
        # Mock-only test doubles predate three-state torque truth and own no
        # physical motor. Real and simulated devices must make it explicit.
        if state is None and state_error is None and self.backend == "mock":
            return
        failure = _device_torque_failure(device, device_name)
        self.record_torque_off_failure(failure, device_name=device_name)
        if state_error is not None:
            raise failure from state_error
        if cause is not None:
            raise failure from cause
        raise failure

    def record_torque_off_failure(self, exc: Exception, *, device_name: str = "follower") -> None:
        """Persist a standalone torque-off failure once, before access is released."""

        self._require_owner_thread()
        if self._released:
            raise RuntimeError(
                "cannot persist a torque-off failure after releasing motor-device locks"
            )
        if self.backend == "mock" or self._torque_failure_recorded:
            return
        global_error: Exception | None = None
        if self.backend == "real":
            try:
                ownership.write_global_energized_intent(
                    self.settings,
                    purpose=f"standalone {device_name} torque-off unconfirmed: {exc}",
                    device_name=(
                        device_name if device_name in {"follower", "leader"} else "follower"
                    ),
                )
            except Exception as error:
                global_error = error
        try:
            self.latches = _persist_torque_off_failure(self.settings, exc, device_name=device_name)
        except Exception as persistence_error:
            if self.backend == "real" and global_error is None:
                # The host-global device marker is the authoritative durable
                # backstop across data directories. Local SQLite/marker loss
                # reduces UI detail but cannot admit another hardware owner.
                log.critical(
                    "standalone torque-off was unconfirmed; the local latch could not be "
                    "saved (%s), but the host-global energized marker is durable",
                    persistence_error,
                )
                self.latches = {
                    "fault": f"standalone {device_name} {exc}",
                    "safety_state": "fault",
                    "torque_unconfirmed": True,
                    "energized": True,
                }
                self._torque_failure_recorded = True
                return
            if self.lock_key is not None:
                _poison_lock(
                    self.lock_key,
                    owner_thread=self._owner_thread,
                    detail=str(persistence_error),
                )
            raise
        if global_error is not None:
            if self.lock_key is not None:
                _poison_lock(
                    self.lock_key,
                    owner_thread=self._owner_thread,
                    detail=str(global_error),
                )
            raise RuntimeError(
                "torque-off was unconfirmed and the host-global energized marker could not "
                f"be saved ({global_error}); refusing to release the motor-device locks"
            ) from global_error
        self._torque_failure_recorded = True

    def prepare_to_energize(self, *, device_name: str = "follower") -> None:
        """Durably record intent before an owned follower can enable torque."""

        self._require_owner_thread()
        self._require_motion_allowed()
        if self._released:
            raise RuntimeError("cannot energize after releasing standalone access")
        if self.backend == "mock" or self._energized_intent_recorded:
            return
        self.latches = _persist_energized_intent(
            self.settings,
            purpose=self.purpose,
            device_name=device_name,
        )
        # A concurrent real latch must still win. The intent we just wrote is
        # expected; every other latch remains a refusal before hardware opens.
        blockers = [item for item in latched_items(self.latches) if item != "arm may be energized"]
        if blockers:
            try:
                _clear_energized_intent(self.settings)
            except Exception:
                pass  # a stale local intent is fail-safe; no hardware was opened
            raise latch_refusal(
                self.latches,
                store=_store_path(self.settings),
                backend=self.backend,
            ) or RuntimeError("safety latch appeared while recording energized intent")
        if self.backend == "real":
            try:
                ownership.write_global_energized_intent(
                    self.settings,
                    purpose=self.purpose,
                    device_name=device_name,
                )
            except Exception:
                try:
                    _clear_energized_intent(self.settings)
                except Exception:
                    pass  # the local false positive keeps this data folder fail-closed
                raise
        self._energized_device_name = device_name
        self._energized_intent_recorded = True

    def clear_energized_intent(self) -> None:
        """Clear write-ahead intent only after follower torque is confirmed off."""

        self._require_owner_thread()
        if (
            self.backend == "mock"
            or not self._energized_intent_recorded
            or self._torque_failure_recorded
        ):
            return
        failures: list[BaseException] = []
        if self.backend == "real":
            try:
                ownership.clear_global_energized_intent(
                    self.settings,
                    purpose="standalone motor torque-off confirmed",
                    device_name=self._energized_device_name,
                )
            except BaseException as exc:
                failures.append(exc)
        try:
            self.latches = _clear_energized_intent(self.settings)
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise RuntimeError(
                "follower torque-off was confirmed, but its energized intent could not be "
                f"cleared everywhere: {'; '.join(str(exc) for exc in failures)}"
            ) from failures[0]
        self._energized_intent_recorded = False

    def halt_follower(self, follower: Any) -> None:
        """End a successful standalone operation with a category-2 hold.

        The follower writes fresh Present as Goal (with the gripper hold
        rule), verifies its torque truth is still on, and closes only the host
        port. The write-ahead energized intent deliberately remains durable,
        so a later process cannot mistake the held arm for torque-off.
        """

        self._require_owner_thread()
        self._require_motion_allowed()
        if self._released:
            raise RuntimeError("standalone access was released before follower Halt")
        if self.backend == "mock" and not callable(getattr(follower, "halt_and_disconnect", None)):
            # Legacy mock-only test doubles have no physical motor and predate
            # category-2 standalone teardown. Keep them source-compatible.
            self.disconnect_follower(follower)
            return
        try:
            follower.halt_and_disconnect()
            state, state_error = _device_torque_state(follower)
            if state_error is not None:
                raise RuntimeError(f"could not read follower torque truth: {state_error}")
            if state != "on":
                raise RuntimeError(f"follower torque is {state!r} after Halt, not confirmed on")
        except BaseException as halt_error:
            try:
                connected = bool(follower.connected)
            except Exception:
                connected = False
            if not connected:
                failure = _device_torque_failure(follower, "follower")
                self.record_torque_off_failure(failure, device_name="follower")
                raise failure from halt_error
            try:
                self.disconnect_follower(follower)
            except BaseException:
                # disconnect_follower already persisted unconfirmed truth (or
                # kept the lock poisoned when neither target was writable).
                raise
            self.latches = _persist_standalone_fault(
                self.settings,
                halt_error,
                device_name="follower Halt",
                torque_unconfirmed=False,
                energized=False,
            )
            raise RuntimeError(
                f"standalone Halt could not hold position; torque was turned off: {halt_error}"
            ) from halt_error

    def release(self) -> None:
        """Give up the access. Idempotent; the last holder releases the device locks."""
        self._require_owner_thread()
        if self._released:
            return
        if self.lock_key is not None:
            _drop_lock(self.lock_key, owner_thread=self._owner_thread)
        self._released = True
        if self._capture_context is not None:
            try:
                self._capture_context.__exit__(None, None, None)
            except OSError as exc:
                log.error("Could not remove completed observation request: %s", exc)
            self._capture_context = None

    def __enter__(self) -> StandaloneAccess:
        if (
            self.backend == "real"
            and getattr(self.settings, "camera_capture_mode", "direct") == "shared"
        ):
            from ..capture_activity import capture_activity

            status = self.settings.data_dir / "mcap/status.json"
            try:
                recorder = json.loads(status.read_text())
            except (OSError, ValueError):
                recorder = {}
            if recorder.get("ready") and time.time() - recorder.get("updated_at", 0) < 5:
                kind = "calibration" if "cal" in self.purpose.casefold() else "diagnostic"
                context = capture_activity(self.settings.data_dir, kind, self.purpose)
                try:
                    request = context.__enter__()
                    self._capture_context = context
                    deadline = time.monotonic() + 6
                    while time.monotonic() < deadline:
                        recorder = json.loads(status.read_text())
                        if recorder.get("active") and recorder.get("activity_id") == request["id"]:
                            break
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("Observation recorder did not acknowledge the activity")
                except BaseException:
                    self.release()
                    raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc is not None:
                from ..hardware.motors import TorqueOffUnconfirmed

                if isinstance(exc, TorqueOffUnconfirmed):
                    self.record_torque_off_failure(
                        exc, device_name=str(getattr(exc, "bus_name", "motor bus"))
                    )
        finally:
            self.release()


def standalone_access(
    purpose: str,
    *,
    settings: Any = None,
    motion: bool = False,
) -> StandaloneAccess:
    """Pass the gate for ``purpose`` (for example ``"so101 teleop"``), or raise.

    Raises :class:`StandaloneRefused` before anything is opened. On the real
    backend the returned access holds the canonical device locks; call ``release()`` (or use
    it as a context manager) when the command has closed the arm. ``motion``
    entry points are refused on real hardware: only RobotManager provides the
    watchdog, software stop listener, supervision grace and timed Collapse.
    """
    settings = settings if settings is not None else get_settings()
    backend = current_backend(settings)
    if backend == "mock":
        return StandaloneAccess(backend, purpose, settings=settings)
    lock_key, lock_path = _take_lock(settings, purpose) if backend == "real" else (None, None)
    try:
        latches = read_latches(settings)
        refusal = latch_refusal(latches, store=_store_path(settings), backend=backend)
        if refusal is not None:
            raise refusal
        if motion:
            refuse_real_motion(purpose, settings=settings)
    except BaseException:
        if lock_key is not None:
            _drop_lock(lock_key, owner_thread=threading.current_thread())
        raise
    return StandaloneAccess(
        backend,
        purpose,
        lock_path=lock_path,
        lock_key=lock_key,
        latches=latches,
        settings=settings,
    )


def adopt_acquired_bus_access(
    lock: ownership.BusLock | ownership.DeviceBusLock,
    *,
    backend: Backend,
    purpose: str,
    settings: Any = None,
) -> StandaloneAccess:
    """Wrap a bus lock an existing bring-up gate already acquired.

    This does not repeat that command's server/latch policy. It gives the
    command the same owned-device teardown and thread-scoped release contract
    as :func:`standalone_access`.
    """

    if not lock.held:
        raise RuntimeError("cannot adopt a motor-bus lock that is not held")
    path = lock.path.resolve()
    key = getattr(lock, "lock_key", (f"lock-file:{path}",))
    with _held_guard:
        held = _held.get(key)
        if held is not None and held.lock.held:
            raise RuntimeError(f"cannot adopt {path}: this process already tracks its owner")
        _held[key] = _HeldLock(lock)
    return StandaloneAccess(
        backend,
        purpose,
        lock_path=path,
        lock_key=key,
        settings=settings if settings is not None else get_settings(),
    )


# --- the latches ----------------------------------------------------------------------


def _store_path(settings: Any) -> Path:
    from ..services.store import control_store_path

    return control_store_path(settings)


def _device_torque_state(device: Any) -> tuple[Any, Exception | None]:
    """Return a device or its bus's torque truth; missing test doubles are unknown."""

    candidates = [device]
    try:
        bus = getattr(device, "bus", None)
    except Exception as exc:
        return "unconfirmed", exc
    if bus is not None and bus is not device:
        candidates.append(bus)
    for candidate in candidates:
        try:
            state = getattr(candidate, "torque_state", None)
        except Exception as exc:
            return "unconfirmed", exc
        if state is not None:
            return state, None
    return None, None


def _device_torque_failure(device: Any, device_name: str) -> Exception:
    from .. import JOINT_NAMES
    from ..hardware.motors import TorqueOffUnconfirmed

    def safe_attr(obj: Any, name: str) -> Any:
        try:
            return getattr(obj, name, None)
        except Exception:
            return None

    bus = safe_attr(device, "bus")
    try:
        name = str(safe_attr(bus, "name") or safe_attr(device, "name") or device_name)
    except Exception:
        name = device_name
    raw_motors = safe_attr(bus, "motors")
    if raw_motors is None:
        raw_motors = safe_attr(device, "motors")
    try:
        motors = [str(motor) for motor in raw_motors] if raw_motors else list(JOINT_NAMES)
    except Exception:
        motors = list(JOINT_NAMES)
    if not motors:
        motors = list(JOINT_NAMES)
    return TorqueOffUnconfirmed(name, motors)


def read_latches(settings: Any) -> dict[str, Any] | None:
    """The persisted latches, None when none were ever saved. Raises StandaloneRefused.

    Reads through the server's own ``ControlStore``, so the schema is the
    server's. A missing store and marker means "never saved"; anything else
    that cannot be read is "unreadable", and the caller refuses.
    """
    from ..services.store import ControlStore, safety_latch_identity

    path = _store_path(settings)
    identity = safety_latch_identity(settings)
    marker = newest_identity_latch_marker(path, identity)
    try:
        saved = None
        if path.exists():
            saved = ControlStore(path, latch_identity=identity).load_latches()
    except Exception as exc:
        raise StandaloneRefused(
            "unreadable",
            f"could not read the persisted safety latches in {path} "
            f"({type(exc).__name__}: {exc}); a command run without the server fails closed, "
            "as the server does (it starts faulted on this file)",
            "Check that file's permissions. If it is damaged, check the arm and its power, "
            "then move the file aside: the latches it held are lost with it.",
        ) from exc
    merged = _merge_latches(saved, marker)
    if identity.backend != "real":
        return merged
    try:
        global_intent = ownership.read_global_energized_intent(settings)
    except Exception as exc:
        raise StandaloneRefused(
            "unreadable",
            f"could not trust the host-global energized state for the configured motor "
            f"devices ({type(exc).__name__}: {exc}); treating the arm as energized",
            "Keep physical power controlled, inspect the global marker and device identity, "
            "then use the Suite fault recovery while it owns the same motor-device locks.",
        ) from exc
    if global_intent is None:
        return merged
    merged = dict(merged or {})
    merged["energized"] = True
    merged["torque_unconfirmed"] = True
    merged["energized_at"] = global_intent.get("written_at")
    merged["energized_by"] = global_intent.get("purpose")
    merged["_global_energized_intent"] = global_intent
    return merged


def latched_items(latches: dict[str, Any] | None) -> list[str]:
    """What is latched, in words; empty when nothing is."""
    if not latches:
        return []
    items = []
    if latches.get("estopped"):
        items.append("E-stop")
    fault = latches.get("fault")
    if fault:
        items.append(f"fault: {fault}")
    if latches.get("torque_unconfirmed"):
        items.append("torque-off unconfirmed")
    if latches.get("energized"):
        items.append("arm may be energized")
    return items


def latch_refusal(
    latches: dict[str, Any] | None, *, store: Path, backend: Backend
) -> StandaloneRefused | None:
    """The refusal for a set latch, naming it and the deliberate reset; None if clear."""
    items = latched_items(latches)
    if not items or latches is None:
        return None
    saved = latches.get("updated_at")
    marker = latches.get("_unsaved_marker")
    global_intent = latches.get("_global_energized_intent")
    location = f" in {store}"
    if isinstance(marker, dict):
        location += f"; the newer fallback marker {marker.get('path')} is also active"
    if isinstance(global_intent, dict):
        location += "; the host-global motor-device energized marker is active"
    detail = (
        f"the arm's safety latch is set ({'; '.join(items)})"
        + (f", saved {saved}" if saved else "")
        + location
        + ". A command run without the server does not clear or skip it"
    )
    resets = []
    if latches.get("estopped"):
        resets.append("POST /api/robot/estop/reset")
    if latches.get("fault") or latches.get("torque_unconfirmed"):
        resets.append("POST /api/robot/fault/reset")
    elif latches.get("energized"):
        resets.append("POST /api/robot/fault/reset")
    steps = ", then ".join(resets)
    remediation = (
        "Clear it on purpose through the Suite: start the server (`so101 serve`) and use "
        f"Reset in the UI, or send {steps}, each with the JSON body {_RESET_BODY}."
    )
    if "POST /api/robot/fault/reset" in resets:
        remediation += (
            ' Add "power_cut_acknowledged": true to the fault reset only after you cut the '
            "arm's power."
        )
    if backend == "real":
        remediation += " Then stop the server (it holds the bus) and re-run this command."
    else:
        remediation += " Then re-run this command."
    return StandaloneRefused("latched", detail, remediation)


# --- device-global bus locks (real backend) ---------------------------------------------


def _take_lock(settings: Any, purpose: str) -> tuple[tuple[str, ...], Path]:
    try:
        identities = ownership.device_lock_identities(settings)
        key = ownership.device_lock_key(settings)
        path = ownership.device_lock_path(identities[0])
    except ownership.BusOwnershipError as exc:
        raise _ownership_refusal(exc) from exc
    devices = ", ".join(identity.canonical_port for identity in identities)
    owner_thread = threading.current_thread()
    with _held_guard:
        held = _held.get(key)
        if held is not None and held.lock.held:
            if held.release_blocked is not None:
                raise StandaloneRefused(
                    "unreadable",
                    f"motor bus {devices} remains locked after a safety latch could not be saved "
                    f"({held.release_blocked})",
                    "Cut physical power, then end this process before trying again.",
                )
            if held.owner_thread is owner_thread:
                held.count += 1
                return key, path
            raise StandaloneRefused(
                "lock",
                f"motor bus {devices} is held by another thread in this process "
                f"({held.lock.purpose}); that thread owns the serial bus",
                "Wait for that command to finish, or stop it, then re-run.",
            )
        try:
            lock, _ = ownership.acquire_bus_access(
                settings, purpose=purpose, server_probe=server_probe
            )
        except ownership.BusOwnershipError as exc:
            raise _ownership_refusal(exc) from exc
        _held[lock.lock_key] = _HeldLock(lock)
        return lock.lock_key, lock.path


def _poison_lock(
    key: tuple[str, ...], *, owner_thread: threading.Thread | None, detail: str
) -> None:
    """Keep an access locked after neither durable latch target could be written."""

    with _held_guard:
        held = _held.get(key)
        if held is not None and owner_thread is not None and held.owner_thread is owner_thread:
            held.release_blocked = detail


def _drop_lock(key: tuple[str, ...], *, owner_thread: threading.Thread | None) -> None:
    with _held_guard:
        held = _held.get(key)
        if held is None:
            return
        current_thread = threading.current_thread()
        if (
            owner_thread is None
            or current_thread is not owner_thread
            or held.owner_thread is not current_thread
        ):
            raise RuntimeError("motor-bus locks must be released by their owning thread")
        if held.release_blocked is not None:
            raise RuntimeError(
                f"refusing to release motor-bus locks after a safety latch could not be saved: "
                f"{held.release_blocked}"
            )
        held.count -= 1
        if held.count <= 0:
            del _held[key]
            held.lock.release()


def _ownership_refusal(exc: ownership.BusOwnershipError) -> StandaloneRefused:
    if exc.reason == "server":
        return StandaloneRefused(
            "server",
            exc.detail,
            "This command opens the arm itself, outside the server's E-stop. Use the running "
            "Suite instead, or stop the server (Ctrl-C in its terminal) and re-run.",
        )
    if exc.reason == "lock":
        return StandaloneRefused(
            "lock",
            f"{exc.detail}; that process owns the serial bus",
            "Wait for it to finish, or stop it (Ctrl-C in its terminal), then re-run.",
        )
    return StandaloneRefused("unsupported", exc.detail, exc.remediation)
