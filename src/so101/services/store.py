"""Durable control-plane audit events, safety latches and agent proposals.

The robot service intentionally keeps high-rate telemetry out of SQLite.  This
store is for the small, important records that must survive a process restart:
operator actions, safety transitions, the safety latches (E-stop, fault,
unconfirmed torque-off and write-ahead energized intent) and bounded agent
proposals.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..atomic_io import atomic_write_json
from ..config import get_settings
from ..hardware.backend import current_backend
from ..logging import get_logger

log = get_logger(__name__)


SAFETY_LATCH_SCHEMA_VERSION = 2
LATCH_MARKER_SCHEMA_VERSION = 2
LATCH_MARKER_SUFFIX = ".latch-unsaved"
LATCH_MARKER_KIND = "so101.latch-unsaved"
LEGACY_UNSCOPED_DEVICE_ID = "__legacy_unscoped__"


@dataclass(frozen=True)
class SafetyLatchIdentity:
    """Backend and logical motor-device identity for one safety snapshot."""

    backend: str
    follower_id: str
    leader_id: str

    @classmethod
    def from_settings(cls, settings: Any, *, backend: str | None = None) -> SafetyLatchIdentity:
        selected = backend or current_backend(settings)
        if selected not in {"real", "mock", "sim"}:
            raise ValueError(f"unsupported safety-latch backend {selected!r}")
        follower_id = str(getattr(settings, "follower_id", ""))
        leader_id = str(getattr(settings, "leader_id", ""))
        if not follower_id or not leader_id:
            raise ValueError("safety-latch device identities must not be empty")
        return cls(selected, follower_id, leader_id)

    def for_real_backend(self) -> SafetyLatchIdentity:
        """The conservative wildcard owner of a pre-identity (v1) latch."""

        return SafetyLatchIdentity(
            "real",
            LEGACY_UNSCOPED_DEVICE_ID,
            LEGACY_UNSCOPED_DEVICE_ID,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "follower_id": self.follower_id,
            "leader_id": self.leader_id,
        }

    def marker_tag(self) -> str:
        raw = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def safety_latch_identity(settings: Any) -> SafetyLatchIdentity:
    return SafetyLatchIdentity.from_settings(settings)


def latch_marker_path(store_path: Path, identity: SafetyLatchIdentity) -> Path:
    """Identity-scoped fallback marker path.

    Every v2 backend/device identity has a disjoint name, so one real logical
    pair cannot replace or remove another's marker. The historical unscoped
    path is read separately by :func:`legacy_latch_marker_path`.
    """

    suffix = f".{identity.backend}-{identity.marker_tag()}{LATCH_MARKER_SUFFIX}"
    return store_path.with_name(store_path.stem + suffix)


def legacy_latch_marker_path(store_path: Path) -> Path:
    """The pre-v2 unscoped marker, conservatively read by real identities only."""

    return store_path.with_name(store_path.stem + LATCH_MARKER_SUFFIX)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


class ControlStore:
    """Tiny SQLite repository with one short-lived connection per operation."""

    def __init__(
        self,
        path: Path,
        *,
        latch_identity: SafetyLatchIdentity | None = None,
    ) -> None:
        self.path = path
        self.latch_identity = latch_identity or safety_latch_identity(get_settings())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS events_created_idx
                    ON events(created_at DESC);
                CREATE INDEX IF NOT EXISTS events_type_id_idx
                    ON events(type, id DESC);

                CREATE TABLE IF NOT EXISTS proposals (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    targets_json TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS proposals_updated_idx
                    ON proposals(updated_at DESC);

                """
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._initialize_safety_latches(connection)
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _create_safety_latches_v2(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_latches (
                schema_version INTEGER NOT NULL,
                backend TEXT NOT NULL,
                follower_id TEXT NOT NULL,
                leader_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                estopped INTEGER NOT NULL,
                fault TEXT,
                safety_state TEXT NOT NULL,
                torque_unconfirmed INTEGER NOT NULL,
                energized INTEGER NOT NULL DEFAULT 0,
                energized_at TEXT,
                energized_by TEXT,
                PRIMARY KEY (backend, follower_id, leader_id),
                CHECK (schema_version = 2),
                CHECK (backend IN ('real', 'mock', 'sim'))
            )
            """
        )

    def _initialize_safety_latches(self, connection: sqlite3.Connection) -> None:
        """Create v2, or migrate the singleton v1 row conservatively to real."""

        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(safety_latches)").fetchall()
        }
        if not columns:
            self._create_safety_latches_v2(connection)
            return
        expected = {
            "schema_version",
            "backend",
            "follower_id",
            "leader_id",
            "version",
            "updated_at",
            "estopped",
            "fault",
            "safety_state",
            "torque_unconfirmed",
            "energized",
            "energized_at",
            "energized_by",
        }
        if expected.issubset(columns):
            return
        legacy = {
            "id",
            "version",
            "updated_at",
            "estopped",
            "fault",
            "safety_state",
            "torque_unconfirmed",
        }
        if not legacy.issubset(columns):
            raise RuntimeError(
                "safety_latches has an unsupported schema; refusing to guess its identity"
            )
        connection.execute("ALTER TABLE safety_latches RENAME TO safety_latches_v1")
        self._create_safety_latches_v2(connection)
        row = connection.execute(
            """
            SELECT version, updated_at, estopped, fault, safety_state, torque_unconfirmed
            FROM safety_latches_v1 WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return
        identity = self.latch_identity.for_real_backend()
        connection.execute(
            """
            INSERT INTO safety_latches(
                schema_version, backend, follower_id, leader_id, version, updated_at,
                estopped, fault, safety_state, torque_unconfirmed, energized,
                energized_at, energized_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)
            """,
            (
                SAFETY_LATCH_SCHEMA_VERSION,
                identity.backend,
                identity.follower_id,
                identity.leader_id,
                int(row["version"]),
                str(row["updated_at"]),
                int(row["estopped"]),
                row["fault"],
                str(row["safety_state"]),
                int(row["torque_unconfirmed"]),
            ),
        )

    def healthy(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False

    def append_event(
        self,
        event_type: str,
        message: str,
        *,
        level: str = "info",
        source: str = "system",
        detail: dict[str, Any] | None = None,
        durable: bool = False,
    ) -> dict[str, Any]:
        """Append an audit event.

        ``durable=True`` uses SQLite ``synchronous=FULL`` for transitions that
        must reach stable storage before autonomous motion is permitted.
        """

        created_at = iso_now()
        payload = json.dumps(detail or {}, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            if durable:
                connection.execute("PRAGMA synchronous=FULL")
            cursor = connection.execute(
                """
                INSERT INTO events(created_at, type, level, source, message, detail_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (created_at, event_type, level, source, message, payload),
            )
            event_id = cursor.lastrowid
        return {
            "id": event_id,
            "created_at": created_at,
            "timestamp": created_at,
            "type": event_type,
            "level": level,
            "source": source,
            "message": message,
            "detail": detail or {},
        }

    def list_events(self, limit: int = 100, after_id: int | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        where = "WHERE id > ?" if after_id is not None else ""
        params: tuple[Any, ...] = (after_id, limit) if after_id is not None else (limit,)
        query = f"""
            SELECT id, created_at, type, level, source, message, detail_json
            FROM events {where} ORDER BY id DESC LIMIT ?
        """
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._event_from_row(row) for row in rows]

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
        created_at = str(row["created_at"])
        return {
            "id": row["id"],
            "created_at": created_at,
            "timestamp": created_at,
            "type": row["type"],
            "level": row["level"],
            "source": row["source"],
            "message": row["message"],
            "detail": json.loads(row["detail_json"]),
        }

    def calibration_events(self, limit: int = 401) -> list[dict[str, Any]]:
        """Read calibration history without unrelated events evicting older attempts."""
        limit = max(1, min(int(limit), 401))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT id, created_at, type, level, source, message, detail_json
                   FROM events
                   WHERE type IN ('calibration.started', 'calibration.saved', 'calibration.failed')
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._event_from_row(row) for row in reversed(rows)]

    # --- safety latches (safety-B8, docs/audit/v0.1/safety.md) ------------
    def save_latches(self, latches: dict[str, Any], *, version: int) -> bool:
        """Store the safety latches unless a newer snapshot is already stored.

        One identity-scoped row holds the latches and the standalone
        write-ahead ``energized`` intent. The upsert only applies when
        ``version`` is greater than that identity's stored one, so a writer
        that was overtaken can never roll it back. The commit runs with
        ``synchronous=FULL``: a latch has to survive a power cut, not only a
        process crash. Returns True when this snapshot was stored.
        """

        identity = self.latch_identity
        row = (
            SAFETY_LATCH_SCHEMA_VERSION,
            identity.backend,
            identity.follower_id,
            identity.leader_id,
            int(version),
            iso_now(),
            1 if latches.get("estopped") else 0,
            latches.get("fault") or None,
            str(latches.get("safety_state") or "disarmed"),
            1 if latches.get("torque_unconfirmed") else 0,
            1 if latches.get("energized") else 0,
            latches.get("energized_at") or None,
            latches.get("energized_by") or None,
        )
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA synchronous=FULL")
            if identity.backend == "real":
                legacy = connection.execute(
                    """
                    SELECT version FROM safety_latches
                    WHERE backend = 'real' AND follower_id = ? AND leader_id = ?
                    """,
                    (LEGACY_UNSCOPED_DEVICE_ID, LEGACY_UNSCOPED_DEVICE_ID),
                ).fetchone()
                if legacy is not None and int(legacy["version"]) >= int(version):
                    return False
            cursor = connection.execute(
                """
                INSERT INTO safety_latches(
                    schema_version, backend, follower_id, leader_id, version, updated_at,
                    estopped, fault, safety_state, torque_unconfirmed, energized,
                    energized_at, energized_by
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(backend, follower_id, leader_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    version = excluded.version,
                    updated_at = excluded.updated_at,
                    estopped = excluded.estopped,
                    fault = excluded.fault,
                    safety_state = excluded.safety_state,
                    torque_unconfirmed = excluded.torque_unconfirmed,
                    energized = excluded.energized,
                    energized_at = excluded.energized_at,
                    energized_by = excluded.energized_by
                WHERE excluded.version > safety_latches.version
                """,
                row,
            )
            return cursor.rowcount > 0

    def load_latches(self) -> dict[str, Any] | None:
        """The stored safety latches, or None when none were ever saved."""

        identity = self.latch_identity
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT schema_version, backend, follower_id, leader_id, version, updated_at,
                       estopped, fault, safety_state, torque_unconfirmed, energized,
                       energized_at, energized_by
                FROM safety_latches
                WHERE (backend = ? AND follower_id = ? AND leader_id = ?)
                   OR (? = 'real' AND backend = 'real' AND follower_id = ? AND leader_id = ?)
                ORDER BY version DESC
                """,
                (
                    identity.backend,
                    identity.follower_id,
                    identity.leader_id,
                    identity.backend,
                    LEGACY_UNSCOPED_DEVICE_ID,
                    LEGACY_UNSCOPED_DEVICE_ID,
                ),
            ).fetchall()
        if not rows:
            return None
        newest_version = int(rows[0]["version"])
        newest = [row for row in rows if int(row["version"]) == newest_version]
        row = newest[0]
        stored_identity = SafetyLatchIdentity(
            str(row["backend"]), str(row["follower_id"]), str(row["leader_id"])
        )
        if int(row["schema_version"]) != SAFETY_LATCH_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported safety-latch schema version {row['schema_version']!r}")
        legacy_identity = identity.for_real_backend()
        if stored_identity not in {identity, legacy_identity}:
            raise RuntimeError(
                f"safety-latch identity mismatch: stored {stored_identity.as_dict()}, "
                f"expected {identity.as_dict()}"
            )
        estopped = any(bool(item["estopped"]) for item in newest)
        torque_unconfirmed = any(bool(item["torque_unconfirmed"]) for item in newest)
        energized = any(bool(item["energized"]) for item in newest)
        fault = next((item["fault"] for item in newest if item["fault"]), None)
        energized_row = next((item for item in newest if item["energized"]), row)
        return {
            "estopped": estopped,
            "fault": fault,
            "safety_state": "estopped" if estopped else ("fault" if fault else "disarmed"),
            "torque_unconfirmed": torque_unconfirmed,
            "energized": energized,
            "energized_at": energized_row["energized_at"],
            "energized_by": energized_row["energized_by"],
            "version": newest_version,
            "updated_at": str(row["updated_at"]),
            "schema_version": int(row["schema_version"]),
            "identity": identity.as_dict(),
            "stored_identity": stored_identity.as_dict(),
            "legacy_unscoped": stored_identity == legacy_identity,
        }

    def create_proposal(
        self,
        *,
        objective: str,
        skill: str,
        constraints: dict[str, Any],
        targets: dict[str, float],
        risk: str,
    ) -> dict[str, Any]:
        proposal_id = uuid.uuid4().hex[:16]
        now = iso_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO proposals(
                    id, created_at, updated_at, objective, skill, status, risk,
                    constraints_json, targets_json, result_json
                ) VALUES (?, ?, ?, ?, ?, 'proposed', ?, ?, ?, '{}')
                """,
                (
                    proposal_id,
                    now,
                    now,
                    objective,
                    skill,
                    risk,
                    json.dumps(constraints, sort_keys=True),
                    json.dumps(targets, sort_keys=True),
                ),
            )
        return self.get_proposal(proposal_id)  # type: ignore[return-value]

    def list_proposals(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM proposals ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._proposal_from_row(row) for row in rows]

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
        return self._proposal_from_row(row) if row is not None else None

    def transition_proposal(
        self,
        proposal_id: str,
        *,
        expected: tuple[str, ...],
        status: str,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Atomically transition a proposal, returning ``None`` on conflict."""

        now = iso_now()
        placeholders = ",".join("?" for _ in expected)
        args: tuple[Any, ...] = (
            status,
            now,
            json.dumps(result or {}, sort_keys=True),
            proposal_id,
            *expected,
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE proposals
                SET status = ?, updated_at = ?, result_json = ?
                WHERE id = ? AND status IN ({placeholders})
                """,
                args,
            )
            if cursor.rowcount != 1:
                return None
        return self.get_proposal(proposal_id)

    @staticmethod
    def _proposal_from_row(row: sqlite3.Row) -> dict[str, Any]:
        targets = json.loads(row["targets_json"])
        constraints = json.loads(row["constraints_json"])
        result = json.loads(row["result_json"])
        if targets:
            steps = [
                "review captured robot and safety state",
                "validate live lease, arm state, limits, and watchdog",
                f"issue bounded {row['skill']} target through the shared safety gate",
                "observe measured state; watchdog stops a stale source",
            ]
        else:
            steps = [
                "inspect captured robot, joint, and safety state",
                "return an observation-only proposal with no motor target",
            ]
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "title": row["objective"],
            "objective": row["objective"],
            "description": f"{row['skill']} proposal; explicit approval is required",
            "skill": row["skill"],
            "status": row["status"],
            "risk": row["risk"],
            "constraints": constraints,
            "targets": targets,
            "steps": steps,
            "result": result,
        }


#: The control store's file in ``<data_dir>``. Standalone CLI commands read the
#: safety latches from it too (so101.bringup.standalone).
CONTROL_STORE_FILENAME = "control_plane.sqlite3"

_store: ControlStore | None = None
_store_lock = threading.Lock()


def control_store_path(settings: Any) -> Path:
    return Path(settings.data_dir) / CONTROL_STORE_FILENAME


def get_store() -> ControlStore:
    global _store
    settings = get_settings()
    path = control_store_path(settings)
    identity = safety_latch_identity(settings)
    with _store_lock:
        if _store is None or _store.path != path or _store.latch_identity != identity:
            _store = ControlStore(path, latch_identity=identity)
        return _store


def reset_store() -> None:
    """Drop the process cache without deleting durable data (mainly for tests)."""

    global _store
    with _store_lock:
        _store = None


# --- standalone safety snapshots ------------------------------------------------------


def read_standalone_latch_marker(
    path: Path,
    *,
    identity: SafetyLatchIdentity | None,
    allow_legacy_real: bool = False,
) -> Any:
    """Read one identity's marker through RobotManager's validated parser."""

    from .robot_manager import _read_latch_marker

    return _read_latch_marker(
        path,
        expected_identity=identity,
        allow_legacy_real=allow_legacy_real,
    )


def read_identity_latch_markers(path: Path, identity: SafetyLatchIdentity) -> list[Any]:
    """Scoped v2 marker plus any legacy marker applicable to this identity."""

    markers = []
    scoped = read_standalone_latch_marker(
        latch_marker_path(path, identity),
        identity=identity,
    )
    if scoped is not None:
        markers.append(scoped)
    if identity.backend == "real":
        legacy = read_standalone_latch_marker(
            legacy_latch_marker_path(path),
            identity=None,
            allow_legacy_real=True,
        )
        if legacy is not None and (legacy.identity is None or legacy.identity == identity):
            markers.append(legacy)
    return markers


def combine_identity_latch_markers(markers: list[Any], *, saved_version: int = 0) -> Any:
    """One conservative view without letting an unreadable file hide known latches."""

    known = [
        marker
        for marker in markers
        if marker.version is not None and marker.version > saved_version
    ]
    unreadable = [marker for marker in markers if marker.version is None]
    if not known and not unreadable:
        return None
    newest_version = max((marker.version for marker in known), default=None)
    newest = [marker for marker in known if marker.version == newest_version]
    base = unreadable[0] if unreadable else newest[0]
    latches: dict[str, Any] = {}
    for marker in newest:
        source = marker.latches or {}
        latches["estopped"] = bool(latches.get("estopped")) or bool(source.get("estopped"))
        latches["torque_unconfirmed"] = bool(latches.get("torque_unconfirmed")) or bool(
            source.get("torque_unconfirmed")
        )
        latches["energized"] = bool(latches.get("energized")) or bool(source.get("energized"))
        if source.get("fault") and not latches.get("fault"):
            latches["fault"] = source.get("fault")
        if source.get("energized"):
            latches["energized_at"] = source.get("energized_at")
            latches["energized_by"] = source.get("energized_by")
    if unreadable:
        # Its exact latch is unknowable. Preserve known E-stop/fault data and
        # add the strongest fault-resettable torque/energized assumptions.
        latches["torque_unconfirmed"] = True
        latches["energized"] = True
    latches["safety_state"] = "estopped" if latches.get("estopped") else "fault"
    errors = [str(marker.error) for marker in [*newest, *unreadable] if marker.error]
    error = "; ".join(errors) or None
    return type(base)(
        base.path,
        None if unreadable else newest_version,
        latches,
        error,
        base.identity,
    )


def newest_identity_latch_marker(path: Path, identity: SafetyLatchIdentity) -> Any:
    """Compatibility wrapper returning the conservative combined active view."""

    return combine_identity_latch_markers(read_identity_latch_markers(path, identity))


def merge_standalone_latches(saved: dict[str, Any] | None, marker: Any) -> dict[str, Any] | None:
    """Apply the server's restart rules to a saved row and fallback marker."""

    saved_version = int(saved.get("version") or 0) if saved is not None else 0
    if marker is None or (marker.version is not None and marker.version <= saved_version):
        return saved

    from .robot_manager import RobotManager

    merged = dict(saved or {})
    marker_latches = marker.latches
    merged["estopped"] = bool(merged.get("estopped")) or bool(
        marker_latches and marker_latches.get("estopped")
    )
    merged["torque_unconfirmed"] = bool(merged.get("torque_unconfirmed")) or (
        marker_latches is None or bool(marker_latches.get("torque_unconfirmed"))
    )
    merged["energized"] = bool(merged.get("energized")) or (
        marker_latches is None or bool(marker_latches.get("energized"))
    )
    if marker_latches is not None and marker_latches.get("energized"):
        merged["energized_at"] = marker_latches.get("energized_at")
        merged["energized_by"] = marker_latches.get("energized_by")
    merged["fault"] = RobotManager._latch_marker_fault(marker, merged.get("fault") or None)
    if marker.version is not None:
        merged["version"] = max(saved_version, marker.version)
    merged["_unsaved_marker"] = {
        "path": str(marker.path),
        "version": marker.version,
        "error": marker.error,
        "unreadable": marker_latches is None,
    }
    return merged


_standalone_version_guard = threading.Lock()
_last_standalone_version = 0


def _next_standalone_latch_version(saved: dict[str, Any] | None, marker: Any) -> int:
    global _last_standalone_version

    saved_version = int(saved.get("version") or 0) if saved is not None else 0
    marker_version = marker.version if marker is not None and marker.version is not None else 0
    with _standalone_version_guard:
        _last_standalone_version = max(
            time.time_ns() // 1_000,
            _last_standalone_version + 1,
            saved_version + 1,
            marker_version + 1,
        )
        return _last_standalone_version


def persist_standalone_torque_off_failure(
    settings: Any, exc: Exception, *, device_name: str
) -> dict[str, Any]:
    """Durably latch a standalone motor device whose torque-off was not confirmed."""

    return persist_standalone_fault(
        settings,
        exc,
        device_name=device_name,
        torque_unconfirmed=True,
        energized=True,
    )


def persist_standalone_fault(
    settings: Any,
    exc: BaseException,
    *,
    device_name: str,
    torque_unconfirmed: bool,
    energized: bool,
) -> dict[str, Any]:
    """Persist a visible standalone teardown/Halt fault for this identity."""

    now = datetime.now(timezone.utc).isoformat()

    def update(merged: dict[str, Any]) -> dict[str, Any]:
        fault = f"standalone {device_name} {exc}"
        previous_fault = merged.get("fault")
        if previous_fault and str(exc) not in str(previous_fault):
            fault += f" (previous fault: {previous_fault})"
        energized_now = bool(merged.get("energized")) or energized
        return {
            **_standalone_latch_snapshot(merged),
            "fault": fault,
            "safety_state": "estopped" if merged.get("estopped") else "fault",
            "torque_unconfirmed": bool(merged.get("torque_unconfirmed")) or torque_unconfirmed,
            "energized": energized_now,
            "energized_at": (merged.get("energized_at") or now) if energized_now else None,
            "energized_by": (
                merged.get("energized_by") or f"standalone {device_name}" if energized_now else None
            ),
        }

    return _persist_standalone_latch_mutation(
        settings,
        update=update,
        satisfied=lambda state: (
            bool(state.get("fault"))
            and (not torque_unconfirmed or bool(state.get("torque_unconfirmed")))
            and (
                not energized
                or bool(state.get("energized"))
                or bool(state.get("torque_unconfirmed"))
            )
        ),
        latching=True,
        failure_context=(
            "torque-off was unconfirmed"
            if torque_unconfirmed
            else "standalone Halt failure could not be saved"
        ),
    )


def persist_standalone_energized_intent(
    settings: Any,
    *,
    purpose: str,
    device_name: str,
) -> dict[str, Any]:
    """Write-ahead intent: durable before a standalone follower can energize."""

    written_at = datetime.now(timezone.utc).isoformat()

    def update(merged: dict[str, Any]) -> dict[str, Any]:
        return {
            **_standalone_latch_snapshot(merged),
            "energized": True,
            "energized_at": written_at,
            "energized_by": purpose or f"standalone {device_name}",
        }

    return _persist_standalone_latch_mutation(
        settings,
        update=update,
        satisfied=lambda state: bool(state.get("energized")),
        latching=True,
        failure_context="standalone energized intent could not be saved",
    )


def clear_standalone_energized_intent(settings: Any) -> dict[str, Any]:
    """Persist confirmed torque-off without weakening any concurrent latch."""

    def update(merged: dict[str, Any]) -> dict[str, Any]:
        return {
            **_standalone_latch_snapshot(merged),
            "energized": False,
            "energized_at": None,
            "energized_by": None,
        }

    return _persist_standalone_latch_mutation(
        settings,
        update=update,
        satisfied=lambda state: not bool(state.get("energized")),
        latching=False,
        failure_context="confirmed torque-off could not clear the standalone energized intent",
    )


def _standalone_latch_snapshot(latches: dict[str, Any] | None) -> dict[str, Any]:
    source = latches or {}
    return {
        "estopped": bool(source.get("estopped")),
        "fault": source.get("fault") or None,
        "safety_state": str(source.get("safety_state") or "disarmed"),
        "torque_unconfirmed": bool(source.get("torque_unconfirmed")),
        "energized": bool(source.get("energized")),
        "energized_at": source.get("energized_at") or None,
        "energized_by": source.get("energized_by") or None,
    }


def _persist_standalone_latch_mutation(
    settings: Any,
    *,
    update: Callable[[dict[str, Any]], dict[str, Any]],
    satisfied: Callable[[dict[str, Any]], bool],
    latching: bool,
    failure_context: str,
) -> dict[str, Any]:
    """CAS one identity's snapshot; latch failures fall back to its marker."""

    identity = safety_latch_identity(settings)
    path = control_store_path(settings)
    fallback = latch_marker_path(path, identity)
    marker = newest_identity_latch_marker(path, identity)
    saved: dict[str, Any] | None = None
    store: ControlStore | None = None
    store_error: Exception | None = None
    try:
        store = ControlStore(path, latch_identity=identity)
        saved = store.load_latches()
    except Exception as error:
        store_error = error

    latches = update(merge_standalone_latches(saved, marker) or {})
    version = _next_standalone_latch_version(saved, marker)
    if store is not None:
        for _attempt in range(2):
            try:
                if store.save_latches(latches, version=version):
                    result = {**latches, "version": version, "identity": identity.as_dict()}
                    if (
                        marker is not None
                        and marker.version is not None
                        and marker.version < version
                        and marker.path == fallback
                    ):
                        try:
                            fallback.unlink(missing_ok=True)
                        except OSError as exc:
                            log.warning(
                                "could not remove obsolete latch marker %s: %s",
                                fallback,
                                exc,
                            )
                    return result
                store_error = RuntimeError(
                    f"a newer safety-latch snapshot displaced standalone version {version}"
                )
            except Exception as error:
                store_error = error
                break
            try:
                newest = store.load_latches()
            except Exception as error:
                store_error = error
                break
            if newest is not None and satisfied(newest):
                return newest
            saved = newest
            latches = update(merge_standalone_latches(saved, marker) or {})
            version = _next_standalone_latch_version(saved, marker)

    assert store_error is not None
    if not latching:
        raise RuntimeError(
            f"{failure_context} in {path} ({store_error}); the durable intent remains set, "
            "so the next process will fail closed"
        ) from store_error
    payload = {
        "kind": LATCH_MARKER_KIND,
        "schema_version": LATCH_MARKER_SCHEMA_VERSION,
        "identity": identity.as_dict(),
        "version": version,
        "latches": latches,
        "error": f"{type(store_error).__name__}: {store_error}",
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        atomic_write_json(fallback, payload)
    except Exception as marker_error:
        raise RuntimeError(
            f"{failure_context} and neither {path} nor its fallback marker could save "
            f"the safety state ({store_error}; {marker_error}); cut physical power"
        ) from marker_error
    log.critical(
        "%s; wrote fallback latch marker %s after %s",
        failure_context,
        fallback,
        store_error,
    )
    return merge_standalone_latches(
        saved,
        newest_identity_latch_marker(path, identity),
    ) or {**latches, "version": version}
