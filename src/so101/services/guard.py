"""Cross-origin and Host guard for the control plane (contract C1).

A browser enforces CORS only on what a page can *read*. A cross-origin "simple"
request (form POST, ``fetch(..., {mode: "no-cors"})``, text/plain body) still
executes on the server. This pure-ASGI middleware therefore decides, before any
routing or body parsing, whether a request may change state:

* Host header: must name this machine (localhost, 127.0.0.1, ::1, a non-wildcard
  bind host) or an entry of ``SO101_ALLOWED_HOSTS``. Defeats DNS rebinding.
* POST/PUT/PATCH/DELETE and every WebSocket handshake: rejected when
  ``Sec-Fetch-Site: cross-site``, ``Origin: null``, or an ``Origin`` that is
  neither in ``SO101_CORS_ORIGINS`` nor same-origin with the Host. Requests that
  carry neither header (CLI, curl, httpx, TestClient) are allowed.
* ``POST /api/robot/estop``, ``/stop`` and ``/halt`` skip both checks: stopping
  must never be blocked. Nor may a body delay one: a stop route that reads its
  body uses :func:`read_stop_body`, which takes at most ``STOP_BODY_MAX_BYTES``
  within ``STOP_BODY_READ_TIMEOUT_S`` and otherwise treats it as absent.

WebSocket violations close with 1008 before ``accept()``, so the route handler
(``/ws/telemetry``) never runs.

Fixes safety-B1 and safety-B3 (the cross-origin half) and safety-B4 (Host and
WebSocket Origin) from docs/audit/v0.1/safety.md, plus F1 from
docs/audit/v0.1/frontend.md. Threat model: docs/control-plane-security.md.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from ..config import Settings, get_settings
from ..hardware.backend import current_backend

logger = logging.getLogger(__name__)

STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Stopping must never be blocked: these POSTs skip the Host and Origin checks
# (and StrictJSONMiddleware's body check). POST /api/robot/collapse is not one
# of them: it commands motion (design doc §5).
STOP_PATHS = frozenset({"/api/robot/estop", "/api/robot/stop", "/api/robot/halt"})
# A stop route that reads its body (only the halt, for operator_id and reason;
# E-stop and the legacy stop read none) reads at most this much, for at most
# this long. A longer or slower body counts as no body, and the stop happens
# anyway: a stop never waits on an upload, and never buffers one.
STOP_BODY_MAX_BYTES = 4096
STOP_BODY_READ_TIMEOUT_S = 0.1
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "::1")
WILDCARD_BIND_HOSTS = frozenset({"", "0.0.0.0", "::", "[::]", "*"})
REJECTED_DETAIL = "cross-origin request rejected"
UNTRUSTED_HOST_DETAIL = "untrusted Host header; add the hostname to SO101_ALLOWED_HOSTS"
WS_POLICY_VIOLATION = 1008
E2E_IDENTITY_PATH = "/api/e2e/identity"
E2E_TOKEN_HEADER = b"x-so101-e2e-token"
E2E_REJECTED_DETAIL = "installed E2E identity rejected"
E2E_REQUIRED_DETAIL = "installed E2E identity token required"
_DEFAULT_PORTS = {"http": 80, "ws": 80, "https": 443, "wss": 443}

Origin = tuple[str, str, int]  # (scheme, hostname, port)


def _parse_port(text: str) -> int:
    if not text.isdigit() or len(text) > 5:
        raise ValueError(f"invalid port {text!r}")
    port = int(text)
    if port > 65535:
        raise ValueError(f"invalid port {text!r}")
    return port


def parse_host_header(value: str) -> tuple[str, int | None] | None:
    """Split a Host header into (lowercase hostname, port). None if malformed.

    IPv6 literals must be bracketed (``[::1]:8000``), as RFC 3986 requires.
    Starlette's TrustedHostMiddleware splits on the first ``:`` instead, which
    can never match ``::1``; that is one reason this guard does its own parsing.
    """
    value = value.strip().lower()
    try:
        if value.startswith("["):
            end = value.find("]")
            if end < 0:
                return None
            host, rest = value[1:end], value[end + 1 :]
            if rest and not rest.startswith(":"):
                return None
            port = _parse_port(rest[1:]) if len(rest) > 1 else None
        else:
            if value.count(":") > 1:
                return None
            host, _, port_text = value.partition(":")
            port = _parse_port(port_text) if port_text else None
    except ValueError:
        return None
    return (host, port) if host else None


def parse_origin(value: str) -> Origin | None:
    """Normalize a serialized origin. None for ``null`` or anything malformed."""
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return None
    if parts.path or parts.query or parts.fragment or "@" in parts.netloc:
        return None
    return scheme, parts.hostname.lower(), port if port is not None else _DEFAULT_PORTS[scheme]


def _normalize_host_entry(entry: str) -> str:
    """``Robot.LAN:8000`` -> ``robot.lan``; ``[fd00::5]`` and ``fd00::5`` -> ``fd00::5``."""
    entry = entry.strip().lower()
    if entry.count(":") > 1 and not entry.startswith("["):
        return entry  # bare IPv6 literal
    parsed = parse_host_header(entry)
    return parsed[0] if parsed else ""


@dataclass(frozen=True)
class GuardPolicy:
    hosts: frozenset[str]
    host_suffixes: frozenset[str]
    origins: frozenset[Origin]

    @classmethod
    def from_settings(cls, settings: Settings) -> GuardPolicy:
        hosts = set(DEFAULT_ALLOWED_HOSTS)
        if settings.host.strip().lower() not in WILDCARD_BIND_HOSTS:
            bind_host = _normalize_host_entry(settings.host)
            if bind_host:
                hosts.add(bind_host)
        suffixes: set[str] = set()
        for raw in settings.allowed_hosts.split(","):
            entry = raw.strip().lower()
            if not entry:
                continue
            if entry.startswith("*.") and "*" not in entry[2:] and len(entry) > 2:
                suffixes.add(entry[1:])
                continue
            if "*" in entry:
                raise ValueError(
                    f"SO101_ALLOWED_HOSTS entry {raw.strip()!r} is not allowed: list each "
                    "hostname or IP explicitly ('*.example.lan' matches subdomains only)"
                )
            host = _normalize_host_entry(entry)
            if not host:
                raise ValueError(f"SO101_ALLOWED_HOSTS entry {raw.strip()!r} is not a hostname")
            hosts.add(host)
        # A "*" CORS entry is ignored here on purpose: cross-origin writes stay
        # blocked even if someone widens CORS for reads.
        origins = frozenset(
            origin
            for origin in (parse_origin(o.rstrip("/")) for o in settings.allowed_origins)
            if origin
        )
        return cls(frozenset(hosts), frozenset(suffixes), origins)

    def host_allowed(self, host: str) -> bool:
        return host in self.hosts or any(host.endswith(s) for s in self.host_suffixes)

    def check(
        self,
        headers: Iterable[tuple[bytes, bytes]],
        *,
        scheme: str,
        check_origin: bool,
    ) -> tuple[int, str] | None:
        """Return (status, detail) for a violation, or None when allowed."""
        host_values: list[bytes] = []
        origins: list[bytes] = []
        cross_site = False
        for key, value in headers:
            if key == b"host":
                host_values.append(value)
            elif key == b"origin":
                origins.append(value)
            elif key == b"sec-fetch-site" and value.strip().lower() == b"cross-site":
                cross_site = True
        host = (
            parse_host_header(host_values[0].decode("latin-1")) if len(host_values) == 1 else None
        )
        if host is None or not self.host_allowed(host[0]):
            return 400, UNTRUSTED_HOST_DETAIL
        if not check_origin:
            return None
        if cross_site or len(origins) > 1:
            return 403, REJECTED_DETAIL
        if not origins:
            return None
        origin = parse_origin(origins[0].decode("latin-1"))
        if origin is None:  # "null", opaque or malformed
            return 403, REJECTED_DETAIL
        if origin in self.origins:
            return None
        hostname, port = host
        if port is None:
            port = _DEFAULT_PORTS.get(scheme, 80)
        origin_scheme = "https" if scheme in {"https", "wss"} else "http"
        if origin == (origin_scheme, hostname, port):
            return None
        return 403, REJECTED_DETAIL


async def read_stop_body(request: Request) -> bytes:
    """A stop route's body, bounded; ``b""`` when it is too long, too slow or broken.

    A declared ``Content-Length`` above ``STOP_BODY_MAX_BYTES`` is not read at
    all. Otherwise the stream is read chunk by chunk until it ends, passes the
    cap or ``STOP_BODY_READ_TIMEOUT_S`` runs out. Memory stays bounded by the
    cap plus one chunk; whatever the client still sends is never read. Never
    raises: a stop is never refused over its body.
    """

    declared = request.headers.get("content-length")
    if declared is not None:
        value = declared.strip().encode("ascii", "ignore")
        if value.isdigit():
            # Compare decimal text instead of calling int(): Python rejects
            # integers longer than sys.int_max_str_digits, and a malformed
            # stop request must never prevent the Halt itself.
            value = value.lstrip(b"0") or b"0"
            limit = str(STOP_BODY_MAX_BYTES).encode("ascii")
            if len(value) > len(limit) or (len(value) == len(limit) and value > limit):
                return b""
    chunks: list[bytes] = []

    async def read() -> bytes:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > STOP_BODY_MAX_BYTES:
                return b""
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        return await asyncio.wait_for(read(), STOP_BODY_READ_TIMEOUT_S)
    except Exception:  # timeout, client gone, anything: no body
        return b""


class GuardMiddleware:
    """Pure ASGI: runs before routing, body parsing and WebSocket accept."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._policy: GuardPolicy | None = None
        self._policy_settings: Settings | None = None

    def policy(self) -> GuardPolicy:
        # Rebuilt only when the cached Settings object changes (tests clear it).
        settings = get_settings()
        if self._policy is None or settings is not self._policy_settings:
            self._policy = GuardPolicy.from_settings(settings)
            self._policy_settings = settings
        return self._policy

    @staticmethod
    def _e2e_verdict(
        settings: Settings,
        headers: Iterable[tuple[bytes, bytes]],
        *,
        requires_token: bool,
    ) -> tuple[int, str] | None:
        """Validate the opt-in installed-E2E identity header.

        With no configured token and no header this returns ``None`` for every
        production request. A supplied header is never ignored: it is accepted
        only when exactly one value matches and this is a non-real backend.
        Every state-changing request, WebSocket and identity probe requires
        the header whenever the token is configured. The token itself is never
        returned or logged.
        """

        values = [value for key, value in headers if key.lower() == E2E_TOKEN_HEADER]
        configured = settings.e2e_token.get_secret_value()
        if not values:
            if requires_token and configured:
                return 403, E2E_REQUIRED_DETAIL
            return None
        if len(values) != 1 or not configured or current_backend(settings) == "real":
            return 403, E2E_REJECTED_DETAIL
        try:
            supplied = values[0].decode("utf-8")
        except UnicodeDecodeError:
            return 403, E2E_REJECTED_DETAIL
        if not secrets.compare_digest(supplied, configured):
            return 403, E2E_REJECTED_DETAIL
        return None

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        method: str,
        path: str,
        status: int,
        detail: str,
    ) -> None:
        # Never include request headers here: the E2E token is secret.
        logger.warning("guard rejected %s %s (%d): %s", method, path, status, detail)
        if scope["type"] == "websocket":
            await WebSocketClose(code=WS_POLICY_VIOLATION, reason=detail)(scope, receive, send)
            return
        await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        is_ws = kind == "websocket"
        method = "GET" if is_ws else scope["method"]
        path = scope.get("path", "")
        headers = tuple(scope.get("headers", ()))
        # Stopping is unconditional. In particular, a stray test-only identity
        # setting or header must never disable E-stop/Halt on a real server.
        # The worst a mismatched E2E client can do here is stop the arm, which
        # is the same deliberate trade-off as the Origin/Host exemption.
        if method == "POST" and path.rstrip("/") in STOP_PATHS:
            await self.app(scope, receive, send)
            return
        e2e_verdict = self._e2e_verdict(
            get_settings(),
            headers,
            requires_token=(
                is_ws or method in STATE_CHANGING_METHODS or path.rstrip("/") == E2E_IDENTITY_PATH
            ),
        )
        if e2e_verdict is not None:
            status, detail = e2e_verdict
            await self._reject(
                scope,
                receive,
                send,
                method=method,
                path=path,
                status=status,
                detail=detail,
            )
            return
        verdict = self.policy().check(
            headers,
            scheme=scope.get("scheme", "http"),
            check_origin=is_ws or method in STATE_CHANGING_METHODS,
        )
        if verdict is None:
            await self.app(scope, receive, send)
            return
        status, detail = verdict
        await self._reject(
            scope,
            receive,
            send,
            method=method,
            path=path,
            status=status,
            detail=detail,
        )
