"""Non-finite and overflowing numbers become 422, never 500.

Python's ``json.loads`` (which FastAPI uses) accepts ``NaN``, ``Infinity`` and
``-Infinity`` and turns ``1e309`` into ``inf``. A non-finite value that then
fails validation crashes FastAPI's default 422 handler, because the response
encoder refuses to echo it back. A finite but huge value such as
``sent_at=1e300`` makes ``datetime.fromtimestamp`` raise OverflowError, which
Pydantic does not convert into a validation error.

Three layers, all global so no route can forget them:

* ``StrictJSONMiddleware`` buffers at most ``JSON_BODY_MAX_BYTES`` from every
  body FastAPI may parse as JSON, rejects a larger body with 413, and rejects
  non-finite constants and out-of-range numbers with 422.
* An OverflowError handler answers 422.
* The RequestValidationError handler renders non-finite inputs as strings.

safety-B11 (docs/audit/v0.1/safety.md) and sim-B19 (docs/audit/v0.1/sim.md).
"""

from __future__ import annotations

import email.message
import json
import math
import sys
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .guard import STOP_PATHS

NON_FINITE_MSG = "JSON body contains a non-finite number (NaN, Infinity or out of range)"
# Request models in the Suite are small control messages. One MiB leaves ample
# room for them while putting a fixed ceiling on StrictJSONMiddleware's copy of
# a body. This is deliberately a product constant, not an SO101_* setting: an
# operator must not accidentally remove a control-plane resource bound.
JSON_BODY_MAX_BYTES = 1024 * 1024
JSON_BODY_TOO_LARGE_MSG = f"JSON body exceeds the {JSON_BODY_MAX_BYTES}-byte limit"


class _NonFinite(ValueError):
    pass


def _reject_constant(name: str) -> float:
    raise _NonFinite(name)


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _NonFinite(text)
    return value


def _float_range_int(text: str) -> int:
    value = int(text)
    if abs(value) > sys.float_info.max:
        raise _NonFinite(text[:32])
    return value


def contains_non_finite(body: bytes) -> bool:
    """True if ``body`` is JSON holding NaN/Infinity or a number beyond float range.

    Malformed JSON returns False; FastAPI reports that itself (json_invalid).
    """
    try:
        json.loads(
            body,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            parse_int=_float_range_int,
        )
    except _NonFinite:
        return True
    except (ValueError, RecursionError):
        return False
    return False


def _may_be_json_body(scope: Scope) -> bool:
    """True if FastAPI may parse this request's body as JSON.

    That is application/json or application/*+json, as FastAPI checks it, or
    no Content-Type at all. FastAPI parses a body with no Content-Type as JSON
    when ``strict_content_type`` is off, and versions without that setting
    always do; pyproject still allows them. Such a body is scanned only if it
    parses as JSON, so other bodies pass unchanged.
    """
    for key, value in scope.get("headers", ()):
        if key == b"content-type":
            message = email.message.Message()
            message["content-type"] = value.decode("latin-1")
            if message.get_content_maintype() != "application":
                return False
            subtype = message.get_content_subtype()
            return subtype == "json" or subtype.endswith("+json")
    return True


def _content_length_exceeds_limit(scope: Scope) -> bool:
    """Whether any well-formed Content-Length is over the JSON body limit.

    Compare decimal text rather than calling ``int`` so even an adversarially
    long header cannot hit Python's integer-string conversion limit. Malformed
    values are left to the ASGI server; the cumulative streaming check below
    remains authoritative.
    """

    limit = str(JSON_BODY_MAX_BYTES).encode("ascii")
    for key, raw_value in scope.get("headers", ()):
        if key != b"content-length":
            continue
        value = raw_value.strip()
        if not value.isdigit():
            continue
        value = value.lstrip(b"0") or b"0"
        if len(value) > len(limit) or (len(value) == len(limit) and value > limit):
            return True
    return False


async def _reject_too_large(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse({"detail": JSON_BODY_TOO_LARGE_MSG}, status_code=413)
    await response(scope, receive, send)


class StrictJSONMiddleware:
    """Bound JSON request bodies, reject non-finite numbers, replay the rest."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _may_be_json_body(scope):
            await self.app(scope, receive, send)
            return
        if scope.get("method") == "POST" and scope.get("path", "").rstrip("/") in STOP_PATHS:
            # A stop is never refused over its body: E-stop and the legacy stop
            # read none, and the halt parses its own leniently.
            await self.app(scope, receive, send)
            return
        if _content_length_exceeds_limit(scope):
            # Reject from headers without asking the server for any body bytes.
            await _reject_too_large(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, _replay(message, receive), send)  # client went away
                return
            chunk = message.get("body", b"")
            if len(chunk) > JSON_BODY_MAX_BYTES - len(body):
                # Do not copy the chunk that crosses the bound and do not read
                # any later messages from the client.
                await _reject_too_large(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        payload = bytes(body)
        if payload and contains_non_finite(payload):
            response = JSONResponse(
                {"detail": [{"type": "json_invalid", "loc": ["body"], "msg": NON_FINITE_MSG}]},
                status_code=422,
            )
            await response(scope, receive, send)
            return
        replay: Message = {"type": "http.request", "body": payload, "more_body": False}
        await self.app(scope, _replay(replay, receive), send)


def _replay(message: Message, receive: Receive) -> Receive:
    pending: Message | None = message

    async def replay_receive() -> Message:
        nonlocal pending
        if pending is not None:
            message = pending
            pending = None
            return message
        return await receive()

    return replay_receive


def _finite_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)  # "nan", "inf", "-inf"
    if isinstance(value, dict):
        return {key: _finite_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_safe(item) for item in value]
    return value


async def _validation_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    detail = _finite_safe(jsonable_encoder(exc.errors()))
    return JSONResponse({"detail": detail}, status_code=422)


async def _overflow_error(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"numeric value out of range: {exc}"}, status_code=422)


def install_numeric_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(OverflowError, _overflow_error)
