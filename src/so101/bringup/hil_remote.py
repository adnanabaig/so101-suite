"""``so101 hil`` against a running Suite server: HilService's calls over ``/api/hil``.

A running server owns the robot, so ``so101 hil run`` drives the server's HIL
service instead of opening anything itself. :class:`HttpHilApi` has the same
method names and return types as :class:`~so101.bringup.hil.HilService`, and maps
HTTP errors back to the same exceptions (404 HilNotFound, 409 HilConflict,
422 HilInvalid with its ``problems``).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .hil import MAX_PROBE_S, HilConflict, HilError, HilInvalid, HilNotFound
from .schemas import (
    AttestRequest,
    HilReport,
    HilRun,
    HilRunSummary,
    HilStepDefinition,
    HilStepRecord,
    ProbeResult,
    StartHilRunRequest,
)

#: Seconds to wait for ordinary calls; a probe may block for its whole window.
REQUEST_TIMEOUT_S = 30.0
PROBE_TIMEOUT_S = MAX_PROBE_S + 60.0


class HilUnavailable(HilError):
    """The server did not answer."""

    status_code = 503


def _segment(value: str) -> str:
    return quote(value, safe="")


def _validation_problem(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
    message = str(item.get("msg", "invalid"))
    return f"{location}: {message}" if location else message


def error_from_response(response: httpx.Response) -> HilError:
    """The HilError a non-2xx ``/api/hil`` reply stands for."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    problems: list[str] = []
    if isinstance(detail, dict):
        message = str(detail.get("message") or f"HTTP {response.status_code}")
        problems = [str(problem) for problem in detail.get("problems") or ()]
    elif isinstance(detail, list):  # FastAPI request validation
        message = "the server rejected the request"
        problems = [_validation_problem(item) for item in detail]
    elif detail:
        message = str(detail)
    else:
        message = f"HTTP {response.status_code}: {response.text[:200]}"
    kinds: dict[int, type[HilError]] = {404: HilNotFound, 409: HilConflict, 422: HilInvalid}
    error = kinds.get(response.status_code, HilError)(message, problems)
    error.status_code = response.status_code
    return error


class HttpHilApi:
    """The HIL service of the server at ``base_url``."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
        probe_timeout_s: float = PROBE_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(trust_env=False)
        self._owns_client = client is None
        self._timeout = timeout_s
        self._probe_timeout = probe_timeout_s

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _call(
        self, method: str, path: str, *, body: Any = None, timeout_s: float | None = None
    ) -> Any:
        url = f"{self.base_url}/api/hil{path}"
        try:
            response = self._client.request(
                method, url, json=body, timeout=timeout_s or self._timeout
            )
        except httpx.HTTPError as exc:
            raise HilUnavailable(
                f"the server at {self.base_url} did not answer: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise error_from_response(response)
        return response.json()

    def steps(self) -> list[HilStepDefinition]:
        return [HilStepDefinition.model_validate(item) for item in self._call("GET", "/steps")]

    def list_runs(self) -> list[HilRunSummary]:
        return [HilRunSummary.model_validate(item) for item in self._call("GET", "/runs")]

    def start_run(self, request: StartHilRunRequest) -> HilRun:
        body = request.model_dump(mode="json")
        return HilRun.model_validate(self._call("POST", "/runs", body=body))

    def get_run(self, run_id: str) -> HilRun:
        return HilRun.model_validate(self._call("GET", f"/runs/{_segment(run_id)}"))

    def probe(self, run_id: str, step_id: str, duration_s: float | None = None) -> ProbeResult:
        path = f"/runs/{_segment(run_id)}/steps/{_segment(step_id)}/probe"
        body = {"duration_s": duration_s} if duration_s is not None else None
        payload = self._call("POST", path, body=body, timeout_s=self._probe_timeout)
        return ProbeResult.model_validate(payload)

    def attest(self, run_id: str, step_id: str, request: AttestRequest) -> HilStepRecord:
        path = f"/runs/{_segment(run_id)}/steps/{_segment(step_id)}/attest"
        payload = self._call("POST", path, body=request.model_dump(mode="json"))
        return HilStepRecord.model_validate(payload)

    def finalize(self, run_id: str) -> HilRun:
        return HilRun.model_validate(self._call("POST", f"/runs/{_segment(run_id)}/finalize"))

    def report(self, run_id: str) -> HilReport:
        return HilReport.model_validate(self._call("GET", f"/runs/{_segment(run_id)}/report"))
