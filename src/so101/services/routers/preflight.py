"""Preflight over HTTP: the checks behind ``so101 preflight``, run inside the server.

The server owns the serial buses, so this path never opens a port on its own:
RobotManager lends each idle bus read-only under its bus lock
(``run_idle_bus_diagnostic``), or refuses while the robot is armed, halted,
collapsing, running a loop, calibrating or E-stopped. On the mock backend the
lent bus is synthetic (a rehearsal). Every report is labeled with its backend
and ``rehearsal`` flag. ``POST /api/preflight/bus`` returns the raw bus facts;
``so101 preflight`` uses it while a server runs.

Running preflight is a POST (``POST /api/preflight``, ``POST /api/preflight/bus``):
it reads the motor buses and ``/api/preflight`` writes a report, so it must pass
the Origin guard (``services/guard.py`` checks state-changing methods only). A GET
to either path is answered 405 and runs nothing. The catalog and the saved
report stay GET (``/checks``, ``/latest``): they only read files.
"""

from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ...bringup import preflight as pf
from ...bringup.schemas import ARMS, PreflightBusFacts, PreflightCheckInfo, PreflightReport
from ...hardware.backend import current_backend
from ..robot_manager import get_manager

router = APIRouter(prefix="/api/preflight", tags=["preflight"])

#: Test seam: the synthetic bus samples positions with real sleeps otherwise.
_sleep = time.sleep


def _sleep_now(seconds: float) -> None:
    _sleep(seconds)


# GET used to run preflight. These answer 405 (also when the built UI's SPA
# fallback, a catch-all GET, is mounted) so an old bookmark or a cross-origin
# page gets an error, not a bus read.
_RUN_IS_POST = (
    "preflight runs with POST (it reads the motor buses and writes a report); "
    "GET runs nothing. Use `so101 preflight`, or POST {path}"
)


@router.get("", include_in_schema=False)
def run_preflight_get() -> JSONResponse:
    return JSONResponse(
        {"detail": _RUN_IS_POST.format(path="/api/preflight")},
        status_code=405,
        headers={"Allow": "POST"},
    )


@router.get("/bus", include_in_schema=False)
def bus_facts_get() -> JSONResponse:
    return JSONResponse(
        {"detail": _RUN_IS_POST.format(path="/api/preflight/bus")},
        status_code=405,
        headers={"Allow": "POST"},
    )


@router.post("", response_model=PreflightReport)
def run_preflight(
    arm: Literal["follower", "leader", "both"] = Query(
        "both", description="Which arm's checks to run"
    ),
    save: bool = Query(True, description="Write the report under <data_dir>/preflight/"),
) -> PreflightReport:
    """Run every check now. Bus checks run on buses RobotManager lends while idle.

    A POST: it reads the buses and (with ``save``) writes a report, so a
    cross-origin page is refused by the Origin guard.
    """
    arms = ARMS if arm == "both" else (arm,)
    manager = get_manager()
    report = pf.run_api_preflight(manager, arms=arms, sleep=_sleep_now)
    if save:
        pf.save_report(manager.settings, report)
    return report


@router.post("/bus", response_model=PreflightBusFacts)
def bus_facts(
    arm: Literal["follower", "leader", "both"] = Query(
        "both", description="Which arm's bus to read"
    ),
) -> PreflightBusFacts:
    """One read-only diagnostic pass per arm on a bus RobotManager lends.

    ``bus_access`` is ``refused`` (with the reason) while the robot is not idle
    or the bus stays busy. Nothing is saved. ``so101 preflight`` calls this
    while a server runs and grades the facts itself. A POST, because it
    borrows the bus.
    """
    arms = ARMS if arm == "both" else (arm,)
    return pf.bus_facts_report(get_manager(), arms=arms, sleep=_sleep_now)


@router.get("/checks", response_model=list[PreflightCheckInfo])
def list_checks() -> list[PreflightCheckInfo]:
    """The check catalog, in run order (for the wizard's step list)."""
    return pf.check_catalog()


@router.get("/latest", response_model=PreflightReport)
def latest_report(
    origin: Literal["any", "cli", "api"] = Query(
        "any", description="'cli' returns the last `so101 preflight` run, which reads the bus"
    ),
) -> PreflightReport:
    """The last saved report of this server's backend, from either the CLI or this API.

    Reports are kept per backend: a real-backend server never answers with a
    sim or mock rehearsal, even though they share the data folder.
    """
    settings = get_manager().settings
    report = pf.load_latest_report(settings, None if origin == "any" else origin)
    if report is None:
        raise HTTPException(
            404,
            f"no saved preflight report for the {current_backend(settings)} backend"
            + ("" if origin == "any" else f" from the {origin}")
            + "; run POST /api/preflight or `so101 preflight`",
        )
    return report
