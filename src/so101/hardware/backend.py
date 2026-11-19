"""Which backend the current settings select: ``real``, ``mock`` or ``sim``.

``Settings.backend`` (SO101_BACKEND) decides. An object without it falls back
to ``SO101_MOCK``.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

Backend = Literal["real", "mock", "sim"]

#: What to do when the sim backend's devices cannot load (diagnostics, preflight).
SIM_EXTRA_REMEDIATION = (
    'Install the sim extra (`pip install -e ".[sim]"`; the installer adds it), '
    "or set SO101_BACKEND=real to use the physical arm."
)


def current_backend(settings: Any) -> Backend:
    explicit = getattr(settings, "backend", None)
    if explicit in ("real", "mock", "sim"):
        return explicit  # type: ignore[no-any-return]
    return "mock" if bool(getattr(settings, "mock", False)) else "real"


def is_rehearsal(backend: Backend) -> bool:
    """Mock and sim runs are rehearsals; only the real backend produces evidence."""
    return backend != "real"


def sim_devices_problem() -> str | None:
    """Why the sim backend's devices cannot load in this process; None when they can.

    Under ``SO101_BACKEND=sim`` the follower, the leader and the cameras come
    from ``so101.simulation.devices``, which imports MuJoCo (the ``sim``
    extra). Without it ``make_bus`` hands out a bus that refuses to connect,
    so arming fails. Any exception counts, not only ImportError: MuJoCo's own
    import can fail in other ways (on macOS it runs ``sysctl``).
    """
    try:
        importlib.import_module("so101.simulation.devices")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None
