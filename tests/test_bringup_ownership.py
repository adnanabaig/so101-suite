"""Bus ownership: the server probe fails closed, the advisory lock refuses a second holder."""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from so101.bringup import ownership as own


def make_settings(tmp_path: Path, host: str = "127.0.0.1", port: int = 8816) -> Any:
    return SimpleNamespace(
        host=host,
        port=port,
        data_dir=tmp_path / "data",
        follower_port=str(tmp_path / "follower-port"),
        leader_port=str(tmp_path / "leader-port"),
    )


_HOLD_DEVICE_LOCK = """
import sys
from pathlib import Path
from types import SimpleNamespace
from so101.bringup import ownership

settings = SimpleNamespace(
    data_dir=sys.argv[1], follower_port=sys.argv[2], leader_port=sys.argv[3]
)
lock_root = Path(sys.argv[5])
ownership._device_lock_root = lambda: lock_root
ownership._validate_lock_namespace_root = lambda _root: None
lock = ownership.DeviceBusLock(settings, purpose=sys.argv[4]).acquire()
print("READY", flush=True)
sys.stdin.read(1)
lock.release()
"""


def _start_device_holder(settings: Any, *, purpose: str) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HOLD_DEVICE_LOCK,
            str(settings.data_dir),
            str(settings.follower_port),
            str(settings.leader_port),
            purpose,
            str(own._device_lock_root()),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    ready = proc.stdout.readline().strip()
    if ready != "READY":
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        raise AssertionError(f"device-lock child did not start: {ready!r}; {stderr}")
    return proc


def _stop_device_holder(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    assert proc.stdin is not None
    proc.stdin.write("x")
    proc.stdin.flush()
    proc.wait(timeout=5)


def _create_fake_ports(settings: Any) -> None:
    for value in (settings.follower_port, settings.leader_port):
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _use_private_marker_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "global-marker-root"
    root.mkdir()
    monkeypatch.setattr(own, "_global_marker_root", lambda: root)
    monkeypatch.setattr(own, "_validate_global_marker_root", lambda _root: None)
    return root


def _run_posix_contract_or_assert_windows_branch() -> bool:
    """No skip budget: Windows exercises its injected msvcrt tests below."""

    if os.name == "posix":
        return True
    assert os.name == "nt" and own._USE_MSVCRT
    return False


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("", "http://127.0.0.1:8816"),
        ("0.0.0.0", "http://127.0.0.1:8816"),
        ("::", "http://127.0.0.1:8816"),
        ("::1", "http://[::1]:8816"),
        ("robot.local", "http://robot.local:8816"),
    ],
)
def test_server_base_url(tmp_path: Path, host: str, expected: str) -> None:
    assert own.server_base_url(make_settings(tmp_path, host=host)) == expected


def _probe(tmp_path: Path, handler: Any) -> own.ServerProbe:
    return own.probe_local_server(make_settings(tmp_path), transport=httpx.MockTransport(handler))


def test_probe_recognizes_a_suite_server(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == own.STATUS_PATH
        return httpx.Response(200, json={"mode": "idle", "estopped": False, "live_cameras": []})

    probe = _probe(tmp_path, handler)
    assert probe.running and probe.http_status == 200
    assert probe.status is not None and probe.status["mode"] == "idle"


def test_probe_treats_any_other_answer_as_in_use(tmp_path: Path) -> None:
    probe = _probe(tmp_path, lambda request: httpx.Response(404, text="nope"))
    assert probe.running is True and probe.status is None
    assert "not a Suite status reply" in probe.detail


def test_probe_connection_refused_means_no_server(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _probe(tmp_path, handler).running is False


@pytest.mark.parametrize(
    ("error", "text"),
    [(httpx.ReadTimeout, "did not answer"), (httpx.RemoteProtocolError, "probe failed")],
)
def test_probe_fails_closed_on_timeouts_and_errors(
    tmp_path: Path, error: type[httpx.HTTPError], text: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("boom", request=request)  # type: ignore[call-arg]

    probe = _probe(tmp_path, handler)
    assert probe.running is True and text in probe.detail


def test_probe_against_a_closed_local_port(tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
    # The default timeouts: Windows needs the longer connect wait to see the refusal.
    probe = own.probe_local_server(make_settings(tmp_path, port=free_port))
    assert probe.running is False, probe.detail


def test_probe_connect_and_read_timeouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        raise httpx.ReadTimeout("slow", request=request)

    monkeypatch.setattr(own, "PROBE_CONNECT_TIMEOUT_S", 3.0)  # the Windows value
    probe = _probe(tmp_path, handler)
    assert seen[-1]["connect"] == 3.0 and seen[-1]["read"] == own.PROBE_READ_TIMEOUT_S
    assert probe.running is True and "within 3 s" in probe.detail

    probe = own.probe_local_server(
        make_settings(tmp_path), timeout=1.5, transport=httpx.MockTransport(handler)
    )
    assert seen[-1]["connect"] == 1.5 and seen[-1]["read"] == 1.5
    assert "within 1.5 s" in probe.detail
    assert own.PROBE_CONNECT_TIMEOUT_S >= own.PROBE_READ_TIMEOUT_S


def test_lock_records_its_holder_and_refuses_a_second_one(tmp_path: Path) -> None:
    path = tmp_path / "data" / own.BUS_LOCK_FILENAME
    first = own.BusLock(path, purpose="so101 preflight").acquire()
    assert first.held and first.acquire() is first
    holder = json.loads(path.read_text(encoding="utf-8"))
    assert holder["pid"] == os.getpid() and holder["purpose"] == "so101 preflight"
    with pytest.raises(own.BusOwnershipError) as caught:
        own.BusLock(path, purpose="so101 rest-pose record").acquire()
    assert caught.value.reason == "lock"
    assert "so101 preflight" in caught.value.detail
    assert caught.value.holder["pid"] == os.getpid()
    first.release()
    assert not first.held and path.exists()  # never deleted
    with own.BusLock(path, purpose="again") as again:
        assert again.held
    assert not again.held


def test_existing_shared_lock_is_opened_without_o_creat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ubuntu protected_regular rejects O_CREAT on another user's /tmp file."""

    path = tmp_path / "shared.lock"
    path.touch(mode=0o666)
    real_open = os.open
    flags_seen: list[int] = []

    def tracked_open(file: Any, flags: int, mode: int = 0o777) -> int:
        if Path(file) == path:
            flags_seen.append(flags)
        return real_open(file, flags, mode)

    monkeypatch.setattr(os, "open", tracked_open)
    lock = own.BusLock(path, purpose="cross-user", shared=True).acquire()
    lock.release()
    assert flags_seen and not (flags_seen[0] & os.O_CREAT)


def test_acquire_refuses_while_a_server_runs_and_creates_no_lock(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    def running(_: Any) -> own.ServerProbe:
        return own.ServerProbe(running=True, url="u", detail="HTTP 200 from u")

    with pytest.raises(own.BusOwnershipError) as caught:
        own.acquire_bus_access(settings, purpose="t", server_probe=running)
    assert caught.value.reason == "server"
    assert "POST /api/preflight" in caught.value.remediation
    assert not own.bus_lock_path(settings).exists()


def test_acquire_takes_the_lock_once(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    def idle(_: Any) -> own.ServerProbe:
        return own.ServerProbe(running=False, url="u", detail="nothing is listening")

    lock, probe = own.acquire_bus_access(settings, purpose="first", server_probe=idle)
    try:
        assert lock.held and probe.running is False
        with pytest.raises(own.BusOwnershipError) as caught:
            own.acquire_bus_access(settings, purpose="second", server_probe=idle)
        assert caught.value.reason == "lock"
    finally:
        lock.release()


def test_server_startup_preserves_unsafe_lock_remediation(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        backend="real",
        mock=False,
        follower_port="",
        leader_port="",
        data_dir=tmp_path,
    )
    with pytest.raises(own.ServerBusLockError) as caught:
        own.take_server_bus_lock(settings)
    message = str(caught.value)
    assert "cannot safely start" in message
    assert "Set SO101_FOLLOWER_PORT" in message
    assert "which owns the serial bus" not in message


def test_same_devices_different_data_dirs_refuse_with_holder(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    settings_a = make_settings(tmp_path / "checkout-a")
    settings_b = make_settings(tmp_path / "checkout-b")
    settings_b.follower_port = settings_a.follower_port
    settings_b.leader_port = settings_a.leader_port
    _create_fake_ports(settings_a)
    proc = _start_device_holder(settings_a, purpose="server from checkout A")
    try:
        with pytest.raises(own.BusOwnershipError) as caught:
            own.acquire_bus_access(
                settings_b,
                purpose="standalone from checkout B",
                server_probe=lambda _settings: own.ServerProbe(False, "u", "idle"),
            )
        assert caught.value.reason == "lock"
        assert caught.value.holder["pid"] == proc.pid
        assert caught.value.holder["purpose"] == "server from checkout A"
        assert "device-global lock" in caught.value.detail
        assert "Wait for" in caught.value.remediation
    finally:
        _stop_device_holder(proc)


def test_disjoint_physical_ports_can_be_locked_concurrently(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    settings_a = make_settings(tmp_path / "arm-a")
    settings_b = make_settings(tmp_path / "arm-b")
    _create_fake_ports(settings_a)
    _create_fake_ports(settings_b)
    proc = _start_device_holder(settings_a, purpose="arm A")
    try:
        lock = own.DeviceBusLock(settings_b, purpose="arm B").acquire()
        assert lock.held
        lock.release()
    finally:
        _stop_device_holder(proc)


def test_unresolved_port_excludes_later_devices_until_holder_restarts(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    unresolved = make_settings(tmp_path / "unplugged")
    known = make_settings(tmp_path / "plugged")
    _create_fake_ports(known)
    proc = _start_device_holder(unresolved, purpose="server started while arm unplugged")
    try:
        with pytest.raises(own.BusOwnershipError) as caught:
            own.DeviceBusLock(known, purpose="later device").acquire()
        assert caught.value.reason == "lock"
        assert "namespace" in caught.value.detail
    finally:
        _stop_device_holder(proc)


def test_alias_and_underlying_device_path_collide_across_processes(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    target = tmp_path / "ttyUSB0"
    target.touch()
    alias = tmp_path / "so101-follower"
    alias.symlink_to(target)
    other_a = tmp_path / "leader-a"
    other_b = tmp_path / "leader-b"
    other_a.touch()
    other_b.touch()
    settings_a = SimpleNamespace(
        data_dir=tmp_path / "data-a",
        follower_port=str(alias),
        leader_port=str(other_a),
    )
    settings_b = SimpleNamespace(
        data_dir=tmp_path / "data-b",
        follower_port=str(target),
        leader_port=str(other_b),
    )
    proc = _start_device_holder(settings_a, purpose="udev alias holder")
    try:
        with pytest.raises(own.BusOwnershipError) as caught:
            own.DeviceBusLock(settings_b, purpose="real-path contender").acquire()
        assert caught.value.reason == "lock"
        assert caught.value.holder["purpose"] == "udev alias holder"
        assert caught.value.holder["canonical_port"] == str(target)
    finally:
        _stop_device_holder(proc)


def test_retargeted_alias_keeps_using_the_endpoint_captured_by_its_lock(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    target_a = tmp_path / "ttyUSB0"
    target_b = tmp_path / "ttyUSB1"
    leader_a = tmp_path / "leader-a"
    leader_b = tmp_path / "leader-b"
    for path in (target_a, target_b, leader_a, leader_b):
        path.touch()
    alias = tmp_path / "so101-follower"
    alias.symlink_to(target_a)
    settings_a = SimpleNamespace(follower_port=str(alias), leader_port=str(leader_a))
    settings_b = SimpleNamespace(follower_port=str(target_b), leader_port=str(leader_b))

    with own.DeviceBusLock(settings_a, purpose="alias owner"):
        alias.unlink()
        alias.symlink_to(target_b)
        # A direct owner of B may proceed, but the alias owner never follows
        # the changed link: its eventual bus construction remains on locked A.
        with own.DeviceBusLock(settings_b, purpose="target B owner"):
            assert own.locked_device_port(str(alias)) == str(target_a)

    assert own.locked_device_port(str(alias)) == str(alias)


def test_two_configured_aliases_for_one_endpoint_are_refused(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    target = tmp_path / "ttyUSB0"
    target.touch()
    follower = tmp_path / "follower"
    leader = tmp_path / "leader"
    follower.symlink_to(target)
    leader.symlink_to(target)
    settings = SimpleNamespace(follower_port=str(follower), leader_port=str(leader))

    with pytest.raises(own.BusOwnershipError) as caught:
        own.DeviceBusLock(settings, purpose="bad two-arm configuration")
    assert caught.value.reason == "unsupported"
    assert "resolve to the same endpoint" in caught.value.detail


def test_hardlinked_device_paths_share_endpoint_lock(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    target = tmp_path / "tty-test"
    alias = tmp_path / "tty-hardlink"
    leader_a = tmp_path / "leader-a"
    leader_b = tmp_path / "leader-b"
    for path in (target, leader_a, leader_b):
        path.touch()
    os.link(target, alias)
    settings_a = SimpleNamespace(follower_port=str(target), leader_port=str(leader_a))
    settings_b = SimpleNamespace(follower_port=str(alias), leader_port=str(leader_b))

    with own.DeviceBusLock(settings_a, purpose="hardlink A"):
        with pytest.raises(own.BusOwnershipError) as caught:
            own.DeviceBusLock(settings_b, purpose="hardlink B").acquire()
    assert caught.value.reason == "lock"


def test_duplicate_character_nodes_share_rdev_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    port_a = tmp_path / "tty-a"
    port_b = tmp_path / "tty-b"
    leader_a = tmp_path / "leader-a"
    leader_b = tmp_path / "leader-b"
    for path in (port_a, port_b, leader_a, leader_b):
        path.touch()
    real_stat = os.stat
    shared_rdev = os.makedev(188, 42)

    def fake_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        info = real_stat(path, *args, **kwargs)
        if Path(path) in {port_a, port_b}:
            return SimpleNamespace(
                st_mode=stat.S_IFCHR | 0o660,
                st_rdev=shared_rdev,
                st_dev=info.st_dev,
                st_ino=info.st_ino,
            )
        return info

    monkeypatch.setattr(os, "stat", fake_stat)
    settings_a = SimpleNamespace(follower_port=str(port_a), leader_port=str(leader_a))
    settings_b = SimpleNamespace(follower_port=str(port_b), leader_port=str(leader_b))
    with own.DeviceBusLock(settings_a, purpose="char node A"):
        with pytest.raises(own.BusOwnershipError) as caught:
            own.DeviceBusLock(settings_b, purpose="char node B").acquire()
    assert caught.value.reason == "lock"


def test_same_configured_follower_and_leader_port_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "ttyUSB0"
    target.touch()
    settings = SimpleNamespace(follower_port=str(target), leader_port=str(target))

    with pytest.raises(own.BusOwnershipError) as caught:
        own.DeviceBusLock(settings, purpose="bad duplicate configuration")
    assert caught.value.reason == "unsupported"
    assert "both configured" in caught.value.detail


def test_feetech_bus_constructed_before_gate_resolves_at_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    from so101.hardware.motors import FeetechBus

    target_a = tmp_path / "ttyUSB0"
    target_b = tmp_path / "ttyUSB1"
    leader = tmp_path / "leader"
    for path in (target_a, target_b, leader):
        path.touch()
    alias = tmp_path / "so101-follower"
    alias.symlink_to(target_a)
    settings = SimpleNamespace(follower_port=str(alias), leader_port=str(leader))
    bus = FeetechBus(str(alias), 1_000_000, "follower")
    opened: list[str] = []

    class ReachedBuild(Exception):
        pass

    def build() -> Any:
        opened.append(bus.port)
        raise ReachedBuild

    monkeypatch.setattr(bus, "_build", build)
    with own.DeviceBusLock(settings, purpose="direct PolicyRunner gate"):
        alias.unlink()
        alias.symlink_to(target_b)
        with pytest.raises(ReachedBuild):
            bus.connect()
    with own.DeviceBusLock(settings, purpose="later PolicyRunner gate"):
        with pytest.raises(ReachedBuild):
            bus.connect()
    assert opened == [str(target_a), str(target_b)]


def test_real_bus_outside_held_device_set_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101.hardware.motors import FeetechBus

    settings = make_settings(tmp_path / "configured")
    _create_fake_ports(settings)
    other = tmp_path / "other-device"
    other.touch()
    bus = FeetechBus(str(other), 1_000_000, "follower")
    reached_build: list[bool] = []
    monkeypatch.setattr(bus, "_build", lambda: reached_build.append(True))

    with own.DeviceBusLock(settings, purpose="configured owner"):
        with pytest.raises(own.BusOwnershipError, match="not covered"):
            bus.connect()
    assert reached_build == []


def test_replaced_device_node_is_refused_at_actual_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    settings = make_settings(tmp_path)
    _use_private_marker_root(tmp_path, monkeypatch)
    _create_fake_ports(settings)
    follower = Path(settings.follower_port)

    with own.DeviceBusLock(settings, purpose="pinned old endpoint"):
        own.write_global_energized_intent(settings, purpose="pre-open durable intent")
        records_before = set(own._global_marker_root().glob("*.json"))
        follower.unlink()
        follower.touch()
        with pytest.raises(own.BusOwnershipError, match="was replaced"):
            own.require_device_lock(settings.follower_port)
        assert set(own._global_marker_root().glob("*.json")) == records_before
        with pytest.raises(own.BusOwnershipError) as contender:
            own.DeviceBusLock(settings, purpose="replacement contender").acquire()
        assert contender.value.reason == "lock"

    with own.DeviceBusLock(settings, purpose="replacement after release"):
        marker = own.read_global_energized_intent(settings)
        assert marker is not None and marker["purpose"] == "pre-open durable intent"


def test_device_lock_is_released_when_holder_process_dies(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)
    proc = _start_device_holder(settings, purpose="process that will die")
    proc.kill()
    proc.wait(timeout=5)

    lock = own.DeviceBusLock(settings, purpose="replacement process").acquire()
    assert lock.held
    lock.release()


def test_global_energized_intent_crosses_data_dirs_for_same_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings_a = make_settings(tmp_path / "data-owner-a")
    settings_b = make_settings(tmp_path / "data-owner-b")
    settings_b.follower_port = settings_a.follower_port
    settings_b.leader_port = settings_a.leader_port
    _create_fake_ports(settings_a)

    with own.DeviceBusLock(settings_a, purpose="standalone A"):
        own.write_global_energized_intent(
            settings_a, purpose="completed standalone Halt", device_name="follower"
        )
    with own.DeviceBusLock(settings_b, purpose="standalone B"):
        marker = own.read_global_energized_intent(settings_b)

    assert marker is not None and marker["energized"] is True
    assert marker["purpose"] == "completed standalone Halt"
    assert marker["backend"] == "real"


def test_retargeted_alias_intent_blocks_alias_and_direct_current_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    _use_private_marker_root(tmp_path, monkeypatch)
    target_a = tmp_path / "tty-a"
    target_b = tmp_path / "tty-b"
    leader_a = tmp_path / "leader-a"
    leader_b = tmp_path / "leader-b"
    for path in (target_a, target_b, leader_a, leader_b):
        path.touch()
    alias = tmp_path / "so101-follower"
    alias.symlink_to(target_a)
    old = SimpleNamespace(
        follower_port=str(alias), leader_port=str(leader_a), follower_id="f", leader_id="l"
    )
    with own.DeviceBusLock(old, purpose="old alias owner"):
        own.write_global_energized_intent(old, purpose="crashed hold")
    alias.unlink()
    alias.symlink_to(target_b)

    for configured in (alias, target_b):
        current = SimpleNamespace(
            follower_port=str(configured),
            leader_port=str(leader_b),
            follower_id="f2",
            leader_id="l2",
        )
        with own.DeviceBusLock(current, purpose=f"current {configured.name}"):
            marker = own.read_global_energized_intent(current)
            assert marker is not None and marker["purpose"] == "crashed hold"


def test_disjoint_ports_do_not_share_global_energized_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings_a = make_settings(tmp_path / "arm-a")
    settings_b = make_settings(tmp_path / "arm-b")
    _create_fake_ports(settings_a)
    _create_fake_ports(settings_b)

    with own.DeviceBusLock(settings_a, purpose="standalone A"):
        own.write_global_energized_intent(settings_a, purpose="Halt", device_name="follower")
    with own.DeviceBusLock(settings_b, purpose="standalone B"):
        assert own.read_global_energized_intent(settings_b) is None


def test_leader_only_global_intent_uses_exact_leader_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)
    leader_identity = own.canonical_device_identity(settings.leader_port)

    with own.DeviceBusLock(settings, purpose="leader calibration"):
        records = own.write_global_energized_intent(
            settings, purpose="leader possible torque", device_name="leader"
        )
        assert len(records) == 1
        assert records[0]["endpoint_key"] == leader_identity.endpoint_key
        assert records[0]["node_key"] == leader_identity.node_key
        assert own.read_global_energized_intent(settings) is not None
        own.clear_global_energized_intent(settings, device_name="leader")
        assert own.read_global_energized_intent(settings) is None


def test_aggregate_intent_keeps_exact_device_scope_after_partial_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)
    with own.DeviceBusLock(settings, purpose="teleop both buses"):
        records = own.write_global_energized_intent(settings, purpose="both", device_name="")
        assert {record["device_name"] for record in records} == {"follower", "leader"}
        own.clear_global_energized_intent(settings, device_name="leader")
        remaining = own.read_global_energized_intent(settings)
        assert remaining is not None
        assert {record["device_name"] for record in remaining["markers"]} == {"follower"}


def test_confirmed_torque_off_clears_global_energized_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path / "arm")
    _create_fake_ports(settings)

    with own.DeviceBusLock(settings, purpose="standalone"):
        own.write_global_energized_intent(settings, purpose="about to connect")
        assert own.read_global_energized_intent(settings) is not None
        own.clear_global_energized_intent(settings)
        assert own.read_global_energized_intent(settings) is None


def test_repeated_intent_cycles_keep_the_common_journal_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path / "arm")
    _create_fake_ports(settings)

    with own.DeviceBusLock(settings, purpose="repeated owner"):
        for cycle in range(20):
            own.write_global_energized_intent(
                settings, purpose=f"cycle {cycle}", device_name="follower"
            )
            own.clear_global_energized_intent(settings, device_name="follower")

    assert list(root.glob("*.json")) == []
    assert len(list((root / "aliases").glob("alias-*.json"))) == 1


def test_confirmed_clear_through_real_path_clears_alias_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    _use_private_marker_root(tmp_path, monkeypatch)
    target = tmp_path / "ttyUSB0"
    alias = tmp_path / "so101-follower"
    leader_a = tmp_path / "leader-a"
    leader_b = tmp_path / "leader-b"
    for path in (target, leader_a, leader_b):
        path.touch()
    alias.symlink_to(target)
    alias_settings = SimpleNamespace(
        follower_port=str(alias),
        leader_port=str(leader_a),
        follower_id="follower-a",
        leader_id="leader-a",
    )
    target_settings = SimpleNamespace(
        follower_port=str(target),
        leader_port=str(leader_b),
        follower_id="follower-b",
        leader_id="leader-b",
    )

    with own.DeviceBusLock(alias_settings, purpose="alias owner"):
        own.write_global_energized_intent(alias_settings, purpose="Halt")
    with own.DeviceBusLock(target_settings, purpose="confirmed-off owner"):
        assert own.read_global_energized_intent(target_settings) is not None
        own.clear_global_energized_intent(target_settings)
    with own.DeviceBusLock(alias_settings, purpose="alias retry"):
        assert own.read_global_energized_intent(alias_settings) is None


def test_power_cut_clears_only_current_alias_facet_not_old_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    _use_private_marker_root(tmp_path, monkeypatch)
    target_a = tmp_path / "tty-a"
    target_b = tmp_path / "tty-b"
    leader = tmp_path / "leader"
    for path in (target_a, target_b, leader):
        path.touch()
    alias = tmp_path / "alias"
    alias.symlink_to(target_a)
    old = SimpleNamespace(
        follower_port=str(alias), leader_port=str(leader), follower_id="old", leader_id="l"
    )
    with own.DeviceBusLock(old, purpose="old owner"):
        own.write_global_energized_intent(old, purpose="old energized arm")
    alias.unlink()
    alias.symlink_to(target_b)
    current = SimpleNamespace(
        follower_port=str(alias), leader_port=str(leader), follower_id="new", leader_id="l"
    )
    with own.DeviceBusLock(current, purpose="new owner"):
        with pytest.raises(own.GlobalEnergizedIntentError, match="different pinned endpoint"):
            own.clear_global_energized_intent(current)
        own.clear_global_energized_intent(current, force_current_aliases=True)
        assert own.read_global_energized_intent(current) is None

    old_direct = SimpleNamespace(
        follower_port=str(target_a),
        leader_port=str(leader),
        follower_id="old-direct",
        leader_id="l",
    )
    with own.DeviceBusLock(old_direct, purpose="old endpoint recovery"):
        marker = own.read_global_energized_intent(old_direct)
        assert marker is not None and marker["purpose"] == "old energized arm"


def test_power_cut_on_reused_path_clears_overlap_but_preserves_old_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    _use_private_marker_root(tmp_path, monkeypatch)
    port = tmp_path / "tty-reused"
    old_node = tmp_path / "tty-old-node"
    leader = tmp_path / "leader"
    port.touch()
    leader.touch()
    os.link(port, old_node)
    old = SimpleNamespace(
        follower_port=str(port), leader_port=str(leader), follower_id="old", leader_id="l"
    )
    with own.DeviceBusLock(old, purpose="old node"):
        own.write_global_energized_intent(old, purpose="old node active")
    port.unlink()
    port.touch()
    replacement = SimpleNamespace(
        follower_port=str(port), leader_port=str(leader), follower_id="new", leader_id="l"
    )
    with own.DeviceBusLock(replacement, purpose="replacement node"):
        with pytest.raises(own.GlobalEnergizedIntentError, match="different pinned endpoint"):
            own.clear_global_energized_intent(replacement)
        own.clear_global_energized_intent(replacement, force_current_aliases=True)
        assert own.read_global_energized_intent(replacement) is None

    preserved = SimpleNamespace(
        follower_port=str(old_node),
        leader_port=str(leader),
        follower_id="preserved-old",
        leader_id="l",
    )
    with own.DeviceBusLock(preserved, purpose="preserved old node"):
        assert own.read_global_energized_intent(preserved) is not None


def test_power_cut_recovers_corrupt_old_alias_for_direct_current_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    _use_private_marker_root(tmp_path, monkeypatch)
    target_a = tmp_path / "tty-a"
    target_b = tmp_path / "tty-b"
    leader = tmp_path / "leader"
    for path in (target_a, target_b, leader):
        path.touch()
    alias = tmp_path / "alias"
    alias.symlink_to(target_a)
    old = SimpleNamespace(
        follower_port=str(alias), leader_port=str(leader), follower_id="old", leader_id="l"
    )
    with own.DeviceBusLock(old, purpose="old alias"):
        own.write_global_energized_intent(old, purpose="old corrupt intent")
        old_event = own.global_energized_intent_paths(old)[0]
        old_event.write_text("corrupt", encoding="utf-8")
    alias.unlink()
    alias.symlink_to(target_b)
    current = SimpleNamespace(
        follower_port=str(target_b),
        leader_port=str(leader),
        follower_id="current",
        leader_id="l",
    )
    with own.DeviceBusLock(current, purpose="current direct B"):
        with pytest.raises(own.GlobalEnergizedIntentError, match="corrupt relevant"):
            own.read_global_energized_intent(current)
        own.clear_global_energized_intent(current, force_current_aliases=True)
        assert own.read_global_energized_intent(current) is None

    old_direct = SimpleNamespace(
        follower_port=str(target_a),
        leader_port=str(leader),
        follower_id="old-direct",
        leader_id="l",
    )
    with own.DeviceBusLock(old_direct, purpose="old corrupt remains"):
        with pytest.raises(own.GlobalEnergizedIntentError, match="corrupt relevant"):
            own.read_global_energized_intent(old_direct)


def test_failed_atomic_global_intent_clear_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path / "arm")
    _create_fake_ports(settings)

    with own.DeviceBusLock(settings, purpose="standalone"):
        own.write_global_energized_intent(settings, purpose="about to connect")
        original = own._atomic_create_marker

        def fail_clear(path: Path, payload: dict[str, Any]) -> None:
            if payload.get("kind") == own._GLOBAL_CLEAR_KIND:
                raise own.GlobalEnergizedIntentError("injected clear crash", path=path)
            original(path, payload)

        monkeypatch.setattr(own, "_atomic_create_marker", fail_clear)
        with pytest.raises(own.GlobalEnergizedIntentError, match="injected clear crash"):
            own.clear_global_energized_intent(settings)
        assert own.read_global_energized_intent(settings) is not None


def test_global_energized_intent_requires_matching_device_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)

    with pytest.raises(own.GlobalEnergizedIntentError, match="matching motor-device lock"):
        own.write_global_energized_intent(settings, purpose="unsafe")


def test_device_lock_held_query_tracks_the_configured_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)

    assert own.device_lock_is_held(settings) is False
    with own.DeviceBusLock(settings, purpose="query"):
        assert own.device_lock_is_held(settings) is True
        assert own.device_lock_is_held(settings, device_name="follower") is True
    assert own.device_lock_is_held(settings) is False


def test_raw_enable_capability_requires_fresh_intent_under_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)
    with own.DeviceBusLock(settings, purpose="raw capability"):
        assert own.enable_intent_is_adopted(settings) is False
        with pytest.raises(own.BusOwnershipError, match="no durable global intent token"):
            own.require_enable_intent(settings.follower_port)
        own.write_global_energized_intent(settings, purpose="approved enable")
        assert own.enable_intent_is_adopted(settings) is True
        own.require_enable_intent(settings.follower_port)
        own.clear_global_energized_intent(settings)
        assert own.enable_intent_is_adopted(settings) is False
        with pytest.raises(own.BusOwnershipError, match="no durable global intent token"):
            own.require_enable_intent(settings.follower_port)


def test_malformed_global_energized_intent_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)

    with own.DeviceBusLock(settings, purpose="reader"):
        own.write_global_energized_intent(settings, purpose="corruption test")
        marker_path = own.global_energized_intent_paths(settings)[0]
        marker_path.write_text("not json", encoding="utf-8")
        with pytest.raises(own.GlobalEnergizedIntentError, match="treat the arm as energized"):
            own.read_global_energized_intent(settings)
        own.clear_global_energized_intent(
            settings, purpose="confirmed off recovers corrupt current record"
        )
        assert own.read_global_energized_intent(settings) is None


def test_confirmed_off_recovers_torn_current_alias_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)
    with own.DeviceBusLock(settings, purpose="alias recovery"):
        own.write_global_energized_intent(settings, purpose="active")
        descriptors = sorted(own._alias_root().glob("alias-*.json"))
        assert descriptors
        descriptors[0].write_text("torn", encoding="utf-8")
        own.clear_global_energized_intent(settings, purpose="confirmed off")
        assert own.read_global_energized_intent(settings) is None


def test_missing_node_identity_is_corrupt_and_scoped_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path)
    _create_fake_ports(settings)
    with own.DeviceBusLock(settings, purpose="node schema"):
        own.write_global_energized_intent(settings, purpose="active")
        event = own.global_energized_intent_paths(settings)[0]
        payload = json.loads(event.read_text(encoding="utf-8"))
        payload.pop("node_key")
        event.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(own.GlobalEnergizedIntentError, match="corrupt relevant"):
            own.read_global_energized_intent(settings)
        own.clear_global_energized_intent(settings)
        assert own.read_global_energized_intent(settings) is None


def test_corrupt_recovery_does_not_clear_unrelated_device_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_private_marker_root(tmp_path, monkeypatch)
    settings_a = make_settings(tmp_path / "arm-a")
    settings_b = make_settings(tmp_path / "arm-b")
    _create_fake_ports(settings_a)
    _create_fake_ports(settings_b)
    with own.DeviceBusLock(settings_a, purpose="arm A"):
        own.write_global_energized_intent(settings_a, purpose="A active")
        marker_a = own.global_energized_intent_paths(settings_a)[0]
        marker_a.write_text("corrupt A", encoding="utf-8")
    with own.DeviceBusLock(settings_b, purpose="arm B"):
        own.write_global_energized_intent(settings_b, purpose="B active")
        own.clear_global_energized_intent(settings_b, purpose="B confirmed off")
        assert own.read_global_energized_intent(settings_b) is None
    with own.DeviceBusLock(settings_a, purpose="A recovery"):
        with pytest.raises(own.GlobalEnergizedIntentError, match="corrupt relevant"):
            own.read_global_energized_intent(settings_a)


def test_marker_publication_never_truncates_or_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _use_private_marker_root(tmp_path, monkeypatch)
    destination = root / "intent--scope--fixed.json"
    destination.write_text("original\n", encoding="utf-8")

    def refuse_truncate(*_args: Any) -> None:
        raise AssertionError("must never truncate")

    monkeypatch.setattr(os, "ftruncate", refuse_truncate)

    with pytest.raises(own.GlobalEnergizedIntentError, match="atomically publish"):
        own._atomic_create_marker(destination, {"kind": "new"})
    assert destination.read_text(encoding="utf-8") == "original\n"


def test_marker_journal_records_are_immutable_single_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _use_private_marker_root(tmp_path, monkeypatch)
    settings = make_settings(tmp_path / "arm")
    _create_fake_ports(settings)
    with own.DeviceBusLock(settings, purpose="writer"):
        own.write_global_energized_intent(settings, purpose="mode check")

    records = list(root.glob("*.json"))
    assert records
    for path in records:
        info = path.stat()
        assert info.st_nlink == 1
        if os.name == "posix":
            assert stat.S_IMODE(info.st_mode) == 0o640
    assert not list(root.glob("*.tmp"))


def test_posix_state_root_is_not_an_age_cleaned_temp_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        assert own._USE_MSVCRT
        return
    monkeypatch.setattr(own, "_USE_MSVCRT", False)
    if sys.platform == "darwin":
        assert own._stable_user_state_root() == Path(
            "/Library/Application Support/SO101/device-ownership-v2"
        )
    else:
        assert own._stable_user_state_root() == Path("/var/lib/so101/device-ownership-v2")


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_rehearsal_servers_do_not_contend_with_real_device_locks(
    tmp_path: Path, backend: str
) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    real = make_settings(tmp_path)
    _create_fake_ports(real)
    proc = _start_device_holder(real, purpose="real owner")
    try:
        rehearsal = make_settings(tmp_path / backend)
        rehearsal.follower_port = real.follower_port
        rehearsal.leader_port = real.leader_port
        rehearsal.backend = backend
        rehearsal.mock = backend == "mock"
        assert own.take_server_bus_lock(rehearsal) is None
    finally:
        _stop_device_holder(proc)


def test_device_lock_refuses_a_symlink_lock_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("do not touch", encoding="utf-8")
    path = tmp_path / "bus.lock"
    path.symlink_to(target)

    with pytest.raises(own.BusOwnershipError) as caught:
        own.BusLock(path, purpose="unsafe contender").acquire()
    assert caught.value.reason == "unsupported"
    assert target.read_text(encoding="utf-8") == "do not touch"


def test_device_lock_refuses_a_hard_link_lock_file(tmp_path: Path) -> None:
    if not _run_posix_contract_or_assert_windows_branch():
        return
    target = tmp_path / "target"
    target.write_text("do not touch", encoding="utf-8")
    path = tmp_path / "bus.lock"
    os.link(target, path)

    with pytest.raises(own.BusOwnershipError) as caught:
        own.BusLock(path, purpose="unsafe contender").acquire()
    assert caught.value.reason == "unsupported"
    assert target.read_text(encoding="utf-8") == "do not touch"


def test_read_lock_holder_tolerates_garbage(tmp_path: Path) -> None:
    path = tmp_path / "bus.lock"
    assert own.read_lock_holder(path) == {}
    path.write_text("not json", encoding="utf-8")
    assert own.read_lock_holder(path) == {}
    path.write_text("[1]", encoding="utf-8")
    assert own.read_lock_holder(path) == {}


def test_windows_com_identity_is_independent_of_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(own, "_USE_MSVCRT", True)
    bare = own.canonical_device_identity("com3")
    namespaced = own.canonical_device_identity(r"\\.\COM3")
    assert bare.key == namespaced.key == r"windows-device:\\.\COM3"


def test_windows_state_root_is_independent_of_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(own, "_USE_MSVCRT", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    monkeypatch.setenv("TEMP", str(tmp_path / "temp-a"))
    first = own._stable_user_state_root()
    monkeypatch.setenv("TEMP", str(tmp_path / "temp-b"))
    second = own._stable_user_state_root()
    assert first == second == tmp_path / "local-app-data" / "SO101" / "device-ownership-v2"


def test_windows_reparse_points_are_refused() -> None:
    info = SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)
    assert own._stat_is_link_or_reparse(info) is True


def test_windows_motor_path_reparse_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(own, "_USE_MSVCRT", True)
    monkeypatch.setattr(
        os,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_file_attributes=0x400,
            st_dev=1,
            st_ino=2,
        ),
    )
    with pytest.raises(own.BusOwnershipError, match="reparse point"):
        own.canonical_device_identity(str(tmp_path / "motor-port"))


class FakeMsvcrt:
    """``msvcrt.locking`` as Windows behaves: byte-range locks, OSError when held."""

    LK_UNLCK, LK_LOCK, LK_NBLCK = 0, 1, 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.held: set[tuple[int, int]] = set()

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        offset = os.lseek(fd, 0, os.SEEK_CUR)  # msvcrt locks from the file position
        key = (os.fstat(fd).st_ino, offset)
        if mode == self.LK_NBLCK:
            self.calls.append(("lock", offset, nbytes))
            if key in self.held:
                raise PermissionError(13, "Permission denied")
            self.held.add(key)
        elif mode == self.LK_UNLCK:
            self.calls.append(("unlock", offset, nbytes))
            self.held.discard(key)
        else:
            raise AssertionError(f"unexpected lock mode {mode}")


def test_lock_uses_a_msvcrt_byte_lock_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMsvcrt()
    monkeypatch.setattr(own, "_USE_MSVCRT", True)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    path = tmp_path / "data" / own.BUS_LOCK_FILENAME

    first = own.BusLock(path, purpose="so101 preflight").acquire()
    # The holder record sits at the start of the file, outside the locked byte,
    # so the refused process can still read who holds the bus.
    assert json.loads(path.read_text(encoding="utf-8"))["purpose"] == "so101 preflight"
    with pytest.raises(own.BusOwnershipError) as caught:
        own.BusLock(path, purpose="so101 rest-pose record").acquire()
    assert caught.value.reason == "lock" and "so101 preflight" in caught.value.detail
    first.release()

    offset = own._WINDOWS_LOCK_OFFSET
    assert fake.calls == [("lock", offset, 1), ("lock", offset, 1), ("unlock", offset, 1)]
    assert fake.held == set()
    with own.BusLock(path, purpose="again") as again:
        assert again.held and fake.held
    assert fake.held == set()


def test_lock_without_file_locking_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(own, "_USE_MSVCRT", False)
    monkeypatch.setitem(sys.modules, "fcntl", None)  # import fcntl -> ImportError
    with pytest.raises(own.BusOwnershipError) as caught:
        own.BusLock(tmp_path / own.BUS_LOCK_FILENAME, purpose="t").acquire()
    assert caught.value.reason == "unsupported"
    assert "Windows, Linux or macOS" in caught.value.remediation
