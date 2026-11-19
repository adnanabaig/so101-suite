"""Fixed-rate control loops (data-B4, docs/audit/v0.1/data.md).

The record/teleop and rollout loops use deadline scheduling: an iteration
takes max(work, period), not work + period. So a 30 fps recording holds 30
frames per second of motion, and a policy with 25 ms inference still runs at
30 Hz. A loop more than one period behind resyncs instead of bursting.

The loops take their clock and sleep from the manager, so these tests run on
injected time (devex-B4, docs/audit/v0.1/devex.md): exact, and independent of
how busy the machine is.
"""

from __future__ import annotations

import time

import pytest

from so101.policy.server import MockPolicy
from so101.services.robot_manager import Mode, RobotManager, _Pacer, get_manager

OPERATOR = "test-operator"
FPS = 30
PERIOD = 1.0 / FPS


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def work(self, seconds: float) -> None:
        self.now += seconds


def _starts(fake: FakeTime, work: list[float]) -> list[float]:
    """Run one paced iteration per entry of ``work``; return their start times."""

    pacer = _Pacer(PERIOD, clock=fake.clock, sleep=fake.sleep)
    starts = []
    for seconds in work:
        starts.append(fake.now)
        fake.work(seconds)
        pacer.wait()
    return starts


# --- the pacer -----------------------------------------------------------------


def test_data_b4_an_iteration_takes_the_max_of_work_and_period_not_the_sum():
    fake = FakeTime()

    starts = _starts(fake, [0.010] * FPS)

    assert fake.now == pytest.approx(1.0)
    assert starts == pytest.approx([index * PERIOD for index in range(FPS)])
    assert fake.sleeps == pytest.approx([PERIOD - 0.010] * FPS)


def test_data_b4_a_late_iteration_is_caught_up_within_the_schedule():
    fake = FakeTime()
    work = [0.010] * FPS
    work[5] = 0.050  # overruns its period, by less than one period

    starts = _starts(fake, work)

    assert fake.now == pytest.approx(1.0)
    assert starts[7:] == pytest.approx([index * PERIOD for index in range(7, FPS)])


def test_data_b4_a_stall_resyncs_instead_of_bursting_to_catch_up():
    fake = FakeTime()
    work = [0.010] * 10
    work[3] = 0.200  # six periods

    starts = _starts(fake, work)

    gaps = [later - earlier for earlier, later in zip(starts, starts[1:], strict=False)]
    assert gaps[3] == pytest.approx(0.200)
    assert min(gaps) == pytest.approx(PERIOD), "no back-to-back catch-up frames"


def test_data_b4_work_longer_than_the_period_runs_at_the_work_rate():
    fake = FakeTime()

    _starts(fake, [0.040] * FPS)

    assert fake.now == pytest.approx(FPS * 0.040)
    assert fake.sleeps == []


def test_pacer_reports_sleep_overshoot_without_changing_deadlines():
    fake = FakeTime()

    def delayed_sleep(seconds):
        fake.sleep(seconds + 0.012)

    pacer = _Pacer(PERIOD, clock=fake.clock, sleep=delayed_sleep)
    fake.work(0.010)
    timing = pacer.wait()
    assert timing["requested_sleep_seconds"] == pytest.approx(PERIOD - 0.010)
    assert timing["deadline_lateness_seconds"] == pytest.approx(0.012)
    assert timing["resynchronized"] is False
    assert pacer._deadline == pytest.approx(2 * PERIOD)


def test_pacer_reports_stall_without_sleeping_or_bursting():
    fake = FakeTime()
    pacer = _Pacer(PERIOD, clock=fake.clock, sleep=fake.sleep)
    fake.work(0.200)
    timing = pacer.wait()
    assert timing["requested_sleep_seconds"] == 0
    assert timing["deadline_lateness_seconds"] == pytest.approx(0.200 - PERIOD)
    assert timing["resynchronized"] is True
    assert fake.sleeps == []
    assert pacer._deadline == pytest.approx(0.200 + PERIOD)


# --- the real loops ----------------------------------------------------------------


@pytest.fixture
def paced(monkeypatch) -> tuple[RobotManager, FakeTime]:
    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    fake = FakeTime()

    def sleep_then_stop_at_one_second(seconds: float) -> None:
        fake.sleep(seconds)
        if fake.now >= 1.0 - 1e-9:
            manager._stop.set()

    monkeypatch.setattr(manager, "_clock", fake.clock)
    monkeypatch.setattr(manager, "_sleep", sleep_then_stop_at_one_second)
    yield manager, fake
    if manager.mode is not Mode.IDLE:
        manager.soft_stop("test cleanup")


def _run_to_idle(manager: RobotManager, start) -> int:
    seq0 = manager.telemetry()["sample_seq"]
    start()
    deadline = time.monotonic() + 5.0
    while manager.mode is not Mode.IDLE and time.monotonic() < deadline:
        time.sleep(0.005)
    assert manager.mode is Mode.IDLE, "the loop never reached its one-second stop"
    assert manager.fault is None
    return manager.telemetry()["sample_seq"] - seq0


def test_data_b4_teleop_and_record_loop_holds_30_hz_with_10_ms_of_work(paced, monkeypatch):
    manager, fake = paced
    cameras = manager.cameras()
    read = cameras.read

    def slow_read():
        fake.work(0.010)  # bus and camera time per frame
        return read()

    monkeypatch.setattr(cameras, "read", slow_read)

    frames = _run_to_idle(manager, lambda: manager.start_teleop(FPS, operator_id=OPERATOR))

    assert frames == FPS
    assert manager.telemetry()["fps"] == pytest.approx(FPS)


def test_data_b4_rollout_holds_30_hz_with_25_ms_of_inference(paced):
    manager, fake = paced

    class SlowPolicy(MockPolicy):
        def select_action(self, observation):
            fake.work(0.025)
            return super().select_action(observation)

    frames = _run_to_idle(
        manager,
        lambda: manager.start_rollout("models/x", FPS, operator_id=OPERATOR, policy=SlowPolicy()),
    )

    assert frames == FPS
    assert manager.telemetry()["fps"] == pytest.approx(FPS)
