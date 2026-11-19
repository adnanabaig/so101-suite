"""The v1 release has predeclared, machine-searchable acceptance thresholds."""

from pathlib import Path

PLAN = Path("docs/plans/2026-10-07-v1-e2e-execution.md")


def test_v1_acceptance_bar_is_numeric_and_covers_every_release_dimension() -> None:
    text = PLAN.read_text(encoding="utf-8")
    section = text.split("### Measurable v1.0 acceptance bar", 1)[1].split(
        "### M8 — v1 release acceptance", 1
    )[0]

    rows: dict[str, str] = {}
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| ---"):
            continue
        area, threshold = (cell.strip() for cell in line.strip("|").split("|", 1))
        if area != "Area":
            rows[area] = threshold

    assert rows == {
        "E-stop": (
            "In each of 3 trials, every motor on every follower and leader bus currently "
            "owned by RobotManager reports torque off and the latch is durable no later than "
            "1.0 s after the G15 accepts the stop request. No motor on an owned bus may receive "
            "a goal or torque-enable write from request ingress until a deliberate E-stop "
            "reset succeeds."
        ),
        "Halt": (
            "In each of 3 trials while moving at 45–50 °/s, worst overshoot is at most 3°, then "
            "worst drift is at most 1° over 10 s; the gripper never opens farther. Supervision "
            "loss starts Collapse within the configured grace plus 3 s, and hold expiry does so "
            "within the configured hold plus 3 s."
        ),
        "Collapse": (
            "Every joint stays at or below 110% of the configured 15 °/s cap in 0.25 s windows, "
            "finishes within `min(max_distance / speed + 2 s, 12 s)` plus one telemetry "
            "interval, ends within 3° of rest with torque reported off, and emits no fallback "
            "event. Pass all 3 trials."
        ),
        "Fault/read-back": (
            "A disconnected or silent servo never produces a confirmed-off result. Within "
            "3.0 s the operation reports a visible fault plus unconfirmed torque, and subsequent "
            "arming remains blocked across a process restart."
        ),
        "Health/control timing": (
            "Across a 10-minute 30 Hz motion-safe HIL sample, servo-health bus-lock hold p99 is "
            "at most 25 ms, control-loop interval p99 is at most 66.7 ms, no interval reaches the "
            "300 ms command TTL, and no false watchdog or supervision stop occurs."
        ),
        "Local command latency": (
            "Across at least 1,000 G15-local commands, API receipt to completed validated bus "
            "write is at most 100 ms at p95 and below the 300 ms TTL at maximum. The measurement "
            "uses the manager's monotonic timestamps, not browser wall clocks."
        ),
        "Reliability": (
            "M4–M7 complete twice consecutively from fresh handoffs with no source edits, "
            "unhandled exception, unexplained warning, skipped mandatory HIL probe or unresolved "
            "critical/high finding. One run includes a 60-minute powered, supervised soak with "
            "no bus-ownership violation or unintended stop."
        ),
        "Task success": (
            "Freeze one real-arm task, start-state distribution and evaluation procedure before "
            "collection. The final checkpoint succeeds in at least 16 of 20 operator-scored "
            "trials (80% observed; Wilson 95% interval reported), with zero collision, limit, "
            "stale-command or stop failures. Interrupted trials are reported and rerun only "
            "when the report records the predeclared non-policy cause."
        ),
    }
    assert "missing measurement" in section
    assert "editing this table after it is known" in section
