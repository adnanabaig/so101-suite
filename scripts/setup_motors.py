#!/usr/bin/env python3
"""Print the SO-101 motor-setup procedure and LeRobot's command for each arm.

This script only prints instructions; it does not talk to the motors. LeRobot's
``lerobot-setup-motors`` assigns the ids and the bus baudrate, one arm at a time.
The commands use the ports and ids from your SO101_* settings (.env).
Run: make setup-motors.

Checked against LeRobot 0.6.1, the version the robot extra pins:
``lerobot/scripts/lerobot_setup_motors.py`` takes either ``--robot.*`` or
``--teleop.*``, ``so101_follower`` and ``so101_leader`` are the registered
types, and ``setup_motors()`` prompts for one motor at a time from the gripper
(id 6) down to shoulder_pan (id 1), setting each to 1,000,000 baud.
"""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import sys
from pathlib import Path

# Import so101 from the checkout this script lives in, whatever the cwd.
_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "so101").is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from so101 import JOINT_NAMES  # noqa: E402
from so101.config import Settings  # noqa: E402

LEROBOT_BAUDRATE = 1_000_000  # lerobot.motors.feetech.feetech.DEFAULT_BAUDRATE


def setup_commands(settings: Settings) -> dict[str, list[str]]:
    """The ``lerobot-setup-motors`` argv for each arm."""
    return {
        "follower": [
            "lerobot-setup-motors",
            "--robot.type=so101_follower",
            f"--robot.port={settings.follower_port}",
            f"--robot.id={settings.follower_id}",
        ],
        "leader": [
            "lerobot-setup-motors",
            "--teleop.type=so101_leader",
            f"--teleop.port={settings.leader_port}",
            f"--teleop.id={settings.leader_id}",
        ],
    }


def main() -> None:
    # Settings() rather than get_settings(): printing must not create data dirs.
    settings = Settings()
    commands = setup_commands(settings)
    print(
        "SO-101 motor setup. This script only prints instructions; it does not talk to the motors."
    )
    print()
    print(f"  follower port: {settings.follower_port}")
    print(f"  leader port:   {settings.leader_port}")
    print(f"  baudrate:      {settings.baudrate}")
    if settings.baudrate != LEROBOT_BAUDRATE:
        print(f"  warning: lerobot-setup-motors sets the motors to {LEROBOT_BAUDRATE} baud")
    print()
    print("Every STS3215 ships with id 1, so each motor needs its own id. LeRobot")
    print("asks for one motor at a time; connect only that motor to the board:")
    for motor_id, name in reversed(list(enumerate(JOINT_NAMES, start=1))):
        print(f"  {name:<14} -> id {motor_id}")
    print()
    print("Run this once per arm, with only that arm's controller board connected:")
    print(f"  follower:  {shlex.join(commands['follower'])}")
    print(f"  leader:    {shlex.join(commands['leader'])}")
    print()
    if shutil.which("lerobot-setup-motors"):
        print("lerobot-setup-motors is on PATH.")
    elif importlib.util.find_spec("lerobot") is not None:
        print("LeRobot is installed, but lerobot-setup-motors is not on PATH; activate its venv.")
    else:
        print("LeRobot is not installed. On the robot host, run: make install-robot")
    print("Then calibrate each arm: so101 calibrate follower, so101 calibrate leader.")


if __name__ == "__main__":
    main()
