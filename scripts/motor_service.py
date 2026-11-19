"""Serve the motor API, or monitor an existing instance without interrupting it."""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def existing():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/robot/status", timeout=3) as reply:
            status = json.load(reply)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return False
        raise
    if status.get("backend") != "real":
        raise RuntimeError("Port 8000 is occupied by a server that is not the real backend")
    return True


def main():
    if not existing():
        launcher = ROOT / "config/launch-real.sh"
        os.execv(str(launcher), [str(launcher)])
    print("Monitoring existing real motor API; its process remains externally managed", flush=True)
    while existing():
        time.sleep(1)
    raise RuntimeError("Existing motor API stopped; run services up to start a managed server")


if __name__ == "__main__":
    main()
