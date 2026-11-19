#!/usr/bin/env python3
"""List physical cameras for SO101_CAMERAS on the real backend.

Run: make find-cameras. RealSense devices are listed with paste-ready
``realsense:<serial>`` entries (pyrealsense2, the realsense extra); OpenCV
indices are opened one at a time (cv2, the robot extra). Either package being
missing prints how to install it. The report comes from
``so101.hardware.camera_discovery``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import so101 from the checkout this script lives in, whatever the cwd.
_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "so101").is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from so101.config import Settings  # noqa: E402
from so101.hardware.camera_discovery import (  # noqa: E402
    camera_discovery_refusal,
    discover_cameras,
    render_report,
)


def main() -> None:
    # Settings(), unlike get_settings(), does not create the data directories:
    # discovery remains a read-only host probe on the real backend.
    refusal = camera_discovery_refusal(Settings())
    if refusal is not None:
        print(f"refused: {refusal}", file=sys.stderr)
        raise SystemExit(3)
    print(render_report(discover_cameras()))


if __name__ == "__main__":
    main()
