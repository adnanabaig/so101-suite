#!/usr/bin/env python3
"""Derive (or re-check) ``so101.simulation.rest_pose.MODEL_REST_POSE``.

Dev-only; needs the ``[sim]`` extra. The scan holds pan, roll and the gripper
(5, nearly closed) fixed and grids shoulder lift, elbow and wrist over the
folded region. Every pose that passes the section 5 criteria (see
``check_rest_pose``) is re-checked under a +-2 deg error on lift, elbow and
wrist, and ranked by:

1. no contact at any perturbed commanded pose (a calibration error must not
   command the arm into itself or the table);
2. fewest perturbed settle failures;
3. smallest settle of the nominal pose;
4. smallest reach from the base axis.

    PYTHONPATH=src python scripts/derive_rest_pose.py          # scan, print top 5
    PYTHONPATH=src python scripts/derive_rest_pose.py --check  # verify MODEL_REST_POSE
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from so101.simulation.rest_pose import MODEL_REST_POSE, check_rest_pose, format_report

FIXED = {"shoulder_pan": 0.0, "wrist_roll": 0.0, "gripper": 5.0}
LIFTS = np.arange(-95.0, -59.0, 5.0)
ELBOWS = np.arange(60.0, 95.0, 0.5)
WRISTS = np.arange(-90.0, 91.0, 5.0)


def scan(top: int) -> list[dict[str, float]]:
    passing = []
    for lift in LIFTS:
        for elbow in ELBOWS:
            for wrist in WRISTS:
                pose = FIXED | {
                    "shoulder_lift": float(lift),
                    "elbow_flex": float(elbow),
                    "wrist_flex": float(wrist),
                }
                report = check_rest_pose(pose, perturbation_deg=0.0)
                if report.ok:
                    passing.append(pose)
    print(f"{len(passing)} of {len(LIFTS) * len(ELBOWS) * len(WRISTS)} poses pass nominally")
    ranked = []
    for pose in passing:
        report = check_rest_pose(pose)
        robust = report.robustness
        worst = max(check.settle for check in report.joints.values())
        key = (robust["commanded_collisions"], robust["settle_failures"], worst, report.reach_m)
        ranked.append((key, pose))
    ranked.sort(key=lambda item: item[0])
    for key, pose in ranked[:top]:
        print(f"{key}  {pose}")
    return [pose for _, pose in ranked[:top]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only check MODEL_REST_POSE")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.check:
        scan(args.top)
    report = check_rest_pose(MODEL_REST_POSE)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
