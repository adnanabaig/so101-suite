"""Measure the scripted expert's success rate per physics task (the numbers in docs/simulation.md).

    PYTHONPATH=src python scripts/measure_sim_expert.py --episodes 100
    PYTHONPATH=src python scripts/measure_sim_expert.py --episodes 50 \
        --noise-deg 1 --waypoint-noise-m 0.004

Every episode goes through the real SafetyGuard and the STS3215 servo profile
(``--servo step`` for plain position targets). Prints one JSON object per task
and a summary table; ``--out`` also writes everything to a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from so101.simulation.expert import ExpertConfig
from so101.simulation.policy import evaluate
from so101.simulation.tasks import TASK_NAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASK_NAMES), choices=TASK_NAMES)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="expert")
    parser.add_argument("--noise-deg", type=float, default=0.0)
    parser.add_argument("--waypoint-noise-m", type=float, default=0.0)
    parser.add_argument("--servo", choices=("sts3215", "step"), default="sts3215")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    config = ExpertConfig(noise_deg=args.noise_deg, waypoint_noise_m=args.waypoint_noise_m)
    reports = []
    for task in args.tasks:
        report = evaluate(
            task,
            args.policy,
            episodes=args.episodes,
            seed=args.seed,
            servo_mode=args.servo,
            expert=config,
        )
        data = report.as_dict()
        data["expert_config"] = vars(config)
        data["failed_seeds"] = [o["seed"] for o in report.outcomes if not o["success"]]
        reports.append(data)
        brief = {k: v for k, v in data.items() if k != "outcomes"}
        print(json.dumps(brief), flush=True)
    header = ("task", "policy", "success", "wilson95", "t_succ", "RTF")
    print("{:12s} {:8s} {:>9s} {:>16s} {:>7s} {:>6s}".format(*header))
    for data in reports:
        low, high = data["wilson95"]
        print(
            f"{data['task']:12s} {data['policy']:8s} "
            f"{data['successes']:>4d}/{data['episodes']:<4d} "
            f"[{low:.3f}, {high:.3f}]   {data['mean_first_success_s'] or float('nan'):6.2f} "
            f"{data['real_time_factor']:6.0f}"
        )
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
