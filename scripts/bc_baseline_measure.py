#!/usr/bin/env python3
"""Measure the numpy BC baseline on the toy reach task, or write the toy dataset.

Run it with so101 installed in the active environment. Reproduce the numbers
in docs/learning.md (about 20 s on a laptop CPU)::

    python scripts/bc_baseline_measure.py --seeds 0,1,2
    python scripts/bc_baseline_measure.py --quick    # ~1 s smoke run

Write the toy demonstrations as a dataset, then train on it like any recording::

    python scripts/bc_baseline_measure.py --write-dataset you/toy_reach
    so101 train you/toy_reach --policy bc_baseline

The toy reach task (``so101.training.toy_reach``) is a kinematic stand-in for
the simulator's reach task. Its success rates show whether the training and
policy plumbing works. They are not simulator or real-arm success rates.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from so101.eval.stats import wilson_interval
from so101.policy.baseline import BaselinePolicy
from so101.training.bc_baseline import BCConfig, fit_bc, with_overrides
from so101.training.toy_reach import (
    ExpertPolicy,
    ToyEpisode,
    evaluate_toy_reach,
    generate_toy_reach_episodes,
    toy_training_data,
    write_toy_reach_dataset,
)


@dataclass
class Variant:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    selection: str = "kept"
    mixed: bool = False  # train on the dataset where some demonstrations miss


VARIANTS = (
    Variant("privileged (default)"),
    Variant("privileged, absolute target", {"target": "absolute"}),
    Variant("privileged, history 3", {"history": 3}),
    Variant("privileged, time feature", {"time_feature": "time"}),
    Variant("state-only", {"state_only": True}),
    Variant("privileged, mixed data, kept", selection="kept", mixed=True),
    Variant("privileged, mixed data, all", selection="all", mixed=True),
)


class HoldPolicy:
    """Commands the current state, so the arm never moves: the lower bound."""

    def reset(self) -> None:
        return None

    def select_action(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(observation["observation.state"], dtype=np.float32)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _interval(successes: int, trials: int) -> str:
    low, high = wilson_interval(successes, trials)
    return f"[{_percent(low)}, {_percent(high)}]"


def _seeds(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"seeds must be integers, not {text!r}") from None
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("give one or more non-negative seeds, e.g. 0,1,2")
    return seeds


def _hidden(text: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part) for part in text.replace("x", ",").split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"hidden must be sizes like 128,128, not {text!r}"
        ) from None
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("hidden must list positive layer sizes, e.g. 128,128")
    return sizes


def _variants(text: str | None) -> list[Variant]:
    if not text:
        return list(VARIANTS)
    chosen = []
    for part in text.split(","):
        try:
            chosen.append(VARIANTS[int(part) - 1])
        except (ValueError, IndexError):
            raise argparse.ArgumentTypeError(
                f"variants are numbers from 1 to {len(VARIANTS)}, not {part!r}"
            ) from None
    return chosen


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the numpy BC baseline on the toy reach task.",
        epilog="Variants: " + "; ".join(f"{i}. {v.name}" for i, v in enumerate(VARIANTS, start=1)),
    )
    parser.add_argument("--episodes", type=int, default=60, help="demonstrations (default 60)")
    parser.add_argument("--steps", type=int, default=3000, help="training steps (default 3000)")
    parser.add_argument("--hidden", type=_hidden, default=(128, 128), help="default 128,128")
    parser.add_argument("--trials", type=int, default=100, help="closed-loop trials (default 100)")
    parser.add_argument("--seeds", type=_seeds, default=[0], help="training seeds (default 0)")
    parser.add_argument("--data-seed", type=int, default=0, help="demonstration seed (default 0)")
    parser.add_argument(
        "--eval-seed", type=int, default=1000, help="closed-loop target seed (default 1000)"
    )
    parser.add_argument(
        "--failure-fraction",
        type=float,
        default=0.3,
        help="share of distracted demonstrations in the mixed dataset and in --write-dataset "
        "(default 0.3)",
    )
    parser.add_argument("--variants", help="comma-separated variant numbers (default all)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke run: 12 episodes, 300 steps, 32x32, 10 trials (overrides those options)",
    )
    parser.add_argument(
        "--write-dataset",
        metavar="REPO_ID",
        help="write the demonstrations as an SO-101 dataset under SO101_DATASETS_DIR and exit",
    )
    parser.add_argument(
        "--no-env-state",
        action="store_true",
        help="with --write-dataset: leave out observation.environment_state",
    )
    args = parser.parse_args(argv)
    try:
        args.variants = _variants(args.variants)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.quick:
        args.episodes, args.steps, args.hidden, args.trials = 12, 300, (32, 32), 10
    if args.episodes < 2 or args.steps < 1 or args.trials < 1:
        parser.error("--episodes must be at least 2, --steps and --trials at least 1")
    if not 0.0 <= args.failure_fraction < 1.0:
        parser.error("--failure-fraction must be from 0 to below 1")
    return args


def write_dataset(args: argparse.Namespace) -> int:
    episodes = generate_toy_reach_episodes(
        args.episodes, seed=args.data_seed, failure_fraction=args.failure_fraction
    )
    privileged = not args.no_env_state
    try:
        root = write_toy_reach_dataset(args.write_dataset, episodes, privileged=privileged)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    failures = sum(not episode.success for episode in episodes)
    kind = "with" if privileged else "without"
    print(
        f"wrote {len(episodes)} toy reach demonstrations ({failures} labeled failure) "
        f"{kind} observation.environment_state to {root}"
    )
    print(f"train a baseline on it: so101 train {args.write_dataset} --policy bc_baseline")
    return 0


@dataclass
class Row:
    name: str
    episodes: int = 0
    seconds: list[float] = field(default_factory=list)
    mae: list[float] = field(default_factory=list)
    successes: list[int] = field(default_factory=list)
    mean_predictor: list[float] = field(default_factory=list)
    hold_predictor: list[float] = field(default_factory=list)


def _measure(
    variant: Variant,
    clean: Sequence[ToyEpisode],
    mixed: Sequence[ToyEpisode],
    args: argparse.Namespace,
) -> Row:
    data = toy_training_data(mixed if variant.mixed else clean, selection=variant.selection)
    row = Row(variant.name, episodes=len(data.episodes))
    base = BCConfig(steps=args.steps, hidden=args.hidden)
    for seed in args.seeds:
        result = fit_bc(data, with_overrides(base, seed=seed, **variant.overrides))
        policy = BaselinePolicy(result.model)
        outcome = evaluate_toy_reach(
            policy,
            args.trials,
            seed=args.eval_seed,
            provide_env_state=policy.privileged_state,
        )
        metrics = result.metrics
        row.seconds.append(float(metrics["wall_seconds"]))
        row.mae.append(float(metrics["mae"]))
        row.successes.append(outcome.successes)
        row.mean_predictor.append(float(metrics["mean_predictor_mae"]))
        row.hold_predictor.append(float(metrics["hold_predictor_mae"]))
    return row


def measure(args: argparse.Namespace) -> int:
    started = time.monotonic()
    clean = generate_toy_reach_episodes(args.episodes, seed=args.data_seed)
    mixed = generate_toy_reach_episodes(
        args.episodes, seed=args.data_seed, failure_fraction=args.failure_fraction
    )
    generated = time.monotonic() - started
    seeds = ",".join(str(seed) for seed in args.seeds)
    print(
        f"toy reach: {len(clean)} demonstrations ({sum(e.success for e in clean)} succeed); "
        f"mixed: {sum(e.distracted for e in mixed)} distracted, "
        f"{sum(not e.success for e in mixed)} labeled failure; generated in {generated:.2f} s"
    )
    hidden = "x".join(str(size) for size in args.hidden)
    print(
        f"training: {args.steps} steps, hidden {hidden}, seeds {seeds}; closed loop: "
        f"{args.trials} trials per policy on targets from seed {args.eval_seed}"
    )
    print()
    expert = evaluate_toy_reach(ExpertPolicy(), args.trials, seed=args.eval_seed)
    hold = evaluate_toy_reach(HoldPolicy(), args.trials, seed=args.eval_seed)
    print(
        f"{'variant':<30} {'episodes':>8} {'train s':>8} {'MAE':>7}  "
        f"{'success per seed':<18} {'rate':>6}  Wilson 95%"
    )
    for name, outcome in (("scripted expert (upper bound)", expert), ("hold (lower bound)", hold)):
        print(
            f"{name:<30} {'-':>8} {'-':>8} {'-':>7}  {outcome.successes:<18} "
            f"{_percent(outcome.success_rate):>6}  {_interval(outcome.successes, args.trials)}"
        )
    rows = []
    for variant in args.variants:
        row = _measure(variant, clean, mixed, args)
        rows.append(row)
        total = sum(row.successes)
        trials = args.trials * len(row.successes)
        print(
            f"{row.name:<30} {row.episodes:>8} {np.mean(row.seconds):>8.2f} "
            f"{np.mean(row.mae):>7.3f}  {' '.join(str(s) for s in row.successes):<18} "
            f"{_percent(total / trials):>6}  {_interval(total, trials)}"
        )
    print()
    if rows:
        first = rows[0]
        print(
            f"MAE: held-out action error in degrees (gripper 0-100), mean over seeds; for "
            f"{first.name}, predicting the mean action scores {np.mean(first.mean_predictor):.3f} "
            f"and holding position {np.mean(first.hold_predictor):.3f}"
        )
    print(
        "rate and Wilson 95%: closed-loop reach success pooled over seeds; toy kinematic "
        "task, not simulator or real-arm success"
    )
    print(f"total {time.monotonic() - started:.1f} s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_dataset:
        return write_dataset(args)
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())
