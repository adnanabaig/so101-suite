"""Small-sample statistics for evaluation success rates.

Robot evals are a handful of binary trials, so the normal approximation
(p +/- 1.96 * sqrt(p(1-p)/n)) collapses to a zero-width interval at 0/10
or 10/10. The Wilson score interval stays inside [0, 1] and keeps an
honest width at small n.
"""

from __future__ import annotations

import math

# Two-sided 95% normal quantile, Phi^-1(0.975).
Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n`` Bernoulli trials.

    Returns ``(low, high)``; ``(0.0, 1.0)`` when ``n == 0`` (no information).
    Raises ``ValueError`` unless ``0 <= successes <= n`` and ``z > 0``.
    """

    if isinstance(successes, bool) or isinstance(n, bool):
        raise ValueError("successes and n must be integers")
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if not (z > 0 and math.isfinite(z)):
        raise ValueError("z must be a positive finite number")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return low, high


def success_summary(successes: int, n: int) -> str:
    """``"8/10 = 80% (95% CI 49-94%)"``: the way eval results are shown."""

    if n == 0:
        return "0 scored trials"
    low, high = wilson_interval(successes, n)
    return f"{successes}/{n} = {successes / n:.0%} (95% CI {round(low * 100)}-{round(high * 100)}%)"
