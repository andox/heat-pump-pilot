"""Small bounded least-squares solver shared by the online learning models."""

from __future__ import annotations

from itertools import product
import math


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def solve(matrix, rhs):
    a = [list(row) + [b] for row, b in zip(matrix, rhs)]
    for i in range(len(a)):
        pivot = max(range(i, len(a)), key=lambda j: abs(a[j][i]))
        if abs(a[pivot][i]) < 1e-10:
            return None
        a[i], a[pivot] = a[pivot], a[i]
        scale = a[i][i]
        a[i] = [v / scale for v in a[i]]
        for j in range(len(a)):
            if i != j:
                scale = a[j][i]
                a[j] = [v - scale * w for v, w in zip(a[j], a[i])]
    return [row[-1] for row in a]


def bounded_fit(observations, bounds, prior=None, weights=None):
    """Fit 2-3 parameters; equal lower/upper bounds freeze a parameter."""
    n = len(bounds)
    xx, xy = [[0.0] * n for _ in range(n)], [0.0] * n
    for x, y in observations:
        for i in range(n):
            xy[i] += x[i] * y
            for j in range(n):
                xx[i][j] += x[i] * x[j]
    for i in range(n):
        weight = weights[i] if weights else 1e-9
        xx[i][i] += weight
        xy[i] += weight * (prior[i] if prior else 0.0)
    best = None
    choices = [(-1,) if lo == hi else (-1, 0, 1) for lo, hi in bounds]
    for active in product(*choices):
        theta = [
            bounds[i][0 if a == -1 else 1] if a else 0.0 for i, a in enumerate(active)
        ]
        free = [i for i, a in enumerate(active) if not a]
        if free:
            fitted = solve(
                [[xx[i][j] for j in free] for i in free],
                [
                    xy[i] - sum(xx[i][j] * theta[j] for j in range(n) if active[j])
                    for i in free
                ],
            )
            if fitted is None:
                continue
            for i, value in zip(free, fitted):
                theta[i] = value
        if any(
            not math.isfinite(v) or v < lo - 1e-8 or v > hi + 1e-8
            for v, (lo, hi) in zip(theta, bounds)
        ):
            continue
        score = sum(
            theta[i] * xx[i][j] * theta[j] for i in range(n) for j in range(n)
        ) - 2 * sum(v * b for v, b in zip(theta, xy))
        if best is None or score < best[0]:
            best = score, theta
    return best[1] if best else None
