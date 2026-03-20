"""
Interpolation methods for discount-curve construction.

Two interpolators are provided:

1. **LogLinearInterpolator** -- piecewise-linear interpolation on log(D(t)),
   which is equivalent to piecewise-constant forward rates between knot points.

2. **MonotoneConvexInterpolator** -- monotone Hermite (PCHIP) interpolation on
   continuously-compounded zero rates, which guarantees non-negative
   instantaneous forward rates.
"""

from __future__ import annotations

import bisect
import math
from typing import List, Sequence

import numpy as np


class LogLinearInterpolator:
    """Piecewise-linear interpolation on log-discount-factors.

    Given knot points ``(t_i, ln D(t_i))``, the interpolated value at *t* is::

        ln D(t) = ln D(t_i) + (t - t_i) / (t_{i+1} - t_i)
                  * (ln D(t_{i+1}) - ln D(t_i))

    For example, with knots at t=1 (lnD=-0.04) and t=2 (lnD=-0.09),
    evaluating at t=1.5 gives lnD = -0.04 + 0.5/1.0 * (-0.09 - (-0.04))
    = -0.04 + 0.5*(-0.05) = -0.065, so D(1.5) = exp(-0.065) ~ 0.9371.

    Parameters
    ----------
    times : sequence of float
        Strictly increasing knot times (year fractions), e.g. [0.0, 0.25, 1.0, 5.0].
    values : sequence of float
        Corresponding log-discount-factors, e.g. [0.0, -0.011, -0.043, -0.215].
    """

    def __init__(self, times: Sequence[float], values: Sequence[float]) -> None:
        if len(times) != len(values):
            raise ValueError("times and values must have the same length")
        if len(times) < 2:
            raise ValueError("Need at least two knot points")
        self._times: List[float] = list(times)
        self._values: List[float] = list(values)

    def __call__(self, t: float) -> float:
        """Interpolate log-discount-factor at year fraction *t*.

        Flat extrapolation outside the knot range.
        """
        if t <= self._times[0]:
            return self._values[0]
        if t >= self._times[-1]:
            # Flat forward extrapolation: extend the last segment's slope
            slope = (self._values[-1] - self._values[-2]) / (
                self._times[-1] - self._times[-2]
            )
            return self._values[-1] + slope * (t - self._times[-1])

        # Binary search for the bracketing interval
        idx = bisect.bisect_right(self._times, t) - 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        v0, v1 = self._values[idx], self._values[idx + 1]
        w = (t - t0) / (t1 - t0)
        return v0 + w * (v1 - v0)


class MonotoneConvexInterpolator:
    """Monotone Hermite (PCHIP) interpolation on zero rates.

    Interpolates continuously-compounded zero rates ``r(t)`` using the
    Fritsch-Carlson monotone piecewise cubic Hermite method, which
    guarantees that the resulting instantaneous forward rate curve
    ``f(t) = r(t) + t * r'(t)`` is non-negative when the input zero
    rates are non-negative.

    Parameters
    ----------
    times : sequence of float
        Strictly increasing knot times (year fractions).
    values : sequence of float
        Corresponding continuously-compounded zero rates (decimal),
        e.g. [0.043, 0.042, 0.044, 0.046].
    """

    def __init__(self, times: Sequence[float], values: Sequence[float]) -> None:
        if len(times) != len(values):
            raise ValueError("times and values must have the same length")
        if len(times) < 2:
            raise ValueError("Need at least two knot points")
        self._times = np.asarray(times, dtype=float)
        self._values = np.asarray(values, dtype=float)
        self._n = len(self._times)
        self._slopes = self._compute_slopes()

    def _compute_slopes(self) -> np.ndarray:
        """Fritsch-Carlson monotone slope computation.

        For interior knots, the slope is the harmonic mean of adjacent
        secants when the secants have the same sign, and zero otherwise.
        """
        n = self._n
        h = np.diff(self._times)  # e.g. [0.25, 0.75, 4.0] for times [0, 0.25, 1, 5]
        delta = np.diff(self._values) / h  # secant slopes

        slopes = np.zeros(n)
        # Interior points
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] > 0:
                # Harmonic mean
                slopes[i] = (
                    2.0 * delta[i - 1] * delta[i] / (delta[i - 1] + delta[i])
                )
            else:
                slopes[i] = 0.0

        # End slopes: one-sided
        slopes[0] = delta[0]
        slopes[-1] = delta[-1]

        # Fritsch-Carlson monotonicity correction
        for i in range(n - 1):
            if abs(delta[i]) < 1e-30:
                slopes[i] = 0.0
                slopes[i + 1] = 0.0
            else:
                alpha = slopes[i] / delta[i]
                beta = slopes[i + 1] / delta[i]
                # Ensure we stay in the monotonicity region
                if alpha**2 + beta**2 > 9.0:
                    tau = 3.0 / math.sqrt(alpha**2 + beta**2)
                    slopes[i] = tau * alpha * delta[i]
                    slopes[i + 1] = tau * beta * delta[i]

        return slopes

    def __call__(self, t: float) -> float:
        """Interpolate zero rate at year fraction *t*.

        Flat extrapolation outside the knot range.
        """
        if t <= self._times[0]:
            return float(self._values[0])
        if t >= self._times[-1]:
            return float(self._values[-1])

        idx = int(np.searchsorted(self._times, t, side="right")) - 1
        t0 = self._times[idx]
        t1 = self._times[idx + 1]
        h = t1 - t0
        s = (t - t0) / h  # local parameter in [0, 1]

        # Cubic Hermite basis
        h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
        h10 = s * (1.0 - s) ** 2
        h01 = s**2 * (3.0 - 2.0 * s)
        h11 = s**2 * (s - 1.0)

        v0 = self._values[idx]
        v1 = self._values[idx + 1]
        m0 = self._slopes[idx] * h
        m1 = self._slopes[idx + 1] * h

        return float(h00 * v0 + h10 * m0 + h01 * v1 + h11 * m1)
