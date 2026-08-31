"""Small mathematical helpers shared by numeric and symbolic code."""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce
from typing import Any

import casadi as ca
import numpy as np

CASADI_TYPES = (ca.MX, ca.SX, ca.DM)


def is_symbolic(value: Any) -> bool:
    return isinstance(value, CASADI_TYPES)


def rotation_2d(angle: Any) -> Any:
    """Return a 2-D rotation matrix for NumPy or CasADi scalars."""

    if is_symbolic(angle):
        return ca.vertcat(
            ca.horzcat(ca.cos(angle), -ca.sin(angle)),
            ca.horzcat(ca.sin(angle), ca.cos(angle)),
        )
    c = np.cos(float(angle))
    s = np.sin(float(angle))
    return np.array([[c, -s], [s, c]], dtype=float)


def smooth_abs(value: Any, epsilon: float) -> Any:
    if is_symbolic(value):
        return ca.sqrt(value * value + epsilon * epsilon)
    return np.sqrt(value * value + epsilon * epsilon)


def smooth_max(a: Any, b: Any, epsilon: float) -> Any:
    return 0.5 * (a + b + smooth_abs(a - b, epsilon))


def smooth_min(a: Any, b: Any, epsilon: float) -> Any:
    return 0.5 * (a + b - smooth_abs(a - b, epsilon))


def smooth_maximum(values: Iterable[Any], epsilon: float) -> Any:
    return reduce(lambda a, b: smooth_max(a, b, epsilon), values)


def smooth_minimum(values: Iterable[Any], epsilon: float) -> Any:
    return reduce(lambda a, b: smooth_min(a, b, epsilon), values)


def as_column(vector: np.ndarray) -> ca.DM:
    return ca.DM(np.asarray(vector, dtype=float).reshape((-1, 1)))


J2_NUMPY = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
J2_CASADI = ca.DM(J2_NUMPY)
