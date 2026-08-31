"""Differentiable corridor estimates used by the predictive controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .math_utils import is_symbolic


@dataclass(frozen=True)
class StraightGapCorridor:
    """Synthetic laser-derived corridor containing one smooth constriction.

    In the prototype this object plays the role of the perception block.  It
    exposes the same differentiable width model to the controller and numeric
    samples to the simulator/visualiser.
    """

    open_width: float = 1.20
    gap_width: float = 0.58
    gap_start: float = 1.25
    gap_end: float = 2.10
    transition_length: float = 0.18
    centre_y: float = 0.0

    def narrowing_factor(self, x_position: Any) -> Any:
        sharpness = 2.5 / self.transition_length
        if is_symbolic(x_position):
            entering = 0.5 * (1.0 + ca.tanh(sharpness * (x_position - self.gap_start)))
            leaving = 0.5 * (1.0 + ca.tanh(sharpness * (x_position - self.gap_end)))
        else:
            entering = 0.5 * (1.0 + np.tanh(sharpness * (x_position - self.gap_start)))
            leaving = 0.5 * (1.0 + np.tanh(sharpness * (x_position - self.gap_end)))
        return entering - leaving

    def full_width(self, x_position: Any) -> Any:
        return self.open_width - (self.open_width - self.gap_width) * self.narrowing_factor(
            x_position
        )

    def half_width(self, x_position: Any) -> Any:
        return 0.5 * self.full_width(x_position)

    def lateral_bounds(self, x_position: Any) -> tuple[Any, Any]:
        half_width = self.half_width(x_position)
        return self.centre_y - half_width, self.centre_y + half_width

    def preview(self, x_positions: np.ndarray) -> np.ndarray:
        return np.asarray(self.full_width(np.asarray(x_positions)), dtype=float)

    def wall_profiles(
        self,
        x_min: float,
        x_max: float,
        samples: int = 500,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_positions = np.linspace(x_min, x_max, samples)
        half_widths = np.asarray(self.half_width(x_positions), dtype=float)
        return (
            x_positions,
            self.centre_y + half_widths,
            self.centre_y - half_widths,
        )
