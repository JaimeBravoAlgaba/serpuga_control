"""Differentiable corridor estimates used by the predictive controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import casadi as ca
import numpy as np

from .math_utils import is_symbolic, smooth_abs, smooth_maximum


@dataclass(frozen=True)
class StraightGapCorridor:
    """Synthetic laser-derived corridor containing one smooth constriction.

    The corridor centreline is parallel to the world x axis and located at
    ``centre_y``.  In the prototype this object plays the role of the future
    perception block: it exposes a differentiable width model to the NMPC and
    numeric samples to the simulator/visualiser.
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
        return self.open_width - (
            self.open_width - self.gap_width
        ) * self.narrowing_factor(x_position)

    def half_width(self, x_position: Any) -> Any:
        return 0.5 * self.full_width(x_position)

    def lateral_bounds(self, x_position: Any) -> tuple[Any, Any]:
        half_width = self.half_width(x_position)
        return self.centre_y - half_width, self.centre_y + half_width

    def clearance_residual(
        self,
        vertices_world: Iterable[Any],
        margin: float,
        epsilon: float,
    ) -> Any:
        """Return one scalar wall-clearance residual for a complete footprint.

        Each vertex is feasible when

        ``abs(y - centre_y) + margin <= half_width(x)``.

        The maximum residual over the footprint is collapsed into one smooth
        scalar, so the NMPC still receives a single geometric inequality while
        respecting the actual world-frame corridor walls and their lateral
        offset.
        """

        residuals = [
            smooth_abs(vertex[1] - self.centre_y, epsilon)
            + margin
            - self.half_width(vertex[0])
            for vertex in vertices_world
        ]
        return smooth_maximum(residuals, epsilon)

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
