"""Centralised, typed configuration for the robot and the controller."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _default_pivots() -> np.ndarray:
    # q=(0, 0) places both tracks parallel and side by side. Coordinates are
    # expressed from the centre of the rigid bar, which is the controlled
    # point of the planar model.
    return np.array([[0.0, 0.24], [0.0, -0.24]], dtype=float)


def _default_offsets() -> np.ndarray:
    return np.array([[0.21, 0.0], [0.21, 0.0]], dtype=float)


def _default_nominal_configuration() -> np.ndarray:
    return np.zeros(2, dtype=float)


def _default_symmetry_coupling() -> np.ndarray:
    return np.ones(2, dtype=float)


@dataclass(frozen=True)
class RobotParameters:
    """Geometric, inertial and actuator parameters.

    The defaults are deliberately illustrative.  All dimensions are in SI
    units and live here so the control code does not depend on the prototype
    geometry.
    """

    pivot_positions: np.ndarray = field(default_factory=_default_pivots)
    track_center_offsets: np.ndarray = field(default_factory=_default_offsets)
    track_length: float = 0.40
    track_width: float = 0.14
    connector_thickness: float = 0.05

    body_mass: float = 6.0
    track_mass: float = 3.0
    body_com: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0], dtype=float)
    )
    com_height: float = 0.24

    # The illustrative mechanism folds symmetrically: q1 <= 0 and q2 >= 0.
    q_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([-70.0, 0.0]))
    )
    q_max: np.ndarray = field(default_factory=lambda: np.deg2rad(np.array([0.0, 70.0])))
    nominal_configuration: np.ndarray = field(
        default_factory=_default_nominal_configuration
    )
    symmetry_coupling: np.ndarray = field(default_factory=_default_symmetry_coupling)
    track_speed_limit: float = 1.10
    articulation_rate_limit: float = np.deg2rad(75.0)
    track_acceleration_limit: float = 2.5
    articulation_acceleration_limit: float = np.deg2rad(210.0)

    longitudinal_slip_weight: float = 1.0
    lateral_slip_weight: float = 6.0

    @classmethod
    def opposed_tracks(cls) -> RobotParameters:
        """Geometry with a transverse bar and antiparallel nominal tracks.

        Track 1 extends along +x and track 2 along -x.  Their pivots lie on
        the body y axis, so the connector is perpendicular to forward motion.
        """

        return cls(
            pivot_positions=np.array([[0.0, 0.24], [0.0, -0.24]], dtype=float),
            q_min=np.deg2rad(np.array([-70.0, 110.0])),
            q_max=np.deg2rad(np.array([0.0, 180.0])),
            nominal_configuration=np.deg2rad(np.array([0.0, 180.0])),
            symmetry_coupling=np.array([-1.0, 1.0], dtype=float),
        )

    def __post_init__(self) -> None:
        if self.pivot_positions.shape != (2, 2):
            raise ValueError("pivot_positions must have shape (2, 2)")
        if not np.allclose(
            np.mean(self.pivot_positions, axis=0),
            np.zeros(2),
            atol=1.0e-9,
        ):
            raise ValueError(
                "pivot_positions must be expressed from the bar centre "
                "(their midpoint must be [0, 0])"
            )
        if self.track_center_offsets.shape != (2, 2):
            raise ValueError("track_center_offsets must have shape (2, 2)")
        if self.q_min.shape != (2,) or self.q_max.shape != (2,):
            raise ValueError("q_min and q_max must have shape (2,)")
        if self.nominal_configuration.shape != (2,):
            raise ValueError("nominal_configuration must have shape (2,)")
        if self.symmetry_coupling.shape != (2,):
            raise ValueError("symmetry_coupling must have shape (2,)")
        if np.linalg.norm(self.symmetry_coupling) <= 0.0:
            raise ValueError("symmetry_coupling must be non-zero")
        if np.any(self.q_min >= self.q_max):
            raise ValueError("Every q_min must be lower than q_max")
        if np.any(self.nominal_configuration < self.q_min) or np.any(
            self.nominal_configuration > self.q_max
        ):
            raise ValueError(
                "nominal_configuration must lie inside the articulation limits"
            )


@dataclass(frozen=True)
class MPCParameters:
    """Prediction, cost and constraint settings for the NMPC."""

    dt: float = 0.15
    horizon_steps: int = 18

    position_weight: float = 8.0
    heading_weight: float = 24.0
    velocity_weight: float = 90.0
    yaw_rate_weight: float = 35.0
    slip_weight: float = 4.0
    scrub_weight: float = 0.08
    articulation_rate_weight: float = 0.08
    track_effort_weight: float = 0.015
    input_rate_weight: float = 0.12
    symmetry_weight: float = 10.0
    track_alignment_weight: float = 0.5
    stability_weight: float = 0.35
    nominal_configuration_weight: float = 0.018
    terminal_position_weight: float = 14.0
    terminal_heading_weight: float = 5.0

    clearance_margin: float = 0.01
    minimum_stability_margin: float = 0.035
    target_stability_margin: float = 0.16
    maximum_heading_error: float = np.deg2rad(2.0)
    body_speed_limit: float = 0.65
    body_yaw_rate_limit: float = 1.2
    maximum_lateral_slip: float = 0.10
    use_zmp: bool = True
    gravity: float = 9.81
    regularisation: float = 1.0e-7
    smooth_epsilon: float = 1.0e-5

    ipopt_max_iterations: int = 160
    ipopt_tolerance: float = 1.0e-5

    @property
    def horizon_time(self) -> float:
        return self.dt * self.horizon_steps


@dataclass(frozen=True)
class SimulationParameters:
    """Closed-loop demo settings."""

    duration: float = 14.0
    desired_speed: float = 0.28
    desired_yaw_rate: float = 0.0
    initial_state: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    stop_position: float | None = 3.05
    random_seed: int = 4
