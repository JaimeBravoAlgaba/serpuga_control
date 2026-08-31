"""Planar body-twist kinematics and analytic inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .config import MPCParameters
from .math_utils import J2_CASADI, J2_NUMPY, as_column, is_symbolic, rotation_2d
from .robot import RobotDescription


@dataclass(frozen=True)
class InverseKinematicsSolution:
    """Analytic track commands for one planar body twist.

    ``articulation_angles`` contains ``[q1, q2]`` and ``track_speeds``
    contains the signed longitudinal belt velocities ``[v1, v2]``. The
    signs let the inverse kinematics select the equivalent track axis closest
    to the current articulation instead of rotating a track by 180 degrees.
    """

    articulation_angles: Any
    track_speeds: Any
    pivot_velocities: Any

    @property
    def actuator_vector(self) -> Any:
        """Return the public actuator order ``[q1, q2, v1, v2]``."""

        if is_symbolic(self.articulation_angles[0]):
            return ca.vertcat(self.articulation_angles, self.track_speeds)
        return np.r_[
            np.asarray(self.articulation_angles, dtype=float).reshape(2),
            np.asarray(self.track_speeds, dtype=float).reshape(2),
        ]


@dataclass
class KinematicModel:
    """Kinematics driven by the twist of the centre of the rigid bar.

    The control is always ``[v_x, v_y, omega]`` expressed in the bar/body
    frame. Track angles and signed belt speeds are generated analytically.
    The state remains ``[X, Y, psi, q1, q2]`` so geometry, steering-rate
    limits and transient contact slip can still be represented.
    """

    robot: RobotDescription
    mpc_parameters: MPCParameters

    control_dimension = 3
    state_dimension = 5

    def body_twist(self, control: Any) -> Any:
        """Return the commanded bar-centre twist ``[v_x, v_y, omega]``."""

        if is_symbolic(control[0]):
            return ca.vertcat(control[0], control[1], control[2])
        return np.asarray(control, dtype=float).reshape(3)

    @staticmethod
    def _nearest_axis_angle(
        local_velocity: Any,
        reference: Any,
        epsilon_squared: float,
    ) -> Any:
        """Choose the velocity axis modulo pi closest to ``reference``.

        The double-angle form avoids evaluating ``atan2(0, 0)`` at standstill
        and is therefore suitable for the symbolic NMPC graph.
        """

        if is_symbolic(local_velocity[0]):
            parallel = (
                ca.cos(reference) * local_velocity[0]
                + ca.sin(reference) * local_velocity[1]
            )
            lateral = (
                -ca.sin(reference) * local_velocity[0]
                + ca.cos(reference) * local_velocity[1]
            )
            delta = 0.5 * ca.atan2(
                2.0 * parallel * lateral,
                parallel**2 - lateral**2 + epsilon_squared,
            )
        else:
            parallel = (
                np.cos(float(reference)) * local_velocity[0]
                + np.sin(float(reference)) * local_velocity[1]
            )
            lateral = (
                -np.sin(float(reference)) * local_velocity[0]
                + np.cos(float(reference)) * local_velocity[1]
            )
            delta = 0.5 * np.arctan2(
                2.0 * parallel * lateral,
                parallel**2 - lateral**2 + epsilon_squared,
            )
        return reference + delta

    def inverse_kinematics(
        self,
        body_twist: Any,
        q_reference: Any | None = None,
    ) -> InverseKinematicsSolution:
        """Solve analytically for ``q1, q2, v1, v2``.

        The velocity of pivot ``i`` is ``u_i = v_B + omega J p_i``. The
        track axis is aligned with that velocity modulo pi and the signed belt
        speed is its projection on the selected axis. If a pivot is
        instantaneously stationary, its previous/reference angle is retained.
        """

        twist = self.body_twist(body_twist)
        symbolic = is_symbolic(twist[0])
        if q_reference is None:
            q_reference = self.robot.parameters.nominal_configuration
        # This value has units of squared speed inside the double-angle
        # expression. Using the configured regularisation directly keeps the
        # IK well conditioned around standstill in the symbolic NLP.
        epsilon_squared = self.mpc_parameters.regularisation

        angles = []
        speeds = []
        pivot_velocities = []
        for index in range(2):
            pivot = self.robot.parameters.pivot_positions[index]
            if symbolic:
                local_velocity = twist[0:2] + twist[2] * (J2_CASADI @ as_column(pivot))
                candidate = self._nearest_axis_angle(
                    local_velocity,
                    q_reference[index],
                    epsilon_squared,
                )
                speed_squared = ca.sumsqr(local_velocity)
                angle = ca.if_else(
                    speed_squared > epsilon_squared,
                    candidate,
                    q_reference[index],
                )
                longitudinal = ca.vertcat(ca.cos(angle), ca.sin(angle))
                speed = ca.dot(longitudinal, local_velocity)
            else:
                local_velocity = twist[0:2] + twist[2] * (J2_NUMPY @ pivot)
                speed_squared = float(np.dot(local_velocity, local_velocity))
                if speed_squared > epsilon_squared:
                    angle = self._nearest_axis_angle(
                        local_velocity,
                        float(q_reference[index]),
                        epsilon_squared,
                    )
                else:
                    angle = float(q_reference[index])
                longitudinal = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                speed = float(np.dot(longitudinal, local_velocity))
            angles.append(angle)
            speeds.append(speed)
            pivot_velocities.append(local_velocity)

        if symbolic:
            return InverseKinematicsSolution(
                articulation_angles=ca.vertcat(*angles),
                track_speeds=ca.vertcat(*speeds),
                pivot_velocities=ca.horzcat(*pivot_velocities).T,
            )
        return InverseKinematicsSolution(
            articulation_angles=np.asarray(angles, dtype=float),
            track_speeds=np.asarray(speeds, dtype=float),
            pivot_velocities=np.asarray(pivot_velocities, dtype=float),
        )

    def actuator_commands(self, q: Any, control: Any) -> Any:
        """Return analytic ``[q1, q2, v1, v2]`` for a body-twist command."""

        return self.inverse_kinematics(control, q).actuator_vector

    def articulation_rates(self, q: Any, control: Any) -> Any:
        """Average steering rates required during one control interval."""

        target = self.inverse_kinematics(control, q).articulation_angles
        return (target - q) / self.mpc_parameters.dt

    def _contact_matrices(self, q: Any) -> tuple[Any, Any, Any]:
        """Return centre-velocity, articulation and belt-direction maps."""

        symbolic = is_symbolic(q[0])
        if symbolic:
            a_blocks = []
            c_matrix = ca.MX.zeros(4, 2)
            e_matrix = ca.MX.zeros(4, 2)
            identity = ca.DM.eye(2)
            j_matrix = J2_CASADI
        else:
            a_blocks = []
            c_matrix = np.zeros((4, 2), dtype=float)
            e_matrix = np.zeros((4, 2), dtype=float)
            identity = np.eye(2)
            j_matrix = J2_NUMPY

        for index in range(2):
            centre = self.robot.track_center_body(q[index], index)
            derivative = self.robot.track_center_derivative(q[index], index)
            rotation = rotation_2d(q[index])
            if symbolic:
                longitudinal = rotation @ ca.DM([1.0, 0.0])
                a_i = ca.horzcat(identity, j_matrix @ centre)
            else:
                longitudinal = rotation @ np.array([1.0, 0.0])
                a_i = np.column_stack((identity, j_matrix @ centre))
            row = slice(2 * index, 2 * index + 2)
            c_matrix[row, index] = derivative
            e_matrix[row, index] = longitudinal
            a_blocks.append(a_i)

        if symbolic:
            return ca.vertcat(*a_blocks), c_matrix, e_matrix
        return np.vstack(a_blocks), c_matrix, e_matrix

    def pivot_slip_vectors(self, q: Any, control: Any) -> Any:
        """Slip at the ideal point contacts used by the analytic inverse."""

        inverse = self.inverse_kinematics(control, q)
        symbolic = is_symbolic(control[0])
        rows = []
        for index in range(2):
            angle = inverse.articulation_angles[index]
            if symbolic:
                longitudinal = ca.vertcat(ca.cos(angle), ca.sin(angle))
            else:
                longitudinal = np.array(
                    [np.cos(angle), np.sin(angle)],
                    dtype=float,
                )
            rows.append(
                inverse.pivot_velocities[index, :].T
                - inverse.track_speeds[index] * longitudinal
            )
        if symbolic:
            return ca.vertcat(*rows)
        return np.asarray(rows, dtype=float)

    def slip_vectors(self, q: Any, control: Any) -> Any:
        """Residual slip at both track centres during steering transients."""

        twist = self.body_twist(control)
        inverse = self.inverse_kinematics(twist, q)
        q_rates = (inverse.articulation_angles - q) / self.mpc_parameters.dt
        a_matrix, c_matrix, e_matrix = self._contact_matrices(q)
        return a_matrix @ twist + c_matrix @ q_rates - e_matrix @ inverse.track_speeds

    def slip_components(self, q: Any, control: Any) -> Any:
        """Return longitudinal/lateral centre slip for each track."""

        slip = self.slip_vectors(q, control)
        symbolic = is_symbolic(q[0])
        rows = []
        for index in range(2):
            rotation = rotation_2d(q[index])
            if symbolic:
                longitudinal = rotation @ ca.DM([1.0, 0.0])
                lateral = rotation @ ca.DM([0.0, 1.0])
                local_slip = slip[2 * index : 2 * index + 2]
                rows.append(
                    ca.horzcat(
                        ca.dot(longitudinal, local_slip),
                        ca.dot(lateral, local_slip),
                    )
                )
            else:
                longitudinal = rotation @ np.array([1.0, 0.0])
                lateral = rotation @ np.array([0.0, 1.0])
                local_slip = slip[2 * index : 2 * index + 2]
                rows.append(
                    [np.dot(longitudinal, local_slip), np.dot(lateral, local_slip)]
                )
        if symbolic:
            return ca.vertcat(*rows)
        return np.asarray(rows, dtype=float)

    def pose_derivative(self, pose: Any, control: Any) -> Any:
        """World derivative of ``[X, Y, psi]`` for a body-frame twist."""

        twist = self.body_twist(control)
        world_velocity = rotation_2d(pose[2]) @ twist[0:2]
        if is_symbolic(pose[0]):
            return ca.vertcat(world_velocity[0], world_velocity[1], twist[2])
        return np.array(
            [world_velocity[0], world_velocity[1], twist[2]],
            dtype=float,
        )

    def state_derivative(self, state: Any, control: Any) -> Any:
        """Continuous diagnostic derivative including average steering rate."""

        pose_rate = self.pose_derivative(state[0:3], control)
        q_rates = self.articulation_rates(state[3:5], control)
        if is_symbolic(state[0]):
            return ca.vertcat(pose_rate, q_rates)
        return np.r_[pose_rate, np.asarray(q_rates, dtype=float).reshape(2)]

    def discrete_step(self, state: Any, control: Any) -> Any:
        """Integrate the bar pose with ZOH and apply the analytic IK target.

        The articulation change over the interval is constrained by the NMPC.
        Assigning the target at the sample boundary is an ideal position-servo
        model; the associated average rate and contact slip remain explicit.
        """

        dt = self.mpc_parameters.dt
        pose = state[0:3]
        k1 = self.pose_derivative(pose, control)
        k2 = self.pose_derivative(pose + 0.5 * dt * k1, control)
        k3 = self.pose_derivative(pose + 0.5 * dt * k2, control)
        k4 = self.pose_derivative(pose + dt * k3, control)
        next_pose = pose + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_q = self.inverse_kinematics(control, state[3:5]).articulation_angles
        if is_symbolic(state[0]):
            return ca.vertcat(next_pose, next_q)
        return np.r_[
            np.asarray(next_pose, dtype=float).reshape(3),
            np.asarray(next_q, dtype=float).reshape(2),
        ]

    def world_velocity(self, state: Any, control: Any) -> Any:
        return rotation_2d(state[2]) @ self.body_twist(control)[0:2]

    def world_velocity_from_twist(self, state: Any, body_twist: Any) -> Any:
        return rotation_2d(state[2]) @ body_twist[0:2]
