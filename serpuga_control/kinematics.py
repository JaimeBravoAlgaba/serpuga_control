"""Weighted-slip planar kinematics for the reconfigurable robot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .config import MPCParameters
from .math_utils import J2_CASADI, J2_NUMPY, as_column, is_symbolic, rotation_2d
from .robot import RobotDescription


@dataclass
class KinematicModel:
    """Forward kinematics based on a minimum-slip rigid-body projection."""

    robot: RobotDescription
    mpc_parameters: MPCParameters

    def _projection_matrices(self, q: Any) -> tuple[Any, Any, Any, Any]:
        symbolic = is_symbolic(q[0])
        params = self.robot.parameters
        if symbolic:
            a_blocks = []
            c_matrix = ca.MX.zeros(4, 2)
            e_matrix = ca.MX.zeros(4, 2)
            w_blocks = []
            identity = ca.DM.eye(2)
            j_matrix = J2_CASADI
        else:
            a_blocks = []
            c_matrix = np.zeros((4, 2), dtype=float)
            e_matrix = np.zeros((4, 2), dtype=float)
            w_blocks = []
            identity = np.eye(2)
            j_matrix = J2_NUMPY

        for index in range(2):
            q_i = q[index]
            centre = self.robot.track_center_body(q_i, index)
            derivative = self.robot.track_center_derivative(q_i, index)
            rotation = rotation_2d(q_i)
            if symbolic:
                longitudinal = rotation @ ca.DM([1.0, 0.0])
                lateral = rotation @ ca.DM([0.0, 1.0])
                a_i = ca.horzcat(identity, j_matrix @ centre)
                weight_i = (
                    params.longitudinal_slip_weight
                    * (longitudinal @ longitudinal.T)
                    + params.lateral_slip_weight * (lateral @ lateral.T)
                )
            else:
                longitudinal = rotation @ np.array([1.0, 0.0])
                lateral = rotation @ np.array([0.0, 1.0])
                a_i = np.column_stack((identity, j_matrix @ centre))
                weight_i = (
                    params.longitudinal_slip_weight
                    * np.outer(longitudinal, longitudinal)
                    + params.lateral_slip_weight * np.outer(lateral, lateral)
                )
            row = slice(2 * index, 2 * index + 2)
            c_matrix[row, index] = derivative
            e_matrix[row, index] = longitudinal
            a_blocks.append(a_i)
            w_blocks.append(weight_i)

        if symbolic:
            a_matrix = ca.vertcat(*a_blocks)
            weight = ca.diagcat(*w_blocks)
        else:
            a_matrix = np.vstack(a_blocks)
            weight = np.block(
                [
                    [w_blocks[0], np.zeros((2, 2))],
                    [np.zeros((2, 2)), w_blocks[1]],
                ]
            )
        return a_matrix, c_matrix, e_matrix, weight

    def body_twist(self, q: Any, control: Any) -> Any:
        """Return [v_x, v_y, omega] in the body frame."""

        track_speeds = control[0:2]
        articulation_rates = control[2:4]
        a_matrix, c_matrix, e_matrix, weight = self._projection_matrices(q)
        regularisation = self.mpc_parameters.regularisation
        if is_symbolic(q[0]):
            normal = a_matrix.T @ weight @ a_matrix + regularisation * ca.DM.eye(3)
            rhs = a_matrix.T @ weight @ (
                e_matrix @ track_speeds - c_matrix @ articulation_rates
            )
            # symbolicqr keeps this small 3x3 solve differentiable and allows
            # CasADi to expand the complete NMPC graph efficiently.
            return ca.solve(normal, rhs, "symbolicqr")
        normal = a_matrix.T @ weight @ a_matrix + regularisation * np.eye(3)
        rhs = a_matrix.T @ weight @ (
            e_matrix @ np.asarray(track_speeds)
            - c_matrix @ np.asarray(articulation_rates)
        )
        return np.linalg.solve(normal, rhs)

    def slip_vectors(self, q: Any, control: Any, body_twist: Any | None = None) -> Any:
        twist = self.body_twist(q, control) if body_twist is None else body_twist
        a_matrix, c_matrix, e_matrix, _ = self._projection_matrices(q)
        track_speeds = control[0:2]
        articulation_rates = control[2:4]
        return (
            a_matrix @ twist
            + c_matrix @ articulation_rates
            - e_matrix @ track_speeds
        )

    def slip_components(
        self,
        q: Any,
        control: Any,
        body_twist: Any | None = None,
    ) -> Any:
        slip = self.slip_vectors(q, control, body_twist)
        symbolic = is_symbolic(q[0])
        rows = []
        for index in range(2):
            rotation = rotation_2d(q[index])
            if symbolic:
                longitudinal = rotation @ ca.DM([1.0, 0.0])
                lateral = rotation @ ca.DM([0.0, 1.0])
                local_slip = slip[2 * index : 2 * index + 2]
                rows.append(ca.horzcat(ca.dot(longitudinal, local_slip), ca.dot(lateral, local_slip)))
            else:
                longitudinal = rotation @ np.array([1.0, 0.0])
                lateral = rotation @ np.array([0.0, 1.0])
                local_slip = slip[2 * index : 2 * index + 2]
                rows.append([np.dot(longitudinal, local_slip), np.dot(lateral, local_slip)])
        if symbolic:
            return ca.vertcat(*rows)
        return np.asarray(rows, dtype=float)

    def state_derivative(self, state: Any, control: Any) -> Any:
        twist = self.body_twist(state[3:5], control)
        body_velocity = twist[0:2]
        world_velocity = rotation_2d(state[2]) @ body_velocity
        if is_symbolic(state[0]):
            return ca.vertcat(
                world_velocity[0],
                world_velocity[1],
                twist[2],
                control[2],
                control[3],
            )
        return np.array(
            [
                world_velocity[0],
                world_velocity[1],
                twist[2],
                control[2],
                control[3],
            ],
            dtype=float,
        )

    def discrete_step(self, state: Any, control: Any) -> Any:
        """Fourth-order Runge--Kutta integration with zero-order hold."""

        dt = self.mpc_parameters.dt
        k1 = self.state_derivative(state, control)
        k2 = self.state_derivative(state + 0.5 * dt * k1, control)
        k3 = self.state_derivative(state + 0.5 * dt * k2, control)
        k4 = self.state_derivative(state + dt * k3, control)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def state_derivative_from_twist(
        self,
        state: Any,
        control: Any,
        body_twist: Any,
    ) -> Any:
        """State derivative when the body twist is an optimisation variable."""

        world_velocity = rotation_2d(state[2]) @ body_twist[0:2]
        if is_symbolic(state[0]):
            return ca.vertcat(
                world_velocity[0],
                world_velocity[1],
                body_twist[2],
                control[2],
                control[3],
            )
        return np.array(
            [
                world_velocity[0],
                world_velocity[1],
                body_twist[2],
                control[2],
                control[3],
            ],
            dtype=float,
        )

    def discrete_step_with_twist(
        self,
        state: Any,
        control: Any,
        body_twist: Any,
    ) -> Any:
        """RK4 step used by the inverse-kinematic NMPC formulation."""

        dt = self.mpc_parameters.dt

        def derivative(current_state: Any) -> Any:
            return self.state_derivative_from_twist(current_state, control, body_twist)

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def world_velocity(self, state: Any, control: Any) -> Any:
        twist = self.body_twist(state[3:5], control)
        return rotation_2d(state[2]) @ twist[0:2]

    def world_velocity_from_twist(self, state: Any, body_twist: Any) -> Any:
        return rotation_2d(state[2]) @ body_twist[0:2]
