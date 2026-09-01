"""Direct planar kinematics for actuator-level track commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .config import MPCParameters
from .math_utils import J2_NUMPY, as_column, is_symbolic, rotation_2d
from .robot import RobotDescription


@dataclass
class KinematicModel:
    """Kinematics driven directly by ``[q1_cmd, q2_cmd, v1, v2]``.

    The MPC no longer computes a body twist and no inverse kinematics is used.
    The commanded track axes and signed belt speeds define four pivot-velocity
    equations.  A regularised least-squares solve returns the rigid-body twist
    that best matches those four equations.
    """

    robot: RobotDescription
    mpc_parameters: MPCParameters

    control_dimension = 4
    state_dimension = 5

    def actuator_commands(self, control: Any) -> Any:
        """Return the public actuator order ``[q1, q2, v1, v2]`` unchanged."""

        if is_symbolic(control[0]):
            return ca.vertcat(control[0], control[1], control[2], control[3])
        return np.asarray(control, dtype=float).reshape(4)

    def articulation_rates(self, q: Any, control: Any) -> Any:
        """Average steering rates required to reach the commanded axes."""

        q_command = control[0:2]
        return (q_command - q) / self.mpc_parameters.dt

    def _rigid_body_matrix(self) -> np.ndarray:
        """Map body twist ``[vx, vy, omega]`` to both pivot velocities."""

        rows: list[np.ndarray] = []
        for pivot in self.robot.parameters.pivot_positions:
            rows.append(
                np.array(
                    [
                        [1.0, 0.0, -pivot[1]],
                        [0.0, 1.0, pivot[0]],
                    ],
                    dtype=float,
                )
            )
        return np.vstack(rows)

    def body_twist(self, control: Any) -> Any:
        """Return the rigid-body twist induced by the four actuator commands.

        For track ``i`` the desired pivot velocity is
        ``v_i [cos(q_i), sin(q_i)]``.  The four scalar equations are generally
        overdetermined for the three body-twist components, so the model uses
        a small Tikhonov regularisation configured by ``regularisation``.
        """

        symbolic = is_symbolic(control[0])
        q = control[0:2]
        belt = control[2:4]
        a_np = self._rigid_body_matrix()
        regularisation = self.mpc_parameters.regularisation

        if symbolic:
            a = ca.DM(a_np)
            desired = ca.vertcat(
                belt[0] * ca.cos(q[0]),
                belt[0] * ca.sin(q[0]),
                belt[1] * ca.cos(q[1]),
                belt[1] * ca.sin(q[1]),
            )
            normal = a.T @ a + regularisation * ca.DM.eye(3)
            return ca.solve(normal, a.T @ desired)

        desired = np.array(
            [
                belt[0] * np.cos(q[0]),
                belt[0] * np.sin(q[0]),
                belt[1] * np.cos(q[1]),
                belt[1] * np.sin(q[1]),
            ],
            dtype=float,
        )
        normal = a_np.T @ a_np + regularisation * np.eye(3)
        return np.linalg.solve(normal, a_np.T @ desired)

    def pivot_residual_vectors(self, control: Any) -> Any:
        """Residual between commanded pivot velocities and one rigid twist."""

        twist = self.body_twist(control)
        q = control[0:2]
        belt = control[2:4]
        symbolic = is_symbolic(control[0])
        rows = []
        for index, pivot in enumerate(self.robot.parameters.pivot_positions):
            if symbolic:
                pivot_velocity = twist[0:2] + twist[2] * ca.DM(
                    [-pivot[1], pivot[0]]
                )
                desired = belt[index] * ca.vertcat(
                    ca.cos(q[index]), ca.sin(q[index])
                )
            else:
                pivot_velocity = twist[0:2] + twist[2] * (J2_NUMPY @ pivot)
                desired = belt[index] * np.array(
                    [np.cos(q[index]), np.sin(q[index])], dtype=float
                )
            rows.append(pivot_velocity - desired)
        if symbolic:
            return ca.vertcat(*rows)
        return np.asarray(rows, dtype=float)

    def slip_components(self, q: Any, control: Any) -> Any:
        """Diagnostic longitudinal/lateral incompatibility at each pivot."""

        residual = self.pivot_residual_vectors(control)
        symbolic = is_symbolic(control[0])
        rows = []
        q_command = control[0:2]
        for index in range(2):
            angle = q_command[index]
            local_residual = residual[2 * index : 2 * index + 2]
            if symbolic:
                longitudinal = ca.vertcat(ca.cos(angle), ca.sin(angle))
                lateral = ca.vertcat(-ca.sin(angle), ca.cos(angle))
                rows.append(
                    ca.horzcat(
                        ca.dot(longitudinal, local_residual),
                        ca.dot(lateral, local_residual),
                    )
                )
            else:
                longitudinal = np.array([np.cos(angle), np.sin(angle)])
                lateral = np.array([-np.sin(angle), np.cos(angle)])
                rows.append(
                    [
                        float(np.dot(longitudinal, local_residual)),
                        float(np.dot(lateral, local_residual)),
                    ]
                )
        if symbolic:
            return ca.vertcat(*rows)
        return np.asarray(rows, dtype=float)

    def pose_derivative(self, pose: Any, control: Any) -> Any:
        """World derivative of ``[X, Y, psi]`` from actuator commands."""

        twist = self.body_twist(control)
        world_velocity = rotation_2d(pose[2]) @ twist[0:2]
        if is_symbolic(pose[0]):
            return ca.vertcat(world_velocity[0], world_velocity[1], twist[2])
        return np.array(
            [world_velocity[0], world_velocity[1], twist[2]], dtype=float
        )

    def state_derivative(self, state: Any, control: Any) -> Any:
        pose_rate = self.pose_derivative(state[0:3], control)
        q_rates = self.articulation_rates(state[3:5], control)
        if is_symbolic(state[0]):
            return ca.vertcat(pose_rate, q_rates)
        return np.r_[pose_rate, np.asarray(q_rates, dtype=float).reshape(2)]

    def discrete_step(self, state: Any, control: Any) -> Any:
        """Integrate body pose with ZOH and apply commanded track angles."""

        dt = self.mpc_parameters.dt
        pose = state[0:3]
        k1 = self.pose_derivative(pose, control)
        k2 = self.pose_derivative(pose + 0.5 * dt * k1, control)
        k3 = self.pose_derivative(pose + 0.5 * dt * k2, control)
        k4 = self.pose_derivative(pose + dt * k3, control)
        next_pose = pose + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_q = control[0:2]
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
