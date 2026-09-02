"""Direct planar kinematics for actuator-level track commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .config import MPCParameters
from .math_utils import J2_NUMPY, is_symbolic, rotation_2d
from .robot import RobotDescription


@dataclass
class KinematicModel:
    """Kinematics driven directly by ``[q1_cmd, q2_cmd, v1, v2]``.

    The articulation commands are position targets for the end of the control
    interval. During the interval the actual track angles are interpolated
    linearly from the measured state to those targets, while belt speeds are
    held constant.
    """

    robot: RobotDescription
    mpc_parameters: MPCParameters

    control_dimension = 4
    state_dimension = 5

    def actuator_commands(self, control: Any) -> Any:
        """Return the public actuator order ``[q1_cmd, q2_cmd, v1, v2]``."""

        if is_symbolic(control[0]):
            return ca.vertcat(control[0], control[1], control[2], control[3])
        return np.asarray(control, dtype=float).reshape(4)

    def articulation_rates(self, q: Any, control: Any) -> Any:
        """Constant average steering rates over one control interval."""

        return (control[0:2] - q) / self.mpc_parameters.dt

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

    def body_twist_from_axes(self, q: Any, belt: Any) -> Any:
        """Rigid-body twist induced by instantaneous track axes and belt speeds."""

        symbolic = is_symbolic(q[0]) or is_symbolic(belt[0])
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

        q_np = np.asarray(q, dtype=float).reshape(2)
        belt_np = np.asarray(belt, dtype=float).reshape(2)
        desired = np.array(
            [
                belt_np[0] * np.cos(q_np[0]),
                belt_np[0] * np.sin(q_np[0]),
                belt_np[1] * np.cos(q_np[1]),
                belt_np[1] * np.sin(q_np[1]),
            ],
            dtype=float,
        )
        normal = a_np.T @ a_np + regularisation * np.eye(3)
        return np.linalg.solve(normal, a_np.T @ desired)

    def body_twist(self, control: Any, q: Any | None = None) -> Any:
        """Return the twist for ``control`` at the supplied actual track angles."""

        axes = control[0:2] if q is None else q
        return self.body_twist_from_axes(axes, control[2:4])

    def interval_axes(self, q: Any, control: Any, fraction: float = 0.5) -> Any:
        """Track angles at a fraction of the current zero-order-hold interval."""

        return q + fraction * (control[0:2] - q)

    def interval_body_twist(self, state: Any, control: Any) -> Any:
        """Representative twist at the midpoint of one control interval."""

        q_mid = self.interval_axes(state[3:5], control, 0.5)
        return self.body_twist(control, q=q_mid)

    def pivot_residual_vectors(self, control: Any, q: Any | None = None) -> Any:
        """Residual between commanded pivot velocities and one rigid twist."""

        axes = control[0:2] if q is None else q
        twist = self.body_twist(control, q=axes)
        belt = control[2:4]
        symbolic = is_symbolic(axes[0]) or is_symbolic(belt[0])
        rows = []
        for index, pivot in enumerate(self.robot.parameters.pivot_positions):
            if symbolic:
                pivot_velocity = twist[0:2] + twist[2] * ca.DM(
                    [-pivot[1], pivot[0]]
                )
                desired = belt[index] * ca.vertcat(
                    ca.cos(axes[index]), ca.sin(axes[index])
                )
            else:
                pivot_velocity = np.asarray(twist[0:2], dtype=float) + float(
                    twist[2]
                ) * (J2_NUMPY @ pivot)
                desired = float(belt[index]) * np.array(
                    [np.cos(float(axes[index])), np.sin(float(axes[index]))],
                    dtype=float,
                )
            rows.append(pivot_velocity - desired)
        if symbolic:
            return ca.vertcat(*rows)
        return np.concatenate([np.asarray(row, dtype=float).reshape(2) for row in rows])

    def slip_components(self, q: Any, control: Any) -> Any:
        """Longitudinal/lateral incompatibility at each pivot for actual ``q``."""

        residual = self.pivot_residual_vectors(control, q=q)
        symbolic = is_symbolic(q[0]) or is_symbolic(control[0])
        rows = []
        for index in range(2):
            angle = q[index]
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
                longitudinal = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                lateral = np.array([-np.sin(angle), np.cos(angle)], dtype=float)
                rows.append(
                    [
                        float(np.dot(longitudinal, local_residual)),
                        float(np.dot(lateral, local_residual)),
                    ]
                )
        if symbolic:
            return ca.vertcat(*rows)
        return np.asarray(rows, dtype=float)

    def pose_derivative(self, pose: Any, q: Any, control: Any) -> Any:
        """World derivative of ``[X, Y, psi]`` at instantaneous angles ``q``."""

        twist = self.body_twist(control, q=q)
        world_velocity = rotation_2d(pose[2]) @ twist[0:2]
        if is_symbolic(pose[0]) or is_symbolic(q[0]):
            return ca.vertcat(world_velocity[0], world_velocity[1], twist[2])
        return np.array(
            [world_velocity[0], world_velocity[1], twist[2]], dtype=float
        )

    def state_derivative(self, state: Any, control: Any) -> Any:
        """Instantaneous derivative at the beginning of the interval."""

        q_rates = self.articulation_rates(state[3:5], control)
        pose_rate = self.pose_derivative(state[0:3], state[3:5], control)
        if is_symbolic(state[0]):
            return ca.vertcat(pose_rate, q_rates)
        return np.r_[pose_rate, np.asarray(q_rates, dtype=float).reshape(2)]

    def intermediate_state(self, state: Any, control: Any, fraction: float) -> Any:
        """Integrate the state up to ``fraction`` of the control interval.

        The articulation interpolation is defined over the full controller
        period, so a fractional RK4 integration uses the corresponding track
        angles at the beginning, midpoint and end of that subinterval.
        """

        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must lie in [0, 1]")
        dt = self.mpc_parameters.dt * fraction
        pose = state[0:3]
        q_start = state[3:5]
        q_delta = control[0:2] - q_start
        q_mid = q_start + 0.5 * fraction * q_delta
        q_end = q_start + fraction * q_delta

        if fraction == 0.0:
            if is_symbolic(state[0]):
                return ca.vertcat(pose, q_start)
            return np.asarray(state, dtype=float).reshape(5).copy()

        k1 = self.pose_derivative(pose, q_start, control)
        k2 = self.pose_derivative(pose + 0.5 * dt * k1, q_mid, control)
        k3 = self.pose_derivative(pose + 0.5 * dt * k2, q_mid, control)
        k4 = self.pose_derivative(pose + dt * k3, q_end, control)
        next_pose = pose + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if is_symbolic(state[0]):
            return ca.vertcat(next_pose, q_end)
        return np.r_[
            np.asarray(next_pose, dtype=float).reshape(3),
            np.asarray(q_end, dtype=float).reshape(2),
        ]

    def discrete_step(self, state: Any, control: Any) -> Any:
        """Integrate one full period while steering linearly to ``q_cmd``."""

        return self.intermediate_state(state, control, 1.0)

    def world_velocity(self, state: Any, control: Any) -> Any:
        """Average world translational velocity over one discrete interval."""

        next_state = self.discrete_step(state, control)
        return (next_state[0:2] - state[0:2]) / self.mpc_parameters.dt

    def world_velocity_from_twist(self, state: Any, body_twist: Any) -> Any:
        return rotation_2d(state[2]) @ body_twist[0:2]
