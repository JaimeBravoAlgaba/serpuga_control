"""Nonlinear MPC with direct actuator-level track commands."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np

from .config import MPCParameters
from .corridor import StraightGapCorridor
from .kinematics import KinematicModel
from .math_utils import J2_NUMPY
from .robot import RobotDescription
from .trajectory import ReferencePreview


@dataclass
class NMPCSolution:
    success: bool
    control: np.ndarray  # [q1_cmd, q2_cmd, v1, v2]
    body_twist: np.ndarray  # representative midpoint [v_x, v_y, omega]
    actuator_command: np.ndarray
    predicted_states: np.ndarray
    predicted_controls: np.ndarray
    predicted_twists: np.ndarray
    predicted_actuator_commands: np.ndarray
    objective: float
    solve_time: float
    status: str


class NMPCController:
    """Multiple-shooting NMPC that optimises the four actuator commands."""

    state_dimension = 5
    control_dimension = 4

    def __init__(
        self,
        robot: RobotDescription,
        model: KinematicModel,
        corridor: StraightGapCorridor,
        parameters: MPCParameters,
    ) -> None:
        self.robot = robot
        self.model = model
        self.corridor = corridor
        self.parameters = parameters
        self._last_states: np.ndarray | None = None
        self._last_controls: np.ndarray | None = None
        self._build_problem()

    def _build_problem(self) -> None:
        p = self.parameters
        r = self.robot.parameters
        n = p.horizon_steps

        self.opti = ca.Opti()
        self.states = self.opti.variable(self.state_dimension, n + 1)
        self.controls = self.opti.variable(self.control_dimension, n)

        self.initial_state = self.opti.parameter(self.state_dimension)
        self.reference_poses = self.opti.parameter(3, n + 1)
        self.reference_speeds = self.opti.parameter(1, n)
        self.reference_yaw_rates = self.opti.parameter(1, n)

        self.opti.subject_to(self.states[:, 0] == self.initial_state)
        self.opti.subject_to(
            self.opti.bounded(
                ca.repmat(ca.DM(r.q_min), 1, n + 1),
                self.states[3:5, :],
                ca.repmat(ca.DM(r.q_max), 1, n + 1),
            )
        )
        self.opti.subject_to(
            self.opti.bounded(
                ca.repmat(ca.DM(r.q_min), 1, n),
                self.controls[0:2, :],
                ca.repmat(ca.DM(r.q_max), 1, n),
            )
        )
        self.opti.subject_to(
            self.opti.bounded(
                -p.track_speed_limit,
                self.controls[2:4, :],
                p.track_speed_limit,
            )
        )

        objective = 0
        for step in range(n):
            state = self.states[:, step]
            control = self.controls[:, step]
            q_start = state[3:5]
            q_mid = self.model.interval_axes(q_start, control, 0.5)
            q_end = control[0:2]
            twist_start = self.model.body_twist(control, q=q_start)
            twist_mid = self.model.body_twist(control, q=q_mid)
            twist_end = self.model.body_twist(control, q=q_end)
            q_rates = self.model.articulation_rates(q_start, control)

            self.opti.subject_to(
                self.opti.bounded(
                    -p.articulation_rate_limit,
                    q_rates,
                    p.articulation_rate_limit,
                )
            )
            for bounded_twist in (twist_start, twist_mid, twist_end):
                self.opti.subject_to(
                    ca.sumsqr(bounded_twist[0:2]) <= p.body_speed_limit**2
                )
                self.opti.subject_to(
                    self.opti.bounded(
                        -p.body_yaw_rate_limit,
                        bounded_twist[2],
                        p.body_yaw_rate_limit,
                    )
                )

            midpoint_state = self.model.intermediate_state(state, control, 0.5)
            next_state = self.model.discrete_step(state, control)
            self.opti.subject_to(self.states[:, step + 1] == next_state)

            world_velocity = self.model.world_velocity(state, control)
            desired_heading = self.reference_poses[2, step]
            desired_world_velocity = self.reference_speeds[0, step] * ca.vertcat(
                ca.cos(desired_heading),
                ca.sin(desired_heading),
            )
            position_error = state[0:2] - self.reference_poses[0:2, step]
            heading_error = state[2] - desired_heading
            velocity_error = world_velocity - desired_world_velocity
            yaw_rate_error = twist_mid[2] - self.reference_yaw_rates[0, step]
            parallelism_error = self.robot.parallelism_residual(q_mid)

            objective += p.position_weight * ca.sumsqr(position_error)
            objective += p.heading_weight * 2.0 * (1.0 - ca.cos(heading_error))
            objective += p.velocity_weight * ca.sumsqr(velocity_error)
            objective += p.yaw_rate_weight * yaw_rate_error**2
            objective += p.parallelism_weight * parallelism_error**2

            self._add_corridor_constraint(state)
            self._add_corridor_constraint(midpoint_state)

        terminal_state = self.states[:, n]
        self._add_corridor_constraint(terminal_state)
        terminal_position_error = terminal_state[0:2] - self.reference_poses[0:2, n]
        terminal_heading_error = terminal_state[2] - self.reference_poses[2, n]
        objective += p.terminal_position_weight * ca.sumsqr(terminal_position_error)
        objective += (
            p.terminal_heading_weight
            * 2.0
            * (1.0 - ca.cos(terminal_heading_error))
        )

        self.opti.minimize(objective)
        self.objective_expression = objective
        self._objective_function = ca.Function(
            "nmpc_objective",
            [
                self.states,
                self.controls,
                self.reference_poses,
                self.reference_speeds,
                self.reference_yaw_rates,
            ],
            [objective],
        )

        plugin_options = {"expand": True, "print_time": False}
        solver_options = {
            "max_iter": p.ipopt_max_iterations,
            "tol": p.ipopt_tolerance,
            "acceptable_tol": 5.0 * p.ipopt_tolerance,
            "print_level": 0,
            "sb": "yes",
            "warm_start_init_point": "yes",
        }
        self.opti.solver("ipopt", plugin_options, solver_options)

    def _add_corridor_constraint(self, state: ca.MX) -> None:
        residual = self.corridor.clearance_residual(
            self.robot.footprint_vertices_world(state),
            margin=self.parameters.clearance_margin,
            epsilon=self.parameters.smooth_epsilon,
        )
        self.opti.subject_to(residual <= 0.0)

    def _set_parameters(self, state: np.ndarray, preview: ReferencePreview) -> None:
        self.opti.set_value(self.initial_state, state)
        self.opti.set_value(self.reference_poses, preview.poses.T)
        self.opti.set_value(self.reference_speeds, preview.speeds.reshape((1, -1)))
        self.opti.set_value(
            self.reference_yaw_rates,
            preview.yaw_rates.reshape((1, -1)),
        )

    def _seed_control(
        self,
        state: np.ndarray,
        desired_world_velocity: np.ndarray,
        desired_yaw_rate: float,
    ) -> np.ndarray:
        p = self.parameters
        q = np.asarray(state[3:5], dtype=float).copy()
        yaw = float(state[2])
        world_to_body = np.array(
            [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]],
            dtype=float,
        )
        desired_body_velocity = world_to_body @ desired_world_velocity
        desired_twist = np.r_[desired_body_velocity, desired_yaw_rate]

        belt = np.zeros(2, dtype=float)
        for index, pivot in enumerate(self.robot.parameters.pivot_positions):
            pivot_velocity = desired_twist[0:2] + desired_twist[2] * (J2_NUMPY @ pivot)
            axis = np.array([np.cos(q[index]), np.sin(q[index])], dtype=float)
            belt[index] = float(np.dot(axis, pivot_velocity))
        belt = np.clip(belt, -p.track_speed_limit, p.track_speed_limit)
        return np.r_[q, belt]

    def _initial_guess(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.parameters.horizon_steps
        state_guess = np.zeros((self.state_dimension, n + 1), dtype=float)
        control_guess = np.zeros((self.control_dimension, n), dtype=float)
        state_guess[:, 0] = state

        for step in range(n):
            current = state_guess[:, step]
            heading = preview.poses[step, 2]
            desired_world_velocity = preview.speeds[step] * np.array(
                [np.cos(heading), np.sin(heading)], dtype=float
            )
            control = self._seed_control(
                current,
                desired_world_velocity,
                float(preview.yaw_rates[step]),
            )
            next_state = np.asarray(
                self.model.discrete_step(current, control), dtype=float
            ).reshape(self.state_dimension)
            control_guess[:, step] = control
            state_guess[:, step + 1] = next_state
        return state_guess, control_guess

    def _apply_warm_start(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> None:
        if self._last_controls is None:
            state_guess, control_guess = self._initial_guess(state, preview)
        else:
            control_guess = np.column_stack(
                (self._last_controls[:, 1:], self._last_controls[:, -1])
            )
            delta = np.clip(
                control_guess[0:2, 0] - state[3:5],
                -self.parameters.articulation_rate_limit * self.parameters.dt,
                self.parameters.articulation_rate_limit * self.parameters.dt,
            )
            control_guess[0:2, 0] = state[3:5] + delta
            state_guess = self._rollout(state, control_guess)
        self.opti.set_initial(self.states, state_guess)
        self.opti.set_initial(self.controls, control_guess)
        self._current_state_guess = state_guess
        self._current_control_guess = control_guess

    def _rollout(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = np.zeros((self.state_dimension, controls.shape[1] + 1), dtype=float)
        states[:, 0] = np.asarray(state, dtype=float).reshape(self.state_dimension)
        for step in range(controls.shape[1]):
            states[:, step + 1] = np.asarray(
                self.model.discrete_step(states[:, step], controls[:, step]),
                dtype=float,
            ).reshape(self.state_dimension)
        return states

    def _evaluate_twists(
        self, states: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray(
                    self.model.interval_body_twist(states[:, step], controls[:, step]),
                    dtype=float,
                ).reshape(3)
                for step in range(controls.shape[1])
            ]
        )

    def _numeric_corridor_residual(self, state: np.ndarray) -> float:
        return float(
            self.corridor.clearance_residual(
                self.robot.footprint_vertices_world(state),
                margin=self.parameters.clearance_margin,
                epsilon=self.parameters.smooth_epsilon,
            )
        )

    def _guess_is_feasible(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        preview: ReferencePreview,
    ) -> bool:
        del preview
        p = self.parameters
        r = self.robot.parameters
        tolerance = max(10.0 * p.ipopt_tolerance, 1.0e-6)
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
            return False
        if np.any(states[3:5, :] < r.q_min[:, None] - tolerance):
            return False
        if np.any(states[3:5, :] > r.q_max[:, None] + tolerance):
            return False
        if np.any(controls[0:2, :] < r.q_min[:, None] - tolerance):
            return False
        if np.any(controls[0:2, :] > r.q_max[:, None] + tolerance):
            return False
        if np.any(np.abs(controls[2:4, :]) > p.track_speed_limit + tolerance):
            return False

        q_rates = (controls[0:2, :] - states[3:5, :-1]) / p.dt
        if np.any(np.abs(q_rates) > p.articulation_rate_limit + tolerance):
            return False

        for step in range(controls.shape[1]):
            state = states[:, step]
            control = controls[:, step]
            for fraction in (0.0, 0.5, 1.0):
                q = self.model.interval_axes(state[3:5], control, fraction)
                twist = np.asarray(
                    self.model.body_twist(control, q=q), dtype=float
                ).reshape(3)
                if np.linalg.norm(twist[0:2]) > p.body_speed_limit + tolerance:
                    return False
                if abs(twist[2]) > p.body_yaw_rate_limit + tolerance:
                    return False
            midpoint_state = np.asarray(
                self.model.intermediate_state(state, control, 0.5), dtype=float
            ).reshape(self.state_dimension)
            if self._numeric_corridor_residual(midpoint_state) > tolerance:
                return False

        for step in range(states.shape[1]):
            if self._numeric_corridor_residual(states[:, step]) > tolerance:
                return False
        return True

    def _solution_from_arrays(
        self,
        preview: ReferencePreview,
        states: np.ndarray,
        controls: np.ndarray,
        elapsed: float,
        status: str,
        objective: float,
    ) -> NMPCSolution:
        del preview
        twists = self._evaluate_twists(states, controls)
        self._last_states = states
        self._last_controls = controls
        return NMPCSolution(
            success=True,
            control=controls[:, 0].copy(),
            body_twist=twists[:, 0].copy(),
            actuator_command=controls[:, 0].copy(),
            predicted_states=states.T,
            predicted_controls=controls.T,
            predicted_twists=twists.T,
            predicted_actuator_commands=controls.T.copy(),
            objective=objective,
            solve_time=elapsed,
            status=status,
        )

    def _fallback_solution(
        self,
        preview: ReferencePreview,
        states: np.ndarray,
        controls: np.ndarray,
        elapsed: float,
        status: str,
    ) -> NMPCSolution:
        objective = float(
            self._objective_function(
                states,
                controls,
                preview.poses.T,
                preview.speeds.reshape((1, -1)),
                preview.yaw_rates.reshape((1, -1)),
            )
        )
        return self._solution_from_arrays(
            preview,
            states,
            controls,
            elapsed,
            f"{status} (feasible actuator fallback)",
            objective,
        )

    def solve(self, state: np.ndarray, preview: ReferencePreview) -> NMPCSolution:
        self._set_parameters(state, preview)
        self._apply_warm_start(state, preview)
        start = perf_counter()
        try:
            solution = self.opti.solve()
            states = np.asarray(solution.value(self.states), dtype=float)
            controls = np.asarray(solution.value(self.controls), dtype=float)
            return self._solution_from_arrays(
                preview,
                states,
                controls,
                perf_counter() - start,
                str(self.opti.stats().get("return_status", "Solve_Succeeded")),
                float(solution.value(self.objective_expression)),
            )
        except RuntimeError as first_error:
            elapsed = perf_counter() - start
            status = str(self.opti.stats().get("return_status", "Solve_Failed"))
            if self._guess_is_feasible(
                self._current_state_guess,
                self._current_control_guess,
                preview,
            ):
                return self._fallback_solution(
                    preview,
                    self._current_state_guess,
                    self._current_control_guess,
                    elapsed,
                    status,
                )

            self._last_states = None
            self._last_controls = None
            state_guess, control_guess = self._initial_guess(state, preview)
            self.opti.set_initial(self.states, state_guess)
            self.opti.set_initial(self.controls, control_guess)
            self._current_state_guess = state_guess
            self._current_control_guess = control_guess
            try:
                solution = self.opti.solve()
                states = np.asarray(solution.value(self.states), dtype=float)
                controls = np.asarray(solution.value(self.controls), dtype=float)
                return self._solution_from_arrays(
                    preview,
                    states,
                    controls,
                    perf_counter() - start,
                    f"{self.opti.stats().get('return_status', 'Solve_Succeeded')} (fresh-start retry)",
                    float(solution.value(self.objective_expression)),
                )
            except RuntimeError as retry_error:
                status = str(self.opti.stats().get("return_status", "Solve_Failed"))
                headline = str(retry_error).splitlines()[0] or str(first_error).splitlines()[0]

            if self._guess_is_feasible(state_guess, control_guess, preview):
                return self._fallback_solution(
                    preview,
                    state_guess,
                    control_guess,
                    perf_counter() - start,
                    status,
                )

            self._last_states = None
            self._last_controls = None
            zero_control = np.r_[state[3:5], np.zeros(2, dtype=float)]
            return NMPCSolution(
                success=False,
                control=zero_control.copy(),
                body_twist=np.zeros(3, dtype=float),
                actuator_command=zero_control.copy(),
                predicted_states=state_guess.T,
                predicted_controls=control_guess.T,
                predicted_twists=self._evaluate_twists(state_guess, control_guess).T,
                predicted_actuator_commands=control_guess.T.copy(),
                objective=float("nan"),
                solve_time=perf_counter() - start,
                status=f"{status}: {headline}",
            )
