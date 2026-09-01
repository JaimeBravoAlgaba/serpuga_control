"""Nonlinear MPC with bar-centre twist controls and analytic track IK."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np

from .config import MPCParameters
from .corridor import StraightGapCorridor
from .kinematics import KinematicModel
from .math_utils import J2_NUMPY, smooth_minimum
from .robot import RobotDescription
from .trajectory import ReferencePreview


@dataclass
class NMPCSolution:
    success: bool
    # Public control order: [v_x, v_y, omega] at the centre of the bar.
    control: np.ndarray
    body_twist: np.ndarray
    # Analytic inverse-kinematics order: [q1, q2, v1, v2].
    actuator_command: np.ndarray
    predicted_states: np.ndarray
    predicted_controls: np.ndarray
    predicted_twists: np.ndarray
    predicted_actuator_commands: np.ndarray
    objective: float
    solve_time: float
    status: str


class NMPCController:
    """Multiple-shooting NMPC driven by the twist of the rigid-bar centre."""

    state_dimension = 5
    control_dimension = 3

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
        self._seed_control_cache: dict[
            tuple[float, ...], tuple[np.ndarray, np.ndarray]
        ] = {}
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
                -p.body_yaw_rate_limit,
                self.controls[2, :],
                p.body_yaw_rate_limit,
            )
        )

        objective = 0
        for step in range(n):
            state = self.states[:, step]
            control = self.controls[:, step]
            twist = self.model.body_twist(control)

            self.opti.subject_to(ca.sumsqr(control[0:2]) <= p.body_speed_limit**2)

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
            yaw_rate_error = twist[2] - self.reference_yaw_rates[0, step]
            parallelism_error = self.robot.parallelism_residual(state[3:5])

            objective += p.position_weight * ca.sumsqr(position_error)
            objective += p.heading_weight * 2.0 * (1.0 - ca.cos(heading_error))
            objective += p.velocity_weight * ca.sumsqr(velocity_error)
            objective += p.yaw_rate_weight * yaw_rate_error**2
            objective += p.parallelism_weight * parallelism_error**2

            self._add_width_constraint(
                state,
                self.reference_poses[0:2, step],
                self.reference_poses[2, step],
            )

        terminal_state = self.states[:, n]
        self._add_width_constraint(
            terminal_state,
            self.reference_poses[0:2, n],
            self.reference_poses[2, n],
        )
        terminal_parallelism_error = self.robot.parallelism_residual(
            terminal_state[3:5]
        )
        objective += p.parallelism_weight * terminal_parallelism_error**2
        terminal_position_error = terminal_state[0:2] - self.reference_poses[0:2, n]
        terminal_heading_error = terminal_state[2] - self.reference_poses[2, n]
        objective += p.terminal_position_weight * ca.sumsqr(terminal_position_error)
        objective += (
            p.terminal_heading_weight * 2.0 * (1.0 - ca.cos(terminal_heading_error))
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

    def _add_width_constraint(
        self,
        state: ca.MX,
        path_position: ca.MX,
        path_heading: ca.MX,
    ) -> None:
        """Require the complete formation to fit in the perceived gap.

        Reference tracking keeps the bar on the corridor centreline.  This
        single geometric inequality only compares the robot envelope along
        the path normal with the locally available width.
        """

        path_normal = ca.vertcat(-ca.sin(path_heading), ca.cos(path_heading))
        robot_width = self.robot.centred_envelope_width_expression(
            state,
            path_position,
            path_normal,
            self.parameters.smooth_epsilon,
        )
        footprint_widths = [
            self.corridor.full_width(vertex[0])
            for vertex in self.robot.footprint_vertices_world(state)
        ]
        available_width = (
            smooth_minimum(footprint_widths, self.parameters.smooth_epsilon)
            - 2.0 * self.parameters.clearance_margin
        )
        self.opti.subject_to(robot_width <= available_width)

    def _set_parameters(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> None:
        self.opti.set_value(self.initial_state, state)
        self.opti.set_value(self.reference_poses, preview.poses.T)
        self.opti.set_value(self.reference_speeds, preview.speeds.reshape((1, -1)))
        self.opti.set_value(
            self.reference_yaw_rates,
            preview.yaw_rates.reshape((1, -1)),
        )

    def _compatible_twist(
        self,
        q_axes: np.ndarray,
        desired_twist: np.ndarray,
    ) -> np.ndarray:
        """Project a desired twist onto the no-lateral-pivot-slip subspace."""

        rows = []
        for index in range(2):
            normal = np.array(
                [-np.sin(q_axes[index]), np.cos(q_axes[index])],
                dtype=float,
            )
            pivot = self.robot.parameters.pivot_positions[index]
            rows.append(np.r_[normal, normal @ (J2_NUMPY @ pivot)])
        constraint = np.asarray(rows, dtype=float)
        _, singular_values, vh = np.linalg.svd(constraint, full_matrices=True)
        rank = int(np.sum(singular_values > 1.0e-9))
        nullspace = vh[rank:].T
        if nullspace.size == 0:
            return np.zeros(3, dtype=float)
        compatible = nullspace @ (nullspace.T @ desired_twist)
        if np.linalg.norm(compatible) <= 1.0e-7 and np.linalg.norm(desired_twist) > 0.0:
            direction = nullspace[:, 0]
            sign = 1.0 if np.dot(direction, desired_twist) >= 0.0 else -1.0
            compatible = 0.05 * sign * direction
        return np.asarray(compatible, dtype=float).reshape(3)

    def _clip_twist(self, twist: np.ndarray) -> np.ndarray:
        clipped = np.asarray(twist, dtype=float).reshape(3).copy()
        linear_norm = np.linalg.norm(clipped[0:2])
        if linear_norm > self.parameters.body_speed_limit:
            clipped[0:2] *= self.parameters.body_speed_limit / linear_norm
        clipped[2] = np.clip(
            clipped[2],
            -self.parameters.body_yaw_rate_limit,
            self.parameters.body_yaw_rate_limit,
        )
        return clipped

    def _initial_guess(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.parameters
        n = p.horizon_steps
        state_guess = np.zeros((self.state_dimension, n + 1), dtype=float)
        control_guess = np.zeros((self.control_dimension, n), dtype=float)
        state_guess[:, 0] = state

        for step in range(n):
            current = state_guess[:, step]
            desired_heading = preview.poses[step, 2]
            desired_world_velocity = preview.speeds[step] * np.array(
                [np.cos(desired_heading), np.sin(desired_heading)],
                dtype=float,
            )
            tangent = np.array(
                [np.cos(desired_heading), np.sin(desired_heading)],
                dtype=float,
            )
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            position_correction = preview.poses[step, 0:2] - current[0:2]
            desired_world_velocity += (
                1.2 * float(np.dot(position_correction, tangent)) * tangent
                + 4.0 * float(np.dot(position_correction, normal)) * normal
            )
            yaw = current[2]
            world_to_body = np.array(
                [
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)],
                ],
                dtype=float,
            )
            desired_twist = np.r_[
                world_to_body @ desired_world_velocity,
                preview.yaw_rates[step]
                + 1.5
                * np.arctan2(
                    np.sin(desired_heading - yaw),
                    np.cos(desired_heading - yaw),
                ),
            ]
            desired_twist = self._clip_twist(desired_twist)
            reference_pose = preview.poses[step + 1]
            desired_control = desired_twist
            desired_next = np.asarray(
                self.model.discrete_step(current, desired_control),
                dtype=float,
            ).reshape(self.state_dimension)
            if (
                self._joint_configuration_feasible(desired_next)
                and
                self._numeric_width_residual(desired_next, reference_pose)
                <= -2.0 * p.smooth_epsilon
            ):
                selected_control = desired_control
                selected_next = desired_next
            else:
                selected_control, selected_next = self._clearance_seed_control(
                    current,
                    desired_twist,
                    reference_pose,
                )
            control_guess[:, step] = selected_control
            state_guess[:, step + 1] = selected_next
        return state_guess, control_guess

    def _numeric_width_residual(
        self,
        state: np.ndarray,
        reference_pose: np.ndarray,
    ) -> float:
        heading = float(reference_pose[2])
        path_normal = np.array([-np.sin(heading), np.cos(heading)], dtype=float)
        required_width = float(
            self.robot.centred_envelope_width_expression(
                state,
                reference_pose[0:2],
                path_normal,
                self.parameters.smooth_epsilon,
            )
        )
        available_width = min(
            float(self.corridor.full_width(vertex[0]))
            for vertex in self.robot.footprint_vertices_world(state)
        ) - 2.0 * self.parameters.clearance_margin
        return required_width - available_width

    def _joint_configuration_feasible(self, state: np.ndarray) -> bool:
        q = np.asarray(state[3:5], dtype=float)
        r = self.robot.parameters
        tolerance = max(10.0 * self.parameters.ipopt_tolerance, 1.0e-6)
        return bool(
            np.all(q >= r.q_min - tolerance)
            and np.all(q <= r.q_max + tolerance)
        )

    def _clearance_seed_control(
        self,
        state: np.ndarray,
        desired_twist: np.ndarray,
        reference_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find a feasible control seed without adding another constraint."""

        r = self.robot.parameters
        cache_key = tuple(
            np.round(
                np.r_[state, desired_twist, reference_pose[1:3]],
                4,
            )
        )
        cached = self._seed_control_cache.get(cache_key)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()

        fallback_control = np.zeros(3, dtype=float)
        fallback_next = np.asarray(
            self.model.discrete_step(state, fallback_control),
            dtype=float,
        ).reshape(self.state_dimension)
        fallback_residual = self._numeric_width_residual(
            fallback_next,
            reference_pose,
        )

        def best_candidate(
            candidates: list[np.ndarray],
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            nonlocal fallback_control, fallback_next, fallback_residual
            best_control: np.ndarray | None = None
            best_next: np.ndarray | None = None
            best_cost = float("inf")
            for q in candidates:
                control = self._clip_twist(
                    self._compatible_twist(q, desired_twist)
                )
                next_state = np.asarray(
                    self.model.discrete_step(state, control),
                    dtype=float,
                ).reshape(self.state_dimension)
                if not self._joint_configuration_feasible(next_state):
                    continue
                residual = self._numeric_width_residual(
                    next_state,
                    reference_pose,
                )
                if residual < fallback_residual:
                    fallback_residual = residual
                    fallback_control = control.copy()
                    fallback_next = next_state.copy()
                parallelism_error = float(self.robot.parallelism_residual(q))
                tracking_error = next_state[0:2] - reference_pose[0:2]
                cost = (
                    parallelism_error**2
                    + 0.2 * float(np.sum((control - desired_twist) ** 2))
                    + 0.5 * float(np.sum(tracking_error**2))
                    + 0.01
                    * float(np.sum((next_state[3:5] - state[3:5]) ** 2))
                )
                if residual <= -2.0 * self.parameters.smooth_epsilon and cost < best_cost:
                    best_cost = cost
                    best_control = control.copy()
                    best_next = next_state.copy()
            return best_control, best_next

        # Search the preferred manifold first: q1-q2 = k*pi. It contains all
        # parallel and antiparallel formations and is both faster and aligned
        # with the only geometric cost in the simplified MPC.
        parallel_candidates: list[np.ndarray] = []
        for q1 in np.linspace(r.q_min[0], r.q_max[0], 41):
            for turns in (-2, -1, 0, 1, 2):
                q2 = q1 + turns * np.pi
                if r.q_min[1] <= q2 <= r.q_max[1]:
                    parallel_candidates.append(np.array([q1, q2], dtype=float))
        for q2 in np.linspace(r.q_min[1], r.q_max[1], 41):
            for turns in (-2, -1, 0, 1, 2):
                q1 = q2 + turns * np.pi
                if r.q_min[0] <= q1 <= r.q_max[0]:
                    parallel_candidates.append(np.array([q1, q2], dtype=float))

        q1_values = np.linspace(r.q_min[0], r.q_max[0], 11)
        q2_values = np.linspace(r.q_min[1], r.q_max[1], 11)
        coarse_candidates = [
            np.array([q1, q2], dtype=float)
            for q1 in q1_values
            for q2 in q2_values
        ]
        best_control, best_next = best_candidate(
            parallel_candidates + coarse_candidates
        )

        if best_control is not None and best_next is not None:
            selected = (best_control, best_next)
        else:
            selected = (fallback_control, fallback_next)
        self._seed_control_cache[cache_key] = (
            selected[0].copy(),
            selected[1].copy(),
        )
        return selected

    def _apply_warm_start(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> None:
        if self._last_states is None or self._last_controls is None:
            state_guess, control_guess = self._initial_guess(state, preview)
        else:
            control_guess = np.column_stack(
                (self._last_controls[:, 1:], self._last_controls[:, -1])
            )
            state_guess = self._rollout(state, control_guess)
        self.opti.set_initial(self.states, state_guess)
        self.opti.set_initial(self.controls, control_guess)
        self._current_state_guess = state_guess
        self._current_control_guess = control_guess

    def _rollout(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Evaluate a dynamically consistent multiple-shooting seed."""

        states = np.zeros(
            (self.state_dimension, controls.shape[1] + 1),
            dtype=float,
        )
        states[:, 0] = np.asarray(state, dtype=float).reshape(self.state_dimension)
        for step in range(controls.shape[1]):
            states[:, step + 1] = np.asarray(
                self.model.discrete_step(states[:, step], controls[:, step]),
                dtype=float,
            ).reshape(self.state_dimension)
        return states

    def _evaluate_actuator_commands(
        self,
        states: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray(
                    self.model.actuator_commands(
                        states[3:5, step],
                        controls[:, step],
                    ),
                    dtype=float,
                ).reshape(4)
                for step in range(controls.shape[1])
            ]
        )

    def _guess_is_feasible(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        preview: ReferencePreview,
    ) -> bool:
        """Check the four hard inequality families on a numeric rollout."""

        p = self.parameters
        r = self.robot.parameters
        tolerance = max(10.0 * p.ipopt_tolerance, 1.0e-6)
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
            return False
        if np.any(states[3:5, :] < r.q_min[:, None] - tolerance):
            return False
        if np.any(states[3:5, :] > r.q_max[:, None] + tolerance):
            return False
        if np.any(np.linalg.norm(controls[0:2, :], axis=0) > p.body_speed_limit + tolerance):
            return False
        if np.any(np.abs(controls[2, :]) > p.body_yaw_rate_limit + tolerance):
            return False
        for step in range(states.shape[1]):
            heading = preview.poses[step, 2]
            normal = np.array([-np.sin(heading), np.cos(heading)], dtype=float)
            robot_width = float(
                self.robot.centred_envelope_width_expression(
                    states[:, step],
                    preview.poses[step, 0:2],
                    normal,
                    p.smooth_epsilon,
                )
            )
            footprint_widths = [
                float(self.corridor.full_width(vertex[0]))
                for vertex in self.robot.footprint_vertices_world(states[:, step])
            ]
            available_width = (
                min(footprint_widths)
                - 2.0 * p.clearance_margin
            )
            buffer = 0.0 if step == 0 else 2.0 * p.smooth_epsilon
            if robot_width > available_width - buffer + tolerance:
                return False
        return True

    def _fallback_solution(
        self,
        preview: ReferencePreview,
        states: np.ndarray,
        controls: np.ndarray,
        elapsed: float,
        status: str,
    ) -> NMPCSolution:
        actuator_commands = self._evaluate_actuator_commands(states, controls)
        self._last_states = states
        self._last_controls = controls
        objective = float(
            self._objective_function(
                states,
                controls,
                preview.poses.T,
                preview.speeds.reshape((1, -1)),
                preview.yaw_rates.reshape((1, -1)),
            )
        )
        return NMPCSolution(
            success=True,
            control=controls[:, 0].copy(),
            body_twist=controls[:, 0].copy(),
            actuator_command=actuator_commands[:, 0].copy(),
            predicted_states=states.T,
            predicted_controls=controls.T,
            predicted_twists=controls.T,
            predicted_actuator_commands=actuator_commands.T,
            objective=objective,
            solve_time=elapsed,
            status=f"{status} (feasible geometric fallback)",
        )

    def _successful_solution(
        self,
        solution: ca.OptiSol,
        elapsed: float,
        *,
        retried: bool = False,
    ) -> NMPCSolution:
        states = np.asarray(solution.value(self.states), dtype=float)
        controls = np.asarray(solution.value(self.controls), dtype=float)
        actuator_commands = self._evaluate_actuator_commands(states, controls)
        self._last_states = states
        self._last_controls = controls
        stats = self.opti.stats()
        status = str(stats.get("return_status", "Solve_Succeeded"))
        if retried:
            status = f"{status} (fresh-start retry)"
        return NMPCSolution(
            success=True,
            control=controls[:, 0].copy(),
            body_twist=controls[:, 0].copy(),
            actuator_command=actuator_commands[:, 0].copy(),
            predicted_states=states.T,
            predicted_controls=controls.T,
            predicted_twists=controls.T,
            predicted_actuator_commands=actuator_commands.T,
            objective=float(solution.value(self.objective_expression)),
            solve_time=elapsed,
            status=status,
        )

    def solve(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> NMPCSolution:
        self._set_parameters(state, preview)
        self._apply_warm_start(state, preview)
        start = perf_counter()
        try:
            solution = self.opti.solve()
            return self._successful_solution(
                solution,
                perf_counter() - start,
            )
        except RuntimeError as first_error:
            elapsed = perf_counter() - start
            first_status = str(
                self.opti.stats().get("return_status", "Solve_Failed")
            )
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
                    first_status,
                )

            # Build one geometry-aware seed. If it already satisfies every
            # hard equation and inequality, applying it is safer and faster
            # than asking IPOPT to repeat a failed restoration phase.
            self._last_states = None
            self._last_controls = None
            state_guess, control_guess = self._initial_guess(state, preview)
            if self._guess_is_feasible(state_guess, control_guess, preview):
                return self._fallback_solution(
                    preview,
                    state_guess,
                    control_guess,
                    elapsed,
                    first_status,
                )

            self.opti.set_initial(self.states, state_guess)
            self.opti.set_initial(self.controls, control_guess)
            self._current_state_guess = state_guess
            self._current_control_guess = control_guess
            try:
                solution = self.opti.solve()
                return self._successful_solution(
                    solution,
                    perf_counter() - start,
                    retried=True,
                )
            except RuntimeError as retry_error:
                error_headline = str(retry_error).splitlines()[0]
                if not error_headline:
                    error_headline = str(first_error).splitlines()[0]

            elapsed = perf_counter() - start
            stats = self.opti.stats()
            return_status = str(stats.get("return_status", "Solve_Failed"))
            try:
                constraint_values = np.asarray(
                    self.opti.debug.value(self.opti.g),
                    dtype=float,
                ).reshape(-1)
                lower_bounds = np.asarray(
                    self.opti.debug.value(self.opti.lbg),
                    dtype=float,
                ).reshape(-1)
                upper_bounds = np.asarray(
                    self.opti.debug.value(self.opti.ubg),
                    dtype=float,
                ).reshape(-1)
                violation = np.maximum(
                    np.maximum(
                        lower_bounds - constraint_values,
                        constraint_values - upper_bounds,
                    ),
                    0.0,
                )
                states = np.asarray(
                    self.opti.debug.value(self.states),
                    dtype=float,
                )
                controls = np.asarray(
                    self.opti.debug.value(self.controls),
                    dtype=float,
                )
                feasible_tolerance = max(10.0 * self.parameters.ipopt_tolerance, 1.0e-5)
                accept_iterate = (
                    return_status
                    in {"Maximum_Iterations_Exceeded", "Solved_To_Acceptable_Level"}
                    and np.all(np.isfinite(states))
                    and np.all(np.isfinite(controls))
                    and np.all(np.isfinite(violation))
                    and float(np.max(violation, initial=0.0)) <= feasible_tolerance
                )
            except (RuntimeError, ValueError):
                accept_iterate = False

            if accept_iterate:
                actuator_commands = self._evaluate_actuator_commands(states, controls)
                self._last_states = states
                self._last_controls = controls
                return NMPCSolution(
                    success=True,
                    control=controls[:, 0].copy(),
                    body_twist=controls[:, 0].copy(),
                    actuator_command=actuator_commands[:, 0].copy(),
                    predicted_states=states.T,
                    predicted_controls=controls.T,
                    predicted_twists=controls.T,
                    predicted_actuator_commands=actuator_commands.T,
                    objective=float(self.opti.debug.value(self.objective_expression)),
                    solve_time=elapsed,
                    status=f"{return_status} (feasible iterate accepted)",
                )

            state_guess = self._current_state_guess
            control_guess = self._current_control_guess
            if self._guess_is_feasible(state_guess, control_guess, preview):
                return self._fallback_solution(
                    preview,
                    state_guess,
                    control_guess,
                    elapsed,
                    return_status,
                )
            actuator_guess = self._evaluate_actuator_commands(
                state_guess,
                control_guess,
            )
            self._last_states = None
            self._last_controls = None
            return NMPCSolution(
                success=False,
                control=np.zeros(self.control_dimension, dtype=float),
                body_twist=np.zeros(3, dtype=float),
                actuator_command=np.r_[state[3:5], np.zeros(2, dtype=float)],
                predicted_states=state_guess.T,
                predicted_controls=control_guess.T,
                predicted_twists=control_guess.T,
                predicted_actuator_commands=actuator_guess.T,
                objective=float("nan"),
                solve_time=elapsed,
                status=f"{return_status}: {error_headline}",
            )
