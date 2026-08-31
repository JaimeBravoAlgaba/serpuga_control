"""Nonlinear MPC with bar-centre twist controls and analytic track IK."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np

from .config import MPCParameters
from .corridor import StraightGapCorridor
from .kinematics import KinematicModel
from .math_utils import J2_NUMPY, smooth_max
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
        self._clearance_seed_cache: dict[tuple[float, ...], np.ndarray] = {}
        self._build_problem()

    def _build_problem(self) -> None:
        p = self.parameters
        r = self.robot.parameters
        n = p.horizon_steps

        self.opti = ca.Opti()
        self.states = self.opti.variable(self.state_dimension, n + 1)
        self.controls = self.opti.variable(self.control_dimension, n)

        self.initial_state = self.opti.parameter(self.state_dimension)
        self.previous_control = self.opti.parameter(self.control_dimension)
        self.previous_track_speeds = self.opti.parameter(2)
        self.previous_articulation_rates = self.opti.parameter(2)
        self.previous_world_velocity = self.opti.parameter(2)
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
                -p.body_speed_limit,
                self.controls[0:2, :],
                p.body_speed_limit,
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
        previous_velocity_expression = self.previous_world_velocity
        previous_track_speed_expression = self.previous_track_speeds
        previous_articulation_rate_expression = self.previous_articulation_rates

        for step in range(n):
            state = self.states[:, step]
            control = self.controls[:, step]
            twist = self.model.body_twist(control)
            inverse = self.model.inverse_kinematics(twist, state[3:5])
            track_speeds = inverse.track_speeds
            articulation_rates = (inverse.articulation_angles - state[3:5]) / p.dt

            self.opti.subject_to(ca.sumsqr(control[0:2]) <= p.body_speed_limit**2)
            self.opti.subject_to(
                self.opti.bounded(
                    -r.track_speed_limit,
                    track_speeds,
                    r.track_speed_limit,
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    -r.articulation_rate_limit,
                    articulation_rates,
                    r.articulation_rate_limit,
                )
            )

            track_speed_change = track_speeds - previous_track_speed_expression
            articulation_rate_change = (
                articulation_rates - previous_articulation_rate_expression
            )
            self.opti.subject_to(
                self.opti.bounded(
                    -r.track_acceleration_limit * p.dt,
                    track_speed_change,
                    r.track_acceleration_limit * p.dt,
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    -r.articulation_acceleration_limit * p.dt,
                    articulation_rate_change,
                    r.articulation_acceleration_limit * p.dt,
                )
            )

            next_state = self.model.discrete_step(state, control)
            self.opti.subject_to(self.states[:, step + 1] == next_state)

            world_velocity = self.model.world_velocity(state, control)
            slip = self.model.slip_components(state[3:5], control)
            self.opti.subject_to(
                self.opti.bounded(
                    -p.maximum_lateral_slip,
                    slip[:, 1],
                    p.maximum_lateral_slip,
                )
            )

            desired_heading = self.reference_poses[2, step]
            desired_world_velocity = self.reference_speeds[0, step] * ca.vertcat(
                ca.cos(desired_heading),
                ca.sin(desired_heading),
            )
            position_error = state[0:2] - self.reference_poses[0:2, step]
            heading_error = state[2] - desired_heading
            heading_excess = smooth_max(
                ca.sqrt(heading_error**2 + p.smooth_epsilon**2)
                - p.maximum_heading_error,
                0.0,
                p.smooth_epsilon,
            )
            velocity_error = world_velocity - desired_world_velocity
            yaw_rate_error = twist[2] - self.reference_yaw_rates[0, step]

            slip_cost = 0
            scrub_cost = 0
            for track_index in range(2):
                slip_cost += (
                    r.longitudinal_slip_weight * slip[track_index, 0] ** 2
                    + r.lateral_slip_weight * slip[track_index, 1] ** 2
                )
                scrub_cost += (twist[2] + articulation_rates[track_index]) ** 2

            if step == 0:
                control_change = control - self.previous_control
            else:
                control_change = control - self.controls[:, step - 1]

            objective += p.position_weight * ca.sumsqr(position_error)
            objective += p.heading_weight * 2.0 * (1.0 - ca.cos(heading_error))
            objective += p.heading_weight * heading_excess**2
            objective += p.velocity_weight * ca.sumsqr(velocity_error)
            objective += p.yaw_rate_weight * yaw_rate_error**2
            objective += p.slip_weight * slip_cost
            objective += p.scrub_weight * scrub_cost
            objective += p.track_effort_weight * ca.sumsqr(track_speeds)
            objective += p.articulation_rate_weight * ca.sumsqr(articulation_rates)
            objective += p.input_rate_weight * ca.sumsqr(control_change)

            for track_index in range(2):
                track_heading_error = (
                    state[2]
                    + state[3 + track_index]
                    - desired_heading
                    - r.nominal_configuration[track_index]
                )
                objective += (
                    p.track_alignment_weight * 2.0 * (1.0 - ca.cos(track_heading_error))
                )
            symmetry_reference = float(
                np.dot(r.symmetry_coupling, r.nominal_configuration)
            )
            symmetry_expression = ca.dot(ca.DM(r.symmetry_coupling), state[3:5])
            objective += (
                p.symmetry_weight * (symmetry_expression - symmetry_reference) ** 2
            )
            objective += p.nominal_configuration_weight * ca.sumsqr(
                state[3:5] - ca.DM(r.nominal_configuration)
            )

            world_acceleration = (world_velocity - previous_velocity_expression) / p.dt
            centre_of_mass = self.robot.centre_of_mass_world(state)
            if p.use_zmp:
                evaluation_point = (
                    centre_of_mass - (r.com_height / p.gravity) * world_acceleration
                )
            else:
                evaluation_point = centre_of_mass
            path_normal = ca.vertcat(-ca.sin(desired_heading), ca.cos(desired_heading))
            lower_margin, upper_margin, stability_margin = (
                self.robot.lateral_stability_margins(
                    state,
                    path_normal,
                    evaluation_point,
                    p.smooth_epsilon,
                )
            )
            self.opti.subject_to(lower_margin >= p.minimum_stability_margin)
            self.opti.subject_to(upper_margin >= p.minimum_stability_margin)
            stability_deficit = smooth_max(
                p.target_stability_margin - stability_margin,
                0.0,
                p.smooth_epsilon,
            )
            objective += p.stability_weight * stability_deficit**2
            objective -= 0.05 * p.stability_weight * stability_margin

            self._add_corridor_constraints(state)
            previous_velocity_expression = world_velocity
            previous_track_speed_expression = track_speeds
            previous_articulation_rate_expression = articulation_rates

        self._add_corridor_constraints(self.states[:, n])

        terminal_position_error = self.states[0:2, n] - self.reference_poses[0:2, n]
        terminal_heading_error = self.states[2, n] - self.reference_poses[2, n]
        objective += p.terminal_position_weight * ca.sumsqr(terminal_position_error)
        objective += (
            p.terminal_heading_weight * 2.0 * (1.0 - ca.cos(terminal_heading_error))
        )

        self.opti.minimize(objective)
        self.objective_expression = objective

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

    def _add_corridor_constraints(self, state: ca.MX) -> None:
        margin = self.parameters.clearance_margin
        for vertex in self.robot.footprint_vertices_world(state):
            lower, upper = self.corridor.lateral_bounds(vertex[0])
            self.opti.subject_to(vertex[1] >= lower + margin)
            self.opti.subject_to(vertex[1] <= upper - margin)

    def _set_parameters(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
        previous_control: np.ndarray,
        previous_world_velocity: np.ndarray,
        previous_track_speeds: np.ndarray | None,
        previous_articulation_rates: np.ndarray | None,
    ) -> None:
        if previous_track_speeds is None or previous_articulation_rates is None:
            inverse = self.model.inverse_kinematics(previous_control, state[3:5])
            if previous_track_speeds is None:
                previous_track_speeds = np.asarray(
                    inverse.track_speeds,
                    dtype=float,
                ).reshape(2)
            if previous_articulation_rates is None:
                previous_articulation_rates = np.zeros(2, dtype=float)

        self.opti.set_value(self.initial_state, state)
        self.opti.set_value(self.previous_control, previous_control)
        self.opti.set_value(self.previous_track_speeds, previous_track_speeds)
        self.opti.set_value(
            self.previous_articulation_rates,
            previous_articulation_rates,
        )
        self.opti.set_value(self.previous_world_velocity, previous_world_velocity)
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
        n = self.parameters.horizon_steps
        r = self.robot.parameters
        p = self.parameters
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
                preview.yaw_rates[step],
            ]

            seed_pose = preview.poses[step + 1].copy()
            seed_pose[2] = current[2]
            clearance_q = self._clearance_seed_configuration(
                seed_pose,
                preferred_configuration=current[3:5],
            )
            maximum_change = r.articulation_rate_limit * p.dt
            q_seed = current[3:5] + np.clip(
                clearance_q - current[3:5],
                -maximum_change,
                maximum_change,
            )
            compatible = self._compatible_twist(q_seed, desired_twist)
            control_guess[:, step] = self._clip_twist(compatible)
            state_guess[:, step + 1] = np.asarray(
                self.model.discrete_step(current, control_guess[:, step]),
                dtype=float,
            ).reshape(self.state_dimension)
        return state_guess, control_guess

    def _clearance_seed_configuration(
        self,
        pose: np.ndarray,
        preferred_configuration: np.ndarray | None = None,
    ) -> np.ndarray:
        r = self.robot.parameters
        preferred = (
            r.nominal_configuration
            if preferred_configuration is None
            else np.asarray(preferred_configuration, dtype=float).reshape(2)
        )
        state = np.r_[pose, preferred]
        path_normal = np.array([-np.sin(pose[2]), np.cos(pose[2])], dtype=float)
        available_width = (
            float(self.corridor.full_width(float(pose[0])))
            - 2.0 * self.parameters.clearance_margin
        )
        preferred_width = self.robot.envelope_width(state, path_normal)
        if preferred_width <= available_width:
            return preferred.copy()

        cache_key = (
            round(float(pose[2]), 4),
            round(available_width, 4),
            round(float(preferred[0]), 3),
            round(float(preferred[1]), 3),
        )
        cached = self._clearance_seed_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        q1_values = np.linspace(r.q_min[0], r.q_max[0], 41)
        q2_values = np.linspace(r.q_min[1], r.q_max[1], 41)
        symmetry_reference = float(np.dot(r.symmetry_coupling, r.nominal_configuration))
        best_q: np.ndarray | None = None
        best_cost = float("inf")
        narrowest_q = preferred.copy()
        narrowest_width = preferred_width
        for q1 in q1_values:
            for q2 in q2_values:
                q = np.array([q1, q2], dtype=float)
                state[3:5] = q
                width = self.robot.envelope_width(state, path_normal)
                distance_cost = float(np.sum((q - preferred) ** 2))
                symmetry_error = float(
                    np.dot(r.symmetry_coupling, q) - symmetry_reference
                )
                cost = distance_cost + 0.2 * symmetry_error**2
                if width < narrowest_width:
                    narrowest_width = width
                    narrowest_q = q.copy()
                if width <= available_width and cost < best_cost:
                    best_cost = cost
                    best_q = q.copy()

        selected = narrowest_q if best_q is None else best_q
        self._clearance_seed_cache[cache_key] = selected.copy()
        return selected

    def _apply_warm_start(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> None:
        if self._last_states is None or self._last_controls is None:
            state_guess, control_guess = self._initial_guess(state, preview)
        else:
            state_guess = np.column_stack(
                (self._last_states[:, 1:], self._last_states[:, -1])
            )
            control_guess = np.column_stack(
                (self._last_controls[:, 1:], self._last_controls[:, -1])
            )
            state_guess[:, 0] = state
        self.opti.set_initial(self.states, state_guess)
        self.opti.set_initial(self.controls, control_guess)
        self._current_state_guess = state_guess
        self._current_control_guess = control_guess

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
        previous_control: np.ndarray,
        previous_world_velocity: np.ndarray,
        previous_track_speeds: np.ndarray | None = None,
        previous_articulation_rates: np.ndarray | None = None,
    ) -> NMPCSolution:
        self._set_parameters(
            state,
            preview,
            previous_control,
            previous_world_velocity,
            previous_track_speeds,
            previous_articulation_rates,
        )
        self._apply_warm_start(state, preview)
        start = perf_counter()
        try:
            solution = self.opti.solve()
            return self._successful_solution(
                solution,
                perf_counter() - start,
            )
        except RuntimeError:
            # A shifted warm start can occasionally lead IPOPT into a poor
            # local restoration path. Retry once from a geometry-aware fresh
            # seed before deciding that the control interval has failed.
            self._last_states = None
            self._last_controls = None
            state_guess, control_guess = self._initial_guess(state, preview)
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
