"""Nonlinear model predictive controller for trajectory and shape tracking."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np

from .config import MPCParameters
from .corridor import StraightGapCorridor
from .kinematics import KinematicModel
from .math_utils import smooth_max
from .robot import RobotDescription
from .trajectory import ReferencePreview


@dataclass
class NMPCSolution:
    success: bool
    control: np.ndarray
    body_twist: np.ndarray
    predicted_states: np.ndarray
    predicted_controls: np.ndarray
    predicted_twists: np.ndarray
    objective: float
    solve_time: float
    status: str


class NMPCController:
    """Multiple-shooting NMPC with geometric and stability constraints."""

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
        self._last_duals: np.ndarray | None = None
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
                -r.track_speed_limit,
                self.controls[0:2, :],
                r.track_speed_limit,
            )
        )
        self.opti.subject_to(
            self.opti.bounded(
                -r.articulation_rate_limit,
                self.controls[2:4, :],
                r.articulation_rate_limit,
            )
        )
        objective = 0
        previous_velocity_expression = self.previous_world_velocity

        for step in range(n):
            state = self.states[:, step]
            control = self.controls[:, step]
            # The rigid-body twist is not an independent actuator.  It is the
            # weighted minimum-slip projection of the two longitudinal track
            # velocity vectors and the articulation rates.
            twist = self.model.body_twist(state[3:5], control)
            next_state = self.model.discrete_step(state, control)
            self.opti.subject_to(self.states[:, step + 1] == next_state)
            self.opti.subject_to(
                self.opti.bounded(
                    -p.body_speed_limit,
                    twist[0:2],
                    p.body_speed_limit,
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    -p.body_yaw_rate_limit,
                    twist[2],
                    p.body_yaw_rate_limit,
                )
            )

            if step == 0:
                control_change = control - self.previous_control
            else:
                control_change = control - self.controls[:, step - 1]
            self.opti.subject_to(
                self.opti.bounded(
                    -r.track_acceleration_limit * p.dt,
                    control_change[0:2],
                    r.track_acceleration_limit * p.dt,
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    -r.articulation_acceleration_limit * p.dt,
                    control_change[2:4],
                    r.articulation_acceleration_limit * p.dt,
                )
            )

            world_velocity = self.model.world_velocity_from_twist(state, twist)
            slip = self.model.slip_components(state[3:5], control, twist)
            self.opti.subject_to(
                self.opti.bounded(
                    -p.maximum_lateral_slip,
                    slip[:, 1],
                    p.maximum_lateral_slip,
                )
            )

            desired_heading = self.reference_poses[2, step]
            desired_world_velocity = self.reference_speeds[0, step] * ca.vertcat(
                ca.cos(desired_heading), ca.sin(desired_heading)
            )
            position_error = state[0:2] - self.reference_poses[0:2, step]
            heading_error = state[2] - desired_heading
            self.opti.subject_to(
                self.opti.bounded(
                    -p.maximum_heading_error,
                    heading_error,
                    p.maximum_heading_error,
                )
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
                scrub_cost += (twist[2] + control[2 + track_index]) ** 2

            objective += p.position_weight * ca.sumsqr(position_error)
            objective += p.heading_weight * 2.0 * (1.0 - ca.cos(heading_error))
            objective += p.velocity_weight * ca.sumsqr(velocity_error)
            objective += p.yaw_rate_weight * yaw_rate_error**2
            objective += p.slip_weight * slip_cost
            objective += p.scrub_weight * scrub_cost
            objective += p.track_effort_weight * ca.sumsqr(control[0:2])
            objective += p.articulation_rate_weight * ca.sumsqr(control[2:4])
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
    ) -> None:
        self.opti.set_value(self.initial_state, state)
        self.opti.set_value(self.previous_control, previous_control)
        self.opti.set_value(self.previous_world_velocity, previous_world_velocity)
        self.opti.set_value(self.reference_poses, preview.poses.T)
        self.opti.set_value(self.reference_speeds, preview.speeds.reshape((1, -1)))
        self.opti.set_value(
            self.reference_yaw_rates,
            preview.yaw_rates.reshape((1, -1)),
        )

    def _initial_guess(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.parameters.horizon_steps
        r = self.robot.parameters
        state_guess = np.zeros((self.state_dimension, n + 1), dtype=float)
        state_guess[0:3] = preview.poses.T
        state_guess[:, 0] = state

        narrowing = np.asarray(
            [self.corridor.narrowing_factor(x) for x in preview.poses[:, 0]],
            dtype=float,
        )
        fold_fraction = np.clip(0.88 * narrowing, 0.0, 0.92)
        for index in range(2):
            state_guess[3 + index] = r.nominal_configuration[index] + fold_fraction * (
                r.narrow_configuration[index] - r.nominal_configuration[index]
            )
        state_guess[2] = preview.poses[:, 2] + fold_fraction * r.narrow_body_yaw
        state_guess[3:5, 0] = state[3:5]
        state_guess[2, 0] = state[2]

        control_guess = np.zeros((self.control_dimension, n), dtype=float)
        control_guess[2:4] = np.clip(
            np.diff(state_guess[3:5], axis=1) / self.parameters.dt,
            -r.articulation_rate_limit,
            r.articulation_rate_limit,
        )
        twist_scaling = np.diag([1.0, 1.0, 0.24])
        for step in range(n):
            q_step = state_guess[3:5, step]
            articulation_rates = control_guess[2:4, step]
            base_control = np.r_[np.zeros(2), articulation_rates]
            base_twist = np.asarray(
                self.model.body_twist(q_step, base_control), dtype=float
            ).reshape(3)
            speed_columns = []
            for track_index in range(2):
                unit_control = base_control.copy()
                unit_control[track_index] = 1.0
                speed_columns.append(
                    np.asarray(
                        self.model.body_twist(q_step, unit_control), dtype=float
                    ).reshape(3)
                    - base_twist
                )
            speed_map = np.column_stack(speed_columns)
            world_velocity_target = (
                state_guess[0:2, step + 1] - state_guess[0:2, step]
            ) / self.parameters.dt
            yaw = state_guess[2, step]
            world_to_body = np.array(
                [
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)],
                ]
            )
            target_twist = np.r_[
                world_to_body @ world_velocity_target,
                (state_guess[2, step + 1] - state_guess[2, step]) / self.parameters.dt,
            ]
            track_speeds = np.linalg.lstsq(
                twist_scaling @ speed_map,
                twist_scaling @ (target_twist - base_twist),
                rcond=None,
            )[0]
            control_guess[0:2, step] = np.clip(
                track_speeds,
                -r.track_speed_limit,
                r.track_speed_limit,
            )
        return state_guess, control_guess

    def _apply_warm_start(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
    ) -> None:
        preview_narrowing = np.asarray(
            [self.corridor.narrowing_factor(x) for x in preview.poses[:, 0]],
            dtype=float,
        )
        use_geometric_seed = (
            abs(self.robot.parameters.narrow_body_yaw) > 1.0e-9
            and float(np.max(preview_narrowing)) > 0.02
        )
        if (
            self._last_states is None
            or self._last_controls is None
            or use_geometric_seed
        ):
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
        # Primal shifting is substantially more robust here than reusing duals:
        # the active wall changes as individual vertices enter/leave the gap.

    def _evaluate_twists(
        self,
        states: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray(
                    self.model.body_twist(states[3:5, step], controls[:, step]),
                    dtype=float,
                ).reshape(3)
                for step in range(controls.shape[1])
            ]
        )

    def solve(
        self,
        state: np.ndarray,
        preview: ReferencePreview,
        previous_control: np.ndarray,
        previous_world_velocity: np.ndarray,
    ) -> NMPCSolution:
        self._set_parameters(state, preview, previous_control, previous_world_velocity)
        self._apply_warm_start(state, preview)
        start = perf_counter()
        try:
            solution = self.opti.solve()
            elapsed = perf_counter() - start
            states = np.asarray(solution.value(self.states), dtype=float)
            controls = np.asarray(solution.value(self.controls), dtype=float)
            twists = self._evaluate_twists(states, controls)
            self._last_states = states
            self._last_controls = controls
            self._last_duals = np.asarray(solution.value(self.opti.lam_g), dtype=float)
            stats = self.opti.stats()
            return NMPCSolution(
                success=True,
                control=controls[:, 0].copy(),
                body_twist=twists[:, 0].copy(),
                predicted_states=states.T,
                predicted_controls=controls.T,
                predicted_twists=twists.T,
                objective=float(solution.value(self.objective_expression)),
                solve_time=elapsed,
                status=str(stats.get("return_status", "Solve_Succeeded")),
            )
        except RuntimeError as error:
            elapsed = perf_counter() - start
            state_guess = self._current_state_guess
            control_guess = self._current_control_guess
            twist_guess = self._evaluate_twists(state_guess, control_guess)
            self._last_states = None
            self._last_controls = None
            self._last_duals = None
            stats = self.opti.stats()
            return_status = str(stats.get("return_status", "Solve_Failed"))
            error_headline = str(error).splitlines()[0]
            return NMPCSolution(
                success=False,
                control=np.zeros(self.control_dimension, dtype=float),
                body_twist=np.zeros(3, dtype=float),
                predicted_states=state_guess.T,
                predicted_controls=control_guess.T,
                predicted_twists=twist_guess.T,
                objective=float("nan"),
                solve_time=elapsed,
                status=f"{return_status}: {error_headline}",
            )
