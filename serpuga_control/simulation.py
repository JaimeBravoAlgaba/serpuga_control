"""Closed-loop simulation and quantitative diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MPCParameters, SimulationParameters
from .corridor import StraightGapCorridor
from .kinematics import KinematicModel
from .nmpc import NMPCController
from .robot import RobotDescription
from .trajectory import ReferenceTrajectory, wrapped_angle_error


@dataclass
class SimulationLog:
    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    reference_poses: np.ndarray
    reference_speeds: np.ndarray
    reference_yaw_rates: np.ndarray
    body_twists: np.ndarray
    world_velocities: np.ndarray
    slips: np.ndarray
    stability_margins: np.ndarray
    clearances: np.ndarray
    robot_widths: np.ndarray
    corridor_widths: np.ndarray
    solve_times: np.ndarray
    objectives: np.ndarray
    solver_statuses: list[str]
    predicted_states: list[np.ndarray]
    completed: bool

    def summary(self) -> dict[str, float | bool]:
        position_error = self.states[:-1, 0:2] - self.reference_poses[:, 0:2]
        heading_error = wrapped_angle_error(
            self.states[:-1, 2] - self.reference_poses[:, 2]
        )
        slip_norm = np.linalg.norm(self.slips, axis=2)
        return {
            "completed": self.completed,
            "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
            "heading_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(np.asarray(heading_error) ** 2)))
            ),
            "integrated_slip_m": float(
                np.sum(slip_norm) * (self.times[1] - self.times[0])
                if self.times.size > 1
                else 0.0
            ),
            "maximum_lateral_slip_mps": float(
                np.max(np.abs(self.slips[:, :, 1]))
            ),
            "minimum_clearance_m": float(np.min(self.clearances)),
            "minimum_stability_margin_m": float(np.min(self.stability_margins)),
            "maximum_fold_deg": float(
                np.max(
                    np.abs(
                        np.rad2deg(self.states[:, 3:5] - self.states[0, 3:5])
                    )
                )
            ),
            "mean_solve_time_s": float(np.mean(self.solve_times)),
            "maximum_solve_time_s": float(np.max(self.solve_times)),
        }


def _minimum_clearance(
    state: np.ndarray,
    robot: RobotDescription,
    corridor: StraightGapCorridor,
) -> float:
    clearances = []
    for vertex in robot.footprint_vertices_world(state):
        lower, upper = corridor.lateral_bounds(vertex[0])
        clearances.extend([vertex[1] - lower, upper - vertex[1]])
    return float(min(clearances))


def run_closed_loop(
    controller: NMPCController,
    model: KinematicModel,
    robot: RobotDescription,
    corridor: StraightGapCorridor,
    trajectory: ReferenceTrajectory,
    mpc_parameters: MPCParameters,
    simulation_parameters: SimulationParameters,
    verbose: bool = True,
) -> SimulationLog:
    """Run the receding-horizon loop against the same kinematic plant."""

    dt = mpc_parameters.dt
    maximum_steps = int(np.ceil(simulation_parameters.duration / dt))
    state = simulation_parameters.initial_state.astype(float).copy()
    previous_control = np.zeros(4, dtype=float)
    previous_world_velocity = np.zeros(2, dtype=float)

    states = [state.copy()]
    times: list[float] = []
    controls: list[np.ndarray] = []
    reference_poses: list[np.ndarray] = []
    reference_speeds: list[float] = []
    reference_yaw_rates: list[float] = []
    body_twists: list[np.ndarray] = []
    world_velocities: list[np.ndarray] = []
    slips: list[np.ndarray] = []
    stability_margins: list[float] = []
    clearances: list[float] = []
    robot_widths: list[float] = []
    corridor_widths: list[float] = []
    solve_times: list[float] = []
    objectives: list[float] = []
    statuses: list[str] = []
    predictions: list[np.ndarray] = []
    completed = True

    for step in range(maximum_steps):
        current_time = step * dt
        preview = trajectory.preview(
            current_time,
            dt,
            mpc_parameters.horizon_steps,
        )
        solution = controller.solve(
            state,
            preview,
            previous_control,
            previous_world_velocity,
        )
        if not solution.success:
            completed = False
            statuses.append(solution.status)
            if verbose:
                print(f"NMPC stopped at t={current_time:.2f}s: {solution.status}")
            break

        control = solution.control
        twist = solution.body_twist.copy()
        world_velocity = np.asarray(
            model.world_velocity_from_twist(state, twist), dtype=float
        ).reshape(2)
        slip = np.asarray(
            model.slip_components(state[3:5], control, twist),
            dtype=float,
        )
        world_acceleration = (world_velocity - previous_world_velocity) / dt
        centre_of_mass = np.asarray(robot.centre_of_mass_world(state), dtype=float)
        evaluation_point = centre_of_mass
        if mpc_parameters.use_zmp:
            evaluation_point = (
                centre_of_mass
                - robot.parameters.com_height
                / mpc_parameters.gravity
                * world_acceleration
            )
        path_normal = np.array(
            [-np.sin(preview.poses[0, 2]), np.cos(preview.poses[0, 2])],
            dtype=float,
        )
        _, _, stability_margin = robot.lateral_stability_margins(
            state,
            path_normal,
            evaluation_point,
            mpc_parameters.smooth_epsilon,
        )

        times.append(current_time)
        controls.append(control.copy())
        reference_poses.append(preview.poses[0].copy())
        reference_speeds.append(float(preview.speeds[0]))
        reference_yaw_rates.append(float(preview.yaw_rates[0]))
        body_twists.append(twist)
        world_velocities.append(world_velocity)
        slips.append(slip)
        stability_margins.append(float(stability_margin))
        clearances.append(_minimum_clearance(state, robot, corridor))
        robot_widths.append(robot.envelope_width(state, np.array([0.0, 1.0])))
        corridor_widths.append(float(corridor.full_width(state[0])))
        solve_times.append(solution.solve_time)
        objectives.append(solution.objective)
        statuses.append(solution.status)
        predictions.append(solution.predicted_states.copy())

        next_state = np.asarray(
            model.discrete_step(state, control), dtype=float
        ).reshape(5)
        states.append(next_state.copy())
        state = next_state
        previous_control = control.copy()
        previous_world_velocity = world_velocity.copy()

        if verbose and (step == 0 or (step + 1) % 10 == 0):
            print(
                f"t={current_time:5.2f}s  x={state[0]:5.2f}m  "
                f"q={np.rad2deg(state[3]):6.1f}deg  "
                f"solve={solution.solve_time:5.3f}s"
            )
        if (
            simulation_parameters.stop_position is not None
            and state[0] >= simulation_parameters.stop_position
        ):
            break

    if not times:
        raise RuntimeError("The controller did not complete a single simulation step")

    final_time = times[-1] + dt
    return SimulationLog(
        times=np.asarray(times),
        states=np.asarray(states),
        controls=np.asarray(controls),
        reference_poses=np.asarray(reference_poses),
        reference_speeds=np.asarray(reference_speeds),
        reference_yaw_rates=np.asarray(reference_yaw_rates),
        body_twists=np.asarray(body_twists),
        world_velocities=np.asarray(world_velocities),
        slips=np.asarray(slips),
        stability_margins=np.asarray(stability_margins),
        clearances=np.asarray(clearances),
        robot_widths=np.asarray(robot_widths),
        corridor_widths=np.asarray(corridor_widths),
        solve_times=np.asarray(solve_times),
        objectives=np.asarray(objectives),
        solver_statuses=statuses,
        predicted_states=predictions,
        completed=bool(
            completed
            and final_time <= simulation_parameters.duration + dt
            and (
                simulation_parameters.stop_position is None
                or state[0] >= simulation_parameters.stop_position
            )
        ),
    )
