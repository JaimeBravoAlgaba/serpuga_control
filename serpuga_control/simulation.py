"""Closed-loop simulation and quantitative diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MPCParameters, SimulationParameters
from .corridor import StraightGapCorridor
from .kinematics import KinematicModel
from .nmpc import NMPCController, NMPCSolution
from .robot import RobotDescription
from .trajectory import ReferenceTrajectory, wrapped_angle_error


@dataclass
class TeleoperationCommand:
    """Live bar-centre twist applied instead of the NMPC when enabled."""

    enabled: bool = False
    body_twist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        body_twist = np.nan_to_num(
            np.asarray(self.body_twist, dtype=float).reshape(3),
            nan=0.0,
        )
        self.body_twist = body_twist


@dataclass
class SimulationLog:
    times: np.ndarray
    states: np.ndarray
    # Body-twist controls [v_x, v_y, omega].
    controls: np.ndarray
    # Analytic IK outputs [q1, q2, v1, v2].
    actuator_commands: np.ndarray
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
    control_modes: list[str] = field(default_factory=list)

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
            "maximum_lateral_slip_mps": float(np.max(np.abs(self.slips[:, :, 1]))),
            "minimum_clearance_m": float(np.min(self.clearances)),
            "minimum_stability_margin_m": float(np.min(self.stability_margins)),
            "maximum_fold_deg": float(
                np.max(np.abs(np.rad2deg(self.states[:, 3:5] - self.states[0, 3:5])))
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


class ClosedLoopSession:
    """Stateful receding-horizon simulation that advances exactly one step.

    The GUI calls :meth:`step` from its worker thread and redraws the returned
    history immediately.  No future control solution is precomputed.
    """

    def __init__(
        self,
        controller: NMPCController,
        model: KinematicModel,
        robot: RobotDescription,
        corridor: StraightGapCorridor,
        trajectory: ReferenceTrajectory,
        mpc_parameters: MPCParameters,
        simulation_parameters: SimulationParameters,
    ) -> None:
        self.controller = controller
        self.model = model
        self.robot = robot
        self.corridor = corridor
        self.trajectory = trajectory
        self.mpc_parameters = mpc_parameters
        self.simulation_parameters = simulation_parameters

        self.dt = float(mpc_parameters.dt)
        self.maximum_steps = int(np.ceil(simulation_parameters.duration / self.dt))
        self.state = simulation_parameters.initial_state.astype(float).copy()
        self.previous_control = np.zeros(3, dtype=float)
        self.previous_track_speeds = np.zeros(2, dtype=float)
        self.previous_articulation_rates = np.zeros(2, dtype=float)
        self.previous_world_velocity = np.zeros(2, dtype=float)

        self.states: list[np.ndarray] = [self.state.copy()]
        self.times: list[float] = []
        self.controls: list[np.ndarray] = []
        self.actuator_commands: list[np.ndarray] = []
        self.reference_poses: list[np.ndarray] = []
        self.reference_speeds: list[float] = []
        self.reference_yaw_rates: list[float] = []
        self.body_twists: list[np.ndarray] = []
        self.world_velocities: list[np.ndarray] = []
        self.slips: list[np.ndarray] = []
        self.stability_margins: list[float] = []
        self.clearances: list[float] = []
        self.robot_widths: list[float] = []
        self.corridor_widths: list[float] = []
        self.solve_times: list[float] = []
        self.objectives: list[float] = []
        self.statuses: list[str] = []
        self.control_modes: list[str] = []
        self.predictions: list[np.ndarray] = []

        self.finished = False
        self.completed = False
        self.failure_status: str | None = None

    @property
    def current_time(self) -> float:
        return len(self.times) * self.dt

    def _manual_control(self, command: TeleoperationCommand) -> np.ndarray:
        p = self.mpc_parameters
        control = np.asarray(command.body_twist, dtype=float).reshape(3).copy()
        linear_norm = np.linalg.norm(control[0:2])
        if linear_norm > p.body_speed_limit:
            control[0:2] *= p.body_speed_limit / linear_norm
        control[2] = np.clip(
            control[2],
            -p.body_yaw_rate_limit,
            p.body_yaw_rate_limit,
        )
        return control

    def _manual_solution(self, command: TeleoperationCommand) -> NMPCSolution:
        control = self._manual_control(command)
        actuator_command = np.asarray(
            self.model.actuator_commands(self.state[3:5], control),
            dtype=float,
        ).reshape(4)
        prediction_state = self.state.copy()
        predicted_states = [prediction_state.copy()]
        predicted_controls = []
        predicted_twists = []
        predicted_actuator_commands = []
        for _ in range(self.mpc_parameters.horizon_steps):
            prediction_control = control.copy()
            prediction_actuator = np.asarray(
                self.model.actuator_commands(
                    prediction_state[3:5],
                    prediction_control,
                ),
                dtype=float,
            ).reshape(4)
            prediction_twist = np.asarray(
                self.model.body_twist(prediction_control),
                dtype=float,
            ).reshape(3)
            prediction_state = np.asarray(
                self.model.discrete_step(prediction_state, prediction_control),
                dtype=float,
            ).reshape(5)
            predicted_controls.append(prediction_control.copy())
            predicted_twists.append(prediction_twist.copy())
            predicted_actuator_commands.append(prediction_actuator.copy())
            predicted_states.append(prediction_state.copy())
        return NMPCSolution(
            success=True,
            control=control,
            body_twist=control.copy(),
            actuator_command=actuator_command,
            predicted_states=np.asarray(predicted_states, dtype=float),
            predicted_controls=np.asarray(predicted_controls, dtype=float).reshape(
                (-1, 3)
            ),
            predicted_twists=np.asarray(predicted_twists, dtype=float).reshape((-1, 3)),
            predicted_actuator_commands=np.asarray(
                predicted_actuator_commands,
                dtype=float,
            ).reshape((-1, 4)),
            objective=0.0,
            solve_time=0.0,
            status="Teleoperation",
        )

    def step(
        self,
        command: TeleoperationCommand | None = None,
        *,
        stop_when_complete: bool = True,
    ) -> bool:
        """Solve and apply one MPC interval; return whether it succeeded."""

        if self.finished:
            return False

        current_time = self.current_time
        preview = self.trajectory.preview(
            current_time,
            self.dt,
            self.mpc_parameters.horizon_steps,
        )
        manual_enabled = command is not None and command.enabled
        if manual_enabled:
            solution = self._manual_solution(command)
            control_mode = "manual"
        else:
            solution = self.controller.solve(
                self.state,
                preview,
                self.previous_control,
                self.previous_world_velocity,
                self.previous_track_speeds,
                self.previous_articulation_rates,
            )
            control_mode = "mpc"
        if not solution.success:
            self.finished = True
            self.completed = False
            self.failure_status = solution.status
            self.statuses.append(solution.status)
            return False

        control = solution.control
        twist = solution.body_twist.copy()
        actuator_command = solution.actuator_command.copy()
        world_velocity = np.asarray(
            self.model.world_velocity_from_twist(self.state, twist), dtype=float
        ).reshape(2)
        slip = np.asarray(
            self.model.slip_components(self.state[3:5], control),
            dtype=float,
        )
        world_acceleration = (world_velocity - self.previous_world_velocity) / self.dt
        centre_of_mass = np.asarray(
            self.robot.centre_of_mass_world(self.state), dtype=float
        )
        evaluation_point = centre_of_mass
        if self.mpc_parameters.use_zmp:
            evaluation_point = (
                centre_of_mass
                - self.robot.parameters.com_height
                / self.mpc_parameters.gravity
                * world_acceleration
            )
        path_normal = np.array(
            [-np.sin(preview.poses[0, 2]), np.cos(preview.poses[0, 2])],
            dtype=float,
        )
        _, _, stability_margin = self.robot.lateral_stability_margins(
            self.state,
            path_normal,
            evaluation_point,
            self.mpc_parameters.smooth_epsilon,
        )

        self.times.append(current_time)
        self.controls.append(control.copy())
        self.actuator_commands.append(actuator_command)
        self.reference_poses.append(preview.poses[0].copy())
        self.reference_speeds.append(float(preview.speeds[0]))
        self.reference_yaw_rates.append(float(preview.yaw_rates[0]))
        self.body_twists.append(twist)
        self.world_velocities.append(world_velocity)
        self.slips.append(slip)
        self.stability_margins.append(float(stability_margin))
        self.clearances.append(
            _minimum_clearance(self.state, self.robot, self.corridor)
        )
        self.robot_widths.append(
            self.robot.envelope_width(self.state, np.array([0.0, 1.0]))
        )
        self.corridor_widths.append(float(self.corridor.full_width(self.state[0])))
        self.solve_times.append(solution.solve_time)
        self.objectives.append(solution.objective)
        self.statuses.append(solution.status)
        self.control_modes.append(control_mode)
        self.predictions.append(solution.predicted_states.copy())

        next_state = np.asarray(
            self.model.discrete_step(self.state, control), dtype=float
        ).reshape(5)
        articulation_rates = (next_state[3:5] - self.state[3:5]) / self.dt
        self.states.append(next_state.copy())
        self.state = next_state
        self.previous_control = control.copy()
        self.previous_track_speeds = actuator_command[2:4].copy()
        self.previous_articulation_rates = articulation_rates
        self.previous_world_velocity = world_velocity.copy()

        reached_stop = (
            self.simulation_parameters.stop_position is not None
            and self.state[0] >= self.simulation_parameters.stop_position
        )
        exhausted_duration = len(self.times) >= self.maximum_steps
        if stop_when_complete and (reached_stop or exhausted_duration):
            self.finished = True
            self.completed = bool(
                reached_stop or self.simulation_parameters.stop_position is None
            )
        return True

    def to_log(self) -> SimulationLog:
        """Return an immutable numeric snapshot of the history so far."""

        return SimulationLog(
            times=np.asarray(self.times, dtype=float),
            states=np.asarray(self.states, dtype=float),
            controls=np.asarray(self.controls, dtype=float).reshape((-1, 3)),
            actuator_commands=np.asarray(
                self.actuator_commands,
                dtype=float,
            ).reshape((-1, 4)),
            reference_poses=np.asarray(self.reference_poses, dtype=float).reshape(
                (-1, 3)
            ),
            reference_speeds=np.asarray(self.reference_speeds, dtype=float),
            reference_yaw_rates=np.asarray(self.reference_yaw_rates, dtype=float),
            body_twists=np.asarray(self.body_twists, dtype=float).reshape((-1, 3)),
            world_velocities=np.asarray(self.world_velocities, dtype=float).reshape(
                (-1, 2)
            ),
            slips=np.asarray(self.slips, dtype=float).reshape((-1, 2, 2)),
            stability_margins=np.asarray(self.stability_margins, dtype=float),
            clearances=np.asarray(self.clearances, dtype=float),
            robot_widths=np.asarray(self.robot_widths, dtype=float),
            corridor_widths=np.asarray(self.corridor_widths, dtype=float),
            solve_times=np.asarray(self.solve_times, dtype=float),
            objectives=np.asarray(self.objectives, dtype=float),
            solver_statuses=list(self.statuses),
            predicted_states=[prediction.copy() for prediction in self.predictions],
            completed=bool(self.completed),
            control_modes=list(self.control_modes),
        )


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
    """Run the same stepwise session to completion for headless workflows."""

    session = ClosedLoopSession(
        controller=controller,
        model=model,
        robot=robot,
        corridor=corridor,
        trajectory=trajectory,
        mpc_parameters=mpc_parameters,
        simulation_parameters=simulation_parameters,
    )
    while not session.finished:
        succeeded = session.step()
        step_count = len(session.times)
        if verbose and succeeded and (step_count == 1 or step_count % 10 == 0):
            print(
                f"t={session.current_time:5.2f}s  x={session.state[0]:5.2f}m  "
                f"q={np.rad2deg(session.state[3]):6.1f}deg  "
                f"solve={session.solve_times[-1]:5.3f}s"
            )
        if not succeeded:
            if verbose:
                print(
                    f"NMPC stopped at t={session.current_time:.2f}s: "
                    f"{session.failure_status}"
                )
            break

    if not session.times:
        raise RuntimeError("The controller did not complete a single simulation step")
    return session.to_log()
