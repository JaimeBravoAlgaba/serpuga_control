import numpy as np

from serpuga_control import (
    KinematicModel,
    MPCParameters,
    RobotDescription,
    RobotParameters,
)
from serpuga_control.corridor import StraightGapCorridor
from serpuga_control.nmpc import NMPCController
from serpuga_control.trajectory import ReferenceTrajectory


def make_controller(parameters: MPCParameters, robot_parameters: RobotParameters | None = None):
    robot = RobotDescription(robot_parameters or RobotParameters())
    model = KinematicModel(robot, parameters)
    corridor = StraightGapCorridor(
        open_width=2.0,
        gap_width=2.0,
        gap_start=10.0,
        gap_end=11.0,
    )
    return robot, model, NMPCController(robot, model, corridor, parameters)


def test_short_horizon_solution_tracks_forward_reference() -> None:
    parameters = MPCParameters(horizon_steps=4)
    _robot, model, controller = make_controller(parameters)
    trajectory = ReferenceTrajectory.straight(2.0, parameters.dt, 0.25)

    solution = controller.solve(
        state=np.zeros(5),
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
    )

    assert solution.success
    assert solution.control.shape == (4,)
    np.testing.assert_allclose(solution.actuator_command, solution.control)
    assert solution.body_twist[0] > 0.20
    assert abs(solution.body_twist[1]) < 0.02
    assert abs(solution.body_twist[2]) < 0.02
    np.testing.assert_allclose(solution.body_twist, model.body_twist(solution.control))


def test_opposed_tracks_directly_output_opposite_belt_speeds() -> None:
    parameters = MPCParameters(horizon_steps=4)
    robot_parameters = RobotParameters.opposed_tracks()
    _robot, _model, controller = make_controller(parameters, robot_parameters)
    trajectory = ReferenceTrajectory.straight(2.0, parameters.dt, 0.25)
    state = np.zeros(5)
    state[3:5] = robot_parameters.nominal_configuration

    solution = controller.solve(
        state=state,
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
    )

    assert solution.success
    np.testing.assert_allclose(solution.control[0:2], solution.actuator_command[0:2])
    assert solution.control[2] > 0.0
    assert solution.control[3] < 0.0
    assert solution.body_twist[0] > 0.20


def test_track_speed_limit_is_enforced() -> None:
    parameters = MPCParameters(horizon_steps=4, track_speed_limit=0.12)
    _robot, _model, controller = make_controller(parameters)
    trajectory = ReferenceTrajectory.straight(2.0, parameters.dt, 0.30)

    solution = controller.solve(
        state=np.zeros(5),
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
    )

    assert solution.success
    assert np.max(np.abs(solution.predicted_controls[:, 2:4])) <= 0.12 + 1e-5


def test_articulation_rate_limit_is_enforced_over_prediction() -> None:
    parameters = MPCParameters(horizon_steps=4, articulation_rate_limit=0.25)
    _robot, _model, controller = make_controller(parameters)
    trajectory = ReferenceTrajectory.straight(2.0, parameters.dt, 0.25)
    state = np.array([0.0, 0.0, 0.0, -0.35, 0.35], dtype=float)

    solution = controller.solve(
        state=state,
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
    )

    assert solution.success
    q_previous = solution.predicted_states[:-1, 3:5]
    q_commands = solution.predicted_controls[:, 0:2]
    rates = (q_commands - q_previous) / parameters.dt
    assert np.max(np.abs(rates)) <= parameters.articulation_rate_limit + 1.0e-5
