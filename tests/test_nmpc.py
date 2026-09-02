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


def make_controller(
    parameters: MPCParameters,
    robot_parameters: RobotParameters | None = None,
    corridor: StraightGapCorridor | None = None,
):
    robot = RobotDescription(robot_parameters or RobotParameters())
    model = KinematicModel(robot, parameters)
    corridor = corridor or StraightGapCorridor(
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
    np.testing.assert_allclose(
        solution.body_twist,
        model.interval_body_twist(solution.predicted_states[0], solution.control),
    )


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


def test_body_speed_limits_hold_at_start_midpoint_and_end_of_each_step() -> None:
    parameters = MPCParameters(
        horizon_steps=4,
        body_speed_limit=0.22,
        track_speed_limit=0.6,
    )
    _robot, model, controller = make_controller(parameters)
    trajectory = ReferenceTrajectory.straight(2.0, parameters.dt, 0.5)

    solution = controller.solve(
        state=np.zeros(5),
        preview=trajectory.preview(0.0, parameters.dt, parameters.horizon_steps),
    )

    assert solution.success
    for state, control in zip(
        solution.predicted_states[:-1], solution.predicted_controls, strict=True
    ):
        for fraction in (0.0, 0.5, 1.0):
            q = model.interval_axes(state[3:5], control, fraction)
            twist = np.asarray(model.body_twist(control, q=q), dtype=float)
            assert np.linalg.norm(twist[:2]) <= parameters.body_speed_limit + 1e-5


def test_corridor_residual_is_referenced_to_real_centre_y() -> None:
    parameters = MPCParameters(horizon_steps=3, clearance_margin=0.01)
    corridor = StraightGapCorridor(
        open_width=0.8,
        gap_width=0.8,
        gap_start=10.0,
        gap_end=11.0,
        centre_y=0.25,
    )
    _robot, _model, controller = make_controller(parameters, corridor=corridor)

    centred = np.zeros(5)
    centred[1] = corridor.centre_y
    shifted = centred.copy()
    shifted[1] += 0.35

    assert controller._numeric_corridor_residual(centred) < 0.0
    assert controller._numeric_corridor_residual(shifted) > 0.0
