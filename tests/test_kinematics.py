import numpy as np

from serpuga_control import (
    KinematicModel,
    MPCParameters,
    RobotDescription,
    RobotParameters,
)


def make_model(parameters: RobotParameters | None = None) -> KinematicModel:
    robot = RobotDescription(parameters or RobotParameters())
    return KinematicModel(robot, MPCParameters(horizon_steps=3))


def test_parallel_forward_twist_has_analytic_zero_pivot_slip_solution() -> None:
    model = make_model()
    q = np.zeros(2)
    control = np.array([0.4, 0.0, 0.0])

    inverse = model.inverse_kinematics(control, q)

    np.testing.assert_allclose(inverse.articulation_angles, [0.0, 0.0])
    np.testing.assert_allclose(inverse.track_speeds, [0.4, 0.4])
    np.testing.assert_allclose(model.pivot_slip_vectors(q, control), 0.0, atol=1e-6)


def test_opposed_track_keeps_its_axis_and_uses_negative_belt_speed() -> None:
    parameters = RobotParameters.opposed_tracks()
    model = make_model(parameters)
    q = parameters.nominal_configuration

    inverse = model.inverse_kinematics(np.array([0.4, 0.0, 0.0]), q)

    np.testing.assert_allclose(inverse.articulation_angles, q, atol=1e-8)
    assert inverse.track_speeds[0] > 0.0
    assert inverse.track_speeds[1] < 0.0
    np.testing.assert_allclose(
        model.pivot_slip_vectors(q, np.array([0.4, 0.0, 0.0])),
        0.0,
        atol=1e-8,
    )


def test_reverse_twist_reverses_belts_without_turning_tracks() -> None:
    parameters = RobotParameters.opposed_tracks()
    model = make_model(parameters)
    q = parameters.nominal_configuration

    inverse = model.inverse_kinematics(np.array([-0.3, 0.0, 0.0]), q)

    np.testing.assert_allclose(inverse.articulation_angles, q, atol=1e-8)
    assert inverse.track_speeds[0] < 0.0
    assert inverse.track_speeds[1] > 0.0


def test_yaw_rate_uses_each_pivot_velocity_in_the_analytic_inverse() -> None:
    model = make_model()
    q = np.zeros(2)
    control = np.array([0.2, 0.05, 0.4])

    inverse = model.inverse_kinematics(control, q)
    pivots = model.robot.parameters.pivot_positions
    expected = np.array(
        [
            [control[0] - control[2] * pivots[0, 1], control[1]],
            [control[0] - control[2] * pivots[1, 1], control[1]],
        ]
    )

    np.testing.assert_allclose(inverse.pivot_velocities, expected)
    np.testing.assert_allclose(model.pivot_slip_vectors(q, control), 0.0, atol=1e-6)


def test_stationary_command_preserves_current_articulation() -> None:
    model = make_model()
    q = np.deg2rad(np.array([-32.0, 41.0]))

    inverse = model.inverse_kinematics(np.zeros(3), q)

    np.testing.assert_allclose(inverse.articulation_angles, q)
    np.testing.assert_allclose(inverse.track_speeds, 0.0)


def test_discrete_step_integrates_bar_twist_and_applies_ik_angles() -> None:
    model = make_model()
    state = np.zeros(5)
    control = np.array([0.25, 0.08, 0.1])
    inverse = model.inverse_kinematics(control, state[3:5])

    next_state = model.discrete_step(state, control)

    assert next_state[0] > 0.0
    assert next_state[1] > 0.0
    np.testing.assert_allclose(next_state[2], control[2] * model.mpc_parameters.dt)
    np.testing.assert_allclose(next_state[3:5], inverse.articulation_angles)


def test_track_centre_model_retains_finite_offset_scrubbing() -> None:
    model = make_model()
    q = np.zeros(2)
    control = np.array([0.2, 0.0, 0.4])

    slip = model.slip_components(q, control)

    assert np.max(np.abs(slip)) > 0.0
