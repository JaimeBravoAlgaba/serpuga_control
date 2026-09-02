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


def test_parallel_track_commands_produce_forward_body_motion() -> None:
    model = make_model()
    control = np.array([0.0, 0.0, 0.4, 0.4])

    twist = model.body_twist(control)

    np.testing.assert_allclose(twist, [0.4, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(model.pivot_residual_vectors(control), 0.0, atol=1e-5)


def test_opposed_tracks_use_opposite_signed_belt_speeds() -> None:
    parameters = RobotParameters.opposed_tracks()
    model = make_model(parameters)
    q = parameters.nominal_configuration
    control = np.r_[q, 0.4, -0.4]

    twist = model.body_twist(control)

    assert twist[0] > 0.39
    assert abs(twist[1]) < 1e-6
    assert abs(twist[2]) < 1e-6


def test_direct_control_is_also_the_actuator_command() -> None:
    model = make_model()
    control = np.array([-0.2, 0.3, 0.25, 0.18])

    np.testing.assert_allclose(model.actuator_commands(control), control)


def test_articulation_rate_uses_commanded_q_directly() -> None:
    model = make_model()
    q = np.array([-0.1, 0.2])
    control = np.array([-0.25, 0.35, 0.0, 0.0])

    rates = model.articulation_rates(q, control)

    np.testing.assert_allclose(rates, (control[0:2] - q) / model.mpc_parameters.dt)


def test_intermediate_state_interpolates_articulation() -> None:
    model = make_model()
    state = np.zeros(5)
    control = np.array([-0.3, 0.3, 0.2, 0.2])

    midpoint = model.intermediate_state(state, control, 0.5)
    final = model.discrete_step(state, control)

    np.testing.assert_allclose(midpoint[3:5], 0.5 * control[0:2])
    np.testing.assert_allclose(final[3:5], control[0:2])


def test_discrete_step_applies_q_commands_and_integrates_forward_motion() -> None:
    model = make_model()
    state = np.zeros(5)
    control = np.array([0.0, 0.0, 0.25, 0.25])

    next_state = model.discrete_step(state, control)

    assert next_state[0] > 0.0
    assert abs(next_state[1]) < 1e-8
    assert abs(next_state[2]) < 1e-8
    np.testing.assert_allclose(next_state[3:5], control[0:2])


def test_pose_integration_does_not_use_final_q_for_whole_interval() -> None:
    model = make_model()
    state = np.zeros(5)
    control = np.array([0.0, np.deg2rad(60.0), 0.3, 0.3])

    integrated = model.discrete_step(state, control)
    final_axis_twist = model.body_twist(control)
    naive_y = model.mpc_parameters.dt * final_axis_twist[1]

    assert not np.isclose(integrated[1], naive_y, atol=1e-5)


def test_incompatible_track_commands_have_flat_residual_and_valid_slip() -> None:
    model = make_model()
    q = np.array([0.0, np.deg2rad(20.0)])
    control = np.array([0.0, np.deg2rad(45.0), 0.3, 0.3])

    residual = model.pivot_residual_vectors(control, q=q)
    slip = model.slip_components(q, control)

    assert residual.shape == (4,)
    assert slip.shape == (2, 2)
    assert np.all(np.isfinite(slip))
    assert np.max(np.abs(residual)) > 1e-3


def test_slip_diagnostic_uses_actual_q_not_only_q_command() -> None:
    model = make_model()
    control = np.array([0.0, np.deg2rad(45.0), 0.3, 0.3])
    slip_at_command = model.slip_components(control[0:2], control)
    slip_before_command = model.slip_components(np.array([0.0, 0.0]), control)

    assert not np.allclose(slip_at_command, slip_before_command)
