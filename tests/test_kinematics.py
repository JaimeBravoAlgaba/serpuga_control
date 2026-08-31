import numpy as np

from serpuga_control import (
    KinematicModel,
    MPCParameters,
    RobotDescription,
    RobotParameters,
)


def make_model() -> KinematicModel:
    robot = RobotDescription(RobotParameters())
    return KinematicModel(robot, MPCParameters(horizon_steps=3))


def test_equal_parallel_track_speeds_generate_straight_motion() -> None:
    model = make_model()
    control = np.array([0.4, 0.4, 0.0, 0.0])
    twist = model.body_twist(np.zeros(2), control)
    np.testing.assert_allclose(twist, [0.4, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(
        model.slip_components(np.zeros(2), control, twist),
        0.0,
        atol=1.0e-6,
    )


def test_parallel_speed_difference_produces_yaw() -> None:
    model = make_model()
    twist = model.body_twist(np.zeros(2), np.array([0.2, 0.6, 0.0, 0.0]))
    assert twist[2] > 0.0


def test_discrete_step_integrates_only_the_projected_track_twist() -> None:
    model = make_model()
    state = np.zeros(5)
    control = np.array([0.25, 0.25, -0.1, 0.1])
    projected_twist = model.body_twist(state[3:5], control)
    next_state = model.discrete_step(state, control)
    assert next_state[0] > 0.0
    np.testing.assert_allclose(
        next_state[0],
        projected_twist[0] * model.mpc_parameters.dt,
        rtol=5.0e-3,
    )
    np.testing.assert_allclose(next_state[3:5], control[2:4] * model.mpc_parameters.dt)


def test_folded_tracks_expose_nonzero_lateral_slip_for_straight_twist() -> None:
    model = make_model()
    q = np.deg2rad(np.array([-45.0, 45.0]))
    twist = np.array([0.25, 0.0, 0.0])
    control = np.array([0.18, 0.18, 0.0, 0.0])
    slip = model.slip_components(q, control, twist)
    assert np.max(np.abs(slip[:, 1])) > 0.05
