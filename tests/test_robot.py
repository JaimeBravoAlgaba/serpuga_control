import numpy as np

from serpuga_control import RobotDescription, RobotParameters


def test_footprint_contains_tracks_and_connector() -> None:
    robot = RobotDescription(RobotParameters())
    vertices = robot.footprint_vertices_world(np.zeros(5))
    assert len(vertices) == 12
    assert all(np.asarray(vertex).shape == (2,) for vertex in vertices)


def test_symmetric_folding_reduces_lateral_envelope() -> None:
    robot = RobotDescription(RobotParameters())
    parallel = np.zeros(5)
    folded = np.array([0.0, 0.0, 0.0, np.deg2rad(-50.0), np.deg2rad(50.0)])
    normal = np.array([0.0, 1.0])
    assert robot.envelope_width(folded, normal) < robot.envelope_width(parallel, normal)


def test_parallelism_residual_accepts_parallel_and_antiparallel_axes() -> None:
    robot = RobotDescription(RobotParameters())

    assert np.isclose(robot.parallelism_residual(np.array([0.3, 0.3])), 0.0)
    assert np.isclose(
        robot.parallelism_residual(np.array([0.3, 0.3 + np.pi])),
        0.0,
        atol=1.0e-12,
    )
    assert np.isclose(
        abs(robot.parallelism_residual(np.array([0.0, 0.5 * np.pi]))),
        1.0,
    )


def test_centred_width_includes_lateral_tracking_offset() -> None:
    robot = RobotDescription(RobotParameters())
    centred = np.zeros(5)
    offset = centred.copy()
    offset[1] = 0.08
    normal = np.array([0.0, 1.0])

    centred_width = robot.centred_envelope_width_expression(
        centred,
        np.zeros(2),
        normal,
        epsilon=1.0e-3,
    )
    offset_width = robot.centred_envelope_width_expression(
        offset,
        np.zeros(2),
        normal,
        epsilon=1.0e-3,
    )

    assert offset_width > centred_width


def test_centre_of_mass_stays_finite_across_configuration() -> None:
    robot = RobotDescription(RobotParameters())
    for angle in np.linspace(0.0, np.deg2rad(60.0), 8):
        state = np.array([1.0, -0.2, 0.1, -angle, angle])
        centre = np.asarray(robot.centre_of_mass_world(state))
        assert np.all(np.isfinite(centre))


def test_opposed_configuration_has_transverse_bar_and_antiparallel_tracks() -> None:
    parameters = RobotParameters.opposed_tracks()
    robot = RobotDescription(parameters)
    q = parameters.nominal_configuration
    first_centre = np.asarray(robot.track_center_body(q[0], 0))
    second_centre = np.asarray(robot.track_center_body(q[1], 1))
    connector_direction = parameters.pivot_positions[1] - parameters.pivot_positions[0]
    first_direction = np.array([np.cos(q[0]), np.sin(q[0])])
    second_direction = np.array([np.cos(q[1]), np.sin(q[1])])

    assert first_centre[0] > 0.0
    assert second_centre[0] < 0.0
    assert np.isclose(connector_direction[0], 0.0)
    assert np.isclose(np.dot(first_direction, second_direction), -1.0)
