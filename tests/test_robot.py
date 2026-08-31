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


def test_centre_of_mass_stays_finite_across_configuration() -> None:
    robot = RobotDescription(RobotParameters())
    for angle in np.linspace(0.0, np.deg2rad(60.0), 8):
        state = np.array([1.0, -0.2, 0.1, -angle, angle])
        centre = np.asarray(robot.centre_of_mass_world(state))
        assert np.all(np.isfinite(centre))

