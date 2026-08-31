from dataclasses import replace

import numpy as np

from serpuga_control.configuration import ConfigurationStore
from serpuga_control.runtime import build_trajectory


def test_initial_robot_yaw_does_not_rotate_position_reference() -> None:
    configuration = ConfigurationStore("configs").load("parallel-gap")
    simulation = replace(
        configuration.simulation,
        initial_state=np.array([0.0, 0.2, np.deg2rad(90.0), 0.0, 0.0]),
    )
    configuration = replace(configuration, simulation=simulation)

    trajectory = build_trajectory(configuration)

    assert trajectory.poses[1, 0] > trajectory.poses[0, 0]
    np.testing.assert_allclose(trajectory.poses[:4, 1], 0.2, atol=1.0e-12)
    np.testing.assert_allclose(trajectory.poses[0, 2], 0.0)
