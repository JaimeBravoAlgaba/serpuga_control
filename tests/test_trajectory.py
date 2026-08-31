import numpy as np

from serpuga_control.trajectory import ReferenceTrajectory


def test_constant_twist_starts_at_selected_pose_and_integrates_yaw_rate() -> None:
    initial = np.array([1.0, -0.5, 0.2])
    trajectory = ReferenceTrajectory.constant_twist(
        duration=2.0,
        integration_dt=0.1,
        speed=0.3,
        yaw_rate=0.15,
        initial_pose=initial,
    )
    np.testing.assert_allclose(trajectory.poses[0], initial)
    assert np.isclose(trajectory.poses[-1, 2], initial[2] + 0.3)
    assert np.allclose(trajectory.speeds, 0.3)
    assert np.allclose(trajectory.yaw_rates, 0.15)
