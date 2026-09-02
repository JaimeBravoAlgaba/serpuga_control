import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from serpuga_control import MPCParameters, RobotDescription, RobotParameters
from serpuga_control.corridor import StraightGapCorridor
from serpuga_control.live_visualization import LiveSimulationPlayer
from serpuga_control.simulation import SimulationLog


def _small_log() -> SimulationLog:
    times = np.arange(3, dtype=float) * 0.15
    states = np.zeros((4, 5), dtype=float)
    states[:, 0] = np.linspace(0.0, 0.12, 4)
    states[:, 3] = np.deg2rad([0.0, -5.0, -10.0, -12.0])
    states[:, 4] = -states[:, 3]
    reference_poses = np.zeros((3, 3), dtype=float)
    reference_poses[:, 0] = states[:3, 0]
    predicted_states = []
    for index in range(3):
        prediction = np.repeat(states[index : index + 1], 3, axis=0)
        prediction[:, 0] += np.arange(3) * 0.04
        predicted_states.append(prediction)
    controls = np.column_stack(
        (
            states[:3, 3:5],
            np.full(3, 0.27),
            np.full(3, 0.27),
        )
    )
    return SimulationLog(
        times=times,
        states=states,
        controls=controls.copy(),
        actuator_commands=controls.copy(),
        reference_poses=reference_poses,
        reference_speeds=np.full(3, 0.27),
        reference_yaw_rates=np.zeros(3),
        body_twists=np.column_stack((np.full(3, 0.27), np.zeros((3, 2)))),
        world_velocities=np.column_stack((np.full(3, 0.27), np.zeros(3))),
        slips=np.zeros((3, 2, 2), dtype=float),
        stability_margins=np.full(3, 0.20),
        clearances=np.full(3, 0.05),
        robot_widths=np.array([0.62, 0.60, 0.58]),
        corridor_widths=np.full(3, 1.20),
        solve_times=np.full(3, 0.02),
        objectives=np.ones(3),
        solver_statuses=["Solve_Succeeded"] * 3,
        predicted_states=predicted_states,
        completed=True,
    )


def test_live_player_navigation_and_screenshot(tmp_path) -> None:
    parameters = MPCParameters()
    player = LiveSimulationPlayer(
        log=_small_log(),
        robot=RobotDescription(RobotParameters()),
        corridor=StraightGapCorridor(),
        mpc_parameters=parameters,
    )

    assert "q1*,q2*" in player.telemetry_text.get_text()
    assert "vx, vy    0.270,  0.000 m/s" in player.telemetry_text.get_text()

    player._on_timer()
    assert player.playback.index == 1
    assert player.step_backward() == 0
    assert not player.playback.playing
    assert player.step_forward() == 1

    player.toggle_playback()
    assert player.playback.playing
    output = player.save_frame(tmp_path / "live-player.png", frame_index=2)
    assert output.exists()
    assert output.stat().st_size > 10_000
    assert player.playback.index == 1
    assert player.playback.playing
    animation = player.save_animation(tmp_path / "live-player.gif", dpi=40)
    assert animation.exists()
    assert animation.stat().st_size > 10_000
    assert player.playback.index == 1
    assert player.playback.playing
    player.close()
