from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg", force=True)

from serpuga_control.configuration import ConfigurationStore
from serpuga_control.online_visualization import OnlineSimulationPlot
from serpuga_control.simulation import SimulationLog

BUILTIN_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_online_plot_shows_initialised_robot_before_run(tmp_path) -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("default")
    figure = Figure(figsize=(9, 6), dpi=80)
    plot = OnlineSimulationPlot(figure, lambda: None)
    plot.reset(configuration)
    output = tmp_path / "online-initial.png"
    figure.savefig(output)
    assert output.exists()
    assert output.stat().st_size > 10_000
    assert len(plot.track_patches) == 2


def test_online_plot_arrows_follow_signed_track_speeds() -> None:
    configuration = ConfigurationStore(BUILTIN_CONFIGS).load("parallel-gap")
    figure = Figure(figsize=(9, 6), dpi=80)
    plot = OnlineSimulationPlot(figure, lambda: None)
    plot.reset(configuration)
    initial = configuration.simulation.initial_state
    states = np.vstack((initial, initial.copy()))
    states[1, 0] = 0.03

    log = SimulationLog(
        times=np.array([0.0]),
        states=states,
        controls=np.array([[0.0, 0.0, 0.0]]),
        actuator_commands=np.array([[0.0, 0.0, 0.2, -0.2]]),
        reference_poses=np.zeros((1, 3)),
        reference_speeds=np.array([0.0]),
        reference_yaw_rates=np.array([0.0]),
        body_twists=np.zeros((1, 3)),
        world_velocities=np.zeros((1, 2)),
        slips=np.zeros((1, 2, 2)),
        stability_margins=np.array([0.2]),
        clearances=np.array([0.1]),
        robot_widths=np.array([0.5]),
        corridor_widths=np.array([1.2]),
        solve_times=np.array([0.01]),
        objectives=np.array([1.0]),
        solver_statuses=["Solve_Succeeded"],
        predicted_states=[states],
        completed=False,
    )

    plot.update_log(log)
    first_x, _ = plot.direction_lines[0].get_data()
    second_x, _ = plot.direction_lines[1].get_data()
    assert first_x[1] > first_x[0]
    assert second_x[1] < second_x[0]
