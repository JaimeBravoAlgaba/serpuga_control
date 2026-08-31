from pathlib import Path

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg", force=True)

from serpuga_control.configuration import ConfigurationStore
from serpuga_control.online_visualization import OnlineSimulationPlot

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
