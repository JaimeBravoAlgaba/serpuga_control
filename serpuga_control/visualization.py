"""Matplotlib visualisation for geometry, predictions and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .config import MPCParameters
from .corridor import StraightGapCorridor
from .robot import RobotDescription
from .simulation import SimulationLog


COLOURS = {
    "ink": "#162033",
    "muted": "#667085",
    "grid": "#DCE2EA",
    "teal": "#00A6A6",
    "teal_dark": "#087F8C",
    "violet": "#6D5BD0",
    "amber": "#F4A261",
    "red": "#D84A5B",
    "wall": "#E8ECF2",
    "white": "#FFFFFF",
}


def _polygon_array(vertices: list[np.ndarray]) -> np.ndarray:
    return np.asarray(vertices, dtype=float).reshape((-1, 2))


def draw_robot(
    axis: plt.Axes,
    state: np.ndarray,
    robot: RobotDescription,
    alpha: float = 1.0,
    label: str | None = None,
    zorder: int = 5,
) -> None:
    connector = _polygon_array(robot.connector_vertices_world(state))
    axis.add_patch(
        Polygon(
            connector,
            closed=True,
            facecolor=COLOURS["ink"],
            edgecolor=COLOURS["white"],
            linewidth=0.8,
            alpha=alpha,
            zorder=zorder,
        )
    )
    track_colours = [COLOURS["teal"], COLOURS["violet"]]
    for index, colour in enumerate(track_colours):
        vertices = _polygon_array(robot.track_vertices_world(state, index))
        axis.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=colour,
                edgecolor=COLOURS["ink"],
                linewidth=1.0,
                alpha=alpha,
                zorder=zorder + 1,
            )
        )
        centre = np.asarray(robot.track_center_body(state[3 + index], index))
        body_rotation = np.array(
            [
                [np.cos(state[2]), -np.sin(state[2])],
                [np.sin(state[2]), np.cos(state[2])],
            ]
        )
        centre_world = state[0:2] + body_rotation @ centre
        direction = body_rotation @ np.array(
            [np.cos(state[3 + index]), np.sin(state[3 + index])]
        )
        axis.arrow(
            centre_world[0],
            centre_world[1],
            0.11 * direction[0],
            0.11 * direction[1],
            width=0.004,
            head_width=0.025,
            color=COLOURS["white"],
            alpha=alpha,
            length_includes_head=True,
            zorder=zorder + 2,
        )
    if label:
        axis.annotate(
            label,
            xy=state[0:2],
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLOURS["ink"],
            weight="semibold",
            zorder=zorder + 3,
        )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(COLOURS["grid"])
    axis.tick_params(colors=COLOURS["muted"], labelsize=8)
    axis.grid(True, color=COLOURS["grid"], linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def plot_simulation_dashboard(
    log: SimulationLog,
    robot: RobotDescription,
    corridor: StraightGapCorridor,
    mpc_parameters: MPCParameters,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    """Create the static visualiser dashboard used in the demo and README."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
        }
    )
    figure = plt.figure(figsize=(16, 10.4), facecolor="#F7F9FC")
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 0.88, 1.0],
        left=0.055,
        right=0.97,
        top=0.90,
        bottom=0.075,
        hspace=0.43,
        wspace=0.22,
    )
    scene = figure.add_subplot(grid[:, 0])
    configuration = figure.add_subplot(grid[0, 1])
    tracking = figure.add_subplot(grid[1, 1])
    diagnostics = figure.add_subplot(grid[2, 1])

    figure.suptitle(
        "SERPUGA · Control predictivo y reconfiguración",
        x=0.055,
        y=0.955,
        ha="left",
        fontsize=20,
        color=COLOURS["ink"],
        weight="bold",
    )
    figure.text(
        0.055,
        0.918,
        "Seguimiento cinemático · mínimo deslizamiento · adaptación al corredor · margen ZMP",
        ha="left",
        fontsize=10.5,
        color=COLOURS["muted"],
    )

    x_min = min(-0.35, float(np.min(log.states[:, 0])) - 0.35)
    x_max = float(np.max(log.states[:, 0])) + 0.45
    has_narrowing = corridor.gap_width < corridor.open_width - 1.0e-9
    if has_narrowing:
        x_max = max(x_max, corridor.gap_end + 0.6)
    wall_x, upper_wall, lower_wall = corridor.wall_profiles(x_min, x_max)
    plot_height = 0.72
    scene.fill_between(
        wall_x,
        upper_wall,
        plot_height,
        color=COLOURS["wall"],
        zorder=0,
    )
    scene.fill_between(
        wall_x,
        -plot_height,
        lower_wall,
        color=COLOURS["wall"],
        zorder=0,
    )
    scene.plot(wall_x, upper_wall, color=COLOURS["ink"], linewidth=1.3)
    scene.plot(wall_x, lower_wall, color=COLOURS["ink"], linewidth=1.3)
    if has_narrowing:
        scene.axvspan(
            corridor.gap_start,
            corridor.gap_end,
            color=COLOURS["amber"],
            alpha=0.10,
            label="Hueco estrecho",
        )
    scene.plot(
        log.reference_poses[:, 0],
        log.reference_poses[:, 1],
        linestyle=(0, (4, 4)),
        color=COLOURS["muted"],
        linewidth=1.5,
        label="Referencia",
    )
    scene.plot(
        log.states[:, 0],
        log.states[:, 1],
        color=COLOURS["teal_dark"],
        linewidth=2.4,
        label="Trayectoria ejecutada",
        zorder=3,
    )

    fold_index = int(np.argmax(np.abs(log.states[:-1, 3])))
    approach_candidates = np.flatnonzero(log.states[:-1, 0] < corridor.gap_start - 0.15)
    exit_candidates = np.flatnonzero(log.states[:-1, 0] > corridor.gap_end + 0.15)
    approach_index = int(approach_candidates[-1]) if approach_candidates.size else 0
    exit_index = int(exit_candidates[0]) if exit_candidates.size else len(log.states) - 1
    snapshot_indices = sorted(
        set([0, approach_index, fold_index, exit_index, len(log.states) - 1])
    )
    for index in snapshot_indices:
        if index == fold_index:
            draw_robot(scene, log.states[index], robot, alpha=0.98, label="Modo estrecho")
        elif index == 0:
            draw_robot(scene, log.states[index], robot, alpha=0.42, label="Inicio")
        elif index == len(log.states) - 1:
            draw_robot(scene, log.states[index], robot, alpha=0.42, label="Final")
        else:
            draw_robot(scene, log.states[index], robot, alpha=0.26)

    if fold_index < len(log.predicted_states):
        prediction = log.predicted_states[fold_index]
        scene.plot(
            prediction[:, 0],
            prediction[:, 1],
            color=COLOURS["violet"],
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
            label="Horizonte MPC",
        )

    scene.set_title("Vista superior del corredor", loc="left", color=COLOURS["ink"], pad=12)
    scene.set_xlabel("x [m]", color=COLOURS["muted"])
    scene.set_ylabel("y [m]", color=COLOURS["muted"])
    scene.set_xlim(x_min, x_max)
    scene.set_ylim(-plot_height, plot_height)
    scene.set_aspect("equal", adjustable="box")
    _style_axis(scene)
    scene.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        fontsize=8,
    )

    time_states = np.arange(log.states.shape[0]) * mpc_parameters.dt
    configuration.plot(
        time_states,
        np.rad2deg(log.states[:, 3]),
        color=COLOURS["teal"],
        linewidth=2.0,
        label=r"$q_1$",
    )
    configuration.plot(
        time_states,
        np.rad2deg(log.states[:, 4]),
        color=COLOURS["violet"],
        linewidth=2.0,
        label=r"$q_2$",
    )
    configuration.set_ylabel("Ángulo [deg]", color=COLOURS["muted"])
    configuration.set_xlabel("Tiempo [s]", color=COLOURS["muted"])
    configuration.set_title("Reconfiguración y anchura", loc="left", color=COLOURS["ink"])
    _style_axis(configuration)
    width_axis = configuration.twinx()
    width_axis.plot(
        log.times,
        log.corridor_widths,
        color=COLOURS["amber"],
        linewidth=1.6,
        alpha=0.9,
        label="Corredor",
    )
    width_axis.plot(
        log.times,
        log.robot_widths,
        color=COLOURS["ink"],
        linewidth=1.3,
        linestyle="--",
        label="Robot",
    )
    width_axis.set_ylabel("Anchura [m]", color=COLOURS["muted"])
    width_axis.tick_params(colors=COLOURS["muted"], labelsize=8)
    width_axis.spines["top"].set_visible(False)
    handles_a, labels_a = configuration.get_legend_handles_labels()
    handles_b, labels_b = width_axis.get_legend_handles_labels()
    configuration.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    reference_tangents = np.column_stack(
        (
            np.cos(log.reference_poses[:, 2]),
            np.sin(log.reference_poses[:, 2]),
        )
    )
    forward_speed = np.sum(log.world_velocities * reference_tangents, axis=1)
    tracking.plot(
        log.times,
        log.reference_speeds,
        color=COLOURS["muted"],
        linewidth=1.3,
        linestyle="--",
        label=r"$v_\mathrm{ref}$",
    )
    tracking.plot(
        log.times,
        forward_speed,
        color=COLOURS["teal_dark"],
        linewidth=2.0,
        label=r"$v$",
    )
    tracking.set_title("Seguimiento de velocidad", loc="left", color=COLOURS["ink"])
    tracking.set_xlabel("Tiempo [s]", color=COLOURS["muted"])
    tracking.set_ylabel("v [m/s]", color=COLOURS["muted"])
    _style_axis(tracking)
    yaw_axis = tracking.twinx()
    yaw_axis.plot(
        log.times,
        log.reference_yaw_rates,
        color=COLOURS["amber"],
        linewidth=1.2,
        linestyle="--",
        label=r"$\omega_\mathrm{ref}$",
    )
    yaw_axis.plot(
        log.times,
        log.body_twists[:, 2],
        color=COLOURS["violet"],
        linewidth=1.6,
        label=r"$\omega$",
    )
    yaw_axis.set_ylabel(r"$\omega$ [rad/s]", color=COLOURS["muted"])
    yaw_axis.tick_params(colors=COLOURS["muted"], labelsize=8)
    yaw_axis.spines["top"].set_visible(False)
    tracking_handles, tracking_labels = tracking.get_legend_handles_labels()
    yaw_handles, yaw_labels = yaw_axis.get_legend_handles_labels()
    tracking.legend(
        tracking_handles + yaw_handles,
        tracking_labels + yaw_labels,
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    slip_magnitude = np.linalg.norm(log.slips, axis=2)
    diagnostics.plot(
        log.times,
        slip_magnitude[:, 0],
        color=COLOURS["teal"],
        linewidth=1.6,
        label="Slip oruga 1",
    )
    diagnostics.plot(
        log.times,
        slip_magnitude[:, 1],
        color=COLOURS["violet"],
        linewidth=1.6,
        label="Slip oruga 2",
    )
    diagnostics.plot(
        log.times,
        log.stability_margins,
        color=COLOURS["amber"],
        linewidth=2.0,
        label="Margen ZMP",
    )
    diagnostics.plot(
        log.times,
        log.clearances,
        color=COLOURS["ink"],
        linewidth=1.3,
        linestyle="--",
        label="Clearance",
    )
    diagnostics.axhline(
        mpc_parameters.minimum_stability_margin,
        color=COLOURS["red"],
        linewidth=1.0,
        linestyle=":",
        label="Margen mínimo",
    )
    diagnostics.set_title("Deslizamiento y seguridad", loc="left", color=COLOURS["ink"])
    diagnostics.set_xlabel("Tiempo [s]", color=COLOURS["muted"])
    diagnostics.set_ylabel("Magnitud [m o m/s]", color=COLOURS["muted"])
    _style_axis(diagnostics)
    diagnostics.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=2.2,
        columnspacing=1.0,
    )

    summary = log.summary()
    figure.text(
        0.97,
        0.947,
        (
            f"clearance mín. {summary['minimum_clearance_m']:.3f} m   ·   "
            f"estabilidad mín. {summary['minimum_stability_margin_m']:.3f} m   ·   "
            f"solve medio {summary['mean_solve_time_s']:.3f} s"
        ),
        ha="right",
        va="center",
        fontsize=9,
        color=COLOURS["muted"],
    )

    figure.savefig(output, dpi=170, facecolor=figure.get_facecolor())
    if show:
        plt.show()
    plt.close(figure)
    return output
