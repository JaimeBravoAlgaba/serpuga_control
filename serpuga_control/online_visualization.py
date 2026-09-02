"""Matplotlib scene used by the online Tk application."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from .configuration import ApplicationConfiguration
from .robot import RobotDescription
from .runtime import build_trajectory
from .simulation import SimulationLog
from .visualization import COLOURS


class OnlineSimulationPlot:
    """Mutable top view and telemetry plots for a stepwise simulation."""

    def __init__(self, figure: Figure, draw: Callable[[], None]) -> None:
        self.figure = figure
        self._draw = draw
        self.configuration: ApplicationConfiguration | None = None

    @staticmethod
    def _style_axis(axis) -> None:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(COLOURS["grid"])
        axis.tick_params(colors=COLOURS["muted"], labelsize=8)
        axis.grid(True, color=COLOURS["grid"], linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)

    def reset(self, configuration: ApplicationConfiguration) -> None:
        self.configuration = configuration
        self.robot = RobotDescription(configuration.robot)
        self.corridor = configuration.corridor
        self.trajectory = build_trajectory(configuration)
        self.duration = configuration.simulation.duration

        self.figure.clear()
        self.figure.set_facecolor("#F7F9FC")
        grid = self.figure.add_gridspec(
            2,
            2,
            height_ratios=[2.25, 1.0],
            left=0.075,
            right=0.965,
            top=0.93,
            bottom=0.09,
            hspace=0.34,
            wspace=0.30,
        )
        self.scene = self.figure.add_subplot(grid[0, :])
        self.tracking = self.figure.add_subplot(grid[1, 0])
        self.safety = self.figure.add_subplot(grid[1, 1])

        self._build_scene()
        self._build_tracking()
        self._build_safety()
        self.update_initial(configuration.simulation.initial_state)

    def _scene_bounds(self) -> tuple[float, float, float, float]:
        active = self.trajectory.times <= self.duration + 1.0e-9
        poses = self.trajectory.poses[active]
        x_min = min(float(np.min(poses[:, 0])), self.corridor.gap_start) - 0.55
        x_max = max(float(np.max(poses[:, 0])), self.corridor.gap_end) + 0.65
        path_min = float(np.min(poses[:, 1]))
        path_max = float(np.max(poses[:, 1]))
        if self.corridor.open_width <= 2.0:
            y_min = min(path_min - 0.55, self.corridor.centre_y - 0.65)
            y_max = max(path_max + 0.55, self.corridor.centre_y + 0.65)
        else:
            y_min = path_min - 0.75
            y_max = path_max + 0.75
        return x_min, x_max, y_min, y_max

    def _build_scene(self) -> None:
        x_min, x_max, y_min, y_max = self._scene_bounds()
        wall_x, upper_wall, lower_wall = self.corridor.wall_profiles(x_min, x_max)
        self.scene.fill_between(
            wall_x, upper_wall, y_max, color=COLOURS["wall"], zorder=0
        )
        self.scene.fill_between(
            wall_x, y_min, lower_wall, color=COLOURS["wall"], zorder=0
        )
        self.scene.plot(wall_x, upper_wall, color=COLOURS["ink"], linewidth=1.2)
        self.scene.plot(wall_x, lower_wall, color=COLOURS["ink"], linewidth=1.2)
        if self.corridor.gap_width < self.corridor.open_width - 1.0e-9:
            self.scene.axvspan(
                self.corridor.gap_start,
                self.corridor.gap_end,
                color=COLOURS["amber"],
                alpha=0.10,
            )

        active = self.trajectory.times <= self.duration + 1.0e-9
        self.scene.plot(
            self.trajectory.poses[active, 0],
            self.trajectory.poses[active, 1],
            linestyle=(0, (4, 4)),
            color=COLOURS["muted"],
            linewidth=1.4,
            label="Referencia",
        )
        (self.executed_path,) = self.scene.plot(
            [], [], color=COLOURS["teal_dark"], linewidth=2.5, label="Robot"
        )
        (self.predicted_path,) = self.scene.plot(
            [],
            [],
            color=COLOURS["violet"],
            linewidth=1.4,
            linestyle="--",
            label="Horizonte MPC",
        )

        initial = self.configuration.simulation.initial_state
        self.connector_patch = Polygon(
            np.asarray(self.robot.connector_vertices_world(initial)),
            closed=True,
            facecolor=COLOURS["ink"],
            edgecolor=COLOURS["white"],
            linewidth=0.9,
            zorder=6,
        )
        self.scene.add_patch(self.connector_patch)
        self.track_patches: list[Polygon] = []
        self.direction_lines: list[Line2D] = []
        for index, colour in enumerate((COLOURS["teal"], COLOURS["violet"])):
            patch = Polygon(
                np.asarray(self.robot.track_vertices_world(initial, index)),
                closed=True,
                facecolor=colour,
                edgecolor=COLOURS["ink"],
                linewidth=1.2,
                zorder=7,
            )
            self.scene.add_patch(patch)
            self.track_patches.append(patch)
            (line,) = self.scene.plot(
                [],
                [],
                color=COLOURS["white"],
                linewidth=2.0,
                marker=">",
                markevery=[1],
                markersize=6,
                zorder=8,
            )
            self.direction_lines.append(line)

        (self.com_marker,) = self.scene.plot(
            [],
            [],
            marker="x",
            markersize=7,
            markeredgewidth=1.8,
            color=COLOURS["amber"],
            linestyle="none",
            zorder=9,
        )
        self.telemetry = self.scene.text(
            0.018,
            0.975,
            "",
            transform=self.scene.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
            color=COLOURS["ink"],
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": COLOURS["white"],
                "edgecolor": COLOURS["grid"],
                "alpha": 0.95,
            },
            zorder=12,
        )
        self.scene.set_title("Simulación online · vista superior", loc="left")
        self.scene.set_xlabel("x [m]")
        self.scene.set_ylabel("y [m]")
        self.scene.set_xlim(x_min, x_max)
        self.scene.set_ylim(y_min, y_max)
        self.scene.set_aspect("equal", adjustable="box")
        self._style_axis(self.scene)
        self.scene.legend(loc="lower right", frameon=False, fontsize=8, ncol=3)

    def _build_tracking(self) -> None:
        self.tracking.set_title("Seguimiento", loc="left", fontsize=10)
        self.tracking.set_xlabel("Tiempo [s]")
        self.tracking.set_ylabel("v [m/s]")
        self.tracking.set_xlim(0.0, self.duration)
        (self.reference_speed_line,) = self.tracking.plot(
            [], [], color=COLOURS["muted"], linestyle="--", label="v ref"
        )
        (self.speed_line,) = self.tracking.plot(
            [], [], color=COLOURS["teal_dark"], linewidth=2.0, label="v"
        )
        self.yaw_axis = self.tracking.twinx()
        (self.reference_yaw_line,) = self.yaw_axis.plot(
            [], [], color=COLOURS["amber"], linestyle="--", label="ω ref"
        )
        (self.yaw_line,) = self.yaw_axis.plot(
            [], [], color=COLOURS["violet"], linewidth=1.6, label="ω"
        )
        self.yaw_axis.set_ylabel("ω [rad/s]")
        self._style_axis(self.tracking)
        handles_a, labels_a = self.tracking.get_legend_handles_labels()
        handles_b, labels_b = self.yaw_axis.get_legend_handles_labels()
        self.tracking.legend(
            handles_a + handles_b,
            labels_a + labels_b,
            loc="upper right",
            frameon=False,
            fontsize=7,
            ncol=2,
        )

    def _build_safety(self) -> None:
        self.safety.set_title("Diagnósticos (no restringidos)", loc="left", fontsize=10)
        self.safety.set_xlabel("Tiempo [s]")
        self.safety.set_ylabel("m o m/s")
        self.safety.set_xlim(0.0, self.duration)
        (self.slip_1_line,) = self.safety.plot(
            [], [], color=COLOURS["teal"], label="Slip 1"
        )
        (self.slip_2_line,) = self.safety.plot(
            [], [], color=COLOURS["violet"], label="Slip 2"
        )
        (self.stability_line,) = self.safety.plot(
            [], [], color=COLOURS["amber"], linewidth=1.8, label="Margen soporte"
        )
        (self.clearance_line,) = self.safety.plot(
            [], [], color=COLOURS["ink"], linestyle="--", label="Holgura"
        )
        self.safety.axhline(0.0, color=COLOURS["red"], linestyle=":", linewidth=1.0)
        self._style_axis(self.safety)
        self.safety.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)

    def _track_segment(
        self,
        state: np.ndarray,
        track_index: int,
        track_speed: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        yaw = state[2]
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=float,
        )
        centre = state[0:2] + rotation @ np.asarray(
            self.robot.track_center_body(state[3 + track_index], track_index)
        )
        local_direction = np.array(
            [np.cos(state[3 + track_index]), np.sin(state[3 + track_index])]
        )
        direction_sign = -1.0 if track_speed < 0.0 else 1.0
        return centre, centre + 0.13 * direction_sign * (rotation @ local_direction)

    def _update_robot(
        self,
        state: np.ndarray,
        track_speeds: np.ndarray | None = None,
    ) -> None:
        self.connector_patch.set_xy(
            np.asarray(self.robot.connector_vertices_world(state), dtype=float)
        )
        if track_speeds is None:
            track_speeds = np.ones(2, dtype=float)
        for index, patch in enumerate(self.track_patches):
            patch.set_xy(
                np.asarray(self.robot.track_vertices_world(state, index), dtype=float)
            )
            start, end = self._track_segment(state, index, float(track_speeds[index]))
            self.direction_lines[index].set_data([start[0], end[0]], [start[1], end[1]])
        centre_of_mass = np.asarray(self.robot.centre_of_mass_world(state), dtype=float)
        self.com_marker.set_data([centre_of_mass[0]], [centre_of_mass[1]])

    def update_initial(self, state: np.ndarray) -> None:
        self._update_robot(np.asarray(state, dtype=float))
        self.executed_path.set_data([state[0]], [state[1]])
        self.predicted_path.set_data([], [])
        q = np.rad2deg(state[3:5])
        self.telemetry.set_text(
            "PREPARADO\n"
            f"x, y    {state[0]: .3f}, {state[1]: .3f} m\n"
            f"yaw     {np.rad2deg(state[2]): .1f} deg\n"
            f"q1, q2  {q[0]: .1f}, {q[1]: .1f} deg"
        )
        self._draw()

    @staticmethod
    def _autoscale_y(axis, minimum_span: float = 0.05) -> None:
        axis.relim()
        axis.autoscale_view(scalex=False, scaley=True)
        low, high = axis.get_ylim()
        if high - low < minimum_span:
            centre = 0.5 * (low + high)
            axis.set_ylim(centre - 0.5 * minimum_span, centre + 0.5 * minimum_span)

    def _expand_time_axis(self, current_time: float) -> None:
        _, high = self.tracking.get_xlim()
        if current_time <= high:
            return
        new_high = current_time + self.configuration.mpc.dt
        self.tracking.set_xlim(0.0, new_high)
        self.safety.set_xlim(0.0, new_high)

    def _expand_scene_to_state(self, state: np.ndarray) -> None:
        x_low, x_high = self.scene.get_xlim()
        y_low, y_high = self.scene.get_ylim()
        x_margin = 0.45
        y_margin = 0.35
        new_x_low = min(x_low, float(state[0]) - x_margin)
        new_x_high = max(x_high, float(state[0]) + x_margin)
        new_y_low = min(y_low, float(state[1]) - y_margin)
        new_y_high = max(y_high, float(state[1]) + y_margin)
        if (
            new_x_low != x_low
            or new_x_high != x_high
            or new_y_low != y_low
            or new_y_high != y_high
        ):
            self.scene.set_xlim(new_x_low, new_x_high)
            self.scene.set_ylim(new_y_low, new_y_high)

    def update_log(self, log: SimulationLog) -> None:
        if log.times.size == 0:
            return
        state = log.states[-1]
        current_control = log.controls[-1]
        track_speeds = current_control[2:4]
        self._expand_scene_to_state(state)
        self._expand_time_axis(float(log.times[-1] + self.configuration.mpc.dt))
        self._update_robot(state, track_speeds)
        self.executed_path.set_data(log.states[:, 0], log.states[:, 1])
        prediction = log.predicted_states[-1]
        self.predicted_path.set_data(prediction[:, 0], prediction[:, 1])

        tangents = np.column_stack(
            (np.cos(log.reference_poses[:, 2]), np.sin(log.reference_poses[:, 2]))
        )
        forward_speed = np.sum(log.world_velocities * tangents, axis=1)
        slip = np.linalg.norm(log.slips, axis=2)
        self.reference_speed_line.set_data(log.times, log.reference_speeds)
        self.speed_line.set_data(log.times, forward_speed)
        self.reference_yaw_line.set_data(log.times, log.reference_yaw_rates)
        self.yaw_line.set_data(log.times, log.body_twists[:, 2])
        self.slip_1_line.set_data(log.times, slip[:, 0])
        self.slip_2_line.set_data(log.times, slip[:, 1])
        self.stability_line.set_data(log.times, log.stability_margins)
        self.clearance_line.set_data(log.times, log.clearances)
        self._autoscale_y(self.tracking)
        self._autoscale_y(self.yaw_axis)
        self._autoscale_y(self.safety)

        q = np.rad2deg(state[3:5])
        q_cmd = np.rad2deg(current_control[0:2])
        index = -1
        mode = log.control_modes[index] if log.control_modes else "mpc"
        label = "MANUAL" if mode == "manual" else "ONLINE"
        body_twist = log.body_twists[index]
        self.telemetry.set_text(
            f"{label} · t={log.times[index] + self.configuration.mpc.dt:05.2f} s\n"
            f"vx, vy   {body_twist[0]: .3f}, {body_twist[1]: .3f} m/s\n"
            f"v1, v2   {track_speeds[0]: .3f}, {track_speeds[1]: .3f} m/s\n"
            f"q1*,q2*  {q_cmd[0]: .1f}, {q_cmd[1]: .1f} deg\n"
            f"v        {forward_speed[index]: .3f} / {log.reference_speeds[index]:.3f} m/s\n"
            f"omega    {body_twist[2]: .3f} / {log.reference_yaw_rates[index]:.3f} rad/s\n"
            f"q1, q2   {q[0]: .1f}, {q[1]: .1f} deg\n"
            f"slip     {slip[index, 0]: .3f}, {slip[index, 1]:.3f} m/s\n"
            f"soporte  {log.stability_margins[index]: .3f} m   "
            f"clear {log.clearances[index]:.3f} m"
        )
        self._draw()
