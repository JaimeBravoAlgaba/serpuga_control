"""Interactive real-time replay of a closed-loop SERPUGA simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button

from .config import MPCParameters
from .corridor import StraightGapCorridor
from .playback import PlaybackController
from .robot import RobotDescription
from .simulation import SimulationLog
from .visualization import COLOURS


def has_interactive_backend() -> bool:
    """Return whether the active Matplotlib backend can create a GUI window."""

    backend = str(matplotlib.get_backend()).lower()
    non_interactive = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
    return backend not in non_interactive and "inline" not in backend


class LiveSimulationPlayer:
    """Matplotlib player with real-time pacing and frame navigation.

    The NMPC result is replayed at the controller sampling period.  Because the
    full history is retained, the user can pause and inspect adjacent frames
    without recomputing the optimisation.
    """

    def __init__(
        self,
        log: SimulationLog,
        robot: RobotDescription,
        corridor: StraightGapCorridor,
        mpc_parameters: MPCParameters,
    ) -> None:
        if log.states.ndim != 2 or log.states.shape[0] < 1:
            raise ValueError("The simulation log must contain at least one state")

        self.log = log
        self.robot = robot
        self.corridor = corridor
        self.mpc_parameters = mpc_parameters
        self.dt = float(mpc_parameters.dt)
        self.playback = PlaybackController(frame_count=log.states.shape[0])
        self._timer: Any | None = None
        self._closed = False

        self._state_times = np.arange(log.states.shape[0], dtype=float) * self.dt
        reference_tangents = np.column_stack(
            (
                np.cos(log.reference_poses[:, 2]),
                np.sin(log.reference_poses[:, 2]),
            )
        )
        self._forward_speed = np.sum(log.world_velocities * reference_tangents, axis=1)
        self._slip_magnitude = np.linalg.norm(log.slips, axis=2)

        self._build_figure()
        self._render_frame(force_draw=True)

    def _build_figure(self) -> None:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "axes.titleweight": "semibold",
                "axes.titlesize": 11,
            }
        )
        self.figure = plt.figure(figsize=(15.6, 9.0), facecolor="#F7F9FC")
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("SERPUGA · Simulación MPC en tiempo real")

        grid = self.figure.add_gridspec(
            3,
            2,
            width_ratios=[1.55, 1.0],
            height_ratios=[1.0, 0.92, 1.0],
            left=0.055,
            right=0.97,
            top=0.86,
            bottom=0.20,
            hspace=0.48,
            wspace=0.24,
        )
        self.scene = self.figure.add_subplot(grid[:, 0])
        self.configuration = self.figure.add_subplot(grid[0, 1])
        self.tracking = self.figure.add_subplot(grid[1, 1])
        self.diagnostics = self.figure.add_subplot(grid[2, 1])

        self.figure.suptitle(
            "SERPUGA · Simulación MPC en tiempo real",
            x=0.055,
            y=0.955,
            ha="left",
            fontsize=19,
            color=COLOURS["ink"],
            weight="bold",
        )
        self.figure.text(
            0.055,
            0.916,
            "Seguimiento · deslizamiento · adaptación al hueco · estabilidad lateral",
            ha="left",
            fontsize=10.5,
            color=COLOURS["muted"],
        )
        self.status_text = self.figure.text(
            0.97,
            0.946,
            "",
            ha="right",
            va="center",
            fontsize=10,
            color=COLOURS["teal_dark"],
            weight="semibold",
        )

        self._build_scene()
        self._build_configuration_plot()
        self._build_tracking_plot()
        self._build_diagnostics_plot()
        self._build_controls()

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

    @staticmethod
    def _style_axis(axis: plt.Axes) -> None:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(COLOURS["grid"])
        axis.tick_params(colors=COLOURS["muted"], labelsize=8)
        axis.grid(True, color=COLOURS["grid"], linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)

    @staticmethod
    def _style_secondary_axis(axis: plt.Axes) -> None:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_color(COLOURS["grid"])
        axis.tick_params(colors=COLOURS["muted"], labelsize=8)

    def _build_scene(self) -> None:
        x_min = min(-0.35, float(np.min(self.log.states[:, 0])) - 0.35)
        x_max = float(np.max(self.log.states[:, 0])) + 0.45
        has_narrowing = self.corridor.gap_width < self.corridor.open_width - 1.0e-9
        if has_narrowing:
            x_max = max(x_max, self.corridor.gap_end + 0.6)
        path_y_min = min(
            float(np.min(self.log.states[:, 1])),
            float(np.min(self.log.reference_poses[:, 1])),
        )
        path_y_max = max(
            float(np.max(self.log.states[:, 1])),
            float(np.max(self.log.reference_poses[:, 1])),
        )
        y_centre = 0.5 * (path_y_min + path_y_max)
        half_height = max(0.72, 0.5 * (path_y_max - path_y_min) + 0.62)
        y_min = y_centre - half_height
        y_max = y_centre + half_height

        wall_x, upper_wall, lower_wall = self.corridor.wall_profiles(x_min, x_max)
        self.scene.fill_between(
            wall_x,
            upper_wall,
            y_max,
            color=COLOURS["wall"],
            zorder=0,
        )
        self.scene.fill_between(
            wall_x,
            y_min,
            lower_wall,
            color=COLOURS["wall"],
            zorder=0,
        )
        self.scene.plot(wall_x, upper_wall, color=COLOURS["ink"], linewidth=1.3)
        self.scene.plot(wall_x, lower_wall, color=COLOURS["ink"], linewidth=1.3)
        if has_narrowing:
            self.scene.axvspan(
                self.corridor.gap_start,
                self.corridor.gap_end,
                color=COLOURS["amber"],
                alpha=0.10,
            )
        self.scene.plot(
            self.log.reference_poses[:, 0],
            self.log.reference_poses[:, 1],
            linestyle=(0, (4, 4)),
            color=COLOURS["muted"],
            linewidth=1.5,
            label="Referencia",
        )
        (self.executed_path,) = self.scene.plot(
            [],
            [],
            color=COLOURS["teal_dark"],
            linewidth=2.5,
            label="Trayectoria ejecutada",
            zorder=3,
        )
        (self.predicted_path,) = self.scene.plot(
            [],
            [],
            color=COLOURS["violet"],
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            label="Horizonte MPC",
            zorder=3,
        )

        initial_state = self.log.states[0]
        connector = np.asarray(self.robot.connector_vertices_world(initial_state))
        self.connector_patch = Polygon(
            connector,
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
                np.asarray(self.robot.track_vertices_world(initial_state, index)),
                closed=True,
                facecolor=colour,
                edgecolor=COLOURS["ink"],
                linewidth=1.2,
                zorder=7,
            )
            self.scene.add_patch(patch)
            self.track_patches.append(patch)
            (direction_line,) = self.scene.plot(
                [],
                [],
                color=COLOURS["white"],
                linewidth=2.0,
                marker=">",
                markevery=[1],
                markersize=6,
                zorder=8,
            )
            self.direction_lines.append(direction_line)

        (self.body_marker,) = self.scene.plot(
            [],
            [],
            marker="o",
            markersize=4,
            color=COLOURS["ink"],
            linestyle="none",
            zorder=9,
        )
        (self.com_marker,) = self.scene.plot(
            [],
            [],
            marker="x",
            markersize=7,
            markeredgewidth=1.8,
            color=COLOURS["amber"],
            linestyle="none",
            label="Centro de masa",
            zorder=9,
        )
        self.telemetry_text = self.scene.text(
            0.025,
            0.975,
            "",
            transform=self.scene.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            color=COLOURS["ink"],
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": COLOURS["white"],
                "edgecolor": COLOURS["grid"],
                "alpha": 0.94,
            },
            zorder=12,
        )

        self.scene.set_title(
            "Vista superior · estado actual", loc="left", color=COLOURS["ink"]
        )
        self.scene.set_xlabel("x [m]", color=COLOURS["muted"])
        self.scene.set_ylabel("y [m]", color=COLOURS["muted"])
        self.scene.set_xlim(x_min, x_max)
        self.scene.set_ylim(y_min, y_max)
        self.scene.set_aspect("equal", adjustable="box")
        self._style_axis(self.scene)
        self.scene.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=4,
            frameon=False,
            fontsize=8,
        )

    def _build_configuration_plot(self) -> None:
        (self.q1_line,) = self.configuration.plot(
            [], [], color=COLOURS["teal"], linewidth=2.0, label=r"$q_1$"
        )
        (self.q2_line,) = self.configuration.plot(
            [], [], color=COLOURS["violet"], linewidth=2.0, label=r"$q_2$"
        )
        self.configuration.set_title(
            "Configuración y anchura", loc="left", color=COLOURS["ink"]
        )
        self.configuration.set_ylabel("Ángulo [deg]", color=COLOURS["muted"])
        self.configuration.set_xlim(0.0, max(self.dt, self._state_times[-1]))
        q_degrees = np.rad2deg(self.log.states[:, 3:5])
        q_padding = 6.0
        self.configuration.set_ylim(
            float(np.min(q_degrees)) - q_padding,
            float(np.max(q_degrees)) + q_padding,
        )
        self._style_axis(self.configuration)
        self.width_axis = self.configuration.twinx()
        (self.robot_width_line,) = self.width_axis.plot(
            [],
            [],
            color=COLOURS["ink"],
            linewidth=1.3,
            linestyle="--",
            label="Robot",
        )
        (self.corridor_width_line,) = self.width_axis.plot(
            [],
            [],
            color=COLOURS["amber"],
            linewidth=1.6,
            label="Corredor",
        )
        maximum_width = max(
            float(np.max(self.log.robot_widths)),
            float(np.max(self.log.corridor_widths)),
        )
        self.width_axis.set_ylim(0.0, maximum_width * 1.18)
        self.width_axis.set_ylabel("Anchura [m]", color=COLOURS["muted"])
        self._style_secondary_axis(self.width_axis)
        self.configuration_cursor = self.configuration.axvline(
            0.0, color=COLOURS["muted"], linewidth=0.9, alpha=0.65
        )
        handles_a, labels_a = self.configuration.get_legend_handles_labels()
        handles_b, labels_b = self.width_axis.get_legend_handles_labels()
        self.configuration.legend(
            handles_a + handles_b,
            labels_a + labels_b,
            loc="lower left",
            ncol=2,
            frameon=False,
            fontsize=7.5,
        )

    def _build_tracking_plot(self) -> None:
        self.tracking.plot(
            self.log.times,
            self.log.reference_speeds,
            color=COLOURS["muted"],
            linewidth=1.2,
            linestyle="--",
            label=r"$v_\mathrm{ref}$",
        )
        (self.speed_line,) = self.tracking.plot(
            [], [], color=COLOURS["teal_dark"], linewidth=2.0, label=r"$v$"
        )
        self.tracking.set_title(
            "Seguimiento de velocidad", loc="left", color=COLOURS["ink"]
        )
        self.tracking.set_ylabel("v [m/s]", color=COLOURS["muted"])
        self.tracking.set_xlim(0.0, max(self.dt, self._state_times[-1]))
        speed_values = np.concatenate((self.log.reference_speeds, self._forward_speed))
        speed_padding = max(0.04, 0.12 * float(np.ptp(speed_values)))
        self.tracking.set_ylim(
            min(0.0, float(np.min(speed_values)) - speed_padding),
            float(np.max(speed_values)) + speed_padding,
        )
        self._style_axis(self.tracking)
        self.yaw_axis = self.tracking.twinx()
        self.yaw_axis.plot(
            self.log.times,
            self.log.reference_yaw_rates,
            color=COLOURS["amber"],
            linewidth=1.2,
            linestyle="--",
            label=r"$\omega_\mathrm{ref}$",
        )
        (self.yaw_rate_line,) = self.yaw_axis.plot(
            [], [], color=COLOURS["violet"], linewidth=1.6, label=r"$\omega$"
        )
        yaw_values = np.concatenate(
            (self.log.reference_yaw_rates, self.log.body_twists[:, 2])
        )
        yaw_padding = max(0.04, 0.12 * float(np.ptp(yaw_values)))
        self.yaw_axis.set_ylim(
            min(0.0, float(np.min(yaw_values)) - yaw_padding),
            max(0.04, float(np.max(yaw_values)) + yaw_padding),
        )
        self.yaw_axis.set_ylabel(r"$\omega$ [rad/s]", color=COLOURS["muted"])
        self._style_secondary_axis(self.yaw_axis)
        self.tracking_cursor = self.tracking.axvline(
            0.0, color=COLOURS["muted"], linewidth=0.9, alpha=0.65
        )
        handles_a, labels_a = self.tracking.get_legend_handles_labels()
        handles_b, labels_b = self.yaw_axis.get_legend_handles_labels()
        self.tracking.legend(
            handles_a + handles_b,
            labels_a + labels_b,
            loc="lower right",
            ncol=2,
            frameon=False,
            fontsize=7.5,
        )

    def _build_diagnostics_plot(self) -> None:
        (self.slip1_line,) = self.diagnostics.plot(
            [], [], color=COLOURS["teal"], linewidth=1.6, label="Slip oruga 1"
        )
        (self.slip2_line,) = self.diagnostics.plot(
            [], [], color=COLOURS["violet"], linewidth=1.6, label="Slip oruga 2"
        )
        (self.stability_line,) = self.diagnostics.plot(
            [], [], color=COLOURS["amber"], linewidth=2.0, label="Margen soporte"
        )
        (self.clearance_line,) = self.diagnostics.plot(
            [],
            [],
            color=COLOURS["ink"],
            linewidth=1.3,
            linestyle="--",
            label="Clearance",
        )
        self.diagnostics.axhline(
            0.0,
            color=COLOURS["red"],
            linewidth=1.0,
            linestyle=":",
            label="Límite del soporte",
        )
        diagnostic_values = np.concatenate(
            (
                self._slip_magnitude.ravel(),
                self.log.stability_margins,
                self.log.clearances,
                np.array([0.0]),
            )
        )
        diagnostic_maximum = max(0.08, float(np.max(diagnostic_values)) * 1.15)
        self.diagnostics.set_ylim(
            min(-0.01, float(np.min(diagnostic_values))), diagnostic_maximum
        )
        self.diagnostics.set_xlim(0.0, max(self.dt, self._state_times[-1]))
        self.diagnostics.set_title(
            "Diagnósticos (no restringidos)", loc="left", color=COLOURS["ink"]
        )
        self.diagnostics.set_xlabel("Tiempo [s]", color=COLOURS["muted"])
        self.diagnostics.set_ylabel("m o m/s", color=COLOURS["muted"])
        self._style_axis(self.diagnostics)
        self.diagnostics_cursor = self.diagnostics.axvline(
            0.0, color=COLOURS["muted"], linewidth=0.9, alpha=0.65
        )
        self.diagnostics.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=7,
            handlelength=2.0,
            columnspacing=0.9,
        )

    def _build_controls(self) -> None:
        self.progress_axis = self.figure.add_axes([0.055, 0.137, 0.915, 0.012])
        self.progress_axis.set_xlim(0.0, 1.0)
        self.progress_axis.set_ylim(0.0, 1.0)
        self.progress_axis.axis("off")
        self.progress_axis.add_patch(
            Rectangle(
                (0.0, 0.16), 1.0, 0.68, facecolor=COLOURS["grid"], edgecolor="none"
            )
        )
        self.progress_patch = Rectangle(
            (0.0, 0.16),
            0.0,
            0.68,
            facecolor=COLOURS["teal_dark"],
            edgecolor="none",
        )
        self.progress_axis.add_patch(self.progress_patch)

        button_y = 0.057
        button_height = 0.050
        button_width = 0.135
        gap = 0.018
        total_width = 3.0 * button_width + 2.0 * gap
        start_x = 0.5 - 0.5 * total_width
        self.backward_axis = self.figure.add_axes(
            [start_x, button_y, button_width, button_height]
        )
        self.play_axis = self.figure.add_axes(
            [start_x + button_width + gap, button_y, button_width, button_height]
        )
        self.forward_axis = self.figure.add_axes(
            [
                start_x + 2.0 * (button_width + gap),
                button_y,
                button_width,
                button_height,
            ]
        )
        self.backward_button = Button(
            self.backward_axis,
            "Paso atrás",
            color=COLOURS["white"],
            hovercolor=COLOURS["wall"],
        )
        self.play_button = Button(
            self.play_axis,
            "Pausa",
            color=COLOURS["teal_dark"],
            hovercolor=COLOURS["teal"],
        )
        self.forward_button = Button(
            self.forward_axis,
            "Paso adelante",
            color=COLOURS["white"],
            hovercolor=COLOURS["wall"],
        )
        self.play_button.label.set_color(COLOURS["white"])
        for button in (self.backward_button, self.play_button, self.forward_button):
            button.label.set_fontsize(9)
        for axis in (self.backward_axis, self.play_axis, self.forward_axis):
            for spine in axis.spines.values():
                spine.set_color(COLOURS["grid"])

        self.backward_button.on_clicked(self._on_backward)
        self.play_button.on_clicked(self._on_toggle)
        self.forward_button.on_clicked(self._on_forward)
        self.figure.text(
            0.97,
            0.081,
            "Teclado: espacio · ← · →",
            ha="right",
            va="center",
            fontsize=8.5,
            color=COLOURS["muted"],
        )

    def _track_direction_segment(
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
        centre_body = np.asarray(
            self.robot.track_center_body(state[3 + track_index], track_index),
            dtype=float,
        )
        centre_world = state[0:2] + rotation @ centre_body
        local_direction = np.array(
            [np.cos(state[3 + track_index]), np.sin(state[3 + track_index])],
            dtype=float,
        )
        direction_sign = -1.0 if track_speed < 0.0 else 1.0
        end_world = centre_world + 0.13 * direction_sign * (rotation @ local_direction)
        return centre_world, end_world

    def _render_frame(self, *, force_draw: bool = False) -> None:
        index = self.playback.index
        state = self.log.states[index]
        current_time = self._state_times[index]
        telemetry_index = min(index, self.log.times.size - 1)
        measurement_stop = min(index + 1, self.log.times.size)
        track_speeds = (
            self.log.actuator_commands[telemetry_index, 2:4]
            if self.log.actuator_commands.size
            else np.ones(2, dtype=float)
        )

        self.executed_path.set_data(
            self.log.states[: index + 1, 0],
            self.log.states[: index + 1, 1],
        )
        if index < len(self.log.predicted_states):
            prediction = self.log.predicted_states[index]
            self.predicted_path.set_data(prediction[:, 0], prediction[:, 1])
        else:
            self.predicted_path.set_data([], [])

        self.connector_patch.set_xy(
            np.asarray(self.robot.connector_vertices_world(state), dtype=float)
        )
        for track_index, patch in enumerate(self.track_patches):
            patch.set_xy(
                np.asarray(
                    self.robot.track_vertices_world(state, track_index), dtype=float
                )
            )
            start, end = self._track_direction_segment(
                state,
                track_index,
                float(track_speeds[track_index]),
            )
            self.direction_lines[track_index].set_data(
                [start[0], end[0]], [start[1], end[1]]
            )
        self.body_marker.set_data([state[0]], [state[1]])
        centre_of_mass = np.asarray(self.robot.centre_of_mass_world(state), dtype=float)
        self.com_marker.set_data([centre_of_mass[0]], [centre_of_mass[1]])

        self.q1_line.set_data(
            self._state_times[: index + 1], np.rad2deg(self.log.states[: index + 1, 3])
        )
        self.q2_line.set_data(
            self._state_times[: index + 1], np.rad2deg(self.log.states[: index + 1, 4])
        )
        measurement_times = self.log.times[:measurement_stop]
        self.robot_width_line.set_data(
            measurement_times, self.log.robot_widths[:measurement_stop]
        )
        self.corridor_width_line.set_data(
            measurement_times, self.log.corridor_widths[:measurement_stop]
        )
        self.speed_line.set_data(
            measurement_times, self._forward_speed[:measurement_stop]
        )
        self.yaw_rate_line.set_data(
            measurement_times, self.log.body_twists[:measurement_stop, 2]
        )
        self.slip1_line.set_data(
            measurement_times, self._slip_magnitude[:measurement_stop, 0]
        )
        self.slip2_line.set_data(
            measurement_times, self._slip_magnitude[:measurement_stop, 1]
        )
        self.stability_line.set_data(
            measurement_times, self.log.stability_margins[:measurement_stop]
        )
        self.clearance_line.set_data(
            measurement_times, self.log.clearances[:measurement_stop]
        )
        for cursor in (
            self.configuration_cursor,
            self.tracking_cursor,
            self.diagnostics_cursor,
        ):
            cursor.set_xdata([current_time, current_time])

        q_degrees = np.rad2deg(state[3:5])
        slip_values = self._slip_magnitude[telemetry_index]
        self.telemetry_text.set_text(
            f"vx, vy   {self.log.controls[telemetry_index, 0]: .3f}, "
            f"{self.log.controls[telemetry_index, 1]: .3f} m/s\n"
            f"v1, v2  {track_speeds[0]: .3f}, {track_speeds[1]: .3f} m/s\n"
            f"v       {self._forward_speed[telemetry_index]: .3f} / "
            f"{self.log.reference_speeds[telemetry_index]:.3f} m/s\n"
            f"omega   {self.log.body_twists[telemetry_index, 2]: .3f} / "
            f"{self.log.reference_yaw_rates[telemetry_index]:.3f} rad/s\n"
            f"q1, q2  {q_degrees[0]: .1f}, {q_degrees[1]: .1f} deg\n"
            f"ancho   {self.log.robot_widths[telemetry_index]: .3f} / "
            f"{self.log.corridor_widths[telemetry_index]:.3f} m\n"
            f"slip    {slip_values[0]: .3f}, {slip_values[1]:.3f} m/s\n"
            f"soporte {self.log.stability_margins[telemetry_index]: .3f} m   "
            f"clear {self.log.clearances[telemetry_index]:.3f} m"
        )

        if self.playback.playing:
            status = "EN MARCHA"
            status_colour = COLOURS["teal_dark"]
            button_label = "Pausa"
        elif self.playback.at_end:
            status = "FINALIZADA"
            status_colour = COLOURS["ink"]
            button_label = "Repetir"
        else:
            status = "PAUSA"
            status_colour = COLOURS["amber"]
            button_label = "Reanudar"
        self.status_text.set_text(
            f"{status}   ·   t = {current_time:05.2f} / {self._state_times[-1]:.2f} s   ·   1x"
        )
        self.status_text.set_color(status_colour)
        self.play_button.label.set_text(button_label)
        self.progress_patch.set_width(self.playback.progress)

        if force_draw:
            self.figure.canvas.draw()
        else:
            self.figure.canvas.draw_idle()

    def toggle_playback(self) -> None:
        self.playback.toggle()
        self._render_frame()

    def step_backward(self) -> int:
        index = self.playback.step_backward()
        self._render_frame()
        return index

    def step_forward(self) -> int:
        index = self.playback.step_forward()
        self._render_frame()
        return index

    def seek(self, index: int, *, pause: bool = True) -> int:
        selected = self.playback.seek(index, pause=pause)
        self._render_frame()
        return selected

    def _on_toggle(self, _event: Any) -> None:
        self.toggle_playback()

    def _on_backward(self, _event: Any) -> None:
        self.step_backward()

    def _on_forward(self, _event: Any) -> None:
        self.step_forward()

    def _on_key_press(self, event: Any) -> None:
        if event.key == " ":
            self.toggle_playback()
        elif event.key == "left":
            self.step_backward()
        elif event.key == "right":
            self.step_forward()

    def _on_timer(self) -> None:
        if self._closed:
            return
        if self.playback.tick():
            self._render_frame()

    def _on_close(self, _event: Any) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.stop()

    def save_frame(
        self, output_path: str | Path, frame_index: int | None = None
    ) -> Path:
        """Save one frame of the live UI, including its playback controls."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if frame_index is None:
            frame_index = int(np.argmin(self.log.robot_widths))
        previous_index = self.playback.index
        was_playing = self.playback.playing
        self.seek(frame_index)
        self.figure.savefig(output, dpi=170, facecolor=self.figure.get_facecolor())
        self.playback.index = previous_index
        self.playback.playing = was_playing
        self._render_frame()
        return output

    def save_animation(self, output_path: str | Path, *, dpi: int = 100) -> Path:
        """Export the same 1x playback shown by the interactive window."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frames_per_second = 1.0 / self.dt
        if output.suffix.lower() == ".mp4":
            writer = FFMpegWriter(
                fps=frames_per_second,
                bitrate=2_000,
                metadata={"title": "SERPUGA MPC simulation"},
            )
        elif output.suffix.lower() == ".gif":
            writer = PillowWriter(fps=frames_per_second)
        else:
            raise ValueError("Animation output must use the .mp4 or .gif extension")

        previous_index = self.playback.index
        was_playing = self.playback.playing
        with writer.saving(self.figure, str(output), dpi=dpi):
            for index in range(self.playback.frame_count):
                self.playback.index = index
                self.playback.playing = index < self.playback.last_index
                self._render_frame(force_draw=True)
                writer.grab_frame(facecolor=self.figure.get_facecolor())
        self.playback.index = previous_index
        self.playback.playing = was_playing
        self._render_frame()
        return output

    def show(self) -> None:
        """Open the player and start advancing one frame every controller step."""

        if not has_interactive_backend():
            raise RuntimeError(
                "Matplotlib is using a non-interactive backend. Run on a desktop "
                "with Tk/Qt support, or pass --headless for a console-only run."
            )
        self._timer = self.figure.canvas.new_timer(
            interval=max(1, round(1000 * self.dt))
        )
        self._timer.add_callback(self._on_timer)
        self._timer.start()
        plt.show()

    def close(self) -> None:
        self._on_close(None)
        plt.close(self.figure)
